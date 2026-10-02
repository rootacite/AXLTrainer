from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm.auto import tqdm

try:
    from config import TrainConfig
    from dataset import LoraImageDataset
    from env import flush_memory
    from utils import sha1_text
    import control
    from device_swap import SwapContext, at_safe_point
except ImportError:
    from trainer.config import TrainConfig
    from trainer.dataset import LoraImageDataset
    from trainer.env import flush_memory
    from trainer.utils import sha1_text
    from trainer import control
    from trainer.device_swap import SwapContext, at_safe_point

_DEFAULT_PREFETCH_WORKERS = os.cpu_count() or 4
# The encode's peak does not follow this number: `prepare_encoding_devices` turns on slicing (which
# splits any batch into single-image slices) and tiling, so one image's tile is the whole working
# set. Measured on a 16GB RX 9070 XT, 9 images in 1408x768 buckets: 5.80 / 5.83 / 5.83 GB for batch
# 2 / 4 / 8, with the largest allocations identical to the byte. 4 keeps the calls fewest for no
# measurable memory; 8 was enough to OOM only while UNet/TEs were still resident, and they are
# offloaded before this runs.
_DEFAULT_ENCODE_BATCH_SIZE = 4


def prepare_encoding_devices(
    vae: torch.nn.Module,
    device: torch.device,
    swap_ctx: Optional[SwapContext] = None,
) -> None:
    """Encoding uses only the VAE. UNet + TEs must not sit on the GPU."""
    if swap_ctx is not None:
        if swap_ctx.denoise is not None:
            swap_ctx.denoise.to("cpu")
        for te in swap_ctx.text_encoders:
            if te is not None:
                te.to("cpu")
    vae.eval()
    vae.to(device=device)
    cfg = getattr(vae, "config", None)
    if cfg is not None and hasattr(cfg, "force_upcast"):
        cfg.force_upcast = False
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    flush_memory(device)


def _latent_generator(seed: int, cache_path: Path, device: torch.device) -> torch.Generator:
    """The generator one image's posterior sample is drawn from.

    Seeded by the run's seed and the image's own cache key, so a cached latent is a function of
    (image, geometry, seed): which batch the image landed in, how many batches went before it, and
    how many draws the pass has made so far no longer reach it. The key carries the absolute image
    path, so the same image in another folder is another entry — as it already was for the file name.
    It also keeps the sample off the global generator — the one the training step's noise, timesteps
    and dropout masks come from — so a pass that has to encode something no longer shifts them.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(int(sha1_text(f"{seed}:{cache_path.name}"), 16))
    return generator


def _sample_latents(latent_dist: Any, items: list[Dict[str, Any]], cfg: TrainConfig,
                    device: torch.device) -> torch.Tensor:
    """One posterior sample per image, each from that image's own generator."""
    seed = int(getattr(cfg, "seed", 0) or 0)
    parameters = getattr(latent_dist, "parameters", None)
    if parameters is not None:
        # The diagonal Gaussian keeps mean and logvar in one tensor, so one row is one image's whole
        # distribution; sampling it there is the same draw the batch-wide `sample()` would make.
        return torch.cat(
            [
                type(latent_dist)(parameters[offset:offset + 1]).sample(
                    generator=_latent_generator(seed, Path(item["cache_path"]), device)
                )
                for offset, item in enumerate(items)
            ],
            dim=0,
        )
    # An encoder that does not hand back the diagonal Gaussian has no per-image parameters to slice:
    # one draw for the whole batch, still from a generator seeded by the batch's own keys.
    keys = "+".join(sorted(Path(item["cache_path"]).name for item in items))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(sha1_text(f"{seed}:{keys}"), 16))
    return latent_dist.sample(generator=generator)


def _plan_batches(entries: list[Dict[str, Any]], batch_size: int) -> list[list[Dict[str, Any]]]:
    """The batches `entries` will be encoded in, in the order they will be encoded.

    Buckets are filled in record order, a batch holds one bucket's images only, and a bucket's
    batches run in the order its first image appears. Planning before encoding is what makes the
    sequence of `vae.encode` calls a property of the dataset instead of a property of the order in
    which the CPU threads happened to finish their images.
    """
    buckets: Dict[tuple[int, int], list[Dict[str, Any]]] = {}
    for entry in entries:
        buckets.setdefault(entry["bucket"], []).append(entry)
    return [
        group[offset:offset + batch_size]
        for group in buckets.values()
        for offset in range(0, len(group), batch_size)
    ]


@torch.no_grad()
def warm_latent_cache(
    dataset: LoraImageDataset,
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    prefetch_workers: Optional[int] = None,
    encode_batch_size: Optional[int] = None,
    swap_ctx: Optional[SwapContext] = None,
) -> bool:
    """Pre-encode image latents to disk when disk caching is enabled.

    The work is planned before a pixel is touched: `dataset.cache_entries()` says which images still
    need encoding and which bucket each belongs to, and `_plan_batches` turns that into the exact
    batches, in the exact order, this pass will encode them. Around that plan:

      1. a thread pool (`os.cpu_count()` by default) verifies the files already on disk and
         decodes/resizes/normalizes the planned batches' images,
      2. this thread encodes each planned batch as soon as its images are ready, front to back,
      3. a writer thread persists finished latents to disk (atomically).

    A file already on disk is only trusted once it really holds this bucket's latent (the dataset's
    own verdict), and one that does not is re-encoded with the rest. Each image's posterior sample
    comes from a generator seeded by its own cache key (`_latent_generator`), so what lands on disk
    is a function of (image, geometry, seed) and the global generator is left alone.
    """
    if not (cfg.cache_latents and cfg.cache_latents_to_disk):
        return True

    prepare_encoding_devices(vae, device, swap_ctx)

    entries = dataset.cache_entries()
    total = len(entries)
    workers = prefetch_workers or _DEFAULT_PREFETCH_WORKERS
    batch_size = max(1, encode_batch_size or _DEFAULT_ENCODE_BATCH_SIZE)
    window = max(workers * 2, batch_size * 2)
    cached = [entry for entry in entries if entry["cached"]]

    pbar = tqdm(total=total, desc="Encoding Latents")
    pbar_lock = threading.Lock()
    processed = 0
    aborted = False
    control.set_encoding(current=0, total=total, done=False)

    def _note_progress() -> None:
        nonlocal processed
        processed += 1
        control.set_encoding(current=processed, total=total, done=False)

    # Stage 3: async disk writer; the bounded queue provides backpressure.
    save_queue: "queue.Queue[Optional[tuple[torch.Tensor, Path]]]" = queue.Queue(maxsize=max(4, workers * 2))
    save_errors: list[BaseException] = []

    def _saver() -> None:
        while True:
            task = save_queue.get()
            if task is None:
                return
            latent, cache_path = task
            try:
                # write to a temp name and rename so a crash never leaves a
                # half-written cache file that would be loaded as valid later
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                torch.save(latent, tmp_path)
                os.replace(tmp_path, cache_path)
            except BaseException as exc:
                save_errors.append(exc)
            with pbar_lock:
                pbar.update(1)
                _note_progress()

    saver = threading.Thread(target=_saver, name="latent-cache-writer", daemon=True)
    saver.start()

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cache-prep")

    def _drain(indices: list[int], handle: Any) -> bool:
        """Feed `indices` through the pool in this order, `handle(index, item)` per result.

        Returns False when a pause/stop arrived, which only `at_safe_point` reports.
        """
        pending: dict[Any, int] = {}
        cursor = 0

        def _fill() -> None:
            nonlocal cursor
            while cursor < len(indices) and len(pending) < window:
                index = indices[cursor]
                pending[executor.submit(dataset.__getitem__, index)] = index
                cursor += 1

        _fill()
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                handle(index, future.result())
                if not at_safe_point("encoding", swap_ctx):
                    return False
            _fill()
        return True

    def _encode_chunk(items: list[Dict[str, Any]]) -> None:
        pixel_values = torch.stack([item["img_data"] for item in items]).to(device=device, dtype=dtype)
        latent_dist = vae.encode(pixel_values).latent_dist
        latents = _sample_latents(latent_dist, items, cfg, device) * vae.config.scaling_factor
        latents = latents.detach().cpu()
        del pixel_values
        for item, latent in zip(items, latents):
            save_queue.put((latent, Path(item["cache_path"])))
        flush_memory(device)

    try:
        # What is already on disk, checked with the same rule the loader applies. A file that is not
        # this bucket's latent comes back as a pixel item and joins the batches below.
        repairs: list[int] = []

        def _check(index: int, item: Dict[str, Any]) -> None:
            if item["img_type"] == "pixel":
                repairs.append(index)
            else:
                with pbar_lock:
                    pbar.update(1)
                    _note_progress()

        if cached and not _drain([entry["index"] for entry in cached], _check):
            aborted = True

        if not aborted:
            by_index = {entry["index"]: entry for entry in entries}
            batches = _plan_batches([entry for entry in entries if not entry["cached"]], batch_size)
            # Repairs go last: an image's latent no longer depends on where its batch sits.
            batches += _plan_batches([by_index[index] for index in repairs], batch_size)

            flat = [entry["index"] for batch in batches for entry in batch]
            starts: list[int] = []
            position = 0
            for batch in batches:
                starts.append(position)
                position += len(batch)

            pending: dict[Any, int] = {}
            settled: set[int] = set()
            ready: dict[int, Dict[str, Any]] = {}
            cursor = 0
            next_batch = 0

            def _fill() -> None:
                nonlocal cursor
                limit = starts[next_batch] + window if next_batch < len(batches) else len(flat)
                while cursor < min(limit, len(flat)) and len(pending) < window:
                    index = flat[cursor]
                    pending[executor.submit(dataset.__getitem__, index)] = index
                    cursor += 1

            def _encode_settled() -> None:
                """Encode every batch whose images have all reported, front to back."""
                nonlocal next_batch
                while next_batch < len(batches) and all(
                    entry["index"] in settled for entry in batches[next_batch]
                ):
                    items = [ready.pop(entry["index"]) for entry in batches[next_batch]
                             if entry["index"] in ready]
                    if items:
                        _encode_chunk(items)
                    next_batch += 1

            _fill()
            while pending:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    item = future.result()
                    settled.add(index)
                    if item["img_type"] == "pixel":
                        ready[index] = item
                    else:  # the cache answered after all; nothing to encode for this image
                        with pbar_lock:
                            pbar.update(1)
                            _note_progress()
                    if not at_safe_point("encoding", swap_ctx):
                        aborted = True
                        break
                if aborted:
                    break
                _fill()
                _encode_settled()
            if not aborted:
                _encode_settled()

        if not aborted:
            save_queue.put(None)
            saver.join()

            if save_errors:
                raise save_errors[0]
            control.set_encoding(current=processed, total=total, done=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        while True:
            try:
                save_queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    save_queue.get_nowait()
                except queue.Empty:
                    pass
        saver.join()
        pbar.close()
        vae.to("cpu")
        flush_memory(device)

    return not aborted
