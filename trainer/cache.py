from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm.auto import tqdm

from config import TrainConfig
from dataset import SDXLLoraDataset
from env import flush_memory

_DEFAULT_PREFETCH_WORKERS = max(2, min(8, os.cpu_count() or 4))
_DEFAULT_ENCODE_BATCH_SIZE = 8


@torch.no_grad()
def warm_latent_cache(
    dataset: SDXLLoraDataset,
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    prefetch_workers: Optional[int] = None,
    encode_batch_size: Optional[int] = None,
) -> None:
    """Pre-encode image latents to disk when disk caching is enabled.

    The serial decode -> encode -> save loop is pipelined into three stages:
      1. a bounded thread pool decodes/resizes/normalizes images on the CPU,
      2. this thread encodes them in batches grouped by bucket size,
      3. a writer thread persists finished latents to disk (atomically).

    Results are identical to the serial version: the skip rule is unchanged
    (already-cached or non-pixel items are never re-encoded) and each latent
    is written to the same per-image, per-bucket cache path.
    """
    if not (cfg.cache_latents and cfg.cache_latents_to_disk):
        return

    vae.eval()
    vae.to(device=device, dtype=dtype)

    total = len(dataset)
    workers = prefetch_workers or _DEFAULT_PREFETCH_WORKERS
    batch_size = max(1, encode_batch_size or _DEFAULT_ENCODE_BATCH_SIZE)
    window = max(workers * 2, batch_size * 2)

    pbar = tqdm(total=total, desc="Encoding Latents")
    pbar_lock = threading.Lock()

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

    saver = threading.Thread(target=_saver, name="latent-cache-writer", daemon=True)
    saver.start()

    # Stage 1: CPU preprocessing in a bounded thread pool.
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cache-prep")
    pending: set[Any] = set()
    submitted = 0
    chunks: Dict[tuple[int, int], list[Dict[str, Any]]] = {}

    def _submit_next() -> None:
        nonlocal submitted
        if submitted >= total:
            return
        pending.add(executor.submit(dataset.__getitem__, submitted))
        submitted += 1

    def _encode_chunk(items: list[Dict[str, Any]]) -> None:
        pixel_values = torch.stack([item["img_data"] for item in items]).to(device=device, dtype=dtype)
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.detach().cpu()
        for item, latent in zip(items, latents):
            save_queue.put((latent, Path(item["cache_path"])))

    def _handle_item(item: Dict[str, Any]) -> None:
        cache_path = Path(item["cache_path"])
        if cache_path.exists() or item["img_type"] != "pixel":
            with pbar_lock:
                pbar.update(1)
            return
        bucket = (item["bucket_w"], item["bucket_h"])
        bucket_items = chunks.setdefault(bucket, [])
        bucket_items.append(item)
        if len(bucket_items) >= batch_size:
            _encode_chunk(chunks.pop(bucket))

    try:
        for _ in range(min(window, total)):
            _submit_next()

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                _handle_item(fut.result())
                _submit_next()

        # flush partially filled buckets
        for bucket_items in chunks.values():
            _encode_chunk(bucket_items)
        chunks.clear()

        # signal the writer to stop and wait for all pending saves
        save_queue.put(None)
        saver.join()

        if save_errors:
            raise save_errors[0]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if not save_queue.empty():
            # error path: make room and signal the writer to stop
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
