#!/usr/bin/env python3
"""Bisect from the faulting side: run the trainer with one file replaced by an edited copy.

`diff_probe.py` removed trainer *config* ingredients and the fault survived all of
them, so the answer is in the code. This harness keeps the whole trainer (loading,
optimizers, DataLoader, environment, config) and swaps one piece for the way
`crash.py` does it, until the fault disappears.

    python patch_mirror.py --list
    python patch_mirror.py                     # the whole ladder
    python patch_mirror.py --row crash-body

Every row is `network_dim = 32` for 12 steps, i.e. the configuration that dies at
step 4 through the unmodified trainer. `control` is the unpatched row and must
fault, otherwise the harness is measuring nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import repro_real as rr

HERE = Path(__file__).resolve().parent
WORK = Path("/tmp/axl-fix2-patch")
OUT = rr.RESOURCES / "patch-mirror.json"
STEPS = 12

# Append-only overrides: the copied file keeps its original text and rebinds the
# name at the end, so no fragile surgery inside the function bodies.
MASK_OFF = '''
# --- patch: hand the loop no loss mask at all (crash.py's plain MSE) ---
_orig_collate_fn = collate_fn


def collate_fn(examples):
    batch = _orig_collate_fn(examples)
    batch["loss_mask"] = None
    return batch
'''

CRASH_BODY = '''
# --- patch: crash.py's step semantics, reusing the trainer's own pipeline ---
def crash_style_epoch(*, artifacts, cfg, global_step, progress=None, total_train_steps=0,
                      swap_ctx=None):
    accelerator = artifacts.accelerator
    modules = artifacts.modules
    denoise = modules.denoise
    text_encoders = list(modules.text_encoders)
    denoise_optimizer = artifacts.denoise_optimizer
    te_optimizer = artifacts.te_optimizer
    device = artifacts.device
    weight_dtype = artifacts.weight_dtype

    denoise.train()
    if hasattr(denoise_optimizer, "train"):
        denoise_optimizer.train()
    for te in text_encoders:
        te.train()
    denoise_clip = [p for p in denoise.parameters() if p.requires_grad]
    te_clip = [p for te in text_encoders for p in te.parameters() if p.requires_grad]

    for batch in artifacts.dataloader:
        batch["loss_mask"] = None            # no mask weighting
        with accelerator.accumulate(denoise, *text_encoders):
            groups = group_indices_by_bucket(batch)
            for _, indices in groups.items():
                prompts, latents, extra = build_group_inputs(
                    indices=indices, batch=batch, family=artifacts.family,
                    vae=modules.vae, cfg=cfg, device=device, weight_dtype=weight_dtype,
                )
                loss = artifacts.family.compute_loss(
                    prompts=prompts, latents=latents, extra=extra, modules=modules,
                    cfg=cfg, device=device, dtype=weight_dtype,
                )
                accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(denoise_clip, cfg.max_grad_norm)
            accelerator.clip_grad_norm_(te_clip, cfg.te_max_grad_norm)
            denoise_optimizer.step()
            te_optimizer.step()
            denoise_optimizer.zero_grad(set_to_none=True)
            te_optimizer.zero_grad(set_to_none=True)
            global_step += 1
            torch.cuda.synchronize()
            print(f"[crash-style] step={global_step} loss={float(loss.detach()):.4f}",
                  flush=True)
    return global_step


train_one_epoch = crash_style_epoch
'''

SETUP_ORDER = '''
# --- patch: create the Accelerator after the pipeline load (crash.py's order) ---
_orig_build_train_objects = build_train_objects


def build_train_objects(cfg):
    family = resolve_family(cfg)
    require_trainable(family)
    setup_migraphx_cache()
    maybe_enable_amp_backends()

    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, weight_dtype)
    accelerator = create_accelerator(cfg)
    device = accelerator.device

    modules = family.unpack(pipe)
    modules.noise_scheduler = family.build_noise_scheduler(pipe, cfg)
    modules.vae.requires_grad_(False)
    modules.denoise.requires_grad_(False)
    modules = family.apply_lora(cfg, modules)
    resume = family.load_lora(cfg, modules)
    modules.vae.to(device=device).eval()
    train_dataset, dataloader = build_dataloader(cfg)
    denoise_optimizer, te_optimizer, te_scheduler = build_optimizers_and_schedulers(
        cfg, modules, dataloader)
    return TrainArtifacts(
        family=family, modules=modules, accelerator=accelerator, device=device,
        weight_dtype=weight_dtype, train_dataset=train_dataset, dataloader=dataloader,
        denoise_optimizer=denoise_optimizer, te_optimizer=te_optimizer,
        te_scheduler=te_scheduler, resume=resume,
    )
'''

PIPELINE_ARGS = '''
# --- patch: load the pipeline the way crash.py does ---
def load_sdxl_pipeline(path, dtype):
    return StableDiffusionXLPipeline.from_pretrained(
        str(path), torch_dtype=dtype, use_safetensors=True,
    )
'''

SNAPSHOT = '''
# --- patch: allocator snapshots, including a sampler that keeps chasing the fault ---
import os as _os
import threading as _threading
import time as _time

_SNAP_DIR = _os.environ.get("AXL_SNAPSHOT_DIR")
_SNAP_STATE = {"n": 0}
if _SNAP_DIR:
    # Recording starts here, before the model is loaded, so the snapshot carries
    # the stack frames of nearly every live block. A small ring keeps the dumps
    # small; live segments are written in full regardless of the ring size.
    torch.cuda.memory._record_memory_history(max_entries=300000)


def _snap() -> None:
    """One dump per step, taken before that step's allocations."""
    if not _SNAP_DIR:
        return
    _SNAP_STATE["n"] += 1
    path = _os.path.join(_SNAP_DIR, f"step{_SNAP_STATE['n']:03d}.pickle")
    try:
        torch.cuda.memory._dump_snapshot(path)
        # Provenance marker: proves this copy of loop.py is the module that ran.
        with open(_os.path.join(_SNAP_DIR, "loop-module.txt"), "w") as fh:
            print(__file__, _os.getpid(), file=fh)
    except Exception as exc:                      # never mask the run being measured
        print(f"snapshot failed: {exc}", flush=True)
    if _SNAP_STATE["n"] == 3:
        _start_sampler()


def _start_sampler() -> None:
    """Chase the fatal kernel: keep replacing `latest.pickle` until the process dies.

    The abort arrives from the runtime, so the last complete file on disk is the
    layout a fraction of a second before the overrunning kernel ran.
    """
    if not _SNAP_DIR:
        return
    tmp = _os.path.join(_SNAP_DIR, ".latest.tmp")
    final = _os.path.join(_SNAP_DIR, "latest.pickle")

    def loop() -> None:
        while True:
            try:
                torch.cuda.memory._dump_snapshot(tmp)
                _os.replace(tmp, final)
            except Exception:
                pass
            _time.sleep(0.6)

    _threading.Thread(target=loop, daemon=True).start()
'''

SNAPSHOT_ANCHOR = "    for batch in artifacts.dataloader:\n"
SNAPSHOT_INSERT = SNAPSHOT_ANCHOR + "        _snap()\n"

NO_FLUSH = '''
# --- patch: no empty_cache anywhere (the trainer's flush_memory calls) ---
import gc as _gc


def flush_memory(device=None) -> None:
    _gc.collect()
'''

NUDGE = '''
# --- patch: perturb the allocator before training, to measure the margin ---
import os as _os

_orig_build_train_objects = build_train_objects


def build_train_objects(cfg):
    artifacts = _orig_build_train_objects(cfg)
    mb = float(_os.environ.get("AXL_NUDGE_MB") or 0)
    if mb:
        n = int(mb * 1024 * 1024)
        blob = torch.empty(n, dtype=torch.uint8, device=artifacts.device)
        print(f"[nudge] {mb} MiB at {hex(blob.data_ptr())} then released", flush=True)
        del blob
        torch.cuda.empty_cache()
    return artifacts
'''

LORA_ORDER = '''
# --- patch: wrap the UNet before the text encoders (crash.py's order) ---
_orig_apply_lora = SdxlFamily.apply_lora


def apply_lora(self, cfg, modules):
    te_cfg = lambda: LoraConfig(
        r=cfg.network_dim, lora_alpha=cfg.network_alpha, lora_dropout=cfg.network_dropout,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    unet_cfg = LoraConfig(
        r=cfg.network_dim, lora_alpha=cfg.network_alpha, lora_dropout=cfg.network_dropout,
        init_lora_weights="gaussian", target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    denoise = get_peft_model(modules.denoise, unet_cfg)
    if bool(getattr(cfg, "gradient_checkpointing_unet", True)):
        denoise.enable_gradient_checkpointing()
    enable_flash_attention(denoise)
    tes = [get_peft_model(te, te_cfg()) for te in modules.text_encoders]
    if bool(getattr(cfg, "gradient_checkpointing_te", True)):
        for te in tes:
            enable_te_gradient_checkpointing(te)
    modules.denoise = denoise
    modules.text_encoders = tes
    return modules


SdxlFamily.apply_lora = apply_lora
'''

NO_LOAD_LORA = '''
# --- patch: skip the resume bookkeeping entirely ---
def load_lora(self, cfg, modules):
    return None
'''

CRASH_SETUP = '''
# --- patch: crash.py's startup order, inside the trainer's lifecycle ---
def build_train_objects(cfg):
    from diffusers import StableDiffusionXLPipeline

    family = resolve_family(cfg)
    require_trainable(family)
    setup_migraphx_cache()
    maybe_enable_amp_backends()

    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(cfg.pretrained_model_name_or_path), torch_dtype=weight_dtype, use_safetensors=True)
    device = torch.device("cuda")

    modules = family.unpack(pipe)
    modules.noise_scheduler = family.build_noise_scheduler(pipe, cfg)
    modules.vae.requires_grad_(False)
    modules.denoise.requires_grad_(False)
    modules = family.apply_lora(cfg, modules)
    modules.vae.to(device=device).eval()

    train_dataset, dataloader = build_dataloader(cfg)
    denoise_optimizer, te_optimizer, te_scheduler = build_optimizers_and_schedulers(
        cfg, modules, dataloader)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision)
    return TrainArtifacts(
        family=family, modules=modules, accelerator=accelerator, device=device,
        weight_dtype=weight_dtype, train_dataset=train_dataset, dataloader=dataloader,
        denoise_optimizer=denoise_optimizer, te_optimizer=te_optimizer,
        te_scheduler=te_scheduler, resume=None)
'''

NO_STATE_WRITE = '''
# --- patch: no per-step state.json traffic (in-memory bookkeeping only) ---
def write_state(state=None, **kwargs) -> None:
    return None
'''

TQDM_STUB = '''class tqdm:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass
'''

NO_PIN = '''
# --- patch: DataLoader without pin_memory / workers (crash.py copies blocking) ---
def build_dataloader(cfg):
    train_dataset = LoraImageDataset(cfg)
    batch_sampler = BucketBatchSampler(
        train_dataset.buckets, batch_size=cfg.train_batch_size, seed=cfg.seed,
    )
    dataloader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, num_workers=0,
        pin_memory=False, persistent_workers=False, collate_fn=make_collate_fn(),
    )
    return train_dataset, dataloader
'''

NO_DATALOADER_ANCHOR = "    for batch in artifacts.dataloader:\n"
NO_DATALOADER_INSERT = """    _ds = artifacts.train_dataset
    _collate = artifacts.dataloader.collate_fn
    _sampler = artifacts.dataloader.batch_sampler
    for _indices in _sampler:
        batch = _collate([_ds[i] for i in _indices])
"""

NO_TRACKERS_ANCHOR = """        if accelerator.is_main_process:
            accelerator.init_trackers(
                project_name=run_id,
                config=vars(cfg),
            )
"""
NO_TRACKERS_INSERT = "        # patch: skip TensorBoard trackers\n"

# crash.py's actual forward: CLIP encode + UNet + plain MSE. Same models, no DataLoader.
TRUE_CRASH = '''
# --- patch: crash.py's forward, same models, no DataLoader ---
def train_one_epoch(*, artifacts, cfg, global_step, progress=None, total_train_steps=0,
                    swap_ctx=None):
    import random as _random
    accelerator = artifacts.accelerator
    modules = artifacts.modules
    denoise = modules.denoise
    tes = list(modules.text_encoders)
    tok1, tok2 = modules.tokenizers
    te1, te2 = tes
    opt_u = artifacts.denoise_optimizer
    opt_t = artifacts.te_optimizer
    device = artifacts.device
    dtype = artifacts.weight_dtype
    ds = artifacts.train_dataset
    sampler = artifacts.dataloader.batch_sampler
    denoise.train()
    if hasattr(opt_u, "train"):
        opt_u.train()
    for te in tes:
        te.train()
    clip_u = [p for p in denoise.parameters() if p.requires_grad]
    clip_t = [p for te in tes for p in te.parameters() if p.requires_grad]
    for indices in sampler:
        with accelerator.accumulate(denoise, *tes):
            prompts = [ds._caption_for(ds.records[i]["path"]) for i in indices]
            latents = []
            extras = []
            for i in indices:
                rec = ds.records[i]
                cache = ds._cache_path(rec["path"], rec["bucket_w"], rec["bucket_h"])
                z = torch.load(cache, map_location="cpu", weights_only=True)
                latents.append(z)
                extras.append(artifacts.family.extra_cond(
                    src_wh=(rec["src_w"], rec["src_h"]),
                    bucket_wh=(rec["bucket_w"], rec["bucket_h"]),
                    device=device, dtype=dtype,
                ))
            latents = torch.stack(latents, 0).to(device=device, dtype=dtype)
            extra = {k: torch.stack([e[k] for e in extras], 0) for k in extras[0]}
            hidden, pooled = artifacts.family.encode_prompts(
                prompts, modules, cfg, device, dtype,
            )
            noise = torch.randn_like(latents)
            if cfg.noise_offset > 0:
                noise = noise + cfg.noise_offset * torch.randn(
                    latents.shape[0], latents.shape[1], 1, 1, device=device, dtype=dtype)
            ts = torch.randint(0, modules.noise_scheduler.config.num_train_timesteps,
                               (latents.shape[0],), device=device, dtype=torch.long)
            noisy = modules.noise_scheduler.add_noise(latents, noise, ts)
            pred = denoise(
                noisy, ts, encoder_hidden_states=hidden,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": extra["time_ids"]},
                return_dict=False,
            )[0]
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(clip_u, cfg.max_grad_norm)
            accelerator.clip_grad_norm_(clip_t, cfg.te_max_grad_norm)
            opt_u.step()
            opt_t.step()
            artifacts.te_scheduler.step()
            opt_u.zero_grad(set_to_none=True)
            opt_t.zero_grad(set_to_none=True)
            global_step += 1
            torch.cuda.synchronize()
            print(f"[true-crash] step={global_step} loss={float(loss.detach()):.4f} "
                  f"M={hidden.shape[0]*hidden.shape[1]}", flush=True)
    return global_step
'''

SIMPLE_CACHE = '''
# --- patch: crash.py serial VAE encode (no thread pool, no extra flush) ---
@torch.no_grad()
def warm_latent_cache(dataset, vae, cfg, device, dtype, prefetch_workers=None,
                      encode_batch_size=None, swap_ctx=None):
    try:
        from utils import image_to_tensor, resize_and_center_crop
    except ImportError:
        from trainer.utils import image_to_tensor, resize_and_center_crop
    from PIL import Image

    vae.eval()
    cfg_v = getattr(vae, "config", None)
    if cfg_v is not None and hasattr(cfg_v, "force_upcast"):
        cfg_v.force_upcast = False
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        vae.enable_slicing()
    vae.to(device=device)
    batch = max(1, encode_batch_size or 2)
    pending = []

    def _flush(chunk):
        pixels = torch.stack([it["pixel"] for it in chunk]).to(device=device, dtype=dtype)
        z = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        for item, latent in zip(chunk, z):
            torch.save(latent.detach().cpu(), item["cache_path"])

    for rec in dataset.records:
        cache_path = dataset._cache_path(rec["path"], rec["bucket_w"], rec["bucket_h"])
        if cache_path.exists():
            continue
        with Image.open(rec["path"]) as img:
            img = img.convert("RGB")
            img = resize_and_center_crop(img, rec["bucket_w"], rec["bucket_h"])
            pixel = image_to_tensor(img)
        pending.append({"pixel": pixel, "cache_path": cache_path})
        if len(pending) >= batch:
            _flush(pending)
            pending = []
    if pending:
        _flush(pending)
    return True
'''

NO_PREPARE = '''
# --- patch: move modules with .to() instead of accelerator.prepare ---
def _prepare_artifacts(artifacts):
    device = artifacts.device
    artifacts.modules.denoise.to(device)
    artifacts.modules.text_encoders = [te.to(device) for te in artifacts.modules.text_encoders]
'''

RUN_CRASH_CHILD = '''
# --- patch: same process as trainer/main.py, but run crash.py's loop ---
def main():
    import sys as _sys
    from pathlib import Path as _Path
    crash_dir = _Path("/home/acite/Deeppin/AxlTrainer/fixes/fix2")
    _sys.path.insert(0, str(crash_dir))
    import family_sdxl as _family_sdxl  # noqa: F401  (same import set as a real run)
    import crash as _crash
    cfg = TrainConfig()
    _crash.STEPS = 8
    _crash.SEED = cfg.seed
    _crash.DIM = cfg.network_dim
    _crash.ALPHA = float(cfg.network_alpha)
    _crash.DROPOUT = float(cfg.network_dropout)
    _crash.BATCH = int(cfg.train_batch_size)
    raise SystemExit(_crash.child_main(cfg.pretrained_model_name_or_path))
'''

PATCHES: dict[str, tuple[str, str, str]] = {
    # name: (file to replace, appended code, what it isolates)
    "control": ("", "", "no patch; must fault or the ladder is meaningless"),
    "mask-off": ("trainer/dataset.py", MASK_OFF, "loss mask dropped instead of weighted"),
    "crash-body": ("trainer/loop.py", CRASH_BODY, "crash.py's step semantics"),
    "accelerator-late": ("trainer/setup.py", SETUP_ORDER, "Accelerator after pipeline load"),
    "pipeline-args": ("trainer/family_sdxl.py", PIPELINE_ARGS, "from_pretrained arguments"),
    "snapshot": ("trainer/loop.py", SNAPSHOT, "instrumentation only; must still fault"),
    "no-flush": ("trainer/env.py", NO_FLUSH, "no empty_cache between phases"),
    "nudge": ("trainer/setup.py", NUDGE, "AXL_NUDGE_MB MiB allocated then freed before training"),
    "lora-order": ("trainer/family_sdxl.py", LORA_ORDER, "UNet wrapped before the text encoders"),
    "no-load-lora": ("trainer/family_sdxl.py", NO_LOAD_LORA, "resume loading skipped"),
    "crash-setup": ("trainer/setup.py", CRASH_SETUP, "crash.py's whole startup order"),
    "no-state-write": ("trainer/control.py", NO_STATE_WRITE, "no state.json writes"),
    "no-progress": ("trainer/main.py", "", "the tqdm progress bar replaced by a stub"),
    "crash-all": ("trainer/setup.py", "# (see CRASH_SETUP / CRASH_BODY)\n",
                  "crash.py's startup AND its step body, inside the trainer's lifecycle"),
    "crash-core": ("trainer/setup.py", "",
                   "crash setup + serial cache + .to(device) + crash.py forward"),
    "prepare-models": ("trainer/main.py", "", "accelerator.prepare on UNet+TEs only; opts stay raw"),
    "prepare-opts": ("trainer/main.py", "", "accelerator.prepare on optimizers only; models .to(device)"),
    "no-pin": ("trainer/setup.py", NO_PIN, "DataLoader pin_memory=False, workers=0"),
    "no-dataloader": ("trainer/loop.py", "", "same collate/loss, no DataLoader pin/prefetch"),
    "no-trackers": ("trainer/main.py", "", "skip accelerator.init_trackers"),
    "true-crash": ("trainer/loop.py", TRUE_CRASH, "crash.py forward, no DataLoader"),
    "simple-cache": ("trainer/cache.py", SIMPLE_CACHE, "serial VAE encode, no thread pool / extra flush"),
    "no-prepare": ("trainer/main.py", NO_PREPARE, "module.to(device) instead of accelerator.prepare"),
    "run-crash-child": ("trainer/main.py", RUN_CRASH_CHILD,
                        "trainer process/imports/cwd, but crash.py's child_main"),
    "stub-main": ("trainer/main.py", "",
                  "python trainer/main.py from the mirror, no trainer imports, crash.py loop"),
    "stub-tqdm": ("trainer/main.py", "", "stub-main + tqdm.auto"),
    "stub-family-sdxl": ("trainer/main.py", "", "stub-main + import family_sdxl"),
    "stub-setup": ("trainer/main.py", "", "stub-main + import setup (pulls Accelerator/dataset)"),
    "stub-cache": ("trainer/main.py", "", "stub-main + import cache"),
    "stub-loop": ("trainer/main.py", "", "stub-main + import loop"),
    "stub-main-imports": ("trainer/main.py", "", "stub-main + exact trainer/main.py import block"),
}

NO_PREPARE_ANCHOR = '''def _prepare_artifacts(artifacts) -> None:
    # Do not prepare the dataloader: Accelerate would device-place metadata
    # tensors and may wrap/replace the bucket batch sampler. This trainer is
    # single-process; latents move to GPU in the train loop.
    n_te = len(artifacts.modules.text_encoders)
    prepared = artifacts.accelerator.prepare(
        artifacts.modules.denoise,
        *artifacts.modules.text_encoders,
        artifacts.denoise_optimizer,
        artifacts.te_optimizer,
        artifacts.te_scheduler,
    )
    artifacts.modules.denoise = prepared[0]
    artifacts.modules.text_encoders = list(prepared[1 : 1 + n_te])
    artifacts.denoise_optimizer = prepared[1 + n_te]
    artifacts.te_optimizer = prepared[2 + n_te]
    artifacts.te_scheduler = prepared[3 + n_te]
'''
NO_PREPARE_INSERT = '''def _prepare_artifacts(artifacts) -> None:
    device = artifacts.device
    artifacts.modules.denoise.to(device)
    artifacts.modules.text_encoders = [te.to(device) for te in artifacts.modules.text_encoders]
'''

REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "snapshot": [(SNAPSHOT_ANCHOR, SNAPSHOT_INSERT)],
    "no-progress": [("from tqdm.auto import tqdm\n", TQDM_STUB)],
    "no-dataloader": [(NO_DATALOADER_ANCHOR, NO_DATALOADER_INSERT)],
    "no-trackers": [(NO_TRACKERS_ANCHOR, NO_TRACKERS_INSERT)],
    "no-prepare": [(NO_PREPARE_ANCHOR, NO_PREPARE_INSERT)],
    "prepare-models": [(NO_PREPARE_ANCHOR, '''def _prepare_artifacts(artifacts) -> None:
    n_te = len(artifacts.modules.text_encoders)
    prepared = artifacts.accelerator.prepare(
        artifacts.modules.denoise, *artifacts.modules.text_encoders,
    )
    artifacts.modules.denoise = prepared[0]
    artifacts.modules.text_encoders = list(prepared[1:])
''')],
    "prepare-opts": [(NO_PREPARE_ANCHOR, '''def _prepare_artifacts(artifacts) -> None:
    device = artifacts.device
    artifacts.modules.denoise.to(device)
    artifacts.modules.text_encoders = [te.to(device) for te in artifacts.modules.text_encoders]
    prepared = artifacts.accelerator.prepare(
        artifacts.denoise_optimizer, artifacts.te_optimizer, artifacts.te_scheduler,
    )
    artifacts.denoise_optimizer = prepared[0]
    artifacts.te_optimizer = prepared[1]
    artifacts.te_scheduler = prepared[2]
''')],
}

_active = {"name": "control"}
_original_make_mirror = rr.make_mirror


def _materialize_mirror(dest: Path) -> None:
    """Replace the mirror's symlinks with real copies.

    `make_mirror` symlinks `trainer/*.py`, so a patched file dropped next to the
    symlink may never be the module Python loads. Real files remove that question.
    """
    import shutil

    for link in list(dest.rglob("*")):
        if link.is_symlink():
            source = link.resolve()
            link.unlink()
            shutil.copy2(source, link)


def patched_make_mirror(dest: Path) -> None:
    _original_make_mirror(dest)
    name = _active["name"]
    STUB_HEAD = (
        "import sys\n"
        "sys.path.insert(0, '/home/acite/Deeppin/AxlTrainer/fixes/fix2')\n"
        "import crash\n"
        "crash.STEPS = 8\n"
    )
    STUB_TAIL = "raise SystemExit(crash.child_main('/opt/models/diffusers/waillu_170'))\n"
    STUB_IMPORTS = {
        "stub-main": "",
        "stub-tqdm": "from tqdm.auto import tqdm\n",
        "stub-family-sdxl": "import family_sdxl\n",
        "stub-setup": "from setup import build_train_objects\n",
        "stub-cache": "from cache import warm_latent_cache\n",
        "stub-loop": "from loop import train_one_epoch\n",
        "stub-main-imports": (
            "from accelerate.utils import set_seed\n"
            "from tqdm.auto import tqdm\n"
            "from config import TrainConfig\n"
            "from models import artifact_root, lora_checkpoint_file\n"
            "from cache import warm_latent_cache\n"
            "from env import flush_memory\n"
            "from loop import train_one_epoch\n"
            "from runs import create_run_dirs\n"
            "from setup import build_train_objects\n"
            "import control\n"
            "from device_swap import SwapContext\n"
        ),
    }
    if name in STUB_IMPORTS:
        _materialize_mirror(dest)
        (dest / "trainer" / "main.py").write_text(
            STUB_HEAD + STUB_IMPORTS[name] + STUB_TAIL, encoding="utf-8"
        )
        return
    if name == "crash-core":
        # crash.py's setup + serial cache + .to(device) + crash.py forward.
        _materialize_mirror(dest)
        for relative, code in (
            ("trainer/setup.py", CRASH_SETUP),
            ("trainer/cache.py", SIMPLE_CACHE),
            ("trainer/loop.py", TRUE_CRASH),
        ):
            target = dest / relative
            target.write_text((rr.REPO_ROOT / relative).read_text(encoding="utf-8") + code,
                              encoding="utf-8")
        target = dest / "trainer" / "main.py"
        text = (rr.REPO_ROOT / "trainer" / "main.py").read_text(encoding="utf-8")
        if NO_PREPARE_ANCHOR not in text:
            raise SystemExit("no-prepare anchor missing for crash-core")
        target.write_text(text.replace(NO_PREPARE_ANCHOR, NO_PREPARE_INSERT, 1), encoding="utf-8")
        return
    if name == "crash-all":
        # Both halves crash.py can own: the startup and the step body. What remains
        # of the trainer is its lifecycle (run dirs, latent cache, prepare, epochs).
        _materialize_mirror(dest)
        for relative, code in (("trainer/setup.py", CRASH_SETUP),
                               ("trainer/loop.py", CRASH_BODY)):
            target = dest / relative
            target.write_text((rr.REPO_ROOT / relative).read_text(encoding="utf-8") + code,
                              encoding="utf-8")
        return
    relative, code, _ = PATCHES[name]
    replacements = REPLACEMENTS.get(name, [])
    if not relative or (not code and not replacements):
        return
    _materialize_mirror(dest)
    target = dest / relative
    text = (rr.REPO_ROOT / relative).read_text(encoding="utf-8")
    for search, replace in replacements:
        if search not in text:
            raise SystemExit(f"anchor missing in {relative} for patch {name!r}")
        text = text.replace(search, replace, 1)
    target.write_text(text + code, encoding="utf-8")


def run_row(name: str, model: str, kill_after_fault: int, index: int,
            env: list[str]) -> dict:
    _active["name"] = name
    # `_materialize_mirror` turns the previous run's symlinks into real files, and
    # `make_mirror` cannot re-link over those, so each row starts from a clean run dir.
    import shutil

    for stale in WORK.glob(f"{index:02d}_*"):
        shutil.rmtree(stale, ignore_errors=True)
    args = SimpleNamespace(
        steps=STEPS, env=list(env), dataset=str(rr.RESOURCES / "dataset"), data_dir="",
        model_path=model, dry_run=False, quiet=True, prune_checkpoints=1,
        kill_after_fault=kill_after_fault,
    )
    return rr.run_variant(name, {"network_dim": 32}, args=args, work=WORK, index=index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", default="", help="run only rows whose name contains this")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--kill-after-fault", type=int, default=3)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    names = [n for n in PATCHES if args.row in n] if args.row else list(PATCHES)
    if args.list:
        for name in names:
            relative, _code, why = PATCHES[name]
            print(f"{name:<18} {relative or '-':<26} {why}")
        return 0
    if not names:
        print(f"no patch matches {args.row!r}; patches: {', '.join(PATCHES)}", file=sys.stderr)
        return 2

    model = rr.model_path(args.model)
    if not Path(model).is_dir():
        print(f"model path not found: {model}", file=sys.stderr)
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    rr.make_mirror = patched_make_mirror

    results = []
    for index, name in enumerate(names, start=1):
        results.append(run_row(name, model, args.kill_after_fault, index, args.env))
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n| {'patch':<18} | {'last step':>9} | result |")
    print("|" + "---|" * 3)
    for name, entry in zip(names, results):
        step = entry["last_step"] if entry["last_step"] is not None else "-"
        print(f"| {name:<18} | {step:>9} | {entry['verdict']} |")
    faults = sum(1 for entry in results if entry["verdict"] == "FAULT")
    print(f"\n{faults}/{len(results)} faulted; json: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
