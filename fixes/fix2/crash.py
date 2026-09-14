#!/usr/bin/env python3
"""Self-contained dim-32 training loop: no AXL trainer, no IPC, no config.toml.

This is the distilled `repro_real.py --row "network_dim 32"` *work*, rewritten so
it only uses torch / diffusers / peft / transformers / accelerate / schedulefree.
It does **not** import `trainer/` or `api.py`.

It matches the crashing row's shapes (batch 3, LoRA r=32, alpha=18, dropout 0.2,
seed 1145141920, 6 packaged images, M=462 for steps 1–4 then M=231 at step 5)
and the trainer's setup order (VAE-encode then offload, dual optimizers,
Accelerator bf16, real CLIP chunking). Measured on this box: it **does not
page-fault**. The same config through unmodified `trainer/main.py` dies at
step 4 every time. The overrun is in Tensile; the fatality is
`accelerator.prepare()` of UNet+TEs after the trainer's load/cache history
(see `README.md`), which this file cannot recreate without that process.

    conda activate axl
    python crash.py
    python crash.py --model /path/to/sdxl-diffusers

Three trainer-only behaviours can be layered on with flags, to find which one the
fault actually needs: `--mask` (alpha-weighted loss, ~9 MB mask H2D per step),
`--nonblocking` (pinned CPU batches copied with `non_blocking=True`) and `--tb`
(TensorBoard scalars). All three are off by default.

For a guaranteed crash use `python repro_real.py --row "network_dim 32"`, or the
`diff_probe.py` ladder that removes one trainer ingredient at a time.

Weights are referenced, never copied. Images come from `resources/dataset/`
next to this file. Nothing is written into that folder (latents live under /tmp).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "resources" / "dataset"
FALLBACK_MODEL = "/opt/models/diffusers/waillu_170"
WORK = Path("/tmp/axl-fix2-crash")
CHILD_FLAG = "AXL_CRASH_CHILD"
FAULT_MARKERS = (
    "HSA_STATUS_ERROR_MEMORY_FAULT",
    "Memory access fault by GPU node",
    "GCVM_L2_PROTECTION_FAULT",
)

DIM = 32
ALPHA = 18.0          # packaged config left network_alpha=18 when dim was patched to 32
DROPOUT = 0.2
BATCH = 3
SEED = 1145141920
STEPS = 8             # dies at 4; a couple of extra steps so a miss is visible
MAX_TOKEN_LENGTH = 225
CLIP_SKIP = 1
NOISE_OFFSET = 0.05
KEEP_TOKENS = 2
BUCKET_STEP = 128
MIN_RESO, MAX_RESO = 768, 1280

# Trainer-only behaviours that can be layered on to find what the fault needs.
USE_MASK = False
USE_NONBLOCKING = False
USE_TB = False
USE_PIN = False
ACCEL_EARLY = False
SNAPSHOT_DIR = ""
IMPORT_TRAINER = False


def _rocm_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "AMD_LOG_LEVEL": "0",
        "CK_LOG_LEVEL": "0",
        "MIOPEN_ENABLE_LOGGING": "0",
        "MIOPEN_ENABLE_LOGGING_CMD": "0",
        "MIOPEN_LOG_LEVEL": "1",
        "MIOPEN_LOG_BUFFER_SIZE": "0",
        # Solver-selection switches start_train.sh sets; they change which MIOpen
        # convolution solvers are candidates, i.e. the workspace allocations.
        "MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS": "0",
        "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR": "0",
        "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK": "0",
        "MIOPEN_CUSTOM_CACHE_DIR": str(Path.home() / ".cache" / "miopen"),
        "MIOPEN_USER_DB_PATH": str(Path.home() / ".config" / "miopen"),
        # Caching allocator is the ingredient that makes the overrun fatal.
        # Do not set PYTORCH_NO_HIP_MEMORY_CACHING.
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
        CHILD_FLAG: "1",
    })
    env.pop("PYTORCH_NO_HIP_MEMORY_CACHING", None)
    for key, value in (extra or {}).items():
        env[key] = str(value)
    return env


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def shuffle_caption(text: str, keep_tokens: int, rng: random.Random) -> str:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) <= keep_tokens:
        return ", ".join(parts)
    head, tail = parts[:keep_tokens], parts[keep_tokens:]
    rng.shuffle(tail)
    return ", ".join(head + tail)


def round_to_step(value: int, step: int) -> int:
    return max(step, (value // step) * step)


def pick_bucket(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    if ar >= 1.0:
        bucket_h = MIN_RESO if h >= MIN_RESO else round_to_step(h, BUCKET_STEP)
        bucket_w = int(round(bucket_h * ar))
    else:
        bucket_w = MIN_RESO if w >= MIN_RESO else round_to_step(w, BUCKET_STEP)
        bucket_h = int(round(bucket_w / ar))
    bucket_w = max(BUCKET_STEP, min(round_to_step(bucket_w, BUCKET_STEP), MAX_RESO))
    bucket_h = max(BUCKET_STEP, min(round_to_step(bucket_h, BUCKET_STEP), MAX_RESO))
    if (ar >= 1.0 and bucket_w < bucket_h) or (ar < 1.0 and bucket_h < bucket_w):
        bucket_w, bucket_h = bucket_h, bucket_w
    return bucket_w, bucket_h


def resize_center_crop(image, target_w: int, target_h: int):
    from PIL import Image

    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return image.crop((left, top, left + target_w, top + target_h))


def image_to_tensor(image, torch):
    import numpy as np

    arr = torch.from_numpy(np.array(image)).float() / 255.0
    if arr.ndim == 2:
        arr = arr.unsqueeze(-1)
    return arr.permute(2, 0, 1) * 2.0 - 1.0


def tokenize_long(text: str, tokenizer, max_token_length: int):
    import torch

    chunk_size = tokenizer.model_max_length - 2
    ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
    ids = ids[:max_token_length]
    chunks = [ids[i:i + chunk_size] for i in range(0, max(1, len(ids)), chunk_size)] or [[]]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad(chunk):
        seq = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
        if len(seq) < tokenizer.model_max_length:
            seq = seq + [pad_id] * (tokenizer.model_max_length - len(seq))
        else:
            seq = seq[:tokenizer.model_max_length]
            seq[-1] = tokenizer.eos_token_id
        return seq

    stacked = torch.tensor([pad(c) for c in chunks], dtype=torch.long)
    return stacked, stacked.shape[0]


def pad_chunk_stack(ids_list, n_chunks, tokenizer, torch):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    empty = torch.tensor(
        [tokenizer.bos_token_id, tokenizer.eos_token_id]
        + [pad_id] * (tokenizer.model_max_length - 2),
        dtype=torch.long,
    )
    padded = []
    for ids in ids_list:
        if ids.shape[0] >= n_chunks:
            padded.append(ids[:n_chunks])
        else:
            extra = empty.unsqueeze(0).expand(n_chunks - ids.shape[0], -1)
            padded.append(torch.cat([ids, extra], dim=0))
    return torch.stack(padded, dim=0)


def encode_prompts(prompts, tok1, tok2, te1, te2, device, dtype, torch):
    ids1, ids2, n_chunks = [], [], 1
    for prompt in prompts:
        a, ca = tokenize_long(prompt, tok1, MAX_TOKEN_LENGTH)
        b, cb = tokenize_long(prompt, tok2, MAX_TOKEN_LENGTH)
        ids1.append(a)
        ids2.append(b)
        n_chunks = max(n_chunks, ca, cb)
    flat1 = pad_chunk_stack(ids1, n_chunks, tok1, torch).to(device)
    flat2 = pad_chunk_stack(ids2, n_chunks, tok2, torch).to(device)
    bsz, n, seq = flat1.shape
    out1 = te1(flat1.reshape(bsz * n, seq), output_hidden_states=True, return_dict=True)
    out2 = te2(flat2.reshape(bsz * n, seq), output_hidden_states=True, return_dict=True)
    # clip_skip=1 → penultimate hidden state (SDXL recipe; last layer NaNs in bf16).
    hs1 = out1.hidden_states[-(CLIP_SKIP + 1)].view(bsz, n, seq, -1)
    hs2 = out2.hidden_states[-(CLIP_SKIP + 1)].view(bsz, n, seq, -1)
    hidden = torch.cat(
        [hs1.reshape(bsz, n * seq, -1), hs2.reshape(bsz, n * seq, -1)], dim=-1,
    ).to(dtype)
    pooled = out2.text_embeds.view(bsz, n, -1)[:, 0, :].to(dtype)
    return hidden, pooled, n


def wrap_lora(module, targets, torch):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=DIM, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none",
        init_lora_weights="gaussian", target_modules=list(targets),
    )
    wrapped = get_peft_model(module, cfg)
    n = sum(1 for name, _ in wrapped.named_modules() if "lora_A" in name)
    if n == 0:
        raise SystemExit(f"peft matched no LoRA modules for {targets}")
    return wrapped, n


def loss_mask_tensor(image_path: Path, bw: int, bh: int, src_w: int, src_h: int, torch):
    """`[1, bh, bw]` float32 in `[0, 1]`, exactly like `trainer/utils.load_loss_mask`.

    Sidecar `{stem}.mask.png` first, else the image's alpha channel, else all ones.
    The packaged kanae PNGs are RGBA, so the trainer moves a ~9 MB pixel-size
    float32 mask to the GPU every step and area-interpolates it in the loss.
    """
    import numpy as np
    from PIL import Image

    def from_l(img):
        mask = img.convert("L")
        if mask.size != (int(src_w), int(src_h)):
            mask = mask.resize((int(src_w), int(src_h)), Image.Resampling.NEAREST)
        mask = resize_center_crop(mask, bw, bh)
        return (torch.from_numpy(np.array(mask)).float() / 255.0).unsqueeze(0)

    sidecar = image_path.with_suffix(".mask.png")
    if sidecar.is_file():
        with Image.open(sidecar) as img:
            return from_l(img)
    with Image.open(image_path) as img:
        if img.mode in {"RGBA", "LA", "PA"} or (img.mode == "P" and "transparency" in img.info):
            return from_l(img.getchannel("A") if img.mode in {"RGBA", "LA"} else img.convert("RGBA"))
        return torch.ones((1, int(bh), int(bw)), dtype=torch.float32)


def load_items(torch):
    from PIL import Image

    images = sorted(
        p for p in DATA.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        and not p.name.endswith(".mask.png")
    )
    if not images:
        raise SystemExit(f"no images in {DATA}")
    items = []
    for path in images:
        with Image.open(path) as im:
            im = im.convert("RGB")
            src_w, src_h = im.size
            bw, bh = pick_bucket(src_w, src_h)
            cropped = resize_center_crop(im, bw, bh)
            pixel = image_to_tensor(cropped, torch)
        caption = path.with_suffix(".txt").read_text(encoding="utf-8").strip()
        items.append({
            "path": path, "src_w": src_w, "src_h": src_h,
            "bw": bw, "bh": bh, "pixel": pixel, "caption": caption,
            "mask": loss_mask_tensor(path, bw, bh, src_w, src_h, torch),
        })
    return items


def batches_for_epoch(n: int, epoch: int) -> list[list[int]]:
    rng = random.Random(SEED + epoch)
    order = list(range(n))
    rng.shuffle(order)
    out = [order[i:i + BATCH] for i in range(0, n, BATCH)]
    rng.shuffle(out)
    return out


def caption_for(item, epoch: int) -> str:
    seed_val = SEED + epoch + int(sha1_text(str(item["path"].resolve())), 16) % 10_000
    return shuffle_caption(item["caption"], KEEP_TOKENS, random.Random(seed_val))


def child_main(model_dir: str) -> int:
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline
    from diffusers.models.attention_processor import AttnProcessor2_0
    from schedulefree import AdamWScheduleFree
    from torch.utils.tensorboard import SummaryWriter

    if not torch.cuda.is_available():
        print("no CUDA/HIP device", file=sys.stderr)
        return 2
    if IMPORT_TRAINER:
        print(f"trainer modules imported first: {import_trainer_modules()}", flush=True)
    if not Path(model_dir).is_dir():
        print(f"model path not found: {model_dir}", file=sys.stderr)
        return 2

    set_seed(SEED)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    early_acc = None
    if ACCEL_EARLY:
        # Trainer creates Accelerator before the VAE pass. prepare() then lands
        # UNet/TE weights in whatever holes that pass punched.
        early_acc = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
        print("Accelerator created before pipeline load", flush=True)

    items = load_items(torch)
    print(f"images={len(items)} buckets={[ (it['bw'], it['bh']) for it in items ]}", flush=True)

    snap_dir = Path(SNAPSHOT_DIR) if SNAPSHOT_DIR else None
    if snap_dir is not None:
        # Record from the very first allocation so the snapshot has frames for everything.
        snap_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=300000)
        print(f"recording allocator history -> {snap_dir}", flush=True)
        _start_sampler(torch, snap_dir)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir, torch_dtype=dtype, use_safetensors=True,
    )
    unet, vae = pipe.unet, pipe.vae
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config, prediction_type="epsilon")
    del pipe
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    unet, n_unet = wrap_lora(unet, ("to_q", "to_k", "to_v", "to_out.0"), torch)
    te1, n_te1 = wrap_lora(te1, ("q_proj", "k_proj", "v_proj", "out_proj"), torch)
    te2, n_te2 = wrap_lora(te2, ("q_proj", "k_proj", "v_proj", "out_proj"), torch)
    unet.enable_gradient_checkpointing()
    unet.set_attn_processor(AttnProcessor2_0())
    print(f"lora adapters: unet={n_unet} te1={n_te1} te2={n_te2} dim={DIM} alpha={ALPHA}",
          flush=True)

    # Encoding uses only the VAE. UNet + TEs stay on CPU so the cache pass
    # punches the same holes in the caching allocator that the trainer does.
    vae.eval()
    if hasattr(vae.config, "force_upcast"):
        vae.config.force_upcast = False
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
        vae.enable_slicing()
    unet.to("cpu")
    te1.to("cpu")
    te2.to("cpu")
    vae.to(device=device)
    cache_dir = WORK / "latents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for start in range(0, len(items), 2):
            chunk = items[start:start + 2]
            pixels = torch.stack([it["pixel"] for it in chunk]).to(device=device, dtype=dtype)
            z = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            for offset, latent in enumerate(z):
                torch.save(latent.cpu(), cache_dir / f"{start + offset}.pt")
    vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"vae encoded {len(items)} latents, VAE offloaded", flush=True)

    unet_opt = AdamWScheduleFree(
        [p for p in unet.parameters() if p.requires_grad],
        lr=2e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01, warmup_steps=10,
    )
    te_params = [p for te in (te1, te2) for p in te.parameters() if p.requires_grad]
    te_opt = torch.optim.AdamW(te_params, lr=2e-5, betas=(0.9, 0.99), weight_decay=0.01)
    te_sched = torch.optim.lr_scheduler.LambdaLR(te_opt, lambda step: 1.0)

    acc = early_acc or Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    unet, te1, te2, unet_opt, te_opt, te_sched = acc.prepare(
        unet, te1, te2, unet_opt, te_opt, te_sched,
    )
    unet.train()
    te1.train()
    te2.train()
    unet_opt.train()
    unet_clip = [p for p in unet.parameters() if p.requires_grad]
    te_clip = [p for te in (te1, te2) for p in te.parameters() if p.requires_grad]
    writer = SummaryWriter(str(WORK / "tb")) if USE_TB else None
    print(f"options: mask={USE_MASK} nonblocking={USE_NONBLOCKING} tb={USE_TB} pin={USE_PIN}",
          flush=True)

    pin_loader = None
    if USE_PIN:
        # Inverse of the trainer no-pin patch: same tensors, but through a pinned DataLoader.
        from torch.utils.data import DataLoader, Dataset

        class _Items(Dataset):
            def __len__(self):
                return len(items)

            def __getitem__(self, i):
                return i

        def _epoch_loader(epoch: int):
            batches = batches_for_epoch(len(items), epoch)
            # one-batch-at-a-time sampler so collate sees the same groups
            return DataLoader(
                _Items(), batch_sampler=batches, num_workers=0, pin_memory=True,
            )

        pin_loader = _epoch_loader

    started = time.time()
    step = 0
    while step < STEPS:
        epoch = step // max(1, math.ceil(len(items) / BATCH))
        epoch_batches = pin_loader(epoch) if pin_loader is not None else batches_for_epoch(len(items), epoch)
        for indices in epoch_batches:
            if pin_loader is not None:
                indices = [int(i) for i in indices]
            if step >= STEPS:
                break
            step += 1
            if snap_dir is not None:
                # Dump before the step's allocations: this is the layout the faulting
                # GEMM will see in this step.
                torch.cuda.memory._dump_snapshot(str(snap_dir / f"step{step:03d}.pickle"))
            with acc.accumulate(unet, te1, te2):
                group = [items[i] for i in indices]
                prompts = [caption_for(it, epoch) for it in group]
                z = torch.stack([
                    torch.load(cache_dir / f"{i}.pt", weights_only=True) for i in indices
                ])
                if USE_NONBLOCKING:
                    z = z.pin_memory().to(device=device, dtype=dtype, non_blocking=True)
                else:
                    z = z.to(device=device, dtype=dtype)
                mask = None
                if USE_MASK:
                    mask = torch.stack([items[i]["mask"] for i in indices])
                    if USE_NONBLOCKING:
                        mask = mask.pin_memory().to(device=device, dtype=torch.float32,
                                                    non_blocking=True)
                    else:
                        mask = mask.to(device=device, dtype=torch.float32)
                hidden, pooled, n_chunks = encode_prompts(
                    prompts, tok1, tok2, te1, te2, device, dtype, torch,
                )
                time_ids = torch.tensor(
                    [[it["src_h"], it["src_w"], 0, 0, it["bh"], it["bw"]] for it in group],
                    device=device, dtype=dtype,
                )
                noise = torch.randn_like(z)
                noise = noise + NOISE_OFFSET * torch.randn(
                    z.shape[0], z.shape[1], 1, 1, device=device, dtype=dtype,
                )
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (z.shape[0],), device=device, dtype=torch.long,
                )
                noisy = scheduler.add_noise(z, noise, timesteps)
                pred = unet(
                    noisy, timesteps, encoder_hidden_states=hidden,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                    return_dict=False,
                )[0]
                if mask is None:
                    loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                else:
                    # trainer/utils.apply_loss_mask: weight the unreduced loss, then mean.
                    err = torch.nn.functional.mse_loss(pred.float(), noise.float(),
                                                       reduction="none")
                    weight = mask
                    if weight.shape[-2:] != err.shape[-2:]:
                        weight = torch.nn.functional.interpolate(
                            weight, size=err.shape[-2:], mode="area"
                        )
                    loss = (err * weight).mean()
                acc.backward(loss)
            acc.clip_grad_norm_(unet_clip, 1.0)
            acc.clip_grad_norm_(te_clip, 0.3)
            unet_opt.step()
            te_opt.step()
            te_sched.step()
            unet_opt.zero_grad(set_to_none=True)
            te_opt.zero_grad(set_to_none=True)
            if writer is not None:
                writer.add_scalar("loss", float(loss.detach()), step)
            torch.cuda.synchronize()
            m = hidden.shape[0] * hidden.shape[1]
            print(f"step={step} loss={float(loss.detach()):.4f} M={m} chunks={n_chunks} "
                  f"{time.time() - started:.1f}s "
                  f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                  flush=True)

    print("SURVIVED: no page fault in", STEPS, "steps", flush=True)
    return 1


def reap(marker: str) -> None:
    import signal

    me, parent = os.getpid(), os.getppid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if marker in cmd:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def parent_main(model: str, child_args: list[str], extra_env: dict[str, str]) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    log_path = WORK / "crash.out"
    print(f"model={model}", flush=True)
    print(f"data={DATA} ({sum(1 for p in DATA.glob('*.png') if not p.name.endswith('.mask.png'))} images)",
          flush=True)
    print(f"log={log_path}", flush=True)
    started = time.time()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(HERE / "crash.py"), "--model", model, *child_args],
            cwd=str(HERE), env=_rocm_env(extra_env),
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        while proc.poll() is None:
            time.sleep(1.0)
            try:
                tail = log_path.read_bytes()[-2048:].decode("utf-8", "replace")
            except OSError:
                continue
            if any(m in tail for m in FAULT_MARKERS):
                # KFD has already printed; don't wait for the ~25 s coredump unwind.
                time.sleep(2.0)
                proc.kill()
                break
        proc.wait()
    reap(str(HERE / "crash.py"))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    last_step = None
    for line in text.splitlines():
        if line.startswith("step="):
            try:
                last_step = int(line.split()[0].split("=")[1])
            except (IndexError, ValueError):
                pass
    fault = any(m in text for m in FAULT_MARKERS)
    print(text[-4000:], end="" if text.endswith("\n") else "\n")
    print(f"\nexit={proc.returncode} last_step={last_step} "
          f"{time.time() - started:.0f}s", flush=True)
    if fault:
        print("FAULT", flush=True)
        return 0
    print("NO-FAULT", flush=True)
    return 1


def import_trainer_modules() -> list[str]:
    """Import the same modules `trainer/main.py` imports, before anything else.

    The two processes differ in nothing else that has survived ablation, so this
    tests the one variable never varied: the import set / import order.
    """
    import importlib

    root = HERE.parent.parent
    if str(root / "trainer") not in sys.path:
        sys.path.insert(0, str(root / "trainer"))
    loaded = []
    for name in ("tensorboard", "safetensors", "accelerate.utils", "family", "models",
                 "cache", "control", "device_swap", "dataset", "loss_log", "runs",
                 "checkpoints", "cleanup", "family_sdxl", "setup", "loop"):
        try:
            importlib.import_module(name)
            loaded.append(name)
        except Exception as exc:                  # a missing module must not hide the test
            loaded.append(f"{name}!{type(exc).__name__}")
    return loaded


def _start_sampler(torch, snap_dir: Path, interval: float = 0.3) -> None:
    """Keep replacing `latest.pickle` so the last complete dump is close to any fault.

    The trainer dies inside a step; the interesting allocations are created and
    destroyed within that step, so a dump at the step boundary cannot see them.
    """
    import threading
    import time

    tmp = snap_dir / ".latest.tmp"
    final = snap_dir / "latest.pickle"

    def loop() -> None:
        while True:
            try:
                torch.cuda.memory._dump_snapshot(str(tmp))
                os.replace(tmp, final)
            except Exception:
                pass
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()


def main() -> int:
    global USE_MASK, USE_NONBLOCKING, USE_TB, USE_PIN, ACCEL_EARLY, STEPS, SNAPSHOT_DIR, IMPORT_TRAINER
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=FALLBACK_MODEL)
    parser.add_argument("--mask", action="store_true",
                        help="weight the loss by the alpha-derived mask, like the trainer "
                             "(moves a ~9 MB float32 mask to the GPU every step)")
    parser.add_argument("--nonblocking", action="store_true",
                        help="pin the CPU latents/mask and copy them with non_blocking=True, "
                             "like the trainer's pinned DataLoader batches")
    parser.add_argument("--tb", action="store_true",
                        help="write TensorBoard scalars, like the trainer's Accelerator tracker")
    parser.add_argument("--pin", action="store_true",
                        help="feed each batch through a pin_memory DataLoader, like the trainer")
    parser.add_argument("--accel-early", action="store_true",
                        help="create Accelerator before the VAE pass, like the trainer")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra child environment (repeatable)")
    parser.add_argument("--steps", type=int, default=STEPS, help="training steps to run")
    parser.add_argument("--snapshot", default="", metavar="DIR",
                        help="dump a CUDA allocator snapshot at the top of every step")
    parser.add_argument("--import-trainer", action="store_true",
                        help="import the trainer's modules first, as trainer/main.py does")
    args = parser.parse_args()

    USE_MASK, USE_NONBLOCKING, USE_TB, USE_PIN = args.mask, args.nonblocking, args.tb, args.pin
    ACCEL_EARLY = args.accel_early
    STEPS = args.steps
    SNAPSHOT_DIR = args.snapshot
    IMPORT_TRAINER = args.import_trainer
    child_args = [flag for flag, on in (("--mask", args.mask),
                                         ("--nonblocking", args.nonblocking),
                                         ("--tb", args.tb),
                                         ("--pin", args.pin),
                                         ("--accel-early", args.accel_early)) if on]
    child_args += ["--steps", str(STEPS)]
    if SNAPSHOT_DIR:
        child_args += ["--snapshot", SNAPSHOT_DIR]
    if args.import_trainer:
        child_args += ["--import-trainer"]
    extra_env = {}
    for item in args.env:
        key, _, value = item.partition("=")
        extra_env[key] = value
    if os.environ.get(CHILD_FLAG) == "1":
        return child_main(args.model)
    return parent_main(args.model, child_args, extra_env)


if __name__ == "__main__":
    raise SystemExit(main())
