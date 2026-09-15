#!/usr/bin/env python3
"""Joint text-encoder + UNet backward, the case `fixes/fix2.txt` records as faulting.

This is the closest model-backed repro to the original failure that still skips the training loop:
real SDXL UNet *and* both CLIP text encoders with LoRA adapters, a real tokenized caption, no VAE,
no DataLoader, no optimizer. Weights are referenced from `config.toml`, never copied.

    python repro_joint.py --batch 3 --caption-index 2   # 2 CLIP chunks -> seq 154 -> M = 462
    python repro_joint.py --batch 1 --caption-index 0   # 1 chunk -> seq 77  -> M = 77
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--caption-index", type=int, default=None,
                        help="use the Nth caption of the configured dataset; default: a synthetic one")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--bucket", default="1024x768")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    import torch

    from trainer import family as family_mod
    from trainer.config import TrainConfig
    from trainer.utils import list_images

    if not torch.cuda.is_available():
        print("no CUDA/HIP device", file=sys.stderr)
        return 2
    device, dtype = torch.device("cuda"), torch.bfloat16
    cfg = TrainConfig()
    if args.model:
        cfg.pretrained_model_name_or_path = args.model
    if not Path(cfg.pretrained_model_name_or_path).is_dir():
        print(f"model path not found: {cfg.pretrained_model_name_or_path}", file=sys.stderr)
        return 2
    bucket_w, bucket_h = (int(value) for value in args.bucket.lower().split("x"))

    prompt = args.prompt
    if prompt is None and args.caption_index is not None:
        images = sorted(list_images(Path(cfg.train_data_dir)))
        caption = images[args.caption_index].with_suffix(".txt")
        prompt = caption.read_text(encoding="utf-8") if caption.is_file() else images[args.caption_index].stem
    if prompt is None:
        prompt = ("solo, 1girl, school uniform, white thighhighs, indoors, classroom, soft lighting, "
                  "looking at viewer, gentle smile, long hair, ribbon, pleated skirt, sitting on desk, "
                  "after school, sunlight from window, detailed background, empty room")

    family = family_mod.resolve_family(cfg)
    pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, dtype)
    modules = family.unpack(pipe)
    modules.noise_scheduler = family.build_noise_scheduler(pipe, cfg)
    modules.vae.requires_grad_(False)
    modules.denoise.requires_grad_(False)
    modules = family.apply_lora(cfg, modules)
    denoise = modules.denoise.to(device=device, dtype=dtype)
    denoise.requires_grad_(False)
    denoise.enable_gradient_checkpointing()
    denoise.train()
    for encoder in modules.text_encoders:
        encoder.to(device=device, dtype=dtype).train()

    print(json.dumps({
        "config": str(REPO_ROOT / "config.toml"),
        "model": cfg.pretrained_model_name_or_path, "network_dim": cfg.network_dim,
        "batch": args.batch, "bucket": args.bucket,
        "gradient_checkpointing_unet": cfg.gradient_checkpointing_unet,
        "gradient_checkpointing_te": cfg.gradient_checkpointing_te,
        "prompt_head": prompt[:80],
    }), flush=True)

    latents = torch.randn(args.batch, 4, bucket_h // 8, bucket_w // 8, device=device, dtype=dtype)
    started = time.time()
    losses, seqs = [], []
    for _ in range(args.iters):
        encoded = family.encode_prompts([prompt] * args.batch, modules, cfg, device, dtype)
        seqs.append(int(encoded[0].shape[1]))
        extra = family.extra_cond(src_wh=(bucket_w, bucket_h), bucket_wh=(bucket_w, bucket_h),
                                  device=device, dtype=dtype)
        extra["time_ids"] = extra["time_ids"].unsqueeze(0).repeat(args.batch, 1)
        loss = family.denoise_loss(latents=latents, encoded=encoded, extra=extra, modules=modules,
                                   cfg=cfg, device=device, dtype=dtype)
        loss.backward()   # both the TE and the UNet LoRA gradients are computed here
        losses.append(float(loss.detach()))
        for module in list(denoise.modules()) + [m for enc in modules.text_encoders for m in enc.modules()]:
            for param in list(module.parameters(recurse=False)):
                param.grad = None
    torch.cuda.synchronize()

    print("RESULT " + json.dumps({
        "batch": args.batch, "seq": seqs[0], "M": args.batch * seqs[0], "losses": losses,
        "seconds": round(time.time() - started, 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
