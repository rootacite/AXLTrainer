#!/usr/bin/env python3
"""Middle-ground repro: the real SDXL UNet, LoRA-wrapped, no text encoders and no VAE.

`fixes/fix2.txt` reports that the UNet backward with *detached* prompt embeds is stable while the
joint TE+UNet backward page-faults. This script keeps the real UNet (and its LoRA adapters, so the
exact `[M, network_dim] @ [network_dim, 2048]` GEMMs run inside a real graph) and feeds a synthetic
cross-attention input that carries gradients — i.e. everything from the failing case except CLIP.

Weights are referenced from `trainer/config.toml` (`pretrained_model_name_or_path`), never copied.

    python repro_unet.py --batch 2 --chunks 1     # M = 462, the shape fix2 pins the fault on
    python repro_unet.py --batch 1 --chunks 1     # M = 231
    python repro_unet.py --batch 2 --chunks 3     # M = 462 with a full 3-chunk prompt
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
    parser.add_argument("--batch", type=int, default=2, help="train_batch_size")
    parser.add_argument("--chunks", type=int, default=1, help="CLIP chunks the caption needed (1-3)")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--bucket", default="1024x768", help="latent source size, like the kanae bucket")
    parser.add_argument("--detach", type=int, default=0, help="1 = detach the cross-attention input")
    parser.add_argument("--no-checkpointing", type=int, default=0)
    parser.add_argument("--model", default=None, help="override pretrained_model_name_or_path")
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)  # trainer/config.py resolves trainer/config.toml relative to the cwd
    import torch
    import torch.nn.functional as F

    from trainer import family as family_mod
    from trainer.config import TrainConfig

    if not torch.cuda.is_available():
        print("no CUDA/HIP device", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = TrainConfig()
    if args.model:
        cfg.pretrained_model_name_or_path = args.model
    if not Path(cfg.pretrained_model_name_or_path).is_dir():
        print(f"model path not found: {cfg.pretrained_model_name_or_path}", file=sys.stderr)
        return 2
    bucket_w, bucket_h = (int(v) for v in args.bucket.lower().split("x"))
    seq_len = 77 * args.chunks
    rows = args.batch * seq_len

    print(json.dumps({"config": str(REPO_ROOT / "trainer" / "config.toml"),
                      "model": cfg.pretrained_model_name_or_path, "network_dim": cfg.network_dim,
                      "batch": args.batch, "chunks": args.chunks, "seq": seq_len, "M": rows,
                      "bucket": args.bucket, "detach": bool(args.detach),
                      "gradient_checkpointing": not args.no_checkpointing}), flush=True)

    family = family_mod.resolve_family(cfg)
    pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, dtype)
    modules = family.unpack(pipe)
    modules.noise_scheduler = family.build_noise_scheduler(pipe, cfg)
    modules.denoise.requires_grad_(False)
    modules = family.apply_lora(cfg, modules)
    denoise = modules.denoise.to(device=device, dtype=dtype)
    if not args.no_checkpointing:
        denoise.enable_gradient_checkpointing()
    denoise.train()

    latents = torch.randn(args.batch, 4, bucket_h // 8, bucket_w // 8, device=device, dtype=dtype)
    encoded = (
        torch.randn(args.batch, seq_len, 2048, device=device, dtype=dtype),
        torch.randn(args.batch, 1280, device=device, dtype=dtype),
    )
    extra = family.extra_cond(src_wh=(bucket_w, bucket_h), bucket_wh=(bucket_w, bucket_h),
                              device=device, dtype=dtype)
    # The training loop stacks per-sample extras into [B, 6] (`_stack_extra`); a bare [6] vector
    # reshapes wrongly inside the UNet for B > 1.
    extra["time_ids"] = extra["time_ids"].unsqueeze(0).repeat(args.batch, 1)

    if args.debug >= 2:
        original_forward = denoise.add_embedding.forward

        def traced_forward(sample, condition=None):
            print(f"      [trace] add_embedding input {tuple(sample.shape)} "
                  f"condition={None if condition is None else tuple(condition.shape)}", flush=True)
            return original_forward(sample, condition)

        denoise.add_embedding.forward = traced_forward
    if args.debug:
        add_embedding = denoise.add_embedding
        print(json.dumps({
            "addition_embed_type": str(getattr(denoise.config, "addition_embed_type", None)),
            "add_embedding": type(add_embedding).__name__,
            "add_embedding.children": {name: str(module) for name, module in add_embedding.named_children()},
            "prompt_embeds": tuple(encoded[0].shape), "pooled": tuple(encoded[1].shape),
            "extra": {key: tuple(value.shape) for key, value in extra.items()},
            "latents": tuple(latents.shape),
        }, indent=2), flush=True)

    started = time.time()
    losses = []
    for _ in range(args.iters):
        prompt_embeds = encoded[0] if args.detach else encoded[0].clone().requires_grad_(True)
        loss = family.denoise_loss(
            latents=latents, encoded=(prompt_embeds, encoded[1]), extra=extra, modules=modules,
            cfg=cfg, device=device, dtype=dtype,
        )
        loss.backward()   # the documented faulting op runs here
        losses.append(float(loss.detach()))
        for module in denoise.modules():
            for param in list(module.parameters(recurse=False)):
                param.grad = None
        latents.grad = None
    torch.cuda.synchronize()

    print("RESULT " + json.dumps({
        "M": rows, "batch": args.batch, "chunks": args.chunks, "detach": bool(args.detach),
        "seconds": round(time.time() - started, 2), "losses": losses,
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }), flush=True)
    # Silence unused-import style complaints while keeping F available for quick local edits.
    _ = F
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
