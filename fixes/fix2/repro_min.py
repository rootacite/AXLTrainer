#!/usr/bin/env python3
"""Simplest possible repro of the gfx1201 Tensile page fault — no axl training code at all.

Nothing here imports `trainer/`: this is `diffusers` + `peft` + `torch`, a real SDXL UNet with LoRA
adapters, and a synthetic batch shaped exactly like the real one. It exists to find out how little of
the stack the abort actually needs; the trainer-side evidence lives in the other scripts in this
directory and in README.md.

    python repro_min.py                      # UNet + LoRA r=32, batch 3 x 2 chunks (M=462), 8 steps
    python repro_min.py --dim 36             # the rank that survives in the trainer until step 101
    python repro_min.py --dim 12             # another rank that aborts in four steps inside training
    python repro_min.py --te 1               # add the two CLIP text encoders to the backward
    python repro_min.py --optimizer adamw    # step an optimizer between iterations

Gradient checkpointing is on by default and should stay on: without it the 1024x768 UNet backward at
batch 3 allocates past 16 GB and dies of OOM (measured: 14.78 GiB allocated before the failed
allocation), which is exactly why the trainer's config has `gradient_checkpointing_unet = true`.

The faulting GEMM is `[M, K] @ [K, 2048]` with `M = batch * 77 * chunks` and `K = network_dim`, so
`--batch 3 --chunks 2 --dim 32` reproduces the shape the trainer aborts on at step 4 of a 12-step
run. The process is expected to die with SIGABRT; the KFD message (kernel name, faulting address)
goes to this script's own log, `repro_min.stderr.log`, because the HIP runtime writes straight to
file descriptor 2.

Weights are referenced, never copied. `--model` defaults to `trainer/config.toml`'s
`pretrained_model_name_or_path` when that file is next to this one, else to the author's SDXL base.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FALLBACK_MODEL = "/opt/models/diffusers/waillu_170"
UNET_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
TE_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")


def default_model() -> str:
    """`trainer/config.toml` if it is reachable, otherwise the author's local SDXL base."""
    config = HERE.parents[1] / "trainer" / "config.toml"
    try:
        import tomllib

        with open(config, "rb") as handle:
            return str(tomllib.load(handle)["environment"]["pretrained_model_name_or_path"])
    except Exception:
        return FALLBACK_MODEL


def redirect_stderr(path: Path) -> None:
    """Point fd 2 at a file: the KFD abort and the HIP warning are written by the runtime itself,
    so Python-level redirection would not catch them (and the process dies before flushing)."""
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(handle, 2)


def build_lora(module, targets, dim: int, alpha: float, dropout: float, prefix: str) -> int:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=dim, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                        target_modules=list(targets))
    wrapped = get_peft_model(module, config)
    count = sum(1 for name, _ in wrapped.named_modules() if "lora_A" in name)
    print(f"  {prefix}: {count} LoRA adapters (r={dim}, alpha={alpha}, targets={list(targets)})",
          flush=True)
    if count == 0:
        raise SystemExit(f"peft matched no module for {prefix} targets {list(targets)}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, default=32, help="LoRA rank = the GEMM's K")
    parser.add_argument("--alpha", type=float, default=None, help="defaults to dim/2")
    parser.add_argument("--batch", type=int, default=3, help="train_batch_size")
    parser.add_argument("--chunks", type=int, default=2, help="CLIP chunks (seq = 77 * chunks)")
    parser.add_argument("--bucket", default="1024x768", help="latent source size")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--te", type=int, default=0, help="1 = also train LoRA in both CLIP encoders")
    parser.add_argument("--checkpointing", type=int, default=1,
                        help="1 = gradient checkpointing (default, like the trainer: without it a "
                             "1024x768 UNet backward at batch 3 does not fit in 16 GB)")
    parser.add_argument("--optimizer", default="none", choices=("none", "adamw", "schedulefree"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1145141920)
    parser.add_argument("--model", default=None)
    parser.add_argument("--log", default=str(HERE / "repro_min.stderr.log"))
    args = parser.parse_args()

    redirect_stderr(Path(args.log))
    import torch

    from diffusers import DDPMScheduler, UNet2DConditionModel

    if not torch.cuda.is_available():
        print("no CUDA/HIP device", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    device, dtype = torch.device("cuda"), torch.bfloat16
    model_dir = args.model or default_model()
    if not Path(model_dir).is_dir():
        print(f"model path not found: {model_dir} (pass --model)", file=sys.stderr)
        return 2

    dim = args.dim
    alpha = args.alpha if args.alpha is not None else dim / 2
    bucket_w, bucket_h = (int(v) for v in args.bucket.lower().split("x"))
    seq_len = 77 * args.chunks
    rows = args.batch * seq_len
    print(json.dumps({"model": model_dir, "dim": dim, "alpha": alpha, "batch": args.batch,
                      "chunks": args.chunks, "seq": seq_len, "M": rows, "K": dim, "N": 2048,
                      "bucket": args.bucket, "text_encoders": bool(args.te),
                      "checkpointing": bool(args.checkpointing), "optimizer": args.optimizer,
                      "iters": args.iters, "log": args.log}), flush=True)

    started = time.time()
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", torch_dtype=dtype)
    adapters = build_lora(unet, UNET_LORA_TARGETS, dim, alpha, args.dropout, "unet")
    unet.to(device=device, dtype=dtype)
    if args.checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()

    text_encoders: list = []
    if args.te:
        from transformers import CLIPTextModel, CLIPTextModelWithProjection

        for subfolder, cls in (("text_encoder", CLIPTextModel),
                               ("text_encoder_2", CLIPTextModelWithProjection)):
            encoder = cls.from_pretrained(model_dir, subfolder=subfolder, torch_dtype=dtype)
            adapters += build_lora(encoder, TE_LORA_TARGETS, dim, alpha, args.dropout, subfolder)
            text_encoders.append(encoder.to(device=device, dtype=dtype).train())

    scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler")
    params = [p for p in unet.parameters() if p.requires_grad]
    for encoder in text_encoders:
        params += [p for p in encoder.parameters() if p.requires_grad]

    optimizer = None
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr)
    elif args.optimizer == "schedulefree":
        try:
            from schedulefree import AdamWScheduleFree

            optimizer = AdamWScheduleFree(params, lr=args.lr)
            optimizer.train()  # schedulefree raises on step() unless it is in train mode
        except ImportError as error:
            print(f"schedulefree unavailable: {error}", file=sys.stderr)
            return 2

    latents = torch.randn(args.batch, 4, bucket_h // 8, bucket_w // 8, device=device, dtype=dtype)
    extra = {
        "text_embeds": torch.randn(args.batch, 1280, device=device, dtype=dtype),
        "time_ids": torch.tensor([bucket_h, bucket_w, 0, 0, bucket_h, bucket_w], device=device,
                                 dtype=dtype).unsqueeze(0).repeat(args.batch, 1),
    }
    if args.te:
        # Real CLIP forward, so the text-encoder LoRAs sit in the graph and their embeddings carry
        # gradients into the UNet cross-attention (the case fix2.txt reports as fatal).
        from transformers import CLIPTokenizer, CLIPTokenizerFast

        tokenizers = [CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer"),
                      CLIPTokenizerFast.from_pretrained(model_dir, subfolder="tokenizer_2")]
        ids = [tok(["a photo of a girl"], padding="max_length", max_length=77,
                   truncation=True, return_tensors="pt").input_ids.to(device) for tok in tokenizers]

        def encode() -> tuple:
            """SDXL's recipe: concatenate both encoders' *penultimate* hidden states (768 + 1280 =
            2048) and take the pooled embedding from the second encoder. The penultimate layer matters:
            the post-final-layer-norm states make the UNet's softmax overflow to NaN in bf16."""
            first = text_encoders[0](ids[0], output_hidden_states=True)
            second = text_encoders[1](ids[1], output_hidden_states=True)
            hidden = torch.cat([first.hidden_states[-2], second.hidden_states[-2]], dim=-1)
            # `chunks` copies of the 77-token caption: the sequence length the failing GEMM's M is
            # built from (the trainer chunks a real caption; only the length matters here).
            return hidden.repeat(args.batch, args.chunks, 1)[:, :seq_len], second.text_embeds
    else:
        cond = torch.randn(args.batch, seq_len, 2048, device=device, dtype=dtype)
        cond.requires_grad_(True)

    losses = []
    for step in range(1, args.iters + 1):
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (args.batch,),
                                  device=device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        if args.te:
            cond, pooled = encode()   # fresh graph each step, so the TE LoRA updates are included
            extra["text_embeds"] = pooled.repeat(args.batch, 1)
        prediction = unet(noisy, timesteps, encoder_hidden_states=cond,
                          added_cond_kwargs=extra).sample
        loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
        loss.backward()   # the documented fault happens inside one of these GEMMs
        losses.append(round(float(loss.detach()), 6))
        if optimizer:
            optimizer.step()
        torch.cuda.synchronize()
        print(f"  step {step}: loss {losses[-1]} "
              f"({time.time() - started:.1f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)",
              flush=True)

    print("RESULT " + json.dumps({
        "M": rows, "K": dim, "N": 2048, "batch": args.batch, "chunks": args.chunks,
        "text_encoders": bool(args.te), "checkpointing": bool(args.checkpointing),
        "optimizer": args.optimizer, "adapters": adapters, "steps_completed": args.iters,
        "losses": losses, "seconds": round(time.time() - started, 1),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "verdict": "ok",
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
