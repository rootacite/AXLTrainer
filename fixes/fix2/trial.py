#!/usr/bin/env python3
"""One trial of the gfx1201 LoRA-backward page-fault repro (see fixes/fix2.txt, fix2-ex.md).

Runs a single configuration in this process and prints one `RESULT {...}` JSON line when it
survives. A trial that hits the fault dies with SIGABRT and cannot report, so the driver
(`repro.py`) spawns one process per configuration and reads the ROCm abort message instead.

The graph mirrors the LoRA backward that faults in the real trainer: a pair of projections
N -> K -> N (K = `network_dim`) in bf16, whose input gradient is the documented
`[M, K] @ [K, N]` GEMM with M = train_batch_size * encoder_seq_len.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def build_stack(width: int, rank: int, layers: int, bias: bool) -> "torch.nn.Module":
    """`layers` residual LoRA-shaped projection pairs: an SDXL UNet attention block repeats this
    pattern dozens of times per step, which is what makes the backward graph large in the trainer."""
    import torch

    class Block(torch.nn.Module):
        """Named projections, so PEFT's `target_modules` cannot match the container by accident."""

        def __init__(self, width: int, rank: int, bias: bool) -> None:
            super().__init__()
            self.down = torch.nn.Linear(width, rank, bias=False)
            self.up = torch.nn.Linear(rank, width, bias=bias)

        def forward(self, x):
            return self.up(self.down(x)) + x

    class Stack(torch.nn.Module):
        def __init__(self, width: int, rank: int, layers: int, bias: bool) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(width, rank, bias) for _ in range(layers)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    return Stack(width, rank, layers, bias)


def build_manual(width: int, rank: int, dtype, device, layers: int):
    model = build_stack(width, rank, layers, bias=False)
    return model.to(device=device, dtype=dtype)


def build_peft(width: int, rank: int, dtype, device, dropout: float, layers: int):
    import torch
    from peft import LoraConfig, get_peft_model

    # bias=True on the up projection: the faulting Tensile kernel has a Bias epilogue.
    base = build_stack(width, rank, layers, bias=True).to(device=device, dtype=dtype)
    for param in base.parameters():
        param.requires_grad_(False)
    config = LoraConfig(
        r=rank, lora_alpha=max(1, rank // 2), lora_dropout=dropout, bias="none",
        target_modules=["down", "up"], task_type=None,
    )
    model = get_peft_model(base, config)
    return model.to(device=device, dtype=dtype)


def stress_allocator(sizes_mb: list[int], device):
    """Odd-sized live allocations, so the caching allocator packs buffers differently.

    fixes/fix2.txt: the Tensile kernel's overrun only aborts when it meets an unmapped page, so the
    allocator's layout decides whether a given shape dies.
    """
    import torch

    return [torch.empty(int(mb * 1e6 // 2), dtype=torch.bfloat16, device=device) for mb in sizes_mb]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True, help="M = batch_size * encoder_seq_len")
    parser.add_argument("--rank", type=int, required=True, help="K = network_dim")
    parser.add_argument("--width", type=int, default=2048, help="N = model hidden width")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--mode", default="peft", choices=["peft", "manual"])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--stress", type=int, default=0, help="1 = perturb the allocator layout first")
    parser.add_argument("--layers", type=int, default=1, help="LoRA projection pairs per step (SDXL: dozens)")
    parser.add_argument("--lived-gb", type=float, default=0.0,
                        help="keep this many GB resident, mimicking the loaded SDXL model")
    parser.add_argument("--rows2", type=int, default=0,
                        help="alternate M with this value across iterations (the trainer switched "
                             "between 231 and 462 whenever the caption's CLIP chunk count changed)")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("no CUDA/HIP device", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(0)

    pins = stress_allocator([3, 7, 13, 29, 61], device) if args.stress else []
    resident = []
    if args.lived_gb:
        resident = [torch.zeros(int(args.lived_gb * 1e9 // 2), dtype=torch.bfloat16, device=device)]
    if args.mode == "peft":
        model = build_peft(args.width, args.rank, dtype, device, args.dropout, args.layers)
    else:
        model = build_manual(args.width, args.rank, dtype, device, args.layers)
    # A dropout-free eval graph would hide the PEFT path; the real trainer runs in train mode.
    model.train()

    started = time.time()
    losses = []
    for index in range(args.iters):
        rows = args.rows
        if args.rows2 and index % 2:
            rows = args.rows2
        x = torch.randn(rows, args.width, device=device, dtype=dtype, requires_grad=True)
        out = model(x)
        loss = out.float().pow(2).mean()
        loss.backward()
        losses.append(float(loss.detach()))
        del x, out, loss
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() if hasattr(torch.cuda, "max_memory_allocated") else 0
    print("RESULT " + json.dumps({
        "rows": args.rows, "rank": args.rank, "width": args.width, "dtype": args.dtype,
        "mode": args.mode, "stress": args.stress, "iters": args.iters,
        "seconds": round(time.time() - started, 3), "first_loss": losses[0],
        "layers": args.layers, "lived_gb": args.lived_gb, "rows2": args.rows2,
        "last_loss": losses[-1], "peak_allocated_mb": round(peak / 1e6, 1),
        "kept_alive_mb": sum(size.numel() * size.element_size() for size in pins) / 1e6 if pins else 0.0,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
