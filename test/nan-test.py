#!/usr/bin/env python3
"""Minimal extract of test/torch-test.py that still hits the two FAILs under the hook.

Stripped: argparse, --time-scale/--mem-scale, --driver-frees, --linear, the cheap
CHECK flags, peak-memory / plateau / churn / transfer / loss verify.

Kept, because dropping them made the FAILs disappear:
  - the seven correctness samples (allocator warmup)
  - 5-iter stress body (1–3 iters, even with the samples, did not fire)

    python test/nan-test.py
    LD_PRELOAD=amdfq/amdfq-tail-rs/target/release/libamdfq_tail_rs.so python test/nan-test.py
"""
from __future__ import annotations

import sys

import torch

ITERS = 5
MATMUL_N = 4096
SCAN_N = 64 * 1024 * 1024
CHUNK_MIB = (64, 96, 128, 192, 256, 320)
CHURN_MIB = (0.25, 0.5, 1, 2, 4, 8, 16, 32)
CHURN_ALLOCS = 96
CHURN_LIVE = 8
RT_MIB = 8
RT_REPS = 4
RT_SMALL = 128
RT_SMALL_KIB = 4
RT_BIAS_SCALE = 2.0
PLATEAU_HOP_ELEMS = 2 * 1024 * 1024
SWEEP_STEP = 2**-8
DTYPE = torch.bfloat16


def log(msg: str = "") -> None:
    print(msg, flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    log(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def err_ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach(), want.detach()
    return float((got - want).abs().max()) / max(float(want.abs().max()), 1e-12)


def _check_cpu_elem(_device: torch.device) -> bool:
    a = torch.randn(1 << 20)
    b = torch.randn(1 << 20)
    got = ((a + b) * 2).sum()
    want = ((a.double() + b.double()) * 2).sum()
    return check(
        "elementwise + reduce vs cpu fp64",
        torch.allclose(got.double(), want, rtol=1e-6),
        f"gpu {got.item():.4f} cpu {want.item():.4f}",
    )


def _check_scan_ones(device: torch.device) -> bool:
    ones = torch.ones(4096, device=device)
    return check(
        "scan: cumsum(ones) == 1..n",
        torch.equal(
            torch.cumsum(ones, 0),
            torch.arange(1, 4097, dtype=torch.float32, device=device),
        ),
    )


def _check_scan_rows(device: torch.device) -> bool:
    rows = torch.randn(257, 999, device=device)
    return check(
        "scan: cumsum(dim=1) matches torch's own reference path",
        torch.allclose(torch.cumsum(rows, 1), rows.cumsum(dim=1)),
    )


def _check_bf16_mm(device: torch.device) -> bool:
    k = 512
    m1 = torch.randn(k, k, device=device, dtype=DTYPE)
    m2 = torch.randn(k, k, device=device, dtype=DTYPE)
    out = m1 @ m2
    log(
        f"  mm va m1={m1.data_ptr():#x} m2={m2.data_ptr():#x} out={out.data_ptr():#x} "
        f"nbytes={m1.nbytes} shape={k}x{k} {DTYPE} layout=split"
    )
    got = out.double().cpu()
    ref = m1.double().cpu() @ m2.double().cpu()
    finite = bool(torch.isfinite(got).all().item())
    err = float((got - ref).abs().max().item())
    return check(
        f"matmul: {DTYPE} {k}x{k} vs cpu fp64",
        finite and torch.allclose(got, ref, rtol=0.05, atol=1e-2),
        f"finite={finite} max abs err {err:.3f}",
    )


def _check_autograd_cube(device: torch.device) -> bool:
    x = torch.randn(64, device=device, dtype=torch.float64, requires_grad=True)
    (x**3).sum().backward()
    return check("autograd: d/dx x^3 == 3x^2", torch.allclose(x.grad, 3 * x.detach() ** 2))


def _check_autograd_mm(device: torch.device) -> bool:
    m, n = 2048, 2048
    xm = torch.randn(m, n, device=device)
    wm = torch.randn(n, n, device=device, requires_grad=True)
    (xm @ wm).pow(2).mean().backward()
    want = 2 * (xm.T @ (xm @ wm).detach()) / (m * n)
    return check(
        "autograd: matmul weight grad vs analytic",
        torch.allclose(wm.grad, want, rtol=1e-4, atol=1e-4),
        f"max rel err {((wm.grad - want).abs() / want.abs().clamp_min(1e-6)).max().item():.2e}",
    )


def _check_autograd_autocast(device: torch.device) -> bool:
    m, n = 2048, 2048
    xm = torch.randn(m, n, device=device)
    wm = torch.randn(n, n, device=device, requires_grad=True)
    with torch.autocast("cuda", dtype=DTYPE):
        mixed = (xm @ wm).pow(2).mean()
    mixed.backward()
    return check(
        "autograd: bf16 autocast backward is finite",
        torch.isfinite(wm.grad).all().item(),
    )


def checks(device: torch.device) -> None:
    for fn in (
        _check_cpu_elem,
        _check_scan_ones,
        _check_scan_rows,
        _check_bf16_mm,
        _check_autograd_cube,
        _check_autograd_mm,
        _check_autograd_autocast,
    ):
        fn(device)


def churn(device: torch.device) -> None:
    live: list[torch.Tensor] = []
    plan = [int(CHURN_MIB[i % len(CHURN_MIB)] * 1024 * 1024) // 4 for i in range(CHURN_ALLOCS)]
    for numel in plan:
        block = torch.full((numel,), 0.25, device=device)
        float(block.sum().item())
        live.append(block)
        if len(live) > CHURN_LIVE:
            live.pop(0)
    del live
    torch.cuda.empty_cache()


def transfers(device: torch.device) -> dict:
    host = torch.randn(RT_MIB * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True)
    dev = host.to(device)
    for _ in range(RT_REPS):
        back = dev.to("cpu")
        dev = back.to(device)
        dev = dev.mul(2.0).div(2.0)
        float(back.sum().item())
    small_elems = RT_SMALL_KIB * 1024
    for i in range(RT_SMALL):
        small = torch.arange(small_elems, dtype=torch.float32, pin_memory=True)
        small.add_(i)
        float(small.to(device).to("cpu").sum().item())
    return {"dev": dev, "host": host}


def main() -> int:
    if not torch.cuda.is_available():
        log("no CUDA/ROCm device")
        return 2
    device = torch.device("cuda")
    torch.manual_seed(0)
    log(f"torch {torch.__version__}  hip {torch.version.hip}  {torch.cuda.get_device_name(0)}")

    checks(device)

    plateau: list[torch.Tensor] = []
    total = 0
    i = 0
    budget = int(3.5 * 2**30)
    while True:
        numel = CHUNK_MIB[i % len(CHUNK_MIB)] * 1024 * 1024 // 2
        if plateau and total + numel * 2 > budget:
            break
        plateau.append(torch.full((numel,), 0.5, dtype=DTYPE, device=device))
        total += numel * 2
        i += 1
    scan = torch.randn(SCAN_N, device=device)
    a = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=DTYPE)
    b = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=DTYPE)
    x = torch.randn(MATMUL_N, MATMUL_N, device=device)
    w1 = torch.randn(MATMUL_N, MATMUL_N, device=device, requires_grad=True)
    w2 = torch.randn(MATMUL_N, MATMUL_N, device=device, requires_grad=True)
    w3 = torch.randn(MATMUL_N, MATMUL_N, device=device, requires_grad=True)
    hop_slice = plateau[0][:PLATEAU_HOP_ELEMS]
    last: dict = {}

    for xx in range(ITERS):
        if xx == 4:
            pass
        w1.grad = w2.grad = w3.grad = None
        churn(device)
        for chunk in plateau:
            chunk.add_(SWEEP_STEP)
        hop_back = hop_slice.to("cpu").to(device)
        float(hop_back.float().sum().item())
        s = torch.cumsum(scan, 0)
        s2 = torch.cumsum(s.view(-1, 1024), 1)
        bias = s[:MATMUL_N].to("cpu").mul(RT_BIAS_SCALE).to(device)
        a_moved = a.to("cpu").to(device)
        h1 = torch.mm(x, w1)
        h2 = torch.mm(h1, w2)
        h3 = torch.mm(h2, w3)
        bfmm = torch.matmul(a_moved, b).float()
        out = h3 + bfmm + bias
        out.pow(2).mean().backward()
        w3.grad = w3.grad.to("cpu").to(device)
        moved = transfers(device)
        torch.cuda.synchronize()
        # Keep the same live set as torch-test.stress(): the previous iter stays
        # allocated until this assignment, which is part of the packing.
        last = {
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "out": out,
            "bfmm": bfmm,
            "bias": bias,
            "a_moved": a_moved,
            "hop_back": hop_back,
            "s": s,
            "s2": s2,
            "moved": moved,
        }

    s, s2, h2, out = last["s"], last["s2"], last["h2"], last["out"]
    rows = s.view(-1, 1024)
    r1 = err_ratio(s2[:, -1], rows.sum(dim=1))
    ok = check("scan: 2-D row tails == row sums", r1 <= 1e-5, f"rel-to-scale {r1:.2e}")
    grad_out = (2 * out / out.numel()).detach()
    r2 = err_ratio(w3.grad, h2.detach().T @ grad_out)
    ok &= check(
        "backward: w3.grad (after its round trip) vs chain rule",
        r2 <= 1e-3,
        f"rel-to-scale {r2:.2e}",
    )
    log(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
