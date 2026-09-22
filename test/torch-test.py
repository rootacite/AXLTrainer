#!/usr/bin/env python3
"""Basic torch exercise for this machine: tensor ops, a scan, backprop, a ~4 GiB memory
plateau held live for the whole run, plus alloc/free churn and host<->device transfers —
then a check that every number the stress loop produced is the number it should have.

Not a unit test — the name does not match `test*.py`, so `unittest discover -s test`
never collects it. Run it directly:

    python test/torch-test.py                     # ~2 s, ~4 GiB plateau
    python test/torch-test.py --time-scale 4      # four times the loop time
    python test/torch-test.py --mem-scale 0.5     # half the plateau
    python test/torch-test.py --time-scale 0      # correctness only, no loop
    python test/torch-test.py --driver-frees      # stage 2: every free is a hipFree
    python test/torch-test.py --driver-frees --linear  # stage 3: F.linear on the chain
    python test/torch-test.py --driver-frees --no-stress   # drop plateau + loop
    python test/torch-test.py --driver-frees --no-stress --max-checks 4
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

# The HIP caching allocator reads this once at import. Stage 2
#: every torch free becomes a hipFree.
if "--driver-frees" in sys.argv:
    os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"

import torch
import torch.nn.functional as F

# Baselines, tuned on the author's machine (RX 9070 XT, torch 2.13.0+rocm10.0.0, HIP
# 7.15.26333): at scale 1.0 they put the measured script time at ~2 s, of which the
# checks and the verification of the loop's output are the fixed ~1.4 s. Nothing here
# is exact, it is a roughly repeatable workload; --time-scale / --mem-scale scale it.
ITERS = 5
PLATEAU_GIB = 3.5
MATMUL_N = 4096
SCAN_N = 64 * 1024 * 1024
# bf16 block sizes for the plateau, cycled until the target size is reached, so the
# allocator sees a spread of sizes rather than one huge block.
CHUNK_MIB = (64, 96, 128, 192, 256, 320)
DTYPE = torch.bfloat16
PLATEAU_FILL = 0.5
# One bf16 spacing at 0.5. Exactly on the grid, so a swept block moves by a step the
# reference in swept_value() can predict — anything smaller rounds away to nothing.
SWEEP_STEP = 2**-8
# Alloc/free churn: this many fresh fp32 blocks per iteration, cycling through these
# sizes, with every block but the newest CHURN_LIVE freed again. empty_cache() at the
# end of the cycle hands the recycled blocks back to the driver, so the next iteration
# starts from a cold pool and the driver really sees hipMalloc/hipFree traffic.
CHURN_MIB = (0.25, 0.5, 1, 2, 4, 8, 16, 32)
CHURN_ALLOCS = 96
CHURN_LIVE = 8
CHURN_FILL = 0.25
# Host<->device churn: a pinned payload moved back and forth RT_REPS times per
# iteration, plus RT_SMALL small tensors moved once each way. Three more hops sit
# inside the compute chain itself: a bf16 operand, the last layer's gradient, and a
# scan slice that is scaled on the host and comes back as a broadcast bias, plus a
# slice of the swept plateau whose sum the device then consumes.
RT_MIB = 8
RT_REPS = 4
RT_SMALL = 128
RT_SMALL_KIB = 4
RT_BIAS_SCALE = 2.0
PLATEAU_HOP_ELEMS = 2 * 1024 * 1024
# Refuse a plateau bigger than this share of the card, so --mem-scale fails loudly
# instead of dying inside the allocator.
MEM_HEADROOM = 0.6


def log(msg: str = "") -> None:
    print(msg, flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    log(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def err_ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    """Largest deviation, relative to the scale of `want`.

    Per-element relative error is useless on prefix sums and matmul entries: both sit
    near zero by cancellation, where the relative error explodes while the absolute
    error stays at the rounding floor.
    """
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
        torch.equal(torch.cumsum(ones, 0), torch.arange(1, 4097, dtype=torch.float32, device=device)),
    )


def _check_scan_rows(device: torch.device) -> bool:
    rows = torch.randn(257, 999, device=device)
    return check(
        "scan: cumsum(dim=1) matches torch's own reference path",
        torch.allclose(torch.cumsum(rows, 1), rows.cumsum(dim=1)),
    )


# Overridden from main() so the mm check can be shrunk without rewriting it.
# MM_PACK: False | True (tight pack) | int MiB (one allocation of that many MiB).
MM_K = 512
MM_DTYPE = DTYPE
MM_PACK = False
MM_PACK_MIB = 0
MM_SYNC = False


def _check_bf16_mm(device: torch.device) -> bool:
    k = MM_K
    dtype = MM_DTYPE
    itemsize = torch.tensor([], dtype=dtype).element_size()
    if MM_PACK:
        # One hipMalloc, three views, 256-byte gaps like the caching allocator.
        nbytes = k * k * itemsize
        # Caching allocator placed the k=91 tensors 16896 B apart (512-aligned).
        stride = (nbytes + 511) // 512 * 512
        n_elem = stride // itemsize
        if MM_PACK_MIB:
            storage = torch.empty((MM_PACK_MIB * 1024 * 1024) // itemsize, device=device, dtype=dtype)
            layout = f"pack-{MM_PACK_MIB}mib"
        elif MM_PACK == "block":
            storage = torch.empty((2 * 1024 * 1024) // itemsize, device=device, dtype=dtype)
            layout = "pack-block"
        else:
            storage = torch.empty(n_elem * 3, device=device, dtype=dtype)
            layout = "pack"
        m1 = storage[0 : k * k].view(k, k)
        m2 = storage[n_elem : n_elem + k * k].view(k, k)
        out = storage[2 * n_elem : 2 * n_elem + k * k].view(k, k)
        m1.normal_()
        m2.normal_()
        torch.mm(m1, m2, out=out)
    else:
        m1 = torch.randn(k, k, device=device, dtype=dtype)
        m2 = torch.randn(k, k, device=device, dtype=dtype)
        out = m1 @ m2
        layout = "split"
    if MM_SYNC:
        torch.cuda.synchronize()
        layout = f"{layout}+sync"
    log(
        f"  mm va m1={m1.data_ptr():#x} m2={m2.data_ptr():#x} out={out.data_ptr():#x} "
        f"nbytes={m1.nbytes} shape={k}x{k} {dtype} layout={layout}"
    )
    got = out.double().cpu()
    ref = m1.double().cpu() @ m2.double().cpu()
    finite = bool(torch.isfinite(got).all().item())
    err = float((got - ref).abs().max().item())
    ok = finite and torch.allclose(got, ref, rtol=0.05, atol=1e-2)
    if not ok:
        diff = (got - ref).abs()
        idx = int(diff.view(-1).argmax().item())
        i, j = divmod(idx, k)
        tiles = (k + 15) // 16
        parts = []
        for ti in range(tiles):
            for tj in range(tiles):
                tile = diff[ti * 16 : (ti + 1) * 16, tj * 16 : (tj + 1) * 16]
                parts.append(f"({ti},{tj})={float(tile.max()):.3f}")
        log(
            f"  mm err peak at [{i},{j}] tile=({i // 16},{j // 16}) "
            f"got={float(got[i, j]):.6g} ref={float(ref[i, j]):.6g}; tiles {' '.join(parts)}"
        )
    return check(
        f"matmul: {dtype} {k}x{k} vs cpu fp64",
        ok,
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
    # First transposed-weight GEMM on this ROCm stack (`F.linear`, `x @ w.T`) costs
    # ~1.6 s of kernel loading; this check stays on `mm`.
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


# Order is the subtraction axis: drop a suffix with --max-checks, a prefix with --skip-checks.
CHECKS = (
    _check_cpu_elem,
    _check_scan_ones,
    _check_scan_rows,
    _check_bf16_mm,
    _check_autograd_cube,
    _check_autograd_mm,
    _check_autograd_autocast,
)


def checks(device: torch.device, *, skip: int = 0, limit: int | None = None) -> bool:
    """Cheap correctness samples; loose tolerances on purpose."""
    selected = CHECKS[skip:]
    if limit is not None:
        selected = selected[:limit]
    log(f"  checks [{skip}:{skip + len(selected)}] of {len(CHECKS)}")
    ok = True
    for fn in selected:
        ok &= fn(device)
    return ok


def plateau_sizes(total_bytes: int) -> list[int]:
    """bf16 element counts summing to about total_bytes."""
    sizes: list[int] = []
    total = 0
    i = 0
    while True:
        numel = CHUNK_MIB[i % len(CHUNK_MIB)] * 1024 * 1024 // 2
        if sizes and total + numel * 2 > total_bytes:
            break
        sizes.append(numel)
        total += numel * 2
        i += 1
    return sizes


def swept_value(iters: int) -> float:
    """What one plateau element holds after `iters` bf16 add_(SWEEP_STEP) calls.

    Rounded to bf16 after every step, the way the GPU add does — this is the reference
    the swept blocks are compared against in verify(). It saturates at 1.0, where the
    bf16 grid halves and the step lands on a tie; the same rounding reproduces that,
    so the comparison holds at any --time-scale.
    """
    value = PLATEAU_FILL
    for _ in range(iters):
        value = float(torch.tensor(value + SWEEP_STEP, dtype=DTYPE))
    return value


def chain_layer(x: torch.Tensor, w: torch.Tensor, linear: bool) -> torch.Tensor:
    """One layer of the stress chain: `torch.mm` or `F.linear` (x @ w.T)."""
    if linear:
        return F.linear(x, w)
    return torch.mm(x, w)


def churn_plan() -> list[int]:
    """fp32 element counts for one churn cycle."""
    return [int(CHURN_MIB[i % len(CHURN_MIB)] * 1024 * 1024) // 4 for i in range(CHURN_ALLOCS)]


def churn(device: torch.device, plan: list[int]) -> float:
    """Allocate blocks of varied sizes, compute on each, free all but the newest.

    Returns the sum of every block, so verify() can still check the values of blocks
    that no longer exist by the time the loop ends.
    """
    live: list[torch.Tensor] = []
    total = 0.0
    for numel in plan:
        block = torch.full((numel,), CHURN_FILL, device=device)
        total += float(block.sum().item())  # compute on the block just allocated
        live.append(block)
        if len(live) > CHURN_LIVE:
            live.pop(0)  # dropping the reference frees the block
    del live
    torch.cuda.empty_cache()
    return total


def transfers(device: torch.device, payload_elems: int, small_elems: int) -> dict:
    """Move data host<->device repeatedly, computing on both sides on the way through.

    `.to()` allocates a tensor per hop, so this is transfer churn and allocation churn
    together; the pinned host buffer is the shape a DataLoader would hand the device.
    """
    host = torch.randn(payload_elems, dtype=torch.float32, pin_memory=True)
    dev = host.to(device)
    for _ in range(RT_REPS):
        back = dev.to("cpu")  # D2H
        dev = back.to(device)  # H2D
        dev = dev.mul(2.0).div(2.0)  # compute between the hops; exact both ways
        float(back.sum().item())  # and a reduction on the host side

    small_sum = 0.0
    for i in range(RT_SMALL):
        # add_ keeps the buffer pinned; `arange(...) + i` would not.
        small = torch.arange(small_elems, dtype=torch.float32, pin_memory=True)
        small.add_(i)
        small_sum += float(small.to(device).to("cpu").sum().item())
    return {"dev": dev, "host": host, "small_sum": small_sum}


def stress(
    device: torch.device,
    plateau,
    scan,
    a,
    b,
    x,
    w1,
    w2,
    w3,
    iters: int,
    plan: list[int],
    payload_elems: int,
    small_elems: int,
    linear: bool = False,
):
    """One iteration: alloc/free churn, plateau sweep, scan, a three-layer fp32 chain
    with a bf16 term, the backward through all three layers, and host<->device transfers
    woven through the middle of it.

    Three layers forward and five matmuls back is deliberate: the backward is the phase
    this script is meant to weigh most. Returns per-iteration times, per-iteration
    losses, and the last iteration's tensors, so verify() can check what it produced.
    """
    times: list[float] = []
    losses: list[float] = []
    churn_total = 0.0
    hop_total = 0.0
    hop_slice = plateau[0][:PLATEAU_HOP_ELEMS]
    last: dict = {}
    for _ in range(iters):
        t0 = time.perf_counter()
        w1.grad = w2.grad = w3.grad = None  # fresh grads each iteration; keeps the peak flat
        churn_total += churn(device, plan)
        for chunk in plateau:
            chunk.add_(SWEEP_STEP)  # read+write the whole plateau
        # Interleaved: a slice of the block just swept goes out to the host and comes
        # back, and the device sums what came back before the scan starts.
        hop_back = hop_slice.to("cpu").to(device)
        hop_total += float(hop_back.float().sum().item())
        s = torch.cumsum(scan, 0)  # scan over 64 Mi elements
        s2 = torch.cumsum(s.view(-1, 1024), 1)  # and along the short axis
        # Mid-chain hop: a slice goes out to the host, is scaled there, and comes back
        # as a bias the matmuls downstream of it consume.
        bias = s[:MATMUL_N].to("cpu").mul(RT_BIAS_SCALE).to(device)
        # And a bf16 operand makes the round trip before its own matmul touches it.
        a_moved = a.to("cpu").to(device)
        h1 = chain_layer(x, w1, linear)  # fp32 chain, layer 1
        h2 = chain_layer(h1, w2, linear)  # layer 2
        h3 = chain_layer(h2, w3, linear)  # layer 3
        bfmm = torch.matmul(a_moved, b).float()  # bf16 matmul
        out = h3 + bfmm + bias
        loss_t = out.pow(2).mean()
        loss_t.backward()  # five matmuls back: d/dw3, d/dh2, d/dw2, d/dh1, d/dw1
        # The last layer's gradient goes out to the host and back, the way an
        # accumulation offload would move it; it has to return bit-identical.
        w3.grad = w3.grad.to("cpu").to(device)
        moved = transfers(device, payload_elems, small_elems)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss_t.detach().item())
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
            "scan": scan,
            "x": x,
            "a": a,
            "b": b,
            "w1": w1,
            "w2": w2,
            "w3": w3,
        }
    last["churn_total"] = churn_total
    last["hop_total"] = hop_total
    last["plan"] = plan
    last["linear"] = linear
    return times, losses, last


def verify(plateau, iters: int, losses: list[float], last: dict) -> bool:
    """Check the loop's results against computations that do not share their path."""
    ok = True

    # Plateau: every block must hold what `iters` bf16 steps produce, and min == max
    # catches a block — or a tail of one — that the sweep never reached.
    expect = swept_value(iters)
    tol = 2 * abs(expect) * float(torch.finfo(DTYPE).eps)
    lows = torch.stack([c.min() for c in plateau])
    highs = torch.stack([c.max() for c in plateau])
    ok &= check(
        f"plateau: {len(plateau)} blocks swept to {expect:.6f} +/- {tol:.6f}",
        bool((highs - lows).abs().max().item() <= tol and (highs - expect).abs().max().item() <= tol),
        f"observed [{lows.min().item():.6f}, {highs.max().item():.6f}]",
    )

    s, s2 = last["s"], last["s2"]
    scan, x, a, b = last["scan"], last["x"], last["a"], last["b"]
    w1, w2, w3 = last["w1"], last["w2"], last["w3"]
    rows = s.view(-1, 1024)  # what the 2-D scan was given

    r = err_ratio(s[:4096].double(), scan[:4096].double().cumsum(0))
    ok &= check("scan: 1-D head vs fp64 prefix sum", r <= 1e-4, f"rel-to-scale {r:.2e}")

    # The last element of a sequential scan and a tree reduction are different sums.
    ok &= check(
        "scan: tail of the 1-D scan == plain sum of the buffer",
        torch.allclose(s[-1], scan.sum(), rtol=1e-2, atol=10.0),
        f"{s[-1].item():.2f} vs {scan.sum().item():.2f}",
    )

    r = err_ratio(s2[:8, :256].double(), rows[:8, :256].double().cumsum(1))
    ok &= check("scan: 2-D head vs fp64 row prefix sums", r <= 1e-4, f"rel-to-scale {r:.2e}")

    r = err_ratio(s2[:, -1], rows.sum(dim=1))
    ok &= check("scan: 2-D row tails == row sums", r <= 1e-5, f"rel-to-scale {r:.2e}")

    h1, h2, h3 = last["h1"], last["h2"], last["h3"]
    bfmm, out = last["bfmm"], last["out"]
    linear = bool(last.get("linear"))
    half = MATMUL_N // 2
    if linear:
        split = F.linear(x[:, :half], w1[:, :half]) + F.linear(x[:, half:], w1[:, half:])
    else:
        split = x[:, :half] @ w1[:half] + x[:, half:] @ w1[half:]
    r = err_ratio(h1, split)
    ok &= check("chain: layer 1 vs split-K recompute", r <= 1e-4, f"rel-to-scale {r:.2e}")

    # Each layer's 32x32 corner against an fp64 cpu reference built from the previous
    # layer, so the chain is checked against something other than itself end to end.
    n, c = 32, 32
    for name, got, lhs, rhs in (
        ("layer 2", h2, h1, w2),
        ("layer 3", h3, h2, w3),
    ):
        lhs64 = lhs[:n].cpu().double()
        rhs64 = rhs.cpu().double()
        want = lhs64 @ rhs64[:c].T if linear else lhs64 @ rhs64[:, :c]
        r = err_ratio(got[:n, :c].double().cpu(), want)
        ok &= check(f"chain: {name} corner vs fp64 cpu", r <= 1e-4, f"rel-to-scale {r:.2e}")
    # The bf16 GEMM reduces its split-K partials in bf16 by default
    # (allow_bf16_reduced_precision_reduction), which costs ~2e-3 of the scale here.
    r = err_ratio(bfmm[:n, :c].double().cpu(), a[:n].cpu().double() @ b[:, :c].cpu().double())
    ok &= check("matmul: bf16 term corner vs fp64 cpu", r <= 1e-2, f"rel-to-scale {r:.2e}")
    ref64 = h3[:n, :c].double().cpu()
    ref64 += a[:n].cpu().double() @ b[:, :c].cpu().double()
    ref64 += last["bias"][:c].double().cpu()
    r = err_ratio(out[:n, :c].double().cpu(), ref64)
    ok &= check("chain: out (layer 3 + bf16 + bias) vs fp64 cpu", r <= 1e-2, f"rel-to-scale {r:.2e}")

    # The transfers that sat inside the chain: what came back has to be what went out.
    ok &= check(
        "transfers: the round-tripped bf16 operand is bit-identical",
        torch.equal(last["a_moved"], a),
    )
    ok &= check(
        f"transfers: the round-tripped bias == {RT_BIAS_SCALE:g} x the scan slice",
        torch.equal(last["bias"], s[:MATMUL_N] * RT_BIAS_SCALE),
    )
    # The plateau slice sat on the block that had been swept that many times, so what
    # came back must hold the swept value, and the sums taken from it must add up.
    hop = last["hop_back"]
    ok &= check(
        "transfers: the round-tripped plateau slice holds the swept value",
        torch.equal(hop, torch.full_like(hop, swept_value(iters))),
    )
    want_hop = hop.numel() * sum(swept_value(k) for k in range(1, iters + 1))
    ok &= check(
        "transfers: the sums computed on that slice match the sweep",
        math.isclose(last["hop_total"], want_hop, rel_tol=1e-5),
        f"{last['hop_total']:.6g} vs {want_hop:.6g}",
    )

    numel = out.numel()
    acc = 0.0
    for i in range(0, MATMUL_N, 1024):  # fp64 accumulation, chunked
        acc += float(out[i : i + 1024].double().pow(2).sum().item())
    want_loss = acc / numel
    ok &= check(
        "loss: mean(out^2) vs fp64 chunked accumulation",
        math.isclose(losses[-1], want_loss, rel_tol=1e-5),
        f"{losses[-1]:.6f} vs {want_loss:.6f}",
    )

    # Every layer's gradient from the chain rule, written out here without autograd.
    # w3.grad is the one that made the host round trip.
    # mm: y = x @ W → dW = x.T @ dy, dx = dy @ W.T
    # F.linear: y = x @ W.T → dW = dy.T @ x, dx = dy @ W
    grad_out = (2 * out / numel).detach()
    if linear:
        grad_h2 = grad_out @ w3.detach()
        grad_h1 = grad_h2 @ w2.detach()
        grad_wants = (
            ("w3.grad (after its round trip)", w3.grad, grad_out.T @ h2.detach()),
            ("w2.grad", w2.grad, grad_h2.T @ h1.detach()),
            ("w1.grad (three layers back)", w1.grad, grad_h1.T @ x),
        )
    else:
        grad_h2 = grad_out @ w3.detach().T
        grad_h1 = grad_h2 @ w2.detach().T
        grad_wants = (
            ("w3.grad (after its round trip)", w3.grad, h2.detach().T @ grad_out),
            ("w2.grad", w2.grad, h1.detach().T @ grad_h2),
            ("w1.grad (three layers back)", w1.grad, x.T @ grad_h1),
        )
    for name, got, want in grad_wants:
        r = err_ratio(got, want)
        ok &= check(f"backward: {name} vs chain rule", r <= 1e-3, f"rel-to-scale {r:.2e}")

    # Alloc/free churn: every block the loop ever allocated summed to its fill value,
    # although only the newest five existed at any moment.
    plan = last["plan"]
    want_churn = iters * sum(plan) * CHURN_FILL
    ok &= check(
        f"churn: {iters * len(plan)} blocks summed to their fill value",
        math.isclose(last["churn_total"], want_churn, rel_tol=1e-5),
        f"{last['churn_total']:.6g} vs {want_churn:.6g}",
    )

    # Transfers: a round trip must not change a bit, and the small tensors' sums must
    # match sum(arange(n) + i), worked out here without torch.
    moved = last["moved"]
    ok &= check(
        f"transfers: {RT_REPS * 2} hops of {RT_MIB} MiB came back bit-identical",
        torch.equal(moved["dev"].detach().cpu(), moved["host"]),
    )
    small_elems = RT_SMALL_KIB * 1024
    want_small = sum(small_elems * (small_elems - 1) / 2 + small_elems * i for i in range(RT_SMALL))
    ok &= check(
        f"transfers: {RT_SMALL} small round trips match the closed-form sum",
        math.isclose(moved["small_sum"], want_small, rel_tol=1e-12),
        f"{moved['small_sum']:.6g} vs {want_small:.6g}",
    )

    # The same work twice must produce the same bits: this is what catches a compute
    # fault that leaves no trace in the kernel log.
    spread = (max(losses) - min(losses)) / abs(losses[-1])
    ok &= check(
        "loop: identical work gives an identical loss every iteration",
        spread <= 1e-6,
        f"relative spread {spread:.2e} over {len(losses)} iterations",
    )
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="torch stress: tensor ops, a scan, backprop and a memory plateau, all verified"
    )
    ap.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help=f"multiplies the baseline iteration count (1.0 = {ITERS} iterations)",
    )
    ap.add_argument(
        "--mem-scale",
        type=float,
        default=1.0,
        help=f"multiplies the baseline plateau (1.0 = {PLATEAU_GIB} GiB)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--driver-frees",
        action="store_true",
        help="stage 2: disable the HIP caching allocator so every torch free is a hipFree",
    )
    ap.add_argument(
        "--linear",
        action="store_true",
        help="stage 3: F.linear (transposed-weight GEMM) on the three-layer chain",
    )
    ap.add_argument(
        "--no-stress",
        action="store_true",
        help="drop the plateau, stress loop, verify(), and the size/peak checks",
    )
    ap.add_argument(
        "--skip-checks",
        type=int,
        default=0,
        metavar="N",
        help="drop the first N correctness samples (subtraction prefix)",
    )
    ap.add_argument(
        "--max-checks",
        type=int,
        default=None,
        metavar="N",
        help="run at most N correctness samples after --skip-checks (default: all)",
    )
    ap.add_argument(
        "--mm-k",
        type=int,
        default=512,
        metavar="N",
        help="square size of the bf16/fp32 matmul check (default 512)",
    )
    ap.add_argument(
        "--mm-dtype",
        choices=("bf16", "fp32", "fp16"),
        default="bf16",
        help="dtype of the matmul check (default bf16)",
    )
    ap.add_argument(
        "--mm-pack",
        action="store_true",
        help="put A, B and C of the mm check in one allocation (one VMM mapping)",
    )
    ap.add_argument(
        "--mm-pack-block",
        action="store_true",
        help="like --mm-pack but the allocation is a full 2 MiB granule (no slack)",
    )
    ap.add_argument(
        "--mm-pack-mib",
        type=int,
        default=0,
        metavar="N",
        help="like --mm-pack but the allocation is N MiB (0 = off)",
    )
    ap.add_argument(
        "--mm-sync",
        action="store_true",
        help="torch.cuda.synchronize() after the mm, before the CPU compare",
    )
    args = ap.parse_args(argv)
    if args.time_scale < 0 or args.mem_scale <= 0:
        ap.error("--time-scale must be >= 0 and --mem-scale must be > 0")
    if args.skip_checks < 0 or args.skip_checks > len(CHECKS):
        ap.error(f"--skip-checks must be in 0..{len(CHECKS)}")
    if args.max_checks is not None and args.max_checks < 0:
        ap.error("--max-checks must be >= 0")
    if args.mm_k < 1:
        ap.error("--mm-k must be >= 1")
    global MM_K, MM_DTYPE, MM_PACK, MM_PACK_MIB, MM_SYNC
    MM_K = args.mm_k
    MM_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.mm_dtype]
    MM_PACK = "block" if args.mm_pack_block else (True if args.mm_pack or args.mm_pack_mib else False)
    MM_PACK_MIB = args.mm_pack_mib if args.mm_pack_mib > 0 else 0
    MM_SYNC = args.mm_sync

    if not torch.cuda.is_available():
        log("no CUDA/ROCm device available — nothing to run")
        return 2
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    t_start = time.perf_counter()

    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    iters = round(ITERS * args.time_scale)
    plateau_gib = PLATEAU_GIB * args.mem_scale
    plan = churn_plan()
    payload_elems = RT_MIB * 1024 * 1024 // 4
    small_elems = RT_SMALL_KIB * 1024
    if iters and plateau_gib + 0.8 > MEM_HEADROOM * total_gib:
        ap.error(
            f"--mem-scale {args.mem_scale} wants a {plateau_gib:.2f} GiB plateau, more than "
            f"{MEM_HEADROOM:.0%} of this card's {total_gib:.2f} GiB"
        )

    log(
        f"torch {torch.__version__}  hip {torch.version.hip}  "
        f"{torch.cuda.get_device_name(0)}  {total_gib:.2f} GiB"
    )
    log(
        f"time-scale {args.time_scale:g} -> {iters} iterations  "
        f"mem-scale {args.mem_scale:g} -> {plateau_gib:.2f} GiB plateau  "
        f"matmul={MATMUL_N}x{MATMUL_N}  scan={SCAN_N / 2**20:.0f} Mi elements  dtype={DTYPE}"
    )
    log(
        f"per iteration: {CHURN_ALLOCS} allocations of {sum(plan) * 4 / 2**20:.0f} MiB total "
        f"(at most {CHURN_LIVE} live) + {RT_REPS * 2} hops of {RT_MIB} MiB "
        f"+ {RT_SMALL} small transfers, and inside the chain a bf16 operand, a sweep slice, "
        f"a 2 x {PLATEAU_HOP_ELEMS / 2**20:.0f} Mi plateau slice and w3.grad each make a host round trip"
    )
    if args.driver_frees:
        log(
            "driver-frees: PYTORCH_NO_HIP_MEMORY_CACHING=1 "
            f"(env={os.environ.get('PYTORCH_NO_HIP_MEMORY_CACHING', '')!r})"
        )
    if args.linear:
        log("linear: stress chain uses F.linear (x @ W.T)")
    if args.no_stress:
        log("no-stress: plateau, loop, verify and size checks are skipped")
    if args.mm_k != 512 or args.mm_dtype != "bf16" or args.mm_pack:
        log(
            f"matmul check: {args.mm_dtype} {args.mm_k}x{args.mm_k}"
            + (" pack=1" if args.mm_pack else "")
        )
    caching_on = torch._C._cuda_cudaCachingAllocator_is_enabled()
    log(
        f"caching_allocator enabled={caching_on} "
        f"PYTORCH_NO_HIP_MEMORY_CACHING={os.environ.get('PYTORCH_NO_HIP_MEMORY_CACHING')!r}"
    )

    ok = True
    log("\ncorrectness samples")
    ok &= checks(device, skip=args.skip_checks, limit=args.max_checks)

    if args.no_stress:
        log(f"\n{'PASS' if ok else 'FAIL'}  script wall time {time.perf_counter() - t_start:.2f} s")
        return 0 if ok else 1

    log("\nmemory plateau + stress")
    sizes = plateau_sizes(int(plateau_gib * 2**30))
    # fill-style init: the plateau is here to occupy mappings, not to test RNG speed.
    plateau = [torch.full((n,), PLATEAU_FILL, dtype=DTYPE, device=device) for n in sizes]
    scan = torch.randn(SCAN_N, device=device)
    a = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=DTYPE)
    b = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=DTYPE)
    x = torch.randn(MATMUL_N, MATMUL_N, device=device)
    w1 = torch.randn(MATMUL_N, MATMUL_N, device=device, requires_grad=True)
    w2 = torch.randn(MATMUL_N, MATMUL_N, device=device, requires_grad=True)
    w3 = torch.randn(MATMUL_N, MATMUL_N, device=device, requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    held = torch.cuda.memory_allocated()
    log(
        f"  {len(plateau)} live blocks, {held / 2**30:.2f} GiB allocated, "
        f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved before the loop"
    )

    losses: list[float] = []
    if iters:
        times, losses, last = stress(
            device,
            plateau,
            scan,
            a,
            b,
            x,
            w1,
            w2,
            w3,
            iters,
            plan,
            payload_elems,
            small_elems,
            linear=args.linear,
        )
        log(
            f"  {iters} iterations: total {sum(times):.2f} s, per-iteration "
            f"min {min(times) * 1e3:.1f} / median {statistics.median(times) * 1e3:.1f} "
            f"/ max {max(times) * 1e3:.1f} ms"
        )
        log("\nverification (loop output vs independent computations)")
        ok &= verify(plateau, iters, losses, last)
        del last
    else:
        log("  0 iterations: the loop and its verification are skipped")

    peak = torch.cuda.max_memory_allocated()
    def _nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    live_bytes = (
        sum(_nbytes(c) for c in plateau)
        + _nbytes(scan)
        + _nbytes(a)
        + _nbytes(b)
        + _nbytes(x)
        + _nbytes(w1)
        + _nbytes(w2)
        + _nbytes(w3)
    )
    log(
        f"\n  peak allocated {peak / 2**30:.2f} GiB "
        f"(reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB); "
        f"live tensors {live_bytes / 2**30:.2f} GiB"
    )
    # PYTORCH_NO_HIP_MEMORY_CACHING makes the caching-allocator counters stay at 0
    # (the tensors are live; verify() already read them). Size the hold from the
    # tensors themselves, and skip the peak-leak check that has nothing to read.
    held_for_size = live_bytes if args.driver_frees else held
    ok &= check(
        "memory plateau holds the requested size",
        plateau_gib * 2**30 <= held_for_size <= (plateau_gib + 1.5) * 2**30,
        f"{held_for_size / 2**30:.2f} GiB held for a {plateau_gib:.2f} GiB plateau",
    )
    if args.driver_frees:
        log("  peak-leak check skipped (caching-allocator stats are zero with --driver-frees)")
    else:
        ok &= check(
            "peak stays within 3 GiB of the plateau (no per-iteration leak)",
            peak <= held + 3.0 * 2**30,
            f"peak {peak / 2**30:.2f} GiB vs held {held / 2**30:.2f} GiB",
        )

    del plateau, scan, a, b, x, w1, w2, w3
    torch.cuda.empty_cache()
    log(
        f"  after teardown: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
        f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved"
    )

    log(f"\n{'PASS' if ok else 'FAIL'}  script wall time {time.perf_counter() - t_start:.2f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
