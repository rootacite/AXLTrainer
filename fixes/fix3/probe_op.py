#!/usr/bin/env python3
"""Name the operation that overruns: log every GEMM the first step launches, then run the trainer.

The KFD abort names a Tensile kernel and a faulting address, but not which `aten` operation asked for
it — and the faulting kernel carries a *forward*-shaped name (`_Bias_`, i.e. `addmm`) whose tile
geometry does not obviously match any single LoRA GEMM in the step. This script replaces the trainer's
entry point (`repro_step1.py --child-script probe_op.py`) and installs a `TorchDispatchMode` that
prints every `addmm` / `mm` / `bmm` / `matmul` with its operand shapes, flushing every line.

Run it under `HIP_LAUNCH_BLOCKING=1`: with kernel launches serialized, the last GEMM line before the
`Memory Fault Error` line is the one whose kernel faulted. Each line carries the autograd state, which
separates the forward pass (`grad=True`) from the backward pass (`grad=False`).

    python repro_step1.py --row "step-1 abort: repo config (dim 48, batch 2)" --in-place \
        --child-script probe_op.py --env HIP_LAUNCH_BLOCKING=1

Everything is written into that row's mirror directory (`probe_op.gemms.log`), which survives the
kill; `<log>.last-gemms.json` keeps a small ring buffer of the most recent calls.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

PRINT_OPS = os.environ.get("AXL_PROBE_PRINT", "1") == "1"
KEEP = int(os.environ.get("AXL_PROBE_KEEP", "30"))
RING_EVERY = int(os.environ.get("AXL_PROBE_RING_EVERY", "10"))
# The faulting kernel is a Tensile GEMM, but MIOpen compiles *convolutions* to Tensile GEMMs too, and
# the UNet is mostly 3x3 convolutions. Logging only `aten.mm`/`addmm` therefore misses the case where
# the overrunning kernel came from a conv — that is exactly what the 2560-workgroup grid suggests,
# since no `mm` in this step has a shape that tiles to 2560.
LOGGED_OPS = {
    "aten.addmm.default", "aten.mm.default", "aten.bmm.default", "aten.matmul.default",
    "aten.addmv.default", "aten.baddbmm.default",
    "aten.convolution.default", "aten.convolution_backward.default",
    "aten._convolution.default", "aten.miopen_convolution.default",
    "aten.miopen_convolution_backward.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten._flash_attention_forward.default",
    "aten._efficient_attention_forward.default",
    "aten._softmax.default",
}
STATE: dict = {"count": 0, "ring": [], "forward": 0, "backward": 0, "backward_passes": 0}


def _describe(args) -> list:
    """Shapes plus the scalars that describe a convolution (stride, padding, dilation, groups).

    For a conv, the interesting numbers are not the input/output shapes but the ones MIOpen turns
    into the GEMM's K: `C_in/groups * kh * kw`.
    """
    out = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            out.append(_describe(arg))
        elif hasattr(arg, "shape"):
            out.append(list(arg.shape))
        else:
            out.append(repr(arg)[:24])
    return out


def install() -> None:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    with open("probe_op.gemms.log", "w", encoding="utf-8") as handle:
        handle.write("# n  pass  op  operands\n")

    class GemmLogger(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if name not in LOGGED_OPS:
                return func(*args, **(kwargs or {}))
            state = torch.is_grad_enabled()
            STATE["count"] += 1
            STATE["forward" if state else "backward"] += 1
            record = {"n": STATE["count"], "pass": "forward" if state else "backward",
                      "op": name, "args": _describe(args)}
            STATE["ring"] = (STATE["ring"] + [record])[-KEEP:]
            if PRINT_OPS:
                # Printed to this process's stdout, which is where MIOpen writes the kernel it selects
                # for a convolution. The op line that immediately precedes a MIOpen candidate dump is
                # the operation whose kernel faulted.
                print(f"OP {record['n']} {record['pass']} {name} {record['args']}", flush=True)
            with open("probe_op.gemms.log", "a", encoding="utf-8") as handle:
                handle.write(f"{record['n']:06d} {record['pass']:<8} {name} {record['args']}\n")
            if STATE["count"] % RING_EVERY == 0:
                with open("probe_op.last-gemms.json", "w", encoding="utf-8") as handle:
                    json.dump({"count": STATE["count"], "forward": STATE["forward"],
                               "backward": STATE["backward"], "last": STATE["ring"]}, handle)
            return func(*args, **(kwargs or {}))

    GemmLogger().__enter__()

    # A python-level boundary: `autograd.backward` is called once per step, so a `BACKWARD START`
    # line in the log places the abort in the backward rather than the forward, independently of
    # what the last GEMM line says.
    real_backward = torch.autograd.backward

    def logged_backward(*args, **kwargs):
        STATE["backward_passes"] += 1
        with open("probe_op.gemms.log", "a", encoding="utf-8") as handle:
            handle.write(f"###### BACKWARD START {STATE['backward_passes']} "
                         f"(after {STATE['forward']} forward GEMMs, {STATE['count']} total)\n")
        return real_backward(*args, **kwargs)

    torch.autograd.backward = logged_backward


if __name__ == "__main__":
    install()
    # `python trainer/main.py` puts `trainer/` on `sys.path[0]`; `runpy` does not, and the trainer's
    # modules import each other as top-level names (`from config import TrainConfig`).
    sys.path.insert(0, str(Path("trainer").resolve()))
    sys.argv = [sys.argv[0]]
    runpy.run_path("trainer/main.py", run_name="__main__")
