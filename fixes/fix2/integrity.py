#!/usr/bin/env python3
"""Did the fault also damage runs that *did not* crash? Compare the numbers byte for byte.

Every captured abort is `RW: 0x0` — the Tensile kernel **reads** one page past a buffer. A read
cannot scribble on a neighbouring tensor, so the silent failure mode to look for is different: if
that page happens to be mapped, the kernel reads garbage into its own GEMM result, the step produces
wrong numbers, and nothing crashes and no dmesg line appears.

Two independent probes, both from the `--grid integrity` rows:

1. **Allocator layout.** The same computation at different addresses. `PYTORCH_NO_HIP_MEMORY_CACHING=1`
   gives every tensor its own `hipMalloc`, and `expandable_segments:True` gives a third layout, so if
   an overread ever fed garbage into a result, these runs could not stay byte-identical.
2. **Sampling on/off.** With `validation.sample_seed = 1` the sample pass uses its own generator
   (`trainer/sampling.py`; with `sample_seed = 0` it draws from the global RNG), so a run with samples
   every 30 steps and a run with none must produce the exact same 120 losses unless the sample pass
   changes training state.

Compares, for every pair of runs: the per-step `Train/Loss` series, the LoRA tensors saved along the
way, and the sample PNGs.

    python integrity.py --work-dir /tmp/axl-fix2-repro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def loss_series(run_root: Path) -> dict[int, float]:
    """`Train/Loss` per step from the run's TensorBoard directory."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logs = run_root / "logs"
    run_dirs = [path for path in logs.iterdir() if path.is_dir()] if logs.is_dir() else []
    if not run_dirs:
        return {}
    accumulator = EventAccumulator(str(max(run_dirs, key=lambda p: p.stat().st_mtime)))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    tag = next((name for name in ("Train/Loss", "train/loss") if name in tags), None)
    if tag is None:
        return {}
    return {event.step: event.value for event in accumulator.Scalars(tag)}


def load_tensors(path: Path) -> dict:
    """Every tensor of a safetensors file, as saved (bf16 stays bf16)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def digest(tensor) -> str:
    """sha256 of the raw bytes; bf16 has no numpy dtype, so go through uint8."""
    # reshape first: a 0-dim tensor (LoRA `alpha`) cannot be viewed as uint8
    return hashlib.sha256(tensor.reshape(-1).contiguous().view(torch.uint8)
                          .numpy().tobytes()).hexdigest()


def checkpoints(run_root: Path) -> dict[str, Path]:
    """Every saved LoRA file of a run, keyed by directory name (step / epoch / final)."""
    outputs = run_root / "outputs"
    found: dict[str, Path] = {}
    for run_dir in outputs.glob("*"):
        if not run_dir.is_dir():
            continue
        for weights in run_dir.rglob("*.safetensors"):
            found[weights.parent.name.replace(run_dir.name + "_", "")] = weights
    return found


def sample_hashes(run_root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for png in (run_root / "outputs").rglob("*.png"):
        digests[png.name] = hashlib.sha256(png.read_bytes()).hexdigest()
    return digests


def compare(left: dict, right: dict, *, label: str, pairs: str) -> list[str]:
    lines: list[str] = []
    left_series, right_series = left["loss"], right["loss"]
    shared = sorted(set(left_series) & set(right_series))
    differing = [step for step in shared if left_series[step] != right_series[step]]
    total = f"{len(shared)} shared steps ({min(shared)}-{max(shared)})" if shared else "no steps"
    if not differing:
        verdict = "byte-identical"
    else:
        first = differing[0]
        verdict = (f"{len(differing)}/{len(shared)} steps differ, first at step {first} "
                   f"({left_series[first]!r} vs {right_series[first]!r})")
    lines.append(f"{pairs:<52} loss: {verdict}  [{total}]")

    left_keys, right_keys = set(left["checkpoints"]), set(right["checkpoints"])
    for key in sorted(left_keys & right_keys):
        left_tensors = load_tensors(left["checkpoints"][key])
        right_tensors = load_tensors(right["checkpoints"][key])
        names = set(left_tensors) & set(right_tensors)
        bad = [name for name in sorted(names) if digest(left_tensors[name]) != digest(right_tensors[name])]
        detail = "all byte-identical"
        if bad:
            worst = max(abs(left_tensors[name].float() - right_tensors[name].float()).max().item()
                        for name in bad)
            detail = f"{len(bad)} of {len(names)} differ (e.g. {bad[:3]}, max |delta| {worst:.3e})"
        lines.append(f"{'':<52} {key}: {len(names)} tensors, {detail}")
    if left_keys ^ right_keys:
        lines.append(f"{'':<52} checkpoints present on one side only: "
                     f"{sorted(left_keys ^ right_keys)}")

    left_pngs, right_pngs = set(left["samples"]), set(right["samples"])
    shared_pngs = sorted(left_pngs & right_pngs)
    bad_pngs = [name for name in shared_pngs if left["samples"][name] != right["samples"][name]]
    lines.append(f"{'':<52} sample PNGs: {len(shared_pngs)} shared, "
                 + ("all byte-identical" if not bad_pngs else f"{len(bad_pngs)} differ"))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default="/tmp/axl-fix2-repro")
    parser.add_argument("--out", default=str(HERE / "resources" / "integrity.json"))
    args = parser.parse_args()

    work = Path(args.work_dir)
    runs: dict[str, dict] = {}
    for run_root in sorted(work.glob("*integrity*")):
        label = run_root.name
        runs[label] = {
            "loss": loss_series(run_root),
            "checkpoints": checkpoints(run_root),
            "samples": sample_hashes(run_root),
        }
    if not runs:
        print(f"no integrity runs under {work}", file=sys.stderr)
        return 1

    print(f"== {len(runs)} runs")
    for label, data in runs.items():
        steps = len(data["loss"])
        last = max(data["loss"]) if data["loss"] else "-"
        print(f"   {label:<52} {steps} loss points (last step {last}), "
              f"{len(data['checkpoints'])} checkpoints, {len(data['samples'])} samples")

    def pick(*needles: str) -> str | None:
        return next((label for label in runs
                     if all(needle in label for needle in needles)), None)

    pairs = [
        (("seed-1145141920", "cached"), ("seed-1145141920", "no-hip"),
         "1920 cached  vs  1920 no-hip"),
        (("seed-1145141919", "cached"), ("seed-1145141919", "no-hip"),
         "1919 cached  vs  1919 no-hip"),
        (("seed-1145141919", "cached"), ("seed-1145141919", "expandable"),
         "1919 cached  vs  1919 expandable"),
        (("seed-1145141919", "cached"), ("seed-1145141919", "sampling-off"),
         "1919 cached  vs  1919 sampling off"),
        # Shared latent cache (`--data-dir`): identical inputs, so any difference below is the
        # allocator/kernel path rather than the VAE encode.
        (("integrity2", "allocator-a"), ("integrity2", "allocator-b"),
         "shared cache: same config twice"),
        (("integrity2", "allocator-a"), ("integrity2", "no-hip"),
         "shared cache: cached  vs  no-hip"),
        (("integrity2", "allocator-a"), ("integrity2", "sampling-off"),
         "shared cache: samples  vs  no samples"),
    ]
    report: list[str] = []
    for left_needles, right_needles, title in pairs:
        left, right = pick(*left_needles), pick(*right_needles)
        if not left or not right:
            report.append(f"{title:<52} skipped (missing run)")
            continue
        report += compare(runs[left], runs[right], label=title, pairs=title)

    print("\n== comparisons")
    for line in report:
        print(line)
    Path(args.out).write_text(json.dumps({"runs": {k: {"steps": sorted(v["loss"])}
                                                  for k, v in runs.items()},
                                          "report": report}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
