#!/usr/bin/env python3
"""Join the scans with the step -> bucket map: which bucket shapes kill which ranks.

`repro_step1.py` records, per row, how many training steps completed before the abort. `probe_batches.py`
records, per step index, the bucket that batch was drawn from (`resources/step-buckets.json`). The
trainer bumps its step counter *after* a step completes (`trainer/loop.py`), so a row that recorded
`N` completed steps died inside step index `N` — which names the bucket:

    fatal bucket = resources/step-buckets.json["steps"][steps_reached]

Rows that survive attribute no fatal bucket; they simply exercise the buckets of steps 0..N-1, which is
what the "exercised" column counts.

    python correlate.py                    # every scan that has data
    python correlate.py --only dim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"

# Which scans to read, and which knob identifies a row within each. The third element is the value a
# row has when it does not override the knob (i.e. the packaged `config.toml`).
SCANS = {
    "dim": ("grid-dim.json", "network_dim", "48"),
    "batch": ("grid-batch.json", "train_batch_size", "2"),
    "seed": ("grid-seed.json", "seed", "1145141919"),
    "stability": ("grid-stability.json", "network_dim", "48"),
}


def load(name: str) -> dict | None:
    path = RESOURCES / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def step_map() -> list[dict]:
    data = load("step-buckets.json")
    return data["steps"] if data else []


def outcome(row: dict, steps: list[dict]) -> tuple[str, dict | None, int]:
    """(verdict, the bucket it died in, how many steps it completed without a fault)."""
    reached = int(row.get("steps_reached") or 0)
    if row.get("fault"):
        bucket = steps[reached] if reached < len(steps) else None
        return "FAULT", bucket, reached
    if row.get("vram_killed"):
        return "vram-kill", None, reached
    if row.get("exit") not in (0, -15):
        return f"exited {row.get('exit')}", None, reached
    return "ok", None, reached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, choices=sorted(SCANS))
    args = parser.parse_args()

    steps = step_map()
    if not steps:
        raise SystemExit("resources/step-buckets.json missing — run probe_batches.py --json first")

    fatal: list[tuple[str, str, int, dict]] = []
    seen_fatal: set[tuple[str, str, int]] = set()
    for scan, (filename, knob, default) in SCANS.items():
        if args.only and args.only != scan:
            continue
        data = load(filename)
        if not data:
            print(f"## {scan}: no data yet (`resources/{filename}`)\n")
            continue

        # bucket -> knob value -> cell text
        cells: dict[str, dict[str, str]] = {}
        exercised: dict[str, set[str]] = {}
        for row in data["rows"]:
            value = str((row.get("overrides") or {}).get(knob, default))
            batch = (row.get("overrides") or {}).get("train_batch_size", 2)
            verdict, bucket, reached = outcome(row, steps)
            for step in steps[:reached]:
                exercised.setdefault(step["bucket"], set()).add(value)
            if bucket is None:
                continue
            label = bucket["bucket"]
            cells.setdefault(label, {})[value] = f"**FAULT** M={bucket['M_lora_gemm']} step {reached}"
            if (label, value, bucket["M_lora_gemm"]) not in seen_fatal:
                seen_fatal.add((label, value, bucket["M_lora_gemm"]))
                fatal.append((label, value, bucket["M_lora_gemm"], bucket))

        values = sorted({v for row in cells.values() for v in row},
                        key=lambda v: (len(v), v))
        buckets = [b["bucket"] for b in steps]
        seen, ordered = set(), []
        for name in buckets:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        print(f"## `{knob}` x bucket (scan `{scan}`)\n")
        print("| bucket | latent | unet tokens | " + " | ".join(f"{knob} {v}" for v in values) + " |")
        print("| --- | --- | --- | " + " | ".join("---" for _ in values) + " |")
        for name in ordered:
            step = next(b for b in steps if b["bucket"] == name)
            row = [f"{step['latent']}", f"{step['unet_tokens']}"]
            for value in values:
                cell = cells.get(name, {}).get(value)
                if cell:
                    row.append(cell)
                elif value in exercised.get(name, set()):
                    row.append("ok")
                else:
                    row.append("—")
            print(f"| {name} | " + " | ".join(row) + " |")
        print()

    if fatal:
        print("## Fatal (bucket, knob) combinations\n")
        print("| bucket | latent | knob value | M | CLIP seq | steps before the abort |")
        print("| --- | --- | --- | --- | --- | --- |")
        for bucket_name, value, m, step in fatal:
            print(f"| {bucket_name} | {step['latent']} | {value} | {m} | {step['seq']} | {step['step']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
