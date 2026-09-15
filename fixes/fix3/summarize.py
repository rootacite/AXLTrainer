#!/usr/bin/env python3
"""Render the scan JSONs in `resources/` as the markdown tables used by `README.md`.

    python summarize.py                # every table, in report order
    python summarize.py --only stability
    python summarize.py --faults-only  # just the kernel/address detail of every fault

Keeping the tables generated means the report cannot drift from the recorded runs: each row here is
one `repro_step1.py` row, and the JSON next to it holds that row's full record (exit code, peak VRAM,
kernel name, log path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"

# (json file, heading) in the order the report uses them.
TABLES = [
    ("grid-stability.json", "Repeat runs: which rows hold their verdict"),
    ("grid-shape.json", "Geometry rows (whole dataset, one knob changed)"),
    ("grid-dim3.json", "Rank scan over three steps (buckets 1280x768, 1408x768, 1408x768)"),
    ("grid-dim.json", "Rank scan over twelve steps"),
    ("grid-batch.json", "`train_batch_size` x `network_dim`"),
    ("grid-seed.json", "Seed scan (three runs per seed)"),
    ("seed-1145141919.json", "The committed seed, six more runs"),
    ("seed-1145141920.json", "A seed that is only sometimes fatal, six more runs"),
    ("grid-bucket.json", "Single-bucket datasets: bucket x rank"),
    ("grid-default.json", "First pass over the whole dataset (superseded by the rows above)"),
    ("packaged-cold.json", "Packaged mini-dataset, cold latent cache"),
    ("packaged-warm.json", "Packaged mini-dataset, warm latent cache"),
]


def verdict(row: dict) -> str:
    if row.get("fault"):
        return "**FAULT**"
    if row.get("vram_killed"):
        return "vram-kill"
    if row.get("timed_out"):
        return "timeout"
    if row.get("oom"):
        return "OOM"
    # -15 is the harness's own stop after the requested steps; 0 is a run that finished its epoch.
    # Anything else exited without a fault, which is a crash to look at rather than a survivor.
    if row.get("exit") not in (0, -15):
        return f"exited {row.get('exit')}"
    return "ok"


def overrides(row: dict) -> str:
    items = row.get("overrides") or {}
    if not items:
        return "(packaged config)"
    return ", ".join(f"`{k}={v}`" for k, v in sorted(items.items()))


def table(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    # Repeated row labels (the stability grid) collapse into one line per row with a column per pass;
    # without repetitions this is a plain one-row-per-line table.
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["row"], []).append(row)
    repeated = max(len(v) for v in grouped.values()) > 1
    if repeated:
        lines = ["| row | " + " | ".join(f"pass {i + 1}" for i in range(max(len(v) for v in grouped.values())))
                 + " | peak VRAM |",
                 "| --- | " + " | ".join("---" for _ in range(max(len(v) for v in grouped.values())))
                 + " | --- |"]
        for label, items in grouped.items():
            cells = [verdict(item) for item in items]
            cells += ["—"] * (max(len(v) for v in grouped.values()) - len(cells))
            peaks = ", ".join(f"{item.get('peak_vram_gb')}" for item in items)
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {peaks} GB |")
        return "\n".join(lines)

    lines = ["| row | steps | verdict | kernel | seconds | peak VRAM |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        kernel = row.get("kernel")
        kernel = "`…MT64x128x16…`" if kernel and "MT64x128x16" in kernel else (kernel or "")
        lines.append(
            f"| {row['row']} | {row.get('steps_reached', '')} | {verdict(row)} | {kernel} | "
            f"{row.get('seconds', '')} | {row.get('peak_vram_gb', '')} GB |")
    return "\n".join(lines)


def fault_detail() -> str:
    """Every fault seen, with its kernel name and faulting address, from the recorded rows."""
    lines = ["| row | kernel (Tensile solution) | faulting address |", "| --- | --- | --- |"]
    seen = []
    for name, _ in TABLES:
        path = RESOURCES / name
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            if not row.get("fault"):
                continue
            root = Path(row["run_root"]) / "train.out"
            address = ""
            if root.is_file():
                for line in root.read_text(encoding="utf-8", errors="replace").splitlines():
                    if "faulting addr:" in line:
                        address = line.split("faulting addr:")[1].split(",")[0].strip()
                        break
            key = (row["row"], address)
            if key in seen:
                continue
            seen.append(key)
            kernel = row.get("kernel") or ""
            lines.append(f"| {row['row']} | `{kernel[:60]}…` | `{address}` |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None)
    parser.add_argument("--faults-only", action="store_true")
    args = parser.parse_args()

    if args.faults_only:
        print("### Every recorded fault\n")
        print(fault_detail())
        return 0

    for name, heading in TABLES:
        if args.only and args.only not in name:
            continue
        path = RESOURCES / name
        if not path.is_file():
            print(f"## {heading}\n\n(no data: `resources/{name}` missing)\n")
            continue
        print(f"### {heading}\n")
        print(table(path))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
