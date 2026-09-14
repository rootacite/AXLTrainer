#!/usr/bin/env python3
"""Merge the `repro_real.py` passes into the single ordered table in `resources/`.

`repro_real.py` writes one JSON file per invocation (`--out`), and its rows are independent, so a
full table can be assembled from several `--only` passes. This normalises the step count field
(the trainer only runs whole epochs, so a row that asked for 3 steps ran one epoch) and orders the
rows the way `repro_real.GRID` lists them.

    python merge_results.py                     # resources/real-baseline-4of4.json + real-first.json
                                                # + real-rest.json -> resources/real-results.json
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
PASSES = ["real-baseline-4of4.json", "real-first.json", "real-rest.json",
          "real-dim32.json", "real-seed919.json", "real-fast.json",
          "real-fast2.json"]

# Table order; mirror of repro_real.GRID, which is also the order the README documents.
ORDER = [
    "original (packaged 6 images)",
    "original, run again",
    "seed 1145141919 (the surviving seed)",
    "no HIP memory caching",
    "sampling every 30 steps disabled",
    "network_dim 32",
    "fix2.txt: batch 2, dim 36, kanae",
    "fix2.txt control: batch 3, dim 36, kanae",
    "fix2.txt: batch 2, dim 36, 3-chunk captions",
    "fast: cadence 2, cheap samples",
]


def main() -> int:
    rows: dict[str, dict] = {}
    for name in PASSES:
        path = RESOURCES / name
        if not path.is_file():
            print(f"skip missing {path}")
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            label = entry["label"].split(" #")[0]
            entry.setdefault("dataset", "packaged")  # the first pass predates the field
            if entry["steps_requested"] <= 3 and entry["last_step"]:
                # The row asked for a handful of steps; the trainer ran a whole epoch instead.
                entry["steps_run"] = entry["last_step"]
            rows.setdefault(label, entry)

    merged = [rows[label] for label in ORDER if label in rows]
    merged += [entry for label, entry in rows.items() if label not in ORDER]
    out = RESOURCES / "real-results.json"
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(f"| {'row':<42} | {'data':>6} | {'batch':>5} | {'dim':>3} | {'seed':>10} | "
          f"{'steps':>5} | result |")
    print("|" + "---|" * 7)
    for entry in merged:
        steps = entry.get("steps_run", entry["steps_requested"])
        data = str(entry.get("dataset", "?"))[:8]
        print(f"| {entry['label']:<42} | {data:>8} | {entry['batch_size']:>5} | "
              f"{entry['network_dim']:>3} | {entry['seed']:>10} | "
              f"{steps:>5} | {entry['verdict']}"
              + (f" (step {entry['last_step']})" if entry["last_step"] else "") + " |")
    faults = sum(entry["verdict"] == "FAULT" for entry in merged)
    print(f"\n{len(merged)} rows, {faults} faulted; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
