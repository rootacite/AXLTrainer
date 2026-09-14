#!/usr/bin/env python3
"""Ablation ladder around the guaranteed fault: which difference decides fatality?

The trainer dies at step 4 with `network_dim = 32`; `crash.py` runs the same shapes
and survives. This script removes one trainer-side ingredient at a time from the
faulting configuration and reports where the fault stops happening, so the missing
ingredient is measured instead of guessed.

    python diff_probe.py                 # the whole ladder (6 rows x 12 steps)
    python diff_probe.py --list
    python diff_probe.py --row "no mask"

Rows are single-variable: each one changes exactly one thing relative to
`baseline`. Output goes to `resources/diff-probe.json`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import repro_real as rr

HERE = Path(__file__).resolve().parent
PROBE = Path("/tmp/axl-diff-probe")
SHARED_DATA = PROBE / "shared-data"
NOALPHA_DATA = PROBE / "dataset-noalpha"
OUT = rr.RESOURCES / "diff-probe.json"
STEPS = 12

NO_WORKERS = {
    "infrastructure.max_data_loader_n_workers": 0,
    "infrastructure.persistent_workers": False,
}

# Every row is the step-4 abort (`network_dim` 32) plus one trainer ingredient removed.
BASE = {"network_dim": 32}


def row(extra_overrides: dict | None = None, **keys) -> dict:
    overrides = dict(BASE)
    overrides.update(keys)
    if extra_overrides:
        merged = dict(overrides.get("set") or {})
        merged.update(extra_overrides)
        overrides["set"] = merged
    return overrides


# label -> (config overrides, dataset override, data-dir override, dataloader workers)
ROWS: list[tuple[str, dict, Path | None, Path | None, int]] = [
    ("dim 32 baseline", row(), None, None, 2),
    ("dim 32, shared data dir (encodes once)", row(), None, SHARED_DATA, 2),
    ("dim 32, shared data dir (latent cache hit, no VAE pass)", row(), None, SHARED_DATA, 2),
    ("dim 32, no dataloader workers", row(NO_WORKERS), None, None, 0),
    ("dim 32, no loss mask (alpha stripped)", row(), NOALPHA_DATA, None, 2),
    ("dim 32, no workers + no mask (alpha stripped)", row(NO_WORKERS), NOALPHA_DATA, None, 0),
]


def stripped_dataset() -> Path:
    """Same pixels without the alpha channel, so `load_loss_mask` returns all ones.

    `LoraImageDataset` calls `img.convert("RGB")` before encoding, so dropping
    alpha cannot change the latents; it only removes the mask multiply.
    """
    from PIL import Image

    source = rr.RESOURCES / "dataset"
    if NOALPHA_DATA.is_dir() and all(
        (NOALPHA_DATA / p.name).is_file() for p in source.iterdir()
    ):
        return NOALPHA_DATA
    NOALPHA_DATA.mkdir(parents=True, exist_ok=True)
    for src in sorted(source.iterdir()):
        if src.suffix.lower() == ".txt":
            shutil.copy2(src, NOALPHA_DATA / src.name)
        elif src.suffix.lower() == ".png" and not src.name.endswith(".mask.png"):
            with Image.open(src) as img:
                img.convert("RGB").save(NOALPHA_DATA / src.name)
    return NOALPHA_DATA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", default="", help="run only rows whose label contains this")
    parser.add_argument("--list", action="store_true", help="print the ladder and exit")
    parser.add_argument("--model", default=None)
    parser.add_argument("--kill-after-fault", type=int, default=3,
                        help="stop the child a few seconds after the KFD abort instead of "
                             "waiting ~25 s for the coredump teardown")
    args = parser.parse_args()

    rows = [row for row in ROWS if args.row in row[0]] if args.row else ROWS
    if args.list:
        for label, overrides, dataset, data_dir, workers in rows:
            print(f"{label:<52} workers={workers} overrides={overrides} "
                  f"dataset={dataset} data_dir={data_dir}")
        return 0
    if not rows:
        print(f"no row contains {args.row!r}", file=sys.stderr)
        return 2

    model = rr.model_path(args.model)
    if not Path(model).is_dir():
        print(f"model path not found: {model}", file=sys.stderr)
        return 2
    PROBE.mkdir(parents=True, exist_ok=True)

    results = []
    for index, (label, overrides, dataset, data_dir, _workers) in enumerate(rows, start=1):
        args_for_row = SimpleNamespace(
            steps=STEPS,
            env=[],
            dataset=str(dataset or (rr.RESOURCES / "dataset")),
            data_dir=str(data_dir or ""),
            model_path=model,
            dry_run=False,
            quiet=True,
            prune_checkpoints=1,
            kill_after_fault=args.kill_after_fault,
        )
        if dataset is not None:
            stripped_dataset()
        results.append(rr.run_variant(label, overrides, args=args_for_row,
                                      work=PROBE, index=index))
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n| {'row':<52} | {'workers':>7} | {'last step':>9} | result |")
    print("|" + "---|" * 4)
    for (label, _o, _d, _dd, workers), entry in zip(rows, results):
        print(f"| {label:<52} | {workers:>7} | "
              f"{entry['last_step'] if entry['last_step'] is not None else '-':>9} | "
              f"{entry['verdict']} |")
    faults = sum(1 for entry in results if entry["verdict"] == "FAULT")
    print(f"\n{faults}/{len(results)} faulted; json: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
