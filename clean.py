"""Interactive cleanup for one training run's artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from trainer.cleanup import discover, run_cleanup
from trainer.config import TrainConfig
from trainer.runs import list_runs


def _format_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _print_runs(runs: list[dict[str, Any]]) -> None:
    print(f"Found {len(runs)} run director{'y' if len(runs) == 1 else 'ies'}:")
    for run in runs:
        output = "yes" if run["has_output"] else "no"
        log = "yes" if run["has_log"] else "no"
        print(
            f"  - {run['run_id']}  {_format_size(run['size_bytes']):>10}"
            f"  output={output}  log={log}"
        )


def _select_run(runs: list[dict[str, Any]], requested: Optional[str]) -> Optional[str]:
    if requested:
        match = next((run for run in runs if run["run_id"] == requested), None)
        if match is None:
            print(f"[Error] unknown run id: {requested}")
            return None
        return match["run_id"]
    _print_runs(runs)
    answer = input(
        f"Run id to clean [enter = newest: {runs[0]['run_id']}]: "
    ).strip()
    if not answer:
        return runs[0]["run_id"]
    match = next((run for run in runs if run["run_id"] == answer), None)
    if match is None:
        print(f"[Error] unknown run id: {answer}")
        return None
    return match["run_id"]


def _confirm_weights(weight_dirs: list[Path]) -> bool:
    if not weight_dirs:
        print("[Info] No checkpoint directories found in this run.")
        return False
    print(f"Found {len(weight_dirs)} checkpoint directory/directories:")
    for directory in weight_dirs:
        print(f"  - {directory.name}/")
    answer = input("Delete these trained weight checkpoints too? (y/N): ").strip().lower()
    if answer in ("y", "yes"):
        return True
    print("[Info] Skipped weights deletion. Safe-saving checkpoints.")
    return False


def clean_project() -> None:
    parser = argparse.ArgumentParser(
        description="Delete one run's samples, TensorBoard logs, and optionally its weights."
    )
    parser.add_argument("--run", help="run id to clean (default: newest / interactive)")
    parser.add_argument(
        "--legacy-flat",
        action="store_true",
        help="clean the legacy flat layout instead of a run directory",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("      LoRA Training Directory Cleaner      ")
    print("=" * 50)

    try:
        cfg = TrainConfig()
    except Exception as e:
        print(f"[Error] Failed to load TrainConfig: {e}")
        return

    output_dir = Path(cfg.output_dir)
    logging_dir = Path(cfg.logging_dir)
    output_name = cfg.output_name

    print(f"Loaded configuration for project: '{output_name}'")
    print(f"Base Output Directory: {output_dir}")
    print(f"Base Logging Directory: {logging_dir}\n")

    run_id: Optional[str] = None
    if not args.legacy_flat:
        runs = list_runs(output_dir, logging_dir, output_name)
        if not runs:
            print(f"[Info] No run directories found for '{output_name}' under {output_dir}.")
            return
        run_id = _select_run(runs, args.run)
        if run_id is None:
            return
        print(f"\nSelected run: {run_id}")

    print("-" * 40)
    print("Step 1: Cleaning sample images...")
    plan = discover(output_dir, logging_dir, output_name, run_id)
    for label, target in (("samples", plan["samples_dir"]), ("logs", plan["log_dir"])):
        if target.exists():
            print(f"[Info] Will remove {label}: {target}")
        else:
            print(f"[Info] No {label} directory at: {target}")

    print("\n" + "-" * 40)
    print("Step 2: Checking for existing weights...")
    delete_weights = _confirm_weights(plan["weight_dirs"])

    result = run_cleanup(
        output_dir,
        logging_dir,
        output_name,
        run_id=run_id,
        delete_weights=delete_weights,
    )
    for path in result["removed"]:
        print(f"[Deleted] {path}")
    for message in result["errors"]:
        print(f"[Error] {message}")

    print("\n" + "=" * 50)
    print("Cleanup task completed.")
    print("=" * 50)


if __name__ == "__main__":
    clean_project()
