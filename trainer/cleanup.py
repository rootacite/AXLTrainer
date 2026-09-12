from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional


def discover(
    output_dir: str | Path,
    logging_dir: str | Path,
    output_name: str,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve one run's artifacts.

    `run_id` selects the run-scoped layout (`{root}/{run_id}/…`). Passing None
    keeps the legacy flat layout (`{output_dir}/{name}_*`, `{logging_dir}/{name}`).
    """
    output_root = Path(output_dir)
    logging_root = Path(logging_dir)
    name = str(output_name)
    samples_name = f"{name}_samples"

    if run_id:
        run_dir = output_root / str(run_id)
        log_dir = logging_root / str(run_id)
    else:
        run_dir = output_root
        log_dir = logging_root / name

    weight_dirs: list[Path] = []
    if run_dir.is_dir():
        for path in sorted(run_dir.iterdir()):
            if path.is_dir() and path.name.startswith(name) and path.name != samples_name:
                weight_dirs.append(path)

    return {
        "run_id": str(run_id) if run_id else None,
        "run_dir": run_dir,
        "samples_dir": run_dir / samples_name,
        "log_dir": log_dir,
        "weight_dirs": weight_dirs,
    }


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()


def run_cleanup(
    output_dir: str | Path,
    logging_dir: str | Path,
    output_name: str,
    *,
    run_id: Optional[str] = None,
    delete_weights: bool = False,
) -> dict[str, Any]:
    plan = discover(output_dir, logging_dir, output_name, run_id)
    removed: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    def attempt(path: Path, enabled: bool) -> None:
        target = str(path)
        if not enabled:
            if path.exists():
                skipped.append(target)
            return
        if not path.exists():
            skipped.append(target)
            return
        try:
            _remove(path)
            removed.append(target)
        except OSError as exc:
            errors.append(f"{target}: {exc}")

    attempt(plan["samples_dir"], True)
    attempt(plan["log_dir"], True)
    for directory in plan["weight_dirs"]:
        attempt(directory, delete_weights)

    if plan["run_id"]:
        run_dir = plan["run_dir"]
        if run_dir.is_dir() and not any(run_dir.iterdir()):
            try:
                run_dir.rmdir()
                removed.append(str(run_dir))
            except OSError as exc:
                errors.append(f"{run_dir}: {exc}")

    return {
        "run_id": plan["run_id"],
        "run_dir": str(plan["run_dir"]),
        "samples_dir": str(plan["samples_dir"]),
        "log_dir": str(plan["log_dir"]),
        "weight_dirs": [str(path) for path in plan["weight_dirs"]],
        "delete_weights": bool(delete_weights),
        "removed": removed,
        "skipped": skipped,
        "errors": errors,
    }
