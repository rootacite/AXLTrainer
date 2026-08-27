from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def discover(
    output_dir: str | Path,
    logging_dir: str | Path,
    output_name: str,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    logging_root = Path(logging_dir)
    name = str(output_name)
    samples_dir = output_root / f"{name}_samples"
    log_dir = logging_root / name
    weight_dirs: list[Path] = []
    if output_root.is_dir():
        for path in sorted(output_root.iterdir()):
            if path.is_dir() and path.name.startswith(name) and path.name != f"{name}_samples":
                weight_dirs.append(path)
    return {
        "samples_dir": samples_dir,
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
    delete_weights: bool = False,
) -> dict[str, Any]:
    plan = discover(output_dir, logging_dir, output_name)
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

    return {
        "samples_dir": str(plan["samples_dir"]),
        "log_dir": str(plan["log_dir"]),
        "weight_dirs": [str(path) for path in plan["weight_dirs"]],
        "delete_weights": bool(delete_weights),
        "removed": removed,
        "skipped": skipped,
        "errors": errors,
    }
