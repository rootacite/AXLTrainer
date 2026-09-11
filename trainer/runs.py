"""Run-scoped artifact directories.

Each training run writes into its own `{output_name}_{YYYYMMDD_HHMMSS}` directory
under `output_dir` (checkpoints, samples) and `logging_dir` (TensorBoard), so a
later run never overwrites an earlier one.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

_RUN_STAMP = "%Y%m%d_%H%M%S"


def safe_name(name: str) -> str:
    """Mirror of `models.safe_output_name`; kept torch-free for api.py."""
    text = (name or "").strip() or "run"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return cleaned or "run"


def run_id_re(output_name: str) -> re.Pattern[str]:
    """Matches only run directories, never flat `{name}_s000010` checkpoint dirs."""
    names = {str(output_name or "").strip(), safe_name(output_name)}
    names.discard("")
    alternatives = "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True)
    ) or re.escape("run")
    return re.compile(rf"^(?:{alternatives})_\d{{8}}_\d{{6}}(?:_\d+)?$")


def make_run_id(output_name: str, *, now: Optional[Union[datetime, float]] = None) -> str:
    if now is None:
        stamp = datetime.now()
    elif isinstance(now, (int, float)):
        stamp = datetime.fromtimestamp(now)
    else:
        stamp = now
    return f"{safe_name(output_name)}_{stamp.strftime(_RUN_STAMP)}"


def create_run_dirs(
    output_dir: Union[str, Path],
    logging_dir: Union[str, Path],
    output_name: str,
    *,
    now: Optional[Union[datetime, float]] = None,
) -> str:
    """Create this run's output + log directories and return its run id."""
    base = make_run_id(output_name, now=now)
    output_root = Path(output_dir)
    logging_root = Path(logging_dir)
    run_id = base
    suffix = 2
    while (output_root / run_id).exists() or (logging_root / run_id).exists():
        run_id = f"{base}_{suffix}"
        suffix += 1
    (output_root / run_id).mkdir(parents=True, exist_ok=True)
    (logging_root / run_id).mkdir(parents=True, exist_ok=True)
    return run_id


def _run_dirs(root: Path, output_name: str) -> list[Path]:
    if not root.is_dir():
        return []
    pattern = run_id_re(output_name)
    return [path for path in root.iterdir() if path.is_dir() and pattern.match(path.name)]


def find_latest_run(logging_dir: Union[str, Path], output_name: str) -> Optional[str]:
    candidates = _run_dirs(Path(logging_dir), output_name)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name)).name


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def list_runs(
    output_dir: Union[str, Path],
    logging_dir: Union[str, Path],
    output_name: str,
) -> list[dict[str, Any]]:
    """Runs known to either root, newest first."""
    output_root = Path(output_dir)
    logging_root = Path(logging_dir)
    known = {path.name: path for path in _run_dirs(output_root, output_name)}
    for path in _run_dirs(logging_root, output_name):
        known.setdefault(path.name, path)

    runs: list[dict[str, Any]] = []
    for run_id, path in known.items():
        output_path = output_root / run_id
        log_path = logging_root / run_id
        anchor = output_path if output_path.is_dir() else path
        try:
            modified = anchor.stat().st_mtime
        except OSError:
            modified = 0.0
        runs.append(
            {
                "run_id": run_id,
                "output_dir": str(output_path),
                "log_dir": str(log_path),
                "has_output": output_path.is_dir(),
                "has_log": log_path.is_dir(),
                "modified": float(modified),
                "size_bytes": _dir_size(output_path) if output_path.is_dir() else 0,
            }
        )
    runs.sort(key=lambda item: (item["modified"], item["run_id"]), reverse=True)
    return runs
