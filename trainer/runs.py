"""Run-scoped artifact directories.

Each training run writes into its own `{output_name}_{YYYYMMDD_HHMMSS}` directory
under `output_dir` (checkpoints, samples) and `logging_dir` (TensorBoard), so a
later run never overwrites an earlier one.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
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


# Any run directory, whatever output name it was created with. `.+` is greedy, so a
# second stamp inside the name itself is still the run id's own suffix.
_ANY_RUN_RE = re.compile(r"^(?P<name>.+)_\d{8}_\d{6}(?:_\d+)?$")
_SAMPLE_STEP_RE = re.compile(r"_(\d+)_(?:p\d+_)?\d+\.png$")
_WEIGHT_STEP_RE = re.compile(r"_s(\d{4,})")


def run_output_name(run_id: str) -> str:
    """The output name a run id was built from, `""` when it is not a run directory."""
    match = _ANY_RUN_RE.match(str(run_id or ""))
    return match.group("name") if match else ""


def validate_output_name(name: str) -> Optional[str]:
    """`None` when the name is usable, else why it is not.

    A run id is `safe_name(output_name)` plus a stamp, and the artifact directories are
    named after the same string, so the name has to be one filename-safe token. A space or
    a slash would make the id (`re_in_…`) disagree with the directories (`re in_samples`),
    and a run's samples could not be found from its id again.
    """
    text = str(name or "")
    if not text.strip():
        return "output_name is empty"
    if safe_name(text) != text:
        return (
            f"output_name must be letters, digits, '-', '_' or '.' only, not {text!r} "
            "(it becomes the run id and the artifact directory names)"
        )
    return None


def find_samples_dir(run_dir: Union[str, Path], output_name: str) -> Path:
    """A run's `{name}_samples` directory, or `run_dir/<something>_samples`.

    A run id carries the sanitized name (`safe_name`) while the artifact directories are
    written with the raw `output_name`, so a name holding a space or a slash cannot be
    recovered from the run id. The run directory only ever holds one sample directory.
    """
    root = Path(run_dir)
    named = root / f"{output_name}_samples"
    if named.is_dir() or not root.is_dir():
        return named
    candidates = sorted(path for path in root.glob("*_samples") if path.is_dir())
    return candidates[0] if len(candidates) == 1 else named


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


def _run_dirs(root: Path, output_name: Optional[str]) -> list[Path]:
    """Run directories under `root`; `output_name=None` accepts every output name."""
    if not root.is_dir():
        return []
    pattern = run_id_re(output_name) if output_name else _ANY_RUN_RE
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


def _max_step(current: Optional[int], candidate: int) -> int:
    return candidate if current is None or candidate > current else current


def _run_stats(run_dir: Path, output_name: str) -> dict[str, Any]:
    """Cheap figures for the run list; no event file is read.

    The newest step is taken from whatever the run managed to write, so a run whose
    TensorBoard directory is gone (or was never there) still reports how far it got.
    """
    samples = 0
    checkpoints = 0
    last_step: Optional[int] = None
    if run_dir.is_dir():
        samples_dir = find_samples_dir(run_dir, output_name)
        if samples_dir.is_dir():
            for image in samples_dir.glob("*.png"):
                samples += 1
                match = _SAMPLE_STEP_RE.search(image.name)
                if match:
                    last_step = _max_step(last_step, int(match.group(1)))
        for child in run_dir.iterdir():
            if not child.is_dir() or child == samples_dir:
                continue
            match = _WEIGHT_STEP_RE.search(child.name)
            if match:
                last_step = _max_step(last_step, int(match.group(1)))
            try:
                checkpoints += sum(1 for _ in child.glob("*.safetensors"))
            except OSError:
                continue
    return {"samples": samples, "last_step": last_step, "checkpoints": checkpoints}


def list_runs(
    output_dir: Union[str, Path],
    logging_dir: Union[str, Path],
    output_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Runs known to either root, newest first.

    `output_name=None` lists every run directory, whatever name it was created with,
    and each entry then carries the name its run id was built from.
    """
    output_root = Path(output_dir)
    logging_root = Path(logging_dir)
    known = {path.name: path for path in _run_dirs(output_root, output_name)}
    for path in _run_dirs(logging_root, output_name):
        known.setdefault(path.name, path)

    runs: list[dict[str, Any]] = []
    for run_id, path in known.items():
        name = str(output_name) if output_name else (run_output_name(run_id) or run_id)
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
                "output_name": name,
                "output_dir": str(output_path),
                "log_dir": str(log_path),
                "has_output": output_path.is_dir(),
                "has_log": log_path.is_dir(),
                "modified": float(modified),
                "size_bytes": _dir_size(output_path) if output_path.is_dir() else 0,
                **_run_stats(output_path, name),
            }
        )
    runs.sort(key=lambda item: (item["modified"], item["run_id"]), reverse=True)
    return runs


STEPS_PER_EPOCH_FILENAME = "steps_per_epoch.json"


def steps_per_epoch_path(log_dir: Union[str, Path]) -> Path:
    """Where a run records how many optimizer steps one epoch took."""
    return Path(log_dir) / STEPS_PER_EPOCH_FILENAME


def write_steps_per_epoch(log_dir: Union[str, Path], steps: int) -> Path:
    """Write `{steps_per_epoch: N}` atomically. The directory is created if needed."""
    value = int(steps)
    if value < 1:
        raise ValueError(f"steps_per_epoch must be positive, not {steps}")
    path = steps_per_epoch_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"steps_per_epoch": value}), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_steps_per_epoch(log_dir: Union[str, Path]) -> Optional[int]:
    """The recorded epoch length, or `None` when the file is missing, broken or not positive.

    A missing file is a run from before this record existed (or one that has not reached the
    dataloader yet). A broken file is warned about and treated the same way: the chart draws no
    epoch lines rather than guessing.
    """
    path = steps_per_epoch_path(log_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Warn] unreadable {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print(f"[Warn] {path} is not an object", file=sys.stderr)
        return None
    try:
        value = int(payload.get("steps_per_epoch"))
    except (TypeError, ValueError):
        print(f"[Warn] {path} has no positive steps_per_epoch", file=sys.stderr)
        return None
    if value < 1:
        return None
    return value


CHART_VIEW_FILENAME = "chart_view.json"
DEFAULT_SMOOTH_EXTRA_DP = 1.2
DEFAULT_OUTLIER_CLIP = 0.15
DEFAULT_STEP_SPAN = 800
SMOOTH_EXTRA_DP_MAX = 6.0
OUTLIER_CLIP_MAX = 0.40
STEP_SPAN_MIN = 100
STEP_SPAN_MAX = 8000


def chart_view_path(log_dir: Union[str, Path]) -> Path:
    """Where a run keeps the Dashboard chart sliders that are not part of `config.toml`."""
    return Path(log_dir) / CHART_VIEW_FILENAME


def default_chart_view() -> dict[str, float | int]:
    """Smoothed stroke extra in dp, the y-clip fraction (0.15 is 15%), and the step window."""
    return {
        "smooth_extra_dp": DEFAULT_SMOOTH_EXTRA_DP,
        "outlier_clip": DEFAULT_OUTLIER_CLIP,
        "step_span": DEFAULT_STEP_SPAN,
    }


def _chart_step_span(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number != int(number):
        return None
    steps = int(number)
    if steps < STEP_SPAN_MIN or steps > STEP_SPAN_MAX:
        return None
    return steps


def _chart_number(value: Any, lo: float, hi: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < lo or number > hi:
        return None
    return number


def read_chart_view(log_dir: Union[str, Path]) -> dict[str, float]:
    """The run's chart sliders, or the defaults when the file is missing or unusable.

    A missing file is a run whose sliders have not been moved. A broken value is warned about and
    replaced with that field's default, so one bad number does not hide the other.
    """
    defaults = default_chart_view()
    path = chart_view_path(log_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Warn] unreadable {path}: {exc}", file=sys.stderr)
        return defaults
    if not isinstance(payload, dict):
        print(f"[Warn] {path} is not an object", file=sys.stderr)
        return defaults
    extra = _chart_number(payload.get("smooth_extra_dp"), 0.0, SMOOTH_EXTRA_DP_MAX)
    clip = _chart_number(payload.get("outlier_clip"), 0.0, OUTLIER_CLIP_MAX)
    span = _chart_step_span(payload.get("step_span", DEFAULT_STEP_SPAN))
    if extra is None or clip is None or span is None:
        print(f"[Warn] {path} has a chart view value outside its range", file=sys.stderr)
    return {
        "smooth_extra_dp": defaults["smooth_extra_dp"] if extra is None else extra,
        "outlier_clip": defaults["outlier_clip"] if clip is None else clip,
        "step_span": defaults["step_span"] if span is None else span,
    }


def write_chart_view(
    log_dir: Union[str, Path],
    smooth_extra_dp: Any,
    outlier_clip: Any,
    step_span: Any = DEFAULT_STEP_SPAN,
) -> Path:
    """Replace the run's chart view atomically. The directory is created if needed."""
    extra = _chart_number(smooth_extra_dp, 0.0, SMOOTH_EXTRA_DP_MAX)
    clip = _chart_number(outlier_clip, 0.0, OUTLIER_CLIP_MAX)
    span = _chart_step_span(step_span)
    if extra is None:
        raise ValueError(
            f"smooth_extra_dp must be between 0 and {SMOOTH_EXTRA_DP_MAX:g}, not {smooth_extra_dp!r}"
        )
    if clip is None:
        raise ValueError(
            f"outlier_clip must be between 0 and {OUTLIER_CLIP_MAX:g}, not {outlier_clip!r}"
        )
    if span is None:
        raise ValueError(
            f"step_span must be an integer between {STEP_SPAN_MIN} and {STEP_SPAN_MAX}, not {step_span!r}"
        )
    path = chart_view_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"smooth_extra_dp": extra, "outlier_clip": clip, "step_span": span}),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path
