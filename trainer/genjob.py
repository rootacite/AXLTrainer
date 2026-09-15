"""Job records for "generate a sample with this checkpoint".

One generation is one JSON file plus one PNG under
`{output_dir}/{run_id}/{output_name}_samples/generated/`. api.py writes the file
before spawning the generator, the generator rewrites it as it progresses, and
Ranko lists the directory to follow a job that outlives the dashboard.

Torch-free (like `runs.py`) so the format and the validation can be tested
without a GPU.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Union

GENERATED_DIRNAME = "generated"

STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"
STATES = (STATE_RUNNING, STATE_DONE, STATE_ERROR)

# Launch limits, mirrored by Ranko's form validation so a rejected click costs no GPU time.
MIN_CFG = 1.0
MAX_CFG = 30.0
MIN_STEPS = 1
MAX_STEPS = 150
MAX_SEED = 2**32 - 1
MIN_SIDE = 256
MAX_SIDE = 4096

_STAMP = "%Y%m%d_%H%M%S"

# `{output_name}_{YYYYMMDD}_{HHMMSS}`: a run directory, not a checkpoint directory.
_RUN_DIR = re.compile(r"^.+_\d{8}_\d{6}(?:_\d+)?$")


def generated_dir(samples_dir: Union[str, Path]) -> Path:
    """`{name}_samples/generated`: inside the sample dir, so api.py's non-recursive sample scan and
    `cleanup.py`'s run-scoped reset both keep working unchanged."""
    return Path(samples_dir) / GENERATED_DIRNAME


def job_stem(checkpoint: Union[str, Path]) -> str:
    """Name a job after its checkpoint directory when there is one (`lllj_s003050`), else its file stem.

    The trainer writes `{output_name}/{output_name}_s003050/{output_name}.safetensors`, so the
    directory carries the step while the file name is the bare output name.
    """
    path = Path(checkpoint)
    stem = path.name.removesuffix(".safetensors") or "checkpoint"
    parent = path.parent.name
    if parent and parent != stem and stem in parent and not _RUN_DIR.match(parent):
        return parent
    return stem


def new_job_id(stem: str, *, now: Optional[Union[datetime, float]] = None) -> str:
    if now is None:
        stamp = datetime.now()
    elif isinstance(now, (int, float)):
        stamp = datetime.fromtimestamp(now)
    else:
        stamp = now
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (stem or "").strip())
    return f"{cleaned or 'sample'}_gen_{stamp.strftime(_STAMP)}"


def job_path(generated: Union[str, Path], job_id: str) -> Path:
    return Path(generated) / f"{job_id}.json"


def image_path(generated: Union[str, Path], job_id: str) -> Path:
    return Path(generated) / f"{job_id}.png"


def log_path(generated: Union[str, Path], job_id: str) -> Path:
    return Path(generated) / f"{job_id}.log"


def atomic_write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    """Write via a temp file + rename, the pattern `control.py` uses for `state.json`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def read_job(path: Union[str, Path]) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_job(generated: Union[str, Path], job: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    atomic_write_json(job_path(generated, str(payload.get("id") or "job")), payload)
    return payload


def update_job(generated: Union[str, Path], job_id: str, **fields: Any) -> dict[str, Any]:
    """Read-modify-write one job; a concurrent api.py spawn cannot lose fields it wrote."""
    path = job_path(generated, job_id)
    payload = read_job(path) or {"id": job_id}
    payload.update(fields)
    payload["updated_at"] = time.time()
    atomic_write_json(path, payload)
    return payload


def list_jobs(generated: Union[str, Path]) -> list[dict[str, Any]]:
    """Every job in the directory, newest first. Malformed files are skipped, never raised."""
    directory = Path(generated)
    if not directory.is_dir():
        return []
    jobs: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        payload = read_job(path)
        if payload is None or not payload.get("id"):
            continue
        if payload.get("state") not in STATES:
            payload["state"] = STATE_ERROR
            payload.setdefault("error", "job file has no valid state")
        jobs.append(payload)
    jobs.sort(key=lambda job: (float(job.get("started_at") or 0.0), str(job.get("id"))), reverse=True)
    return jobs


def running_job(generated: Union[str, Path]) -> Optional[dict[str, Any]]:
    return next((job for job in list_jobs(generated) if job.get("state") == STATE_RUNNING), None)


def _number(params: Mapping[str, Any], key: str, default: Any) -> Any:
    """A key that is absent (or explicitly null) falls back to the config; an empty one is passed on
    so an intentionally cleared prompt is rejected instead of silently replaced."""
    value = params.get(key)
    return default if value is None else value


def normalize_request(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one generation request.

    Raises ValueError with a message Ranko can show as-is. `defaults` carries the config's sample
    settings, so a client that sends nothing still gets the run's usual prompt and CFG.
    """
    prompt = str(_number(params, "prompt", defaults.get("prompt", "")) or "").strip()
    if not prompt:
        raise ValueError("prompt must not be empty")
    negative = str(_number(params, "negative_prompt", defaults.get("negative_prompt", "")) or "")

    try:
        cfg = float(_number(params, "cfg", defaults.get("cfg", 6.0)))
    except (TypeError, ValueError) as exc:
        raise ValueError("cfg must be a number") from exc
    if not (MIN_CFG <= cfg <= MAX_CFG):
        raise ValueError(f"cfg must be between {MIN_CFG:g} and {MAX_CFG:g}")

    try:
        steps = int(_number(params, "steps", defaults.get("steps", 30)))
    except (TypeError, ValueError) as exc:
        raise ValueError("steps must be an integer") from exc
    if not (MIN_STEPS <= steps <= MAX_STEPS):
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")

    try:
        seed = int(_number(params, "seed", defaults.get("seed", 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc
    if not (0 <= seed <= MAX_SEED):
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")

    width, height = defaults.get("width", 1024), defaults.get("height", 1024)
    for key, fallback in (("width", width), ("height", height)):
        try:
            value = int(_number(params, key, fallback))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if not (MIN_SIDE <= value <= MAX_SIDE):
            raise ValueError(f"{key} must be between {MIN_SIDE} and {MAX_SIDE}")
        if key == "width":
            width = value
        else:
            height = value

    step = params.get("step")
    try:
        step = None if step is None or step == "" else int(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("step must be an integer") from exc

    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "cfg": cfg,
        "steps": steps,
        "seed": seed,
        "width": width,
        "height": height,
        "step": step,
    }


def new_job(
    request: Mapping[str, Any],
    *,
    run_id: str,
    output_name: str,
    checkpoint: str,
    pid: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The job record api.py writes before spawning the generator."""
    job: dict[str, Any] = {
        "id": new_job_id(job_stem(checkpoint)),
        "state": STATE_RUNNING,
        "run_id": run_id,
        "output_name": output_name,
        "checkpoint": checkpoint,
        "prompt": request.get("prompt", ""),
        "negative_prompt": request.get("negative_prompt", ""),
        "cfg": request.get("cfg"),
        "steps": request.get("steps"),
        "seed": request.get("seed"),
        "width": request.get("width"),
        "height": request.get("height"),
        "step": request.get("step"),
        "current_step": 0,
        "total_steps": request.get("steps"),
        "image_path": None,
        "error": None,
        "pid": pid,
        "started_at": time.time(),
    }
    if extra:
        job.update(dict(extra))
    return job


def checkpoint_step(metadata: Mapping[str, str]) -> Optional[int]:
    """Step a checkpoint was saved at, from its own kohya metadata."""
    raw = str(metadata.get("ss_steps") or "").strip()
    return int(raw) if re.fullmatch(r"-?\d+", raw) else None
