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
# Asked to stop (a second `cancel_generation` or a stray signal) and finished stopping. A job that
# is cancelled still keeps whatever `files` it had written; it is not a failure.
STATE_CANCELLED = "cancelled"
STATES = (STATE_RUNNING, STATE_DONE, STATE_ERROR, STATE_CANCELLED)

# What one job renders: a single ad-hoc image with its own prompt, every `[[validation.samples]]`
# set of the config for one checkpoint, that same sets pass for each checkpoint of a step range
# (a batch, which drives its own `sets` jobs — one per checkpoint — from one process), or an
# evaluation of one checkpoint (top the sample count up to a depth, tag every image, score it).
MODE_SINGLE = "single"
MODE_SETS = "sets"
MODE_BATCH = "batch"
MODE_EVALUATE = "evaluate"
MODES = (MODE_SINGLE, MODE_SETS, MODE_BATCH, MODE_EVALUATE)

# An evaluation's own progress: which stage it is in. `images_done` / `total_images` are that
# stage's counters — images rendered while `rendering`, images tagged while `tagging`.
PHASE_RENDERING = "rendering"
PHASE_TAGGING = "tagging"
PHASE_SCORING = "scoring"
PHASE_DONE = "done"
PHASES = (PHASE_RENDERING, PHASE_TAGGING, PHASE_SCORING, PHASE_DONE)

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


def new_job_id(stem: str, *, mode: str = MODE_SINGLE, now: Optional[Union[datetime, float]] = None) -> str:
    if now is None:
        stamp = datetime.now()
    elif isinstance(now, (int, float)):
        stamp = datetime.fromtimestamp(now)
    else:
        stamp = now
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (stem or "").strip())
    # `_sets` marks a full `[[validation.samples]]` pass, so the two kinds never share a job id.
    marker = "" if mode == MODE_SINGLE else f"_{mode}"
    return f"{cleaned or 'sample'}{marker}_gen_{stamp.strftime(_STAMP)}"


def job_path(generated: Union[str, Path], job_id: str) -> Path:
    return Path(generated) / f"{job_id}.json"


def image_path(generated: Union[str, Path], job_id: str) -> Path:
    return Path(generated) / f"{job_id}.png"


def set_image_path(
    generated: Union[str, Path],
    job_id: str,
    set_index: int,
    repeat_idx: int,
) -> Path:
    """One image of a `sets` job: `{job_id}_p{set}_{repeat}.png`, sets counting from 0 like the
    run's own `{output_name}_{step:06d}_p{set}_{repeat}.png` samples."""
    return Path(generated) / f"{job_id}_p{int(set_index)}_{int(repeat_idx)}.png"


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
        if payload.get("mode") not in MODES:
            payload["mode"] = MODE_SINGLE
        payload.setdefault("cancel_requested", False)
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
    mode: str = MODE_SINGLE,
    total_images: int = 1,
    pid: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The job record api.py writes before spawning the generator."""
    if mode not in MODES:
        raise ValueError(f"unknown job mode: {mode}")
    total_images = max(1, int(total_images))
    job: dict[str, Any] = {
        "id": new_job_id(job_stem(checkpoint), mode=mode),
        "state": STATE_RUNNING,
        "mode": mode,
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
        # Never null: the client declares this an Int, and an explicit null is not a missing key
        # for it (the reply of a `sets` job, which has no per-image `steps`, would fail to parse
        # and take the whole generated-samples list down with it). A `single` job sets it below
        # through `normalize_request`'s validated step count.
        "total_steps": int(request.get("steps") or 0),
        "image_path": None,
        # Set by `cancel_generation`; the job stays `running` until its process is really gone, so
        # the card can say "cancelling…" and another generation cannot start on the same card yet.
        "cancel_requested": False,
        # A `sets` job writes one image per (set, repeat); a `single` job one, recorded in
        # `image_path` as before so a job file from the old build still reads.
        "files": [],
        "images_done": 0,
        "total_images": total_images,
        "error": None,
        "pid": pid,
        "started_at": time.time(),
    }
    if extra:
        job.update(dict(extra))
    return job


def new_batch_job(
    *,
    run_id: str,
    output_name: str,
    checkpoints: list[dict[str, Any]],
    from_step: int,
    to_step: int,
    images_per_checkpoint: int,
    config_log_dir: str = "",
    sample_sets: Optional[list[Mapping[str, Any]]] = None,
    now: Optional[Union[datetime, float]] = None,
) -> dict[str, Any]:
    """The plan api.py writes before spawning a range batch.

    `checkpoints` is the ordered work list (`path` + `step`); the runner creates one `sets` job per
    entry, so the images of each checkpoint are named, shown and followed exactly as a manual pass
    from that checkpoint would be. This record is the batch's own bookkeeping: what is left, which
    checkpoint is being rendered, and what failed. `config_log_dir` is the run directory the prompts
    come from (its own saved config, or the sets the Dashboard edited for it), which the runner
    resolves once for the whole range.
    """
    stem = f"{output_name}_s{from_step}-{to_step}"
    return {
        "id": new_job_id(stem, mode=MODE_BATCH, now=now),
        "state": STATE_RUNNING,
        "mode": MODE_BATCH,
        "run_id": run_id,
        "output_name": output_name,
        "checkpoints": [dict(entry) for entry in checkpoints],
        "from_step": int(from_step),
        "to_step": int(to_step),
        "config_log_dir": str(config_log_dir or ""),
        # The prompt sets the range renders with, recorded the way a single-checkpoint pass records
        # them: whoever planned the batch had them resolved in hand, and the runner must not derive
        # them again from a config file that may have moved on since. Empty on an older record or a
        # hand-written spec, which is when the runner resolves the run's own config itself.
        "sample_sets": [dict(entry) for entry in (sample_sets or [])],
        "checkpoint_index": 0,
        "total_checkpoints": len(checkpoints),
        "current_checkpoint": None,
        "job_ids": [],
        "failed": [],
        "images_done": 0,
        # Every checkpoint of one run renders the same `[[validation.samples]]` sets, so this is an
        # exact total rather than a guess; the runner only moves `images_done`.
        "total_images": len(checkpoints) * max(0, int(images_per_checkpoint)),
        "error": None,
        "cancel_requested": False,
        "started_at": time.time(),
    }


def checkpoint_step(metadata: Mapping[str, str]) -> Optional[int]:
    """Step a checkpoint was saved at, from its own kohya metadata."""
    raw = str(metadata.get("ss_steps") or "").strip()
    return int(raw) if re.fullmatch(r"-?\d+", raw) else None


def new_evaluation_job(
    *,
    run_id: str,
    output_name: str,
    checkpoint: str,
    step: Optional[int],
    depth: int,
    threshold: float,
    categories: list[str],
    config_source: str,
    config_log_dir: str,
    plan: Mapping[str, Any],
    images: list[Mapping[str, Any]],
    sample_sets: list[Mapping[str, Any]],
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """The record api.py writes before spawning an evaluation.

    The plan and the image list are the work order: `plan` says which `(set, repeat)` slots to
    render (none, when the checkpoint already has `depth` images) and `images` is the set to tag and
    score, each entry already carrying the prompt it was rendered from. `sample_sets` is what the
    top-up renders with, recorded so the pass does not re-read a config that may have moved on;
    `config_source` is the file the prompts came from (the run's own copy, or the hparams it recorded
    at startup) and `config_log_dir` the run directory they were resolved in, which is what the
    runner resolves again for the model side. The
    record starts in the phase the plan implies — `rendering` when there is something to draw,
    `tagging` when there is not, which is what keeps a re-evaluation from loading the diffusion
    model at all. `tags` narrows the scoring to the tags the caller picked (empty = all of them).
    """
    needed = bool(plan.get("needed"))
    phase = PHASE_RENDERING if needed else PHASE_TAGGING
    return new_job(
        {"step": step},
        run_id=run_id,
        output_name=output_name,
        checkpoint=str(checkpoint),
        mode=MODE_EVALUATE,
        # The counters belong to the phase the record starts in; the tagging phase resets them to
        # the size of the whole scored set (the run's own samples included) when it begins.
        total_images=int(plan.get("render_total") or 0) if needed else len(images),
        extra={
            "depth": int(depth),
            "threshold": float(threshold),
            "categories": [str(category) for category in categories],
            # The tags the scoring is narrowed to; empty means every tag a prompt asks for.
            "tags": [str(tag) for tag in (tags or [])],
            "config_source": str(config_source),
            "config_log_dir": str(config_log_dir),
            "plan": dict(plan),
            "images": [dict(image) for image in images],
            "sample_sets": [dict(entry) for entry in sample_sets],
            "phase": phase,
            "scores": None,
        },
    )
