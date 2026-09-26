import argparse
import ipaddress
import json
import logging
import os
import re
import subprocess
import sys
import threading
import traceback
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer.amdfq_patch import resolve_preload
from trainer.checkpoints import discover_checkpoints, resolve_resume_path
from trainer.config import TrainConfig, _load_toml_config, resolve_sample_sets, resolve_train_data_entries
from trainer.family import require_trainable, resolve_family
from trainer.cleanup import run_cleanup
from trainer.control import (
    is_pid_alive,
    log_path,
    mark_starting,
    reconcile,
    request as request_train_command,
    reset_to_idle,
    status_payload,
)
from trainer.hardware import collect_hardware_status
from trainer.runs import find_latest_run
from trainer import blobcodec, fsrpc, genjob

_TAG_BLOCKED = frozenset(
    {
        "starting",
        "encoding",
        "training",
        "sampling",
        "pausing",
        "resuming",
        "stopping",
    }
)
from trainer.loss_log import synthesize_avg_loss


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _train_config_dict() -> dict[str, Any]:
    cfg = TrainConfig()
    data = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    data.update(_load_toml_config())
    return _json_safe(data)


def _get_tensorboard_metrics(
    log_dir: str,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
) -> dict:
    if not os.path.exists(log_dir):
        return {}

    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        return {}

    latest_log_dir = str(max(event_files, key=os.path.getmtime).parent)
    ea = EventAccumulator(latest_log_dir, size_guidance={"scalars": 0})
    ea.Reload()

    metrics: dict = {}
    if "scalars" in ea.Tags():
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            filtered = [
                {"step": e.step, "value": float(e.value), "wall_time": float(e.wall_time)}
                for e in events
                if (start_step is None or e.step >= start_step)
                and (end_step is None or e.step <= end_step)
            ]
            metrics[tag] = filtered

    return metrics


def _run_name(params: dict[str, Any], cfg: dict[str, Any]) -> str:
    return str(params.get("name") or cfg.get("output_name") or "default")


def _resolve_run_id(params: dict[str, Any], cfg: dict[str, Any]) -> Optional[str]:
    """Explicit param → the run recorded in state.json → the newest run directory."""
    explicit = params.get("run_id")
    if explicit:
        return str(explicit)
    name = _run_name(params, cfg)
    current = reconcile()
    state_run = current.get("run_id")
    if state_run and str(current.get("output_name") or name) == name:
        return str(state_run)
    logging_dir = cfg.get("logging_dir")
    if not logging_dir:
        return None
    return find_latest_run(str(logging_dir), name)


def handle_ping(_params: dict[str, Any]) -> dict[str, str]:
    return {"status": "ok"}


def handle_dashboard(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    logging_dir = cfg.get("logging_dir", "./logs")
    run_id = _resolve_run_id(params, cfg)

    start_step = params.get("start_step")
    end_step = params.get("end_step")
    metrics: dict = {}
    if run_id:
        target_log_dir = os.path.join(str(logging_dir), str(run_id))
        metrics = _get_tensorboard_metrics(target_log_dir, start_step, end_step)
    if not metrics.get("Train/Avg_Loss") and metrics.get("Train/Loss"):
        metrics["Train/Avg_Loss"] = synthesize_avg_loss(metrics["Train/Loss"])

    latest_stats: dict[str, Any] = {}
    for tag, data in metrics.items():
        if data:
            latest_stats[tag] = data[-1]["value"]
            latest_stats["current_step"] = data[-1]["step"]

    # The flat `sample_*` keys mirror the first `[[validation.samples]]` entry, so the
    # "generate a sample" form keeps a single place to read its defaults from.
    sample_sets = _sample_sets_payload()
    if sample_sets:
        first = sample_sets[0]
        cfg = {
            **cfg,
            "sample_prompts": first["prompt"],
            "sample_negative": first["negative"],
            "sample_width": first["width"],
            "sample_height": first["height"],
            "sample_steps": first["steps"],
            "sample_seed": first["seed"],
            "sample_repeat": first["repeat"],
            "guidance_scale": first["guidance_scale"],
        }

    return {
        "config": cfg,
        "run_id": run_id,
        "latest_stats": latest_stats,
        "metrics": metrics,
        "sample_sets": sample_sets,
    }


# `{name}_{step:06d}_p{set}_{repeat}.png` (multi-prompt runs); the two-number form is
# what runs before `[[validation.samples]]` wrote, and still maps to set 0.
_SAMPLE_NAME_SET = re.compile(r"_(\d+)_p(\d+)_(\d+)\.png$")
_SAMPLE_NAME = re.compile(r"_(\d+)_(\d+)\.png$")


def scan_samples(sample_dir: Path) -> dict[str, list]:
    if not sample_dir.exists():
        return {}

    grouped: dict[int, list] = defaultdict(list)
    for img_path in sample_dir.glob("*.png"):
        match = _SAMPLE_NAME_SET.search(img_path.name)
        if match:
            step, set_index, repeat_idx = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        else:
            legacy = _SAMPLE_NAME.search(img_path.name)
            if legacy:
                step, set_index, repeat_idx = int(legacy.group(1)), 0, int(legacy.group(2))
            else:
                step, set_index, repeat_idx = -1, 0, 0
        grouped[step].append(
            {
                "filename": img_path.name,
                "set_index": set_index,
                "repeat_idx": repeat_idx,
                "path": str(img_path.resolve()),
            }
        )

    for step in grouped:
        grouped[step] = sorted(grouped[step], key=lambda x: (x["set_index"], x["repeat_idx"]))

    return {str(k): grouped[k] for k in sorted(grouped.keys(), reverse=True)}


def _sample_sets_payload() -> list[dict[str, Any]]:
    """The resolved `[[validation.samples]]` sets, for the dashboard's sample defaults.

    A broken entry must not take the dashboard down with it: the charts and the run
    status keep working, and the trainer reports the config error when it starts.
    """
    try:
        return [asdict(sample_set) for sample_set in resolve_sample_sets(_train_config_dict())]
    except Exception as exc:
        print(f"[Warn] validation.samples ignored: {exc}", file=sys.stderr)
        return []


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def handle_train_status(_params: dict[str, Any]) -> dict[str, Any]:
    return status_payload()


def handle_train_start(_params: dict[str, Any]) -> dict[str, Any]:
    current = reconcile()
    if is_pid_alive(current.get("pid")):
        raise ValueError("training already running")

    root = _repo_root()
    script = root / "start_train.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing launcher: {script}")

    cfg_obj = TrainConfig()
    require_trainable(resolve_family(cfg_obj))
    # A bad `[[validation.samples]]` entry would otherwise only surface once sampling starts.
    resolve_sample_sets(cfg_obj)

    resume_raw = str(getattr(cfg_obj, "resume_lora_path", "") or "").strip()
    if resume_raw:
        try:
            resolve_resume_path(resume_raw)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    try:
        resolve_preload(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    cfg = _train_config_dict()
    output_name = str(cfg.get("output_name") or "default")
    log_file = log_path()
    with open(log_file, "ab") as log_handle:
        proc = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    mark_starting(proc.pid, output_name)
    return status_payload()


def _require_alive() -> dict[str, Any]:
    current = reconcile()
    if not is_pid_alive(current.get("pid")):
        raise ValueError("no running training process")
    return current


def handle_train_pause(_params: dict[str, Any]) -> dict[str, Any]:
    _require_alive()
    request_train_command("pause")
    return status_payload()


def handle_train_resume(_params: dict[str, Any]) -> dict[str, Any]:
    _require_alive()
    request_train_command("resume")
    return status_payload()


def handle_train_stop(_params: dict[str, Any]) -> dict[str, Any]:
    _require_alive()
    request_train_command("stop")
    return status_payload()


_RESET_BLOCKED = frozenset(
    {
        "starting",
        "encoding",
        "training",
        "sampling",
        "pausing",
        "paused",
        "resuming",
    }
)


def handle_train_reset(params: dict[str, Any]) -> dict[str, Any]:
    current = reconcile()
    if current.get("status") in _RESET_BLOCKED and is_pid_alive(current.get("pid")):
        raise ValueError("cannot reset while training is running")
    cfg = _train_config_dict()
    run_id = _resolve_run_id(params, cfg)
    delete_weights = bool(params.get("delete_weights"))
    if run_id:
        cleanup: dict[str, Any] = run_cleanup(
            cfg.get("output_dir", "./output"),
            cfg.get("logging_dir", "./logs"),
            _run_name(params, cfg),
            run_id=run_id,
            delete_weights=delete_weights,
        )
    else:
        # No run directory to clean: legacy flat artifacts are only reachable
        # through `python clean.py --legacy-flat`.
        cleanup = {
            "run_id": None,
            "run_dir": None,
            "samples_dir": None,
            "log_dir": None,
            "weight_dirs": [],
            "delete_weights": delete_weights,
            "removed": [],
            "skipped": [],
            "errors": [],
        }
    reset_to_idle()
    payload = status_payload()
    payload["run_id"] = run_id
    payload["cleanup"] = cleanup
    return payload


def handle_list_samples(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    output_dir = cfg.get("output_dir", "./output")
    run_id = _resolve_run_id(params, cfg)
    if not run_id:
        return {"run_id": None, "samples": {}}
    sample_dir = Path(str(output_dir)) / str(run_id) / f"{_run_name(params, cfg)}_samples"
    return {"run_id": run_id, "samples": scan_samples(sample_dir)}


def handle_list_checkpoints(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    output_dir = params.get("output_dir") or cfg.get("output_dir", "./output")
    return {
        "checkpoints": discover_checkpoints(str(output_dir), _run_name(params, cfg)),
    }


def run_tagger_process(
    directory: str,
    threshold: float,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Spawn tagger/main.py with the same interpreter as api.py (the axl env)."""
    script = _repo_root() / "tagger" / "main.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing tagger: {script}")
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(script),
            str(directory),
            "--threshold",
            str(threshold),
            "--batch-size",
            str(int(batch_size)),
            "--json",
        ],
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    raw_out = (proc.stdout or "").strip()
    raw_err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = raw_err or raw_out or f"tagger exited {proc.returncode}"
        raise RuntimeError(detail[-2000:])
    if not raw_out:
        raise RuntimeError(raw_err[-2000:] if raw_err else "tagger produced no output")
    try:
        payload = json.loads(raw_out.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"tagger returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or "error" in payload and "processed" not in payload:
        raise RuntimeError(str(payload.get("error") if isinstance(payload, dict) else payload))
    return payload


def handle_hardware_status(_params: dict[str, Any]) -> dict[str, Any]:
    """nvtop -s snapshot plus sysfs CPU/GPU temps. Never raises; Ranko keeps training UI up."""
    return _json_safe(collect_hardware_status())


def _samples_dir(cfg: dict[str, Any], run_id: str, output_name: str) -> Path:
    output_dir = Path(str(cfg.get("output_dir") or ".")).expanduser()
    return output_dir / str(run_id) / f"{output_name}_samples"


def _generated_dir(params: dict[str, Any], cfg: dict[str, Any]) -> Optional[tuple[str, str, Path]]:
    """(run_id, output_name, generated dir) for the resolved run, or None when no run exists."""
    run_id = _resolve_run_id(params, cfg)
    if not run_id:
        return None
    output_name = _run_name(params, cfg)
    return run_id, output_name, genjob.generated_dir(_samples_dir(cfg, run_id, output_name))


def _reconcile_generated(generated: Path) -> list[dict[str, Any]]:
    """A generator killed with its job still `running` (SIGKILL, reboot) must not block the next one."""
    for job in genjob.list_jobs(generated):
        if job.get("state") == genjob.STATE_RUNNING and not is_pid_alive(job.get("pid")):
            genjob.update_job(
                generated,
                str(job["id"]),
                state=genjob.STATE_ERROR,
                error="the generator exited before finishing (see the job's .log)",
            )
    return genjob.list_jobs(generated)


def handle_list_generated_samples(params: dict[str, Any]) -> dict[str, Any]:
    """Generated samples for a run, newest first. Read-only and empty-safe."""
    cfg = _train_config_dict()
    resolved = _generated_dir(params, cfg)
    if resolved is None:
        return {"run_id": None, "jobs": []}
    run_id, _output_name, generated = resolved
    return {"run_id": run_id, "jobs": _json_safe(_reconcile_generated(generated))}


def _generator_script() -> Path:
    script = _repo_root() / "trainer" / "generate_sample.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing generator: {script}")
    return script


def handle_generate_sample(params: dict[str, Any]) -> dict[str, Any]:
    """Start one "sample with this checkpoint" job. Returns immediately; Ranko follows the job file.

    The GPU is single-tenant: a live trainer (running *or* paused) refuses the request outright,
    because a second SDXL would have to load into the same 16 GB.
    """
    current = reconcile()
    if is_pid_alive(current.get("pid")):
        raise ValueError(
            "training is still running; finish or stop the run before generating "
            "(the GPU is in use)"
        )

    cfg = _train_config_dict()
    resolved = _generated_dir(params, cfg)
    if resolved is None:
        raise ValueError("no run to attach the sample to; finish a run first")
    run_id, output_name, generated = resolved

    checkpoint = Path(str(params.get("checkpoint") or "")).expanduser()
    if not checkpoint.is_file():
        raise ValueError(f"not a checkpoint file: {checkpoint}")

    first_set = _sample_sets_payload()
    defaults = first_set[0] if first_set else {"prompt": None}
    request = genjob.normalize_request(
        params,
        defaults={
            "prompt": defaults.get("prompt") or cfg.get("sample_prompts"),
            "negative_prompt": defaults.get("negative") or cfg.get("sample_negative"),
            "cfg": defaults.get("guidance_scale", cfg.get("guidance_scale")),
            "steps": defaults.get("steps", cfg.get("sample_steps")),
            "seed": defaults.get("seed", cfg.get("sample_seed")),
            "width": defaults.get("width", cfg.get("sample_width")),
            "height": defaults.get("height", cfg.get("sample_height")),
        },
    )

    generated.mkdir(parents=True, exist_ok=True)
    # Reconcile first: a generator killed with its job still `running` must not block this one.
    running = next(
        (job for job in _reconcile_generated(generated) if job.get("state") == genjob.STATE_RUNNING),
        None,
    )
    if running is not None:
        raise ValueError(f"a generation is already running ({running.get('id')})")

    job = genjob.new_job(
        request,
        run_id=run_id,
        output_name=output_name,
        checkpoint=str(checkpoint),
    )
    spec_path = genjob.job_path(generated, job["id"])
    log = genjob.log_path(generated, job["id"])

    # Write the record before spawning so a click that arrives while the process starts still lists it.
    genjob.write_job(generated, job)
    with open(log, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(_generator_script()), "--spec", str(spec_path)],
            cwd=str(_repo_root()),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    job = genjob.update_job(generated, job["id"], pid=proc.pid)
    return {"job": _json_safe(job), "log_path": str(log)}


def handle_dataset_tag(params: dict[str, Any]) -> dict[str, Any]:
    current = reconcile()
    if current.get("status") in _TAG_BLOCKED and is_pid_alive(current.get("pid")):
        raise ValueError("cannot tag while training is using the GPU")

    cfg = _train_config_dict()
    directory = params.get("directory")
    if not directory:
        # No directory asked for: tag the first configured dataset folder, i.e. the one the
        # `train_data_dir` scalar mirrors. Resolving the list here also accepts a hand-written
        # config that carries `[[environment.train_data]]` blocks and no scalar.
        try:
            entries = resolve_train_data_entries(cfg)
        except ValueError as exc:
            raise ValueError(f"cannot resolve the training data directory: {exc}") from exc
        directory = entries[0].path if entries else None
    if not directory:
        raise ValueError("missing directory")
    directory_path = Path(str(directory)).expanduser()
    if not directory_path.is_dir():
        raise ValueError(f"not a directory: {directory_path}")

    try:
        threshold = float(params["threshold"]) if params.get("threshold") is not None else 0.35
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a number") from exc
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0.0 and 1.0")

    batch_size = params.get("batch_size", 1)
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("batch_size must be an integer") from exc
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    return run_tagger_process(str(directory_path), threshold, batch_size=batch_size)


def handle_config_get(_params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.config_get()


def handle_config_save(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.config_save(str(params.get("text") or ""))


def handle_profile_list(_params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.profile_list()


def handle_profile_get(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.profile_get(str(params.get("name") or ""))


def handle_profile_save(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.profile_save(
        str(params.get("name") or ""),
        str(params.get("text") or ""),
        bool(params.get("overwrite")),
    )


def handle_profile_delete(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.profile_delete(str(params.get("name") or ""))


def handle_tag_lexicon(_params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.tag_lexicon()


def handle_dataset_list(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    directory = fsrpc.require_dataset_dir(params.get("directory"), cfg)
    return fsrpc.dataset_list(directory)


def handle_caption_write(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    directory = fsrpc.require_dataset_dir(params.get("directory"), cfg)
    stem = fsrpc.require_stem(params.get("stem"))
    text = params.get("text")
    if text is None:
        raise ValueError("missing text")
    return fsrpc.caption_write(directory, stem, str(text))


def handle_dataset_drop(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    directory = fsrpc.require_dataset_dir(params.get("directory"), cfg)
    try:
        rate = float(params.get("rate"))
    except (TypeError, ValueError) as exc:
        raise ValueError("rate must be a number") from exc
    seed = params.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
    stems = params.get("stems")
    if stems is not None:
        if not isinstance(stems, list):
            raise ValueError("stems must be an array")
        stems = [str(s) for s in stems]
    return fsrpc.dataset_drop(directory, rate, seed, stems)


def handle_dataset_shuffle(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    directory = fsrpc.require_dataset_dir(params.get("directory"), cfg)
    seed = params.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
    return fsrpc.dataset_shuffle(directory, seed)


def handle_mask_get(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    directory = fsrpc.require_dataset_dir(params.get("directory"), cfg)
    stem = fsrpc.require_stem(params.get("stem"))
    return fsrpc.mask_get(directory, stem)


def handle_mask_write(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    directory = fsrpc.require_dataset_dir(params.get("directory"), cfg)
    stem = fsrpc.require_stem(params.get("stem"))
    return fsrpc.mask_write(directory, stem, str(params.get("png_base64") or ""))


def handle_mask_delete(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    directory = fsrpc.require_dataset_dir(params.get("directory"), cfg)
    stem = fsrpc.require_stem(params.get("stem"))
    return fsrpc.mask_delete(directory, stem)


def _blob_request(params: dict[str, Any]) -> tuple[list[str], int, int, str]:
    raw = params.get("paths")
    if not isinstance(raw, list):
        raise ValueError("paths must be an array")
    paths = [str(item) for item in raw]
    max_edge, quality, fmt = blobcodec.parse_encode_params(params)
    return paths, max_edge, quality, fmt


def _filter_blob_paths(paths: list[str], cfg: dict[str, Any]) -> list[dict[str, Any] | None]:
    """None means allowed; a dict is the per-item error placeholder."""
    marks: list[dict[str, Any] | None] = []
    for raw in paths:
        path = Path(raw)
        if not fsrpc.blob_path_allowed(path, cfg):
            marks.append({"path": raw, "error": "path is not an allowed image"})
        else:
            marks.append(None)
    return marks


def handle_blob_stat(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    paths, max_edge, quality, fmt = _blob_request(params)
    marks = _filter_blob_paths(paths, cfg)
    allowed = [p for p, mark in zip(paths, marks) if mark is None]
    resolved = blobcodec.resolve_blobs(allowed, max_edge, quality, fmt, include_payload=False)
    items: list[dict[str, Any]] = []
    cursor = 0
    for mark in marks:
        if mark is not None:
            items.append(mark)
        else:
            items.append(resolved[cursor])
            cursor += 1
    return {"items": items}


def handle_blob_batch(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    paths, max_edge, quality, fmt = _blob_request(params)
    marks = _filter_blob_paths(paths, cfg)
    allowed = [p for p, mark in zip(paths, marks) if mark is None]
    resolved = blobcodec.resolve_blobs(allowed, max_edge, quality, fmt, include_payload=True)
    items: list[dict[str, Any]] = []
    cursor = 0
    for mark in marks:
        if mark is not None:
            items.append(mark)
        else:
            items.append(resolved[cursor])
            cursor += 1
    return {"items": items}


def handle_checkpoint_export(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    source = Path(str(params.get("source") or ""))
    dest = Path(str(params.get("dest") or ""))
    if not source.as_posix() or not dest.as_posix():
        raise ValueError("source and dest are required")
    if not fsrpc.checkpoint_source_allowed(source, cfg):
        raise ValueError(f"not an allowed checkpoint: {source}")
    return fsrpc.checkpoint_export(source, dest)


def handle_fs_listdir(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.fs_listdir(params.get("path"))


def handle_fs_roots(_params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.fs_roots(_train_config_dict())


_HANDLERS = {
    "ping": handle_ping,
    "dashboard": handle_dashboard,
    "list_samples": handle_list_samples,
    "list_checkpoints": handle_list_checkpoints,
    "train_status": handle_train_status,
    "train_start": handle_train_start,
    "train_pause": handle_train_pause,
    "train_resume": handle_train_resume,
    "train_stop": handle_train_stop,
    "train_reset": handle_train_reset,
    "dataset_tag": handle_dataset_tag,
    "hardware_status": handle_hardware_status,
    "generate_sample": handle_generate_sample,
    "list_generated_samples": handle_list_generated_samples,
    "config_get": handle_config_get,
    "config_save": handle_config_save,
    "profile_list": handle_profile_list,
    "profile_get": handle_profile_get,
    "profile_save": handle_profile_save,
    "profile_delete": handle_profile_delete,
    "tag_lexicon": handle_tag_lexicon,
    "dataset_list": handle_dataset_list,
    "caption_write": handle_caption_write,
    "dataset_drop": handle_dataset_drop,
    "dataset_shuffle": handle_dataset_shuffle,
    "mask_get": handle_mask_get,
    "mask_write": handle_mask_write,
    "mask_delete": handle_mask_delete,
    "blob_stat": handle_blob_stat,
    "blob_batch": handle_blob_batch,
    "checkpoint_export": handle_checkpoint_export,
    "fs_listdir": handle_fs_listdir,
    "fs_roots": handle_fs_roots,
}

_CONTROL_METHODS = frozenset(
    {
        "train_start",
        "train_pause",
        "train_resume",
        "train_stop",
        "train_reset",
        "dataset_tag",
        "generate_sample",
        "config_save",
        "profile_save",
        "profile_delete",
        "caption_write",
        "dataset_drop",
        "dataset_shuffle",
        "mask_write",
        "mask_delete",
        "checkpoint_export",
    }
)
_CONTROL_LOCK = threading.Lock()
_WS_MAX_SIZE = 32 * 1024 * 1024
_WS_DEFAULT_PORT = 18765
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}


def dispatch(method: str, params: Optional[dict[str, Any]] = None) -> Any:
    handler = _HANDLERS.get(method)
    if handler is None:
        raise ValueError(f"unknown method: {method}")
    payload = params or {}
    if method in _CONTROL_METHODS:
        with _CONTROL_LOCK:
            return handler(payload)
    return handler(payload)


def _reply_for(req: dict[str, Any]) -> dict[str, Any]:
    req_id = req.get("id")
    try:
        method = req.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("missing method")
        params = req.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        result = dispatch(method, params)
        return {"id": req_id, "ok": True, "result": _json_safe(result)}
    except Exception as exc:
        traceback.print_exc()
        return {"id": req_id, "ok": False, "error": str(exc)}


def parse_allow_networks(entries: list[str]) -> list[ipaddress._BaseNetwork]:
    networks: list[ipaddress._BaseNetwork] = []
    for raw in entries:
        text = raw.strip()
        if not text:
            continue
        try:
            if "/" in text:
                networks.append(ipaddress.ip_network(text, strict=False))
            else:
                ip = ipaddress.ip_address(text)
                networks.append(ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}"))
        except ValueError as exc:
            raise SystemExit(f"invalid --allow-ip {raw!r}: {exc}") from exc
    return networks


def client_ip_allowed(remote: str, networks: list[ipaddress._BaseNetwork]) -> bool:
    host = remote.split("%")[0]
    if host.startswith("::ffff:"):
        host = host[7:]
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    return any(ip in net for net in networks)


def _peer_host(connection: Any) -> str:
    addr = getattr(connection, "remote_address", None)
    if isinstance(addr, tuple) and addr:
        return str(addr[0])
    return str(addr or "")


class _QuietWsClose(logging.Filter):
    """Browsers drop sockets without a close frame; handshake probes die mid-request."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        if exc is not None:
            from websockets.exceptions import ConnectionClosed, InvalidMessage

            if isinstance(exc, (ConnectionClosed, InvalidMessage, EOFError)):
                return False
        msg = record.getMessage()
        return "opening handshake failed" not in msg and "connection handler failed" not in msg


def _quiet_websockets_log() -> None:
    filt = _QuietWsClose()
    # Filters on a parent logger are not applied to child loggers; the handshake
    # traceback is emitted on websockets.server.
    for name in ("websockets", "websockets.server", "websockets.client"):
        log = logging.getLogger(name)
        if not any(isinstance(item, _QuietWsClose) for item in log.filters):
            log.addFilter(filt)


def run_ws_loop(host: str, port: int, allow_networks: Optional[list[ipaddress._BaseNetwork]] = None) -> None:
    # Keep stdout unused for JSON: Ranko talks over the socket. Logs go to stderr.
    sys.stdout = sys.stderr
    _quiet_websockets_log()
    networks = list(allow_networks or [])
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    def handler(connection) -> None:
        try:
            peer = _peer_host(connection)
            if not client_ip_allowed(peer, networks):
                print(f"api.py rejected {peer}", file=sys.stderr)
                connection.close()
                return
            for raw in connection:
                if not raw or (isinstance(raw, str) and not raw.strip()):
                    continue
                req_id: Any = None
                try:
                    req = json.loads(raw)
                    if not isinstance(req, dict):
                        raise ValueError("request must be an object")
                    req_id = req.get("id")
                    connection.send(json.dumps(_reply_for(req), ensure_ascii=False, allow_nan=False))
                except ConnectionClosed:
                    return
                except Exception as exc:
                    traceback.print_exc()
                    try:
                        connection.send(
                            json.dumps(
                                {"id": req_id, "ok": False, "error": str(exc)},
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                        )
                    except Exception:
                        return
        except ConnectionClosed:
            return

    with serve(handler, host, port, max_size=_WS_MAX_SIZE, origins=None) as server:
        print(f"api.py websocket on ws://{host}:{port}", file=sys.stderr)
        server.serve_forever()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="AXLTrainer dashboard helper")
    parser.add_argument(
        "--websocket",
        action="store_true",
        help="accepted and ignored: WebSocket is the only transport",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AXL_WS_HOST", "127.0.0.1"),
        help="WebSocket bind address (default 127.0.0.1; 0.0.0.0 for LAN)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AXL_WS_PORT", str(_WS_DEFAULT_PORT))),
        help="WebSocket bind port",
    )
    parser.add_argument(
        "--allow-ip",
        action="append",
        default=[],
        help="Client IP or CIDR allowed to connect (repeatable). Loopback is always allowed.",
    )
    args = parser.parse_args(argv)
    from_env = [part.strip() for part in os.environ.get("AXL_WS_ALLOW", "").split(",") if part.strip()]
    networks = parse_allow_networks(list(args.allow_ip) + from_env)
    run_ws_loop(args.host, args.port, networks)


if __name__ == "__main__":
    main()

