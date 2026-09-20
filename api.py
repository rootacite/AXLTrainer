import json
import os
import re
import subprocess
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer.amdfq_patch import resolve_preload
from trainer.checkpoints import discover_checkpoints, resolve_resume_path
from trainer.config import TrainConfig, _load_toml_config, resolve_sample_sets
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
from trainer import genjob

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

_IPC_STDOUT = sys.stdout


def _write(payload: dict[str, Any]) -> None:
    _IPC_STDOUT.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    _IPC_STDOUT.flush()


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
    directory = params.get("directory") or cfg.get("train_data_dir")
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
}


def dispatch(method: str, params: Optional[dict[str, Any]] = None) -> Any:
    handler = _HANDLERS.get(method)
    if handler is None:
        raise ValueError(f"unknown method: {method}")
    return handler(params or {})


def run_ipc_loop() -> None:
    # Keep stdout exclusive for NDJSON IPC. All logs go to stderr.
    sys.stdout = sys.stderr
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            if not isinstance(method, str) or not method:
                raise ValueError("missing method")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params must be an object")
            result = dispatch(method, params)
            _write({"id": req_id, "ok": True, "result": _json_safe(result)})
        except Exception as exc:
            traceback.print_exc()
            _write({"id": req_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    run_ipc_loop()
