import argparse
import functools
import ipaddress
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer.amdfq_patch import resolve_preload
from trainer.checkpoints import (
    discover_checkpoints,
    parse_checkpoint_dir,
    pin_entry,
    pins_path,
    read_lora_metadata,
    read_pins,
    require_resume_network_type,
    resolve_resume_path,
    unpin_entry,
    write_pins,
)
from trainer.config import (
    TrainConfig,
    _load_toml_config,
    clear_sample_override,
    resolve_sample_sets,
    resolve_train_data_entries,
    run_config_mapping,
    sample_override_path,
    write_sample_override,
)
from trainer.family import require_trainable, resolve_family
from trainer import estimate
from trainer.cleanup import run_cleanup
from trainer.control import (
    LIVE_STATUSES,
    LiveSettings,
    is_pid_alive,
    log_path,
    mark_starting,
    publish_settings,
    read_settings,
    reconcile,
    request as request_train_command,
    request_settings,
    reset_to_idle,
    status_payload,
)
from trainer.hardware import collect_hardware_status
from trainer import orphans
from trainer.runs import find_samples_dir, list_runs, run_output_name, safe_name
from trainer import automation, blobcodec, comfy, evaluation, fsrpc, genjob

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


def _gpu_busy(current: dict[str, Any]) -> bool:
    """True while a live trainer is using the GPU.

    `paused` is deliberately free: pause has offloaded the UNet, both text encoders, the
    optimizers and the VAE to CPU, so a one-off generation can run next to it.
    """
    return current.get("status") in _TAG_BLOCKED and is_pid_alive(current.get("pid"))


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


_TB_CACHE_LOCK = threading.Lock()
_TB_CACHE: dict[str, "_TbEntry"] = {}
_TB_CACHE_MAX = 8
_TB_BUILDS = 0


@dataclass
class _TbEntry:
    """One run directory's reader: the event file it read last, how big it was, and the reader."""

    path: Path
    size: int
    accumulator: Any


def reset_tensorboard_cache() -> None:
    """Drop every cached reader; the next read starts from an empty file."""
    global _TB_BUILDS
    with _TB_CACHE_LOCK:
        _TB_CACHE.clear()
        _TB_BUILDS = 0


def tensorboard_cache_stats() -> dict[str, int]:
    """`{entries, builds}`: how many run directories are cached, and how many readers were built."""
    with _TB_CACHE_LOCK:
        return {"entries": len(_TB_CACHE), "builds": _TB_BUILDS}


def _tensorboard_reader(log_dir: str) -> Optional[Any]:
    """This run directory's `EventAccumulator`, reloaded in place. Caller holds `_TB_CACHE_LOCK`.

    A poll asks for the whole history every time (`dashboard`) and the history only grows, so a
    fresh `EventAccumulator(...).Reload()` per call re-read the file from the start — 343 ms at
    24 k points, and worse as the run got longer. Reusing the reader and reloading it reads only
    what was appended. It is rebuilt when the newest event file changes (a new run in the same
    directory, or a second writer) or gets smaller (a rewrite), so a stale reader cannot outlive
    the file it was reading.
    """
    global _TB_BUILDS
    key = str(Path(log_dir).resolve())
    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        _TB_CACHE.pop(key, None)
        return None
    latest = max(event_files, key=os.path.getmtime)
    try:
        size = latest.stat().st_size
    except OSError:
        return None
    entry = _TB_CACHE.get(key)
    if entry is not None and entry.path == latest and size >= entry.size:
        entry.size = size
        entry.accumulator.Reload()
        return entry.accumulator
    accumulator = EventAccumulator(str(latest.parent), size_guidance={"scalars": 0})
    accumulator.Reload()
    _TB_BUILDS += 1
    _TB_CACHE[key] = _TbEntry(path=latest, size=size, accumulator=accumulator)
    while len(_TB_CACHE) > _TB_CACHE_MAX:
        _TB_CACHE.pop(next(iter(_TB_CACHE)))
    return accumulator


def _get_tensorboard_metrics(
    log_dir: str,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
) -> dict:
    if not os.path.exists(log_dir):
        return {}

    # One lock for the whole read: the reader is a mutable object, and two polls of the same
    # directory reloading it at once would interleave. Nothing else in the helper holds a lock, so
    # this cannot hold up a call of another kind.
    with _TB_CACHE_LOCK:
        ea = _tensorboard_reader(log_dir)
        if ea is None:
            return {}

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


def _resolve_run(params: dict[str, Any], cfg: dict[str, Any]) -> tuple[Optional[str], str]:
    """(run_id, output_name) for this request.

    The resolved run's own name wins over the config's: a run directory is only ever
    scanned once, and its `{name}_samples` lives under the name it was created with.
    The config's spelling is kept when it is that name, because artifact directories
    are written with the raw name while a run id carries the sanitized one.
    """
    run_id = _resolve_run_id(params, cfg)
    explicit = params.get("name")
    if explicit:
        return run_id, str(explicit)
    derived = run_output_name(run_id or "")
    configured = str(cfg.get("output_name") or "")
    if derived and (not configured or safe_name(configured) == derived):
        return run_id, configured or derived
    return run_id, derived or configured or "default"


def _resolve_run_id(params: dict[str, Any], cfg: dict[str, Any]) -> Optional[str]:
    """Explicit `run_id` → the run recorded in state.json → the newest run of an explicitly named one.

    A request that names nothing (`dashboard` / `list_samples` / `train_reset` as the
    dashboard sends them while it follows the trainer) means *the run the trainer is on*,
    so it stops at `state.json`. It deliberately does not fall back to the newest run
    directory: that made "current" mean "the last run on disk", so a dashboard with no
    run recorded showed a stopped run's step count and size under a `Current run` heading.
    Such a run is reached by naming it — every run directory of both roots is listed by
    `list_runs`, so a run whose TensorBoard directory is gone (Reset used to delete it, or
    the config's `logging_dir` moved since) is still picked from the history list.
    """
    explicit = params.get("run_id")
    if explicit:
        return str(explicit)
    state_run = reconcile().get("run_id")
    if state_run:
        return str(state_run)
    name = str(params.get("name") or "")
    if not name:
        return None
    runs = list_runs(cfg.get("output_dir", "./output"), cfg.get("logging_dir", "./logs"), None)
    known = next((run for run in runs if run["output_name"] == name), None)
    return str(known["run_id"]) if known else None


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
    # "generate a sample" form keeps a single place to read its defaults from. The sets are the
    # shown run's own (its saved config, or the prompts the Dashboard edited for it).
    sample_sets = _sample_sets_payload(_log_dir(cfg, run_id) if run_id else None)
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


def _state_run_entry(
    cfg: dict[str, Any],
    current: dict[str, Any],
    run_id: str,
    live: bool,
) -> dict[str, Any]:
    """The run `state.json` is on, for a root that holds no directory for it."""
    name = str(current.get("output_name") or "") or run_output_name(run_id) or run_id
    return {
        "run_id": run_id,
        "output_name": name,
        "output_dir": os.path.join(str(cfg.get("output_dir") or "./output"), run_id),
        "log_dir": os.path.join(str(cfg.get("logging_dir") or "./logs"), run_id),
        "has_output": False,
        "has_log": False,
        "modified": float(current.get("started_at") or current.get("updated_at") or 0.0),
        "size_bytes": 0,
        "samples": 0,
        "last_step": None,
        "checkpoints": 0,
        "current": True,
        "live": live,
    }


def handle_list_runs(_params: dict[str, Any]) -> dict[str, Any]:
    """Every run directory under `output_dir` / `logging_dir`, newest first.

    `current` marks the run `state.json` is on and `live` that its PID still holds it,
    which is what tells a stopped run from the running one in the dashboard's history
    list. Runs are listed whatever `output_name` they were created with; each entry
    carries the name its run id was built from, so samples and checkpoints resolve.
    """
    cfg = _train_config_dict()
    current = reconcile()
    current_run = str(current.get("run_id") or "")
    live = bool(current_run) and is_pid_alive(current.get("pid")) and current.get("status") in LIVE_STATUSES

    runs: list[dict[str, Any]] = []
    for run in list_runs(cfg.get("output_dir", "./output"), cfg.get("logging_dir", "./logs"), None):
        is_current = run["run_id"] == current_run
        runs.append({**run, "current": is_current, "live": is_current and live})
    if current_run and not any(run["run_id"] == current_run for run in runs):
        runs.insert(0, _state_run_entry(cfg, current, current_run, live))
    return {"runs": runs}


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


def _sample_sets_payload(log_dir: Optional[str] = None) -> list[dict[str, Any]]:
    """The resolved prompt sets of one run, for the dashboard's sample defaults.

    `log_dir` names the run being shown, so the defaults are the prompts that run samples with —
    its own saved config, or the sets edited for it. Without one (no run resolves) it is today's
    repo `config.toml`, which is also what a run from before the snapshots falls back to.

    A broken entry must not take the dashboard down with it: the charts and the run
    status keep working, and the trainer reports the config error when it starts.
    """
    try:
        mapping = run_config_mapping(log_dir)[0] if log_dir else _train_config_dict()
        return [asdict(sample_set) for sample_set in resolve_sample_sets(mapping)]
    except Exception as exc:
        print(f"[Warn] validation.samples ignored: {exc}", file=sys.stderr)
        return []


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _train_status_payload() -> dict[str, Any]:
    """`status_payload()` plus the settings the run has been asked for but has not adopted yet.

    The trainer reads `settings.json` once per optimizer step, so a switch flipped while a sample
    pass is running stays a request for a while. Reporting the request next to the effective
    values is what lets the dashboard answer a click immediately ("off from the next sample pass")
    instead of showing the old value until the trainer publishes the change.
    """
    payload = status_payload()
    payload["requested"] = _requested_settings(payload)
    return payload


def _requested_settings(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The outstanding `settings.json` request, or None when there is nothing to report.

    Nothing to report means: the run is not live (a request nobody will adopt), no request was
    ever made, or the request already equals the values the trainer published - which is how the
    marker clears itself once the change lands.
    """
    if not is_pid_alive(payload.get("pid")):
        return None
    requested = read_settings()
    if requested is None:
        return None
    effective = payload.get("settings") or {}
    if all(
        requested.get(key) == effective.get(key)
        for key in ("save_every_n_steps", "sampling_enabled")
    ):
        return None
    return requested


def handle_train_status(_params: dict[str, Any]) -> dict[str, Any]:
    return _train_status_payload()


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
            source = resolve_resume_path(resume_raw)
            require_resume_network_type(cfg_obj, read_lora_metadata(source), source)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    try:
        resolve_preload(root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    # A fresh run starts from config.toml: whatever the last run was tuned to must not leak in.
    start_settings = LiveSettings.from_config(cfg_obj)
    request_settings(
        save_every_n_steps=start_settings.save_every_n_steps,
        sampling_enabled=start_settings.sampling_enabled,
    )

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
    # `mark_starting` cleared the state, so publish the values this run will use right away: the
    # card would otherwise read the empty placeholder until the trainer's first optimizer step.
    publish_settings(start_settings)
    return _train_status_payload()


def _require_alive() -> dict[str, Any]:
    current = reconcile()
    if not is_pid_alive(current.get("pid")):
        raise ValueError("no running training process")
    return current


def handle_train_pause(_params: dict[str, Any]) -> dict[str, Any]:
    _require_alive()
    request_train_command("pause")
    return _train_status_payload()


def handle_train_resume(_params: dict[str, Any]) -> dict[str, Any]:
    _require_alive()
    # A one-off generation may have been started while the run was paused; resuming now would
    # put a second SDXL on the same card.
    running = _running_generation(_output_dir(_train_config_dict()))
    if running is not None:
        raise ValueError(
            f"a sample generation is using the GPU ({running.get('id')}); "
            "wait for it to finish, then resume"
        )
    request_train_command("resume")
    return _train_status_payload()


def handle_train_stop(_params: dict[str, Any]) -> dict[str, Any]:
    _require_alive()
    request_train_command("stop")
    return _train_status_payload()


def handle_train_settings(params: dict[str, Any]) -> dict[str, Any]:
    """Change the checkpoint cadence / sampling switch of the run in progress.

    The request lands in the runtime `settings.json`; the trainer adopts it at its next
    optimizer step and publishes the effective values back through `state.json`. This reply
    carries both: `settings` is still what the run is doing now, `requested` is the change that
    has been accepted and is waiting for the trainer (`_requested_settings`). Nothing is written
    to `config.toml`: the next run starts from the file.

    A field the request leaves out keeps the run's value: from `settings.json` when it is there
    (`train_start` keeps it complete), else from the settings the run published (`baseline`), so a
    lone switch flip on a hand-started run cannot arrive as "and stop writing checkpoints".
    """
    current = _require_alive()
    steps = params.get("save_every_n_steps")
    if steps is not None:
        try:
            steps = int(steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("save_every_n_steps must be an integer") from exc
        if steps < 0:
            raise ValueError("save_every_n_steps must be >= 0 (0 disables checkpoints)")
    enabled = params.get("sampling_enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("sampling_enabled must be a boolean")
    if steps is None and enabled is None:
        raise ValueError("nothing to change: pass save_every_n_steps and/or sampling_enabled")
    request_settings(
        save_every_n_steps=steps,
        sampling_enabled=enabled,
        baseline=current.get("settings"),
    )
    return _train_status_payload()


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
    """Clear the run's state so Start can launch a new one.

    Nothing is deleted: the sample images and TensorBoard logs are what the dashboard's run
    history shows afterwards, and the LoRA weight directories are what the run produced.
    `python clean.py` is the tool for wiping a run.
    """
    current = reconcile()
    if current.get("status") in _RESET_BLOCKED and is_pid_alive(current.get("pid")):
        raise ValueError("cannot reset while training is running")
    cfg = _train_config_dict()
    run_id, output_name = _resolve_run(params, cfg)
    if run_id:
        # A `delete_weights` param an older client may still send is ignored: Reset never removes
        # a weight directory, whatever the request asks for.
        cleanup: dict[str, Any] = run_cleanup(
            cfg.get("output_dir", "./output"),
            cfg.get("logging_dir", "./logs"),
            output_name,
            run_id=run_id,
            delete_weights=False,
            delete_samples=False,
            delete_logs=False,
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
            "delete_weights": False,
            "delete_samples": False,
            "delete_logs": False,
            "removed": [],
            "skipped": [],
            "errors": [],
        }
    reset_to_idle()
    payload = _train_status_payload()
    payload["run_id"] = run_id
    payload["cleanup"] = cleanup
    return payload


def _sample_prompts_payload(params: dict[str, Any]) -> dict[str, Any]:
    """One run's effective prompt sets, where they were resolved from, and whether it is live.

    The sets are the same ones `evaluate_checkpoint` and the manual sample passes render with
    (`run_config_mapping`), so the panel, the pass and the scoring all answer with one set of
    prompts. Never fails for a config it cannot use: `sets` comes back empty with `reason` set.
    """
    cfg = _train_config_dict()
    run_id, output_name = _resolve_run(params, cfg)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "output_name": output_name,
        "file": None,
        "edited": False,
        "config_source": "",
        "sets": [],
        "live": False,
        "reason": "",
    }
    if not run_id:
        payload["reason"] = "no run to read sampling prompts for"
        return payload

    log_dir = _log_dir(cfg, run_id)
    override = sample_override_path(log_dir)
    mapping, source = run_config_mapping(log_dir)
    if override.is_file():
        payload["edited"] = True
        payload["file"] = str(override)
    payload["config_source"] = source
    current = reconcile()
    payload["live"] = str(current.get("run_id") or "") == str(run_id) and is_pid_alive(current.get("pid"))
    try:
        sets = resolve_sample_sets(mapping)
    except ValueError as exc:
        payload["reason"] = str(exc)[-500:]
        return payload
    payload["sets"] = [_json_safe(asdict(sample_set)) for sample_set in sets]
    return payload


def handle_sample_prompts(params: dict[str, Any]) -> dict[str, Any]:
    """Read-only: the prompts one run samples with. See `handle_sample_prompts_set`."""
    return _sample_prompts_payload(params)


def handle_sample_prompts_set(params: dict[str, Any]) -> dict[str, Any]:
    """Save the prompt sets a run should sample with, or drop them so it uses its config again.

    A non-empty `sets` list is validated and stored whole in the run's own log directory
    (`sample_sets.json`), which every read path layers over the run's config snapshot — the
    trainer's own sample points included, so an edit made while the run is live lands on the next
    checkpoint it writes. `sets: null` deletes that file. Nothing is written to `config.toml`, and
    the snapshot itself is never rewritten: it stays the record of what the run trained with.

    An entry that omits a key is filled from the run's own config, not from the repo file, so a
    partial request can never mix two runs' settings.
    """
    cfg = _train_config_dict()
    run_id, output_name = _resolve_run(params, cfg)
    if not run_id:
        raise ValueError("no run to set sampling prompts on")
    if "sets" not in params:
        raise ValueError("sets is required (null resets the run to its own config)")

    raw = params.get("sets")
    log_dir = _log_dir(cfg, run_id)
    if raw is None:
        clear_sample_override(log_dir)
    else:
        if not isinstance(raw, list) or not raw:
            raise ValueError("sets must be a non-empty array of tables")
        mapping, _source = run_config_mapping(log_dir)
        try:
            resolved = resolve_sample_sets({**mapping, "samples": raw})
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        write_sample_override(log_dir, resolved)
    return _sample_prompts_payload({"name": output_name, "run_id": run_id})


def handle_clear_checkpoint_samples(params: dict[str, Any]) -> dict[str, Any]:
    """Remove one checkpoint's sample images: the run's own at its step, and the passes on its card.

    Detached generations and evaluations live under `{name}_samples/generated/`, each as a job
    record plus its PNGs, so a card is cleared by removing the images of every job that belongs to
    it (the checkpoint path it names, or its step for a record that names none — the same rule the
    section groups by) together with those records and their logs. The run's own samples of that
    step go too, and nothing else: another step's samples, another checkpoint's passes and the
    job records of a range batch stay where they are.

    Refused while a live trainer is using the GPU (a paused one is free) or another generation is
    running, because both write the directory being cleaned.
    """
    current = reconcile()
    if _gpu_busy(current):
        raise ValueError(
            "training is using the GPU; pause the run (or stop it) before clearing samples"
        )

    cfg = _train_config_dict()
    checkpoint = Path(str(params.get("checkpoint") or "")).expanduser()
    if not checkpoint.is_file():
        raise ValueError(f"not a checkpoint file: {checkpoint}")
    running = _running_generation(_output_dir(cfg))
    if running is not None:
        raise ValueError(f"a generation is still using this card ({running.get('id')})")

    run_id, output_name, samples_dir = _checkpoint_run(params, cfg, checkpoint)
    if not run_id:
        raise ValueError("no run to clear samples for")
    step = _checkpoint_step(checkpoint, output_name)

    removed: list[str] = []
    images = 0
    jobs: list[str] = []

    def drop(path: Path, *, image: bool) -> None:
        nonlocal images
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            print(f"[Warn] could not remove {path}: {exc}", file=sys.stderr)
            return
        removed.append(str(path))
        if image:
            images += 1

    if step is not None:
        for entry in scan_samples(samples_dir).get(str(step), []):
            drop(Path(str(entry["path"])), image=True)

    generated = genjob.generated_dir(samples_dir)
    for job in genjob.list_jobs(generated):
        if not _job_belongs_to(job, checkpoint, step):
            continue
        job_id = str(job.get("id"))
        names = [str(path) for path in (job.get("files") or [])]
        single = str(job.get("image_path") or "")
        if single:
            names.append(single)
        for name in names:
            drop(Path(name), image=True)
        drop(genjob.job_path(generated, job_id), image=False)
        drop(genjob.log_path(generated, job_id), image=False)
        jobs.append(job_id)

    return {
        "run_id": run_id,
        "output_name": output_name,
        "checkpoint": str(checkpoint),
        "step": step,
        "files": removed,
        "images": images,
        "jobs": jobs,
    }


def _job_belongs_to(job: dict[str, Any], checkpoint: Path, step: Optional[int]) -> bool:
    """The card rule of `checkpointRows`: the checkpoint a record names, else its step."""
    recorded = str(job.get("checkpoint") or "")
    if recorded:
        return recorded == str(checkpoint)
    return step is not None and job.get("step") == step


def handle_list_samples(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    run_id, output_name = _resolve_run(params, cfg)
    if not run_id:
        return {"run_id": None, "samples": {}}
    return {"run_id": run_id, "samples": scan_samples(_samples_dir(cfg, run_id, output_name))}


def handle_list_checkpoints(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    output_dir = params.get("output_dir") or cfg.get("output_dir", "./output")
    _run_id, output_name = _resolve_run(params, cfg)
    return {
        "checkpoints": discover_checkpoints(str(output_dir), output_name),
    }


def handle_checkpoint_pins(params: dict[str, Any]) -> dict[str, Any]:
    """The run's pinned checkpoints, read from its own `checkpoint_pins.json`. Read-only.

    Pins belong to one run, so a request that names nothing resolves the run `state.json` is on,
    exactly like `list_samples`; without a run there is nothing to read and nothing to show.
    """
    cfg = _train_config_dict()
    run_id, _output_name = _resolve_run(params, cfg)
    if not run_id:
        return {"run_id": None, "file": None, "pins": []}
    log_dir = _log_dir(cfg, run_id)
    return {"run_id": run_id, "file": str(pins_path(log_dir)), "pins": read_pins(log_dir)}


def handle_checkpoint_pin_set(params: dict[str, Any]) -> dict[str, Any]:
    """Pin or unpin one checkpoint of a run, and answer with the run's whole pin list.

    Unpinning is allowed for a path whose file is gone (a Reset deletes weights and keeps the pin
    file): a stale entry has to stay removable. Pinning checks the file exists, so a typo cannot
    leave a pin nothing will ever show.
    """
    cfg = _train_config_dict()
    run_id, _output_name = _resolve_run(params, cfg)
    if not run_id:
        raise ValueError("no run to pin a checkpoint to")
    if "pinned" not in params:
        raise ValueError("pinned is required")
    target = str(params.get("path") or "").strip()
    if not target:
        raise ValueError("path is empty")

    log_dir = _log_dir(cfg, run_id)
    pins = read_pins(log_dir)
    if bool(params.get("pinned")):
        if not Path(target).is_file():
            raise ValueError(f"checkpoint not found: {target}")
        pins = pin_entry(pins, target, dir_name=params.get("dir"), step=params.get("step"))
    else:
        pins = unpin_entry(pins, target)
    return {"run_id": run_id, "file": str(write_pins(log_dir, run_id, pins)), "pins": pins}


def _tagger_script() -> Path:
    script = _repo_root() / "tagger2" / "main.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing tagger: {script}")
    return script


def _spawn_tagger(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a tagger process with the same interpreter as api.py (the axl env)."""
    return subprocess.run(
        [sys.executable, "-u", str(_tagger_script()), *argv],
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def run_tagger_info() -> dict[str, Any]:
    """`tagger2/main.py --info`: the categories and calibrated thresholds, read from the model."""
    proc = _spawn_tagger(["--info"])
    raw_out = (proc.stdout or "").strip()
    if proc.returncode != 0 or not raw_out:
        detail = (proc.stderr or raw_out or f"tagger exited {proc.returncode}").strip()
        raise RuntimeError(detail[-2000:])
    try:
        payload = json.loads(raw_out.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"tagger returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"tagger returned {type(payload).__name__}, expected an object")
    return payload


def run_tagger_process(
    directory: str,
    threshold: float,
    batch_size: int | None = None,
    categories: str | None = None,
    only_tags: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Caption a folder with the Pixai tagger; `categories` is a comma-joined list of names.

    `only_tags` switches the pass to partial tagging: the named tags are added to the captions that
    show them and nothing else is touched, and the categories do not take part (a requested tag is
    looked up in all of them). The two are mutually exclusive by construction here.
    """
    argv = [str(directory), "--threshold", str(threshold), "--json"]
    if only_tags:
        argv += ["--only-tags", ",".join(str(tag) for tag in only_tags)]
    elif categories:
        argv += ["--categories", str(categories)]
    # No `--batch-size` when the caller did not ask for one: the script's own default applies.
    if batch_size is not None:
        argv += ["--batch-size", str(int(batch_size))]

    proc = _spawn_tagger(argv)
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
    return find_samples_dir(output_dir / str(run_id), output_name)


def _output_dir(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("output_dir") or ".")).expanduser()


def _log_dir(cfg: dict[str, Any], run_id: str) -> Path:
    """A run's TensorBoard directory, which is also where its pinned checkpoints live."""
    return Path(str(cfg.get("logging_dir") or "./logs")).expanduser() / str(run_id)


def _checkpoint_step(checkpoint: Path, output_name: str) -> Optional[int]:
    """The step a checkpoint was saved at: its directory name, else its own kohya metadata.

    The step is what attaches a generated image to a checkpoint in the Dashboard's Checkpoints
    section, so a job without one would render into the run's `generated/` directory and never be
    shown anywhere.
    """
    parsed = parse_checkpoint_dir(checkpoint.parent.name, output_name)
    if parsed.get("step") is not None:
        return int(parsed["step"])
    try:
        return genjob.checkpoint_step(read_lora_metadata(checkpoint))
    except ValueError:
        return None


def _running_generation(output_dir: Path) -> Optional[dict[str, Any]]:
    """Any generation job still running under `output_dir`, whatever run it belongs to.

    The GPU is single-tenant, so a job started for another run blocks a new one just as much as
    a job of the run being viewed. A job whose process is gone is closed as an error on the way
    past, or a kill would block generations forever.
    """
    if not output_dir.is_dir():
        return None
    for spec in sorted(output_dir.glob("*/*_samples/generated/*.json")):
        job = genjob.read_job(spec)
        if job is None or job.get("state") != genjob.STATE_RUNNING:
            continue
        if is_pid_alive(job.get("pid")):
            # A cancel asked to stop is still using the card until its process is really gone.
            return job
        if job.get("cancel_requested"):
            genjob.update_job(spec.parent, str(job.get("id") or spec.stem), state=genjob.STATE_CANCELLED, error=None)
        else:
            genjob.update_job(
                spec.parent,
                str(job.get("id") or spec.stem),
                state=genjob.STATE_ERROR,
                error="the generator exited before finishing (see the job's .log)",
            )
    return None


def _generated_dir(params: dict[str, Any], cfg: dict[str, Any]) -> Optional[tuple[str, str, Path]]:
    """(run_id, output_name, generated dir) for the resolved run, or None when no run exists."""
    run_id, output_name = _resolve_run(params, cfg)
    if not run_id:
        return None
    return run_id, output_name, genjob.generated_dir(_samples_dir(cfg, run_id, output_name))


def _close_dead_jobs(generated: Path) -> None:
    """A generator whose process is gone must not leave its job `running`.

    A job that was asked to cancel closes as `cancelled` (not a failure); anything else died on its
    own and closes as `error` with the log to look at.
    """
    for job in genjob.list_jobs(generated):
        if job.get("state") != genjob.STATE_RUNNING or is_pid_alive(job.get("pid")):
            continue
        if job.get("cancel_requested"):
            genjob.update_job(generated, str(job["id"]), state=genjob.STATE_CANCELLED, error=None)
        else:
            genjob.update_job(
                generated,
                str(job["id"]),
                state=genjob.STATE_ERROR,
                error="the generator exited before finishing (see the job's .log)",
            )


def _reconcile_generated(generated: Path) -> list[dict[str, Any]]:
    """Every job of the run, with the ones whose generator is gone closed first."""
    _close_dead_jobs(generated)
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


def _spawn_generator(generated: Path, job: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Write the job file, start `generate_sample.py` detached on it, return (job, log path)."""
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
    return genjob.update_job(generated, job["id"], pid=proc.pid), str(log)


def _claim_generation_run(params: dict[str, Any]) -> tuple[dict[str, Any], str, str, Path]:
    """Shared gate + run resolution of every generation entry point.

    Refuses while a live trainer is using the GPU (a paused one is fine) and while any other
    generation is running, and returns (cfg, run_id, output_name, generated dir).
    """
    current = reconcile()
    if _gpu_busy(current):
        raise ValueError(
            "training is using the GPU; pause the run (or stop it) before generating samples"
        )

    cfg = _train_config_dict()
    resolved = _generated_dir(params, cfg)
    if resolved is None:
        raise ValueError("no run to attach the sample to; finish a run first")
    run_id, output_name, generated = resolved

    running = _running_generation(_output_dir(cfg))
    if running is not None:
        raise ValueError(f"a generation is already running ({running.get('id')})")
    return cfg, run_id, output_name, generated


def _claim_generation(params: dict[str, Any]) -> tuple[dict[str, Any], str, str, Path, Path]:
    """`_claim_generation_run` plus the checkpoint one of the two forms renders from."""
    cfg, run_id, output_name, generated = _claim_generation_run(params)

    checkpoint = Path(str(params.get("checkpoint") or "")).expanduser()
    if not checkpoint.is_file():
        raise ValueError(f"not a checkpoint file: {checkpoint}")
    return cfg, run_id, output_name, generated, checkpoint


def handle_generate_sample(params: dict[str, Any]) -> dict[str, Any]:
    """Start one "sample with this checkpoint" job. Returns immediately; Ranko follows the job file.

    The image keeps its own prompt/CFG/seed and lands in the run's `_samples/generated/`; the
    form's defaults come from the run's own prompts, and `generate_checkpoint_samples` renders the
    run's whole set list.
    """
    cfg, run_id, output_name, generated, checkpoint = _claim_generation(params)

    first_set = _sample_sets_payload(_log_dir(cfg, run_id))
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

    if request.get("step") is None:
        request["step"] = _checkpoint_step(checkpoint, output_name)

    generated.mkdir(parents=True, exist_ok=True)
    job = genjob.new_job(
        request,
        run_id=run_id,
        output_name=output_name,
        checkpoint=str(checkpoint),
    )
    job, log = _spawn_generator(generated, job)
    return {"job": _json_safe(job), "log_path": log}


def handle_generate_checkpoint_samples(params: dict[str, Any]) -> dict[str, Any]:
    """Render this checkpoint's `[[validation.samples]]` sets as one detached job.

    This is the "one checkpoint, one full sample pass" the Dashboard offers for any checkpoint
    once the GPU is free. The prompts, size, steps, CFG, seed and repeat are the ones the run
    samples with — its own saved config, or the sets the Dashboard edited for it — recorded on the
    job so the pass renders what it was planned with; network type/dim/alpha, `clip_skip`,
    `max_token_length` and the base model come from the checkpoint's own metadata. Images land in
    `_samples/generated/` next to the run's own samples, so a training-produced sample is never
    overwritten.
    """
    cfg, run_id, output_name, generated, checkpoint = _claim_generation(params)

    log_dir = _log_dir(cfg, run_id)
    try:
        sets = resolve_sample_sets(run_config_mapping(log_dir)[0])
    except ValueError as exc:
        raise ValueError(f"this run has no sample prompts to render: {exc}") from exc

    generated.mkdir(parents=True, exist_ok=True)
    job = genjob.new_job(
        {"step": _checkpoint_step(checkpoint, output_name)},
        run_id=run_id,
        output_name=output_name,
        checkpoint=str(checkpoint),
        mode=genjob.MODE_SETS,
        total_images=sum(sample_set.repeat for sample_set in sets),
        extra={
            "sample_sets": [asdict(sample_set) for sample_set in sets],
            "config_log_dir": str(log_dir),
        },
    )
    job, log = _spawn_generator(generated, job)
    return {"job": _json_safe(job), "log_path": log}


def _signal_generator(pid: Any) -> None:
    """Ask one generator process (and its group) to stop, without touching anything else.

    The generator is spawned with `start_new_session=True`, so its pid is its own process group
    *and* its session — verified through `/proc` before signalling a group, because a reused pid
    must not take an unrelated group down.
    """
    if not is_pid_alive(pid):
        return
    pid_i = int(pid)
    if orphans.session_of(pid_i) == pid_i:
        try:
            os.killpg(pid_i, signal.SIGTERM)
            return
        except OSError:
            pass
    try:
        os.kill(pid_i, signal.SIGTERM)
    except OSError:
        pass


def _find_running_job(output_dir: Path, job_id: str = "") -> Optional[tuple[Path, dict[str, Any]]]:
    """`(generated dir, job)` of the running job called `job_id`, or of the running one when no id is
    given, wherever it lives under `output_dir`.

    An id names one job (its stem carries the checkpoint and a timestamp), so a cancel does not have
    to know which run it belongs to: the page can be showing a past run that `state.json` does not
    carry at all. With no id the GPU is single-tenant, so there is at most one job to mean. A job
    whose process is gone is closed on the way past, exactly as `_close_dead_jobs` does.
    """
    if not output_dir.is_dir():
        return None
    for spec in sorted(output_dir.glob("*/*_samples/generated/*.json")):
        job = genjob.read_job(spec)
        if job is None:
            continue
        if job_id and str(job.get("id") or spec.stem) != job_id:
            continue
        _close_dead_jobs(spec.parent)
        job = genjob.read_job(spec)
        if job is not None and job.get("state") == genjob.STATE_RUNNING:
            return spec.parent, job
        if job_id:
            return None
    return None


def handle_cancel_generation(params: dict[str, Any]) -> dict[str, Any]:
    """Ask a running generation job to stop, the batch one included.

    The job keeps its state until its process is gone (so the card can say `cancelling…` and no new
    generation starts on the same card meanwhile); whatever images it had written stay in `files`.
    No GPU gate: this has to work while the trainer is using the card. The job is looked for under
    every run, by `id` when the request names one and by "the running one" otherwise — the page's
    own run (`name` / `run_id`, else `state.json`) says nothing about a pass started from a past
    run's card, and a helper whose `state.json` carries no run can still stop the only running job.
    """
    cfg = _train_config_dict()
    wanted = str(params.get("id") or "").strip()
    resolved = _generated_dir(params, cfg)
    generated: Optional[Path] = resolved[2] if resolved is not None else None

    job: Optional[dict[str, Any]] = None
    if generated is not None:
        job = next(
            (
                item
                for item in _reconcile_generated(generated)
                if item.get("state") == genjob.STATE_RUNNING
                and (not wanted or str(item.get("id")) == wanted)
            ),
            None,
        )
    if job is None:
        found = _find_running_job(_output_dir(cfg), wanted)
        if found is not None:
            generated, job = found

    if generated is None:
        raise ValueError("no run to cancel a generation for")
    if job is None:
        if wanted:
            raise ValueError(f"{wanted} is not running")
        raise ValueError("no generation is running for this run")

    _signal_generator(job.get("pid"))
    updated = genjob.update_job(generated, str(job["id"]), cancel_requested=True)
    return {"job": _json_safe(updated), "cancelled": True}


def _range_bounds(params: dict[str, Any]) -> tuple[int, int]:
    """`(from_step, to_step)` a batch covers, validated before any work is planned."""
    bounds: list[int] = []
    for key in ("from_step", "to_step"):
        raw = params.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if value < 0:
            raise ValueError(f"{key} must be >= 0")
        bounds.append(value)
    from_step, to_step = bounds
    if from_step > to_step:
        raise ValueError(f"from_step ({from_step}) must not be greater than to_step ({to_step})")
    return from_step, to_step


def handle_generate_checkpoint_samples_batch(params: dict[str, Any]) -> dict[str, Any]:
    """Render the run's sample sets for every checkpoint of one step range, in one job.

    The run's own checkpoints whose step falls inside `from_step..to_step` are rendered oldest
    first by a single detached process, which gives each of them its own `generate_checkpoint_samples`
    job (so the images land beside the run's samples and are shown under that checkpoint's card).
    The prompts are the run's own, recorded on the batch as `config_log_dir` so the runner resolves
    them once for the whole range.
    """
    cfg, run_id, output_name, generated = _claim_generation_run(params)
    from_step, to_step = _range_bounds(params)

    candidates = [
        item
        for item in discover_checkpoints(str(_output_dir(cfg)), output_name)
        if item.get("run_id") == run_id
        and item.get("step") is not None
        and from_step <= int(item["step"]) <= to_step
    ]
    if not candidates:
        raise ValueError(
            f"no checkpoints between step {from_step} and {to_step} for this run"
        )
    candidates.sort(key=lambda item: (int(item["step"]), str(item["dir"])))

    log_dir = _log_dir(cfg, run_id)
    try:
        sets = resolve_sample_sets(run_config_mapping(log_dir)[0])
    except ValueError as exc:
        raise ValueError(f"this run has no sample prompts to render: {exc}") from exc
    images_per_checkpoint = sum(sample_set.repeat for sample_set in sets)

    generated.mkdir(parents=True, exist_ok=True)
    job = genjob.new_batch_job(
        run_id=run_id,
        output_name=output_name,
        checkpoints=[
            {"path": str(item["path"]), "step": int(item["step"]), "dir": str(item["dir"])}
            for item in candidates
        ],
        from_step=from_step,
        to_step=to_step,
        images_per_checkpoint=images_per_checkpoint,
        config_log_dir=str(log_dir),
    )
    job, log = _spawn_generator(generated, job)
    return {"job": _json_safe(job), "log_path": log}


def _checkpoint_run(
    params: dict[str, Any],
    cfg: dict[str, Any],
    checkpoint: Path,
) -> tuple[str, str, Path]:
    """`(run_id, output_name, samples dir)` of the run a checkpoint belongs to.

    A checkpoint sits at `{output_dir}/{run_id}/{name}_sXXX/{name}.safetensors`, so the path names
    its own run: an evaluation of a past run's checkpoint is held to *that* run's config and topped
    up beside *that* run's samples, even while the Dashboard is showing another run. A checkpoint
    outside the configured `output_dir` (a copied file) falls back to the run this request resolved.
    """
    output_root = Path(str(cfg.get("output_dir") or ".")).expanduser()
    resolved_run, resolved_name = _resolve_run(params, cfg)
    candidate = ""
    try:
        parts = checkpoint.expanduser().resolve().relative_to(output_root.resolve()).parts
        if len(parts) > 1:
            candidate = str(parts[0])
    except ValueError:  # not under output_dir at all
        candidate = ""
    if candidate and run_output_name(candidate):
        name = run_output_name(candidate)
        return candidate, name, find_samples_dir(output_root / candidate, name)
    return resolved_run, resolved_name, _samples_dir(cfg, resolved_run, resolved_name)


def handle_evaluate_checkpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Start one evaluation of this checkpoint: top its samples up to Depth, tag them, score them.

    Returns immediately (`{job, log_path}`), like the generation entries; Ranko follows the job file
    and the reply's record already carries the plan (which slots to render, and which images are to
    be scored with which prompt). Prompts and sampling values come from the config the run that
    trained this checkpoint saved beside its logs, falling back to today's `config.toml` for a run
    from before snapshots existed — `config_source` says which one was used. A checkpoint that
    already holds `depth` images renders nothing and goes straight to tagging.
    """
    cfg, _run_id, _output_name, _generated, checkpoint = _claim_generation(params)

    depth = evaluation.normalize_depth(params.get("depth"))
    raw_threshold = params.get("threshold")
    threshold = evaluation.normalize_threshold(0.35 if raw_threshold is None else raw_threshold)
    categories = _clean_tagger_categories(params.get("categories")) or ["general"]
    # The tags the scoring is narrowed to; empty means every tag a prompt asks for.
    tags = sorted(evaluation.selected_tags(_clean_string_list(params.get("tags"), "tags")))

    run_id, output_name, samples_dir = _checkpoint_run(params, cfg, checkpoint)
    config_log_dir = _log_dir(cfg, run_id)
    mapping, config_source = run_config_mapping(config_log_dir)
    sets = resolve_sample_sets(mapping)
    step = _checkpoint_step(checkpoint, output_name)

    images = evaluation.collect_images(
        samples_dir=samples_dir,
        generated_dir=genjob.generated_dir(samples_dir),
        checkpoint=checkpoint,
        step=step,
        sets=sets,
    )
    plan = evaluation.expansion_plan(sets, depth, images)

    generated = genjob.generated_dir(samples_dir)
    generated.mkdir(parents=True, exist_ok=True)
    job = genjob.new_evaluation_job(
        run_id=run_id,
        output_name=output_name,
        checkpoint=str(checkpoint),
        step=step,
        depth=depth,
        threshold=threshold,
        categories=categories,
        config_source=config_source,
        config_log_dir=str(config_log_dir),
        plan=plan,
        images=[image.to_dict() for image in images],
        sample_sets=[asdict(sample_set) for sample_set in sets],
        tags=tags,
    )
    job, log = _spawn_generator(generated, job)
    return {"job": _json_safe(job), "log_path": log}


def handle_evaluation_prompts(params: dict[str, Any]) -> dict[str, Any]:
    """Read-only: the prompts (and their tag frequencies) an evaluation of this checkpoint would use.

    The panel asks this before anything is rendered, so the operator can pick which tags the scoring
    should look at. It resolves the run exactly as `evaluate_checkpoint` does — the checkpoint's own
    run wins when its path sits under `output_dir` — and reads the same config snapshot or hparams
    fallback, so what the picker offers is what the pass will score against.

    Never fails for a config it cannot use: `tags` comes back empty with `reason` set, and the panel
    falls back to "score every tag the prompt asks for".
    """
    cfg = _train_config_dict()
    raw = str(params.get("checkpoint") or "").strip()
    if not raw:
        raise ValueError("checkpoint is empty")
    checkpoint = resolve_resume_path(raw)
    run_id, output_name, _samples_dir = _checkpoint_run(params, cfg, checkpoint)
    mapping, config_source = run_config_mapping(_log_dir(cfg, run_id))
    payload: dict[str, Any] = {
        "run_id": run_id,
        "output_name": output_name,
        "checkpoint": str(checkpoint),
        "config_source": config_source,
        "sample_sets": [],
        "tags": [],
        "reason": "",
    }
    try:
        sets = resolve_sample_sets(mapping)
    except ValueError as exc:
        payload["reason"] = str(exc)[-500:]
        return payload
    payload["sample_sets"] = [_json_safe(asdict(sample_set)) for sample_set in sets]
    payload["tags"] = evaluation.prompt_tag_counts(sets)
    return payload


def handle_dataset_tag(params: dict[str, Any]) -> dict[str, Any]:
    if _gpu_busy(reconcile()):
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

    # The script refuses a name the model does not have; this only cleans the request up.
    categories = _clean_tagger_categories(params.get("categories"))
    only_tags = _clean_string_list(params.get("only_tags"), "only_tags")

    batch_raw = params.get("batch_size")
    batch_size: int | None = None
    if batch_raw is not None:
        try:
            batch_size = int(batch_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("batch_size must be an integer") from exc
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    return run_tagger_process(
        str(directory_path),
        threshold,
        batch_size=batch_size,
        categories=",".join(categories) if categories else None,
        only_tags=only_tags or None,
    )


def _clean_string_list(raw: Any, field: str) -> list[str]:
    """A comma-separated string or a list of names as a clean, ordered, de-duplicated list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        items = [str(item) for item in raw]
    else:
        raise ValueError(f"{field} must be a list of names or a comma-separated string")
    ordered: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def _clean_tagger_categories(raw: Any) -> list[str]:
    """Requested tagger categories as a clean, de-duplicated list; empty means the caller's default."""
    return _clean_string_list(raw, "categories")


def handle_tagger_info(_params: dict[str, Any]) -> dict[str, Any]:
    """Read-only: what the tagger can write. Never raises, so the card can show why it is empty."""
    try:
        return _json_safe(run_tagger_info())
    except Exception as exc:  # noqa: BLE001 - the reply carries the reason instead of an error frame
        return {
            "available": False,
            "engine": "",
            "model": "",
            "model_path": "",
            "cache_dir": "",
            "categories": [],
            "default_categories": [],
            "reason": str(exc)[-500:],
        }


def handle_dataset_counts(params: dict[str, Any]) -> dict[str, Any]:
    """Read-only: how many images each training folder holds, and the per-epoch sample total.

    The Utils → Training section asks this so its step estimate follows unsaved edits: the caller
    hands over the folders the form currently holds (`[{path, repeat}]`), and a request that names
    none falls back to the config's own `[[environment.train_data]]` entries. Counting is a
    directory walk; the epoch/batch/GA arithmetic stays on the client, so editing those costs no
    round trip. A folder that is missing answers with its reason instead of failing the call.
    """
    raw = params.get("dirs")
    if raw is None:
        entries: list[Any] = resolve_train_data_entries(_train_config_dict())
    elif isinstance(raw, (list, tuple)):
        entries = [item for item in raw if isinstance(item, dict)]
        if len(entries) != len(raw):
            raise ValueError("dirs must be a list of {path, repeat} objects")
    else:
        raise ValueError("dirs must be a list of {path, repeat} objects")
    return _json_safe(estimate.count_train_images(entries))


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


def handle_prompt_matrix(_params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.prompt_matrix()


def handle_prompt_profile_list(_params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.prompt_profile_list()


def handle_prompt_profile_get(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.prompt_profile_get(str(params.get("name") or ""))


def handle_prompt_profile_save(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.prompt_profile_save(
        str(params.get("name") or ""),
        str(params.get("text") or ""),
        bool(params.get("overwrite")),
    )


def handle_prompt_profile_delete(params: dict[str, Any]) -> dict[str, Any]:
    return fsrpc.prompt_profile_delete(str(params.get("name") or ""))


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
        if not fsrpc.blob_path_allowed(path, cfg, extra_roots=_automation_roots()):
            marks.append({"path": raw, "error": "path is not an allowed image"})
        else:
            marks.append(None)
    return marks


def _automation_roots() -> list[Path]:
    """Folders the Automation page may show images from, on top of the dataset and output roots."""
    roots = [automation.automation_root()]
    try:
        roots.append(Path(automation.load_settings()["output_dir"]).expanduser())
    except (ValueError, OSError, KeyError):
        pass
    return roots


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


# --- Automation page: settings, workflows, prompt sets, ComfyUI jobs ---


def _automation_object_info(server: str = "") -> Optional[dict[str, Any]]:
    """The instance's node definitions, when one can be reached; None means "check skipped"."""
    try:
        settings = automation.load_settings()
        url = server or settings["server"]
        found = comfy.discover(server=url, budget=2.0) if not url else comfy.discover(server=url)
        if not found["found"]:
            return None
        return comfy.ComfyClient(found["url"]).object_info()
    except (comfy.ComfyError, ValueError, OSError):
        return None


def _automation_workflow_report(path: Path, positive_node: str = "", object_info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    payload = automation.load_workflow(path)
    report = automation.validate_workflow(payload, positive_node=positive_node, object_info=object_info)
    return {"name": path.stem, "path": str(path), **report}


def handle_automation_config_get(_params: dict[str, Any]) -> dict[str, Any]:
    settings = automation.load_settings()
    return {
        "settings": settings,
        "default_output_dir": str(automation.default_output_dir()),
        "paths": {
            "root": str(automation.automation_root()),
            "workflows": str(automation.workflows_dir()),
            "prompts": str(automation.prompts_dir()),
            "jobs": str(automation.jobs_dir()),
        },
    }


def handle_automation_config_save(params: dict[str, Any]) -> dict[str, Any]:
    payload = params.get("settings")
    if not isinstance(payload, dict):
        raise ValueError("settings must be an object")
    return {"settings": automation.save_settings(payload)}


def handle_automation_discover(params: dict[str, Any]) -> dict[str, Any]:
    server = str(params.get("server") or "")
    if server:
        # An explicit address is probed on its own; discovery would not add anything to it.
        client = comfy.ComfyClient(server)
        version = client.version()
        if version is None:
            return {
                "found": False,
                "url": comfy.normalize_server(server),
                "version": "",
                "queue_running": 0,
                "queue_pending": 0,
                "checked": [{"url": comfy.normalize_server(server), "ok": False, "reason": "no answer"}],
                "probed_all": True,
            }
        queue: dict[str, Any] = {}
        try:
            queue = client.queue()
        except comfy.ComfyError:
            queue = {}
        running = queue.get("queue_running")
        pending = queue.get("queue_pending")
        return {
            "found": True,
            "url": client.server,
            "version": version,
            "queue_running": len(running) if isinstance(running, list) else 0,
            "queue_pending": len(pending) if isinstance(pending, list) else 0,
            "checked": [],
            "probed_all": True,
        }
    return comfy.discover()


def handle_automation_workflow_list(_params: dict[str, Any]) -> dict[str, Any]:
    settings = automation.load_settings()
    object_info = _automation_object_info(settings["server"])
    workflows: list[dict[str, Any]] = []
    for item in automation.workflow_list():
        path = Path(item["path"])
        if not item["valid"]:
            workflows.append({"name": item["name"], "path": str(path), "valid": False, "error": item["error"]})
            continue
        try:
            workflows.append(
                _automation_workflow_report(path, positive_node=settings["positive_node"], object_info=object_info)
            )
        except ValueError as exc:
            workflows.append({"name": item["name"], "path": str(path), "valid": False, "error": str(exc)})
    return {
        "workflows": workflows,
        "default_workflow": settings["workflow"],
        "model_check": object_info is not None,
    }


def handle_automation_workflow_validate(params: dict[str, Any]) -> dict[str, Any]:
    raw = str(params.get("path") or "")
    if not raw:
        raise ValueError("path is required")
    settings = automation.load_settings()
    object_info = _automation_object_info(settings["server"])
    report = _automation_workflow_report(
        Path(raw).expanduser(),
        positive_node=str(params.get("positive_node") or settings["positive_node"]),
        object_info=object_info,
    )
    report["model_check"] = object_info is not None
    return report


def handle_automation_workflow_save(params: dict[str, Any]) -> dict[str, Any]:
    return automation.workflow_save(str(params.get("name") or ""), str(params.get("text") or ""))


def handle_automation_workflow_delete(params: dict[str, Any]) -> dict[str, Any]:
    return automation.workflow_delete(str(params.get("name") or ""))


def handle_automation_prompt_list(_params: dict[str, Any]) -> dict[str, Any]:
    return {"prompts": automation.prompt_set_list()}


def handle_automation_prompt_save(params: dict[str, Any]) -> dict[str, Any]:
    return automation.prompt_set_save(str(params.get("name") or ""), params.get("text"))


def handle_automation_prompt_get(params: dict[str, Any]) -> dict[str, Any]:
    return automation.prompt_set_get(str(params.get("name") or ""))


def handle_automation_prompt_delete(params: dict[str, Any]) -> dict[str, Any]:
    return automation.prompt_set_delete(str(params.get("name") or ""))


def _automation_runner_script() -> Path:
    script = _repo_root() / "trainer" / "run_automation.py"
    if not script.is_file():
        raise FileNotFoundError(f"missing runner: {script}")
    return script


def _spawn_automation_job(
    job_id: str,
    output_dir: str,
    only_failed: bool = False,
    image: str = "",
    append: Optional[int] = None,
    append_all: bool = False,
    images: int = 1,
) -> int:
    spec_path = automation.job_path(job_id, output_dir)
    log = automation.log_path(job_id, output_dir)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(_automation_runner_script()), "--spec", str(spec_path)]
    if only_failed:
        command.append("--only-failed")
    if image:
        command += ["--image", str(image)]
    elif append is not None:
        command += ["--append", str(int(append)), "--images", str(int(images))]
    elif append_all:
        command += ["--append-all", "--images", str(int(images))]
    with open(log, "a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            command,
            cwd=str(_repo_root()),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return proc.pid


def handle_automation_job_start(params: dict[str, Any]) -> dict[str, Any]:
    """Write the job record, then spawn the runner detached. Returns immediately."""
    settings = automation.load_settings()
    merged = dict(settings)
    for key in ("server", "workflow", "positive_node", "count", "poll", "output_dir"):
        if params.get(key) not in (None, ""):
            merged[key] = params[key]
    settings = automation.normalize_settings(merged)

    prompts_payload = params.get("prompts")
    if prompts_payload in (None, "", []):
        set_name = str(params.get("prompt_set") or "")
        if not set_name:
            raise ValueError("prompts or a prompt set is required")
        prompts = automation.normalize_prompts(automation.prompt_set_get(set_name)["text"])
    else:
        prompts = automation.normalize_prompts(prompts_payload)

    workflow_raw = settings["workflow"]
    if not workflow_raw:
        raise ValueError("pick a workflow first")
    workflow_path = Path(workflow_raw).expanduser()
    if not workflow_path.is_file():
        workflow_path = automation.workflow_path(Path(workflow_raw).stem)
    report = _automation_workflow_report(workflow_path, positive_node=settings["positive_node"])
    if not report["valid"]:
        raise ValueError(f"workflow is not usable: {report['error']}")
    if settings["count"] > 1 and not report["batch_size_nodes"]:
        raise ValueError("this workflow has no numeric batch_size input, so images per prompt cannot apply")
    if not report["positive_node"]:
        raise ValueError("pick the CLIPTextEncode node that receives the prompt")

    output_dir = settings["output_dir"]
    running = next(
        (job for job in automation.reconcile_jobs(output_dir) if job.get("state") == automation.STATE_RUNNING),
        None,
    )
    if running is not None:
        raise ValueError(f"a job is already running ({running.get('id')})")

    stem = Path(workflow_path).stem or "automation"
    job_id = automation.new_job_id(stem)
    now = time.time()
    job = {
        "id": job_id,
        "state": automation.STATE_RUNNING,
        "created_at": now,
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "pid": None,
        "server": settings["server"],
        "comfy_url": "",
        "workflow": str(workflow_path),
        "workflow_path": str(workflow_path),
        "positive_node": report["positive_node"],
        "count": settings["count"],
        "poll": settings["poll"],
        "output_dir": output_dir,
        "error": None,
        "prompts": [
            {"index": index, "text": text, "state": automation.PROMPT_STATE_PENDING, "images": []}
            for index, text in enumerate(prompts)
        ],
    }
    automation.write_job(job, output_dir)
    pid = _spawn_automation_job(job_id, output_dir)
    job = automation.update_job(job_id, output_dir, pid=pid, started_at=now)
    return {"job": _json_safe(automation.job_summary(job)), "log_path": str(automation.log_path(job_id, output_dir))}


def handle_automation_job_list(_params: dict[str, Any]) -> dict[str, Any]:
    settings = automation.load_settings()
    jobs = automation.reconcile_jobs(settings["output_dir"])
    return {"jobs": [_json_safe(automation.job_summary(job)) for job in jobs]}


def handle_automation_job_get(params: dict[str, Any]) -> dict[str, Any]:
    settings = automation.load_settings()
    return _automation_job_detail(_automation_job(str(params.get("id") or ""), settings), settings)


def _automation_log_tail(path: Path, lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def handle_automation_job_cancel(params: dict[str, Any]) -> dict[str, Any]:
    """Stop a running job: signal its own process group, then mark it cancelled."""
    job_id = str(params.get("id") or "")
    settings = automation.load_settings()
    for job in automation.reconcile_jobs(settings["output_dir"]):
        if str(job.get("id")) != job_id:
            continue
        pid = job.get("pid")
        if job.get("state") == automation.STATE_RUNNING and is_pid_alive(pid):
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError, TypeError, ValueError):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError, TypeError, ValueError):
                    pass
            for _ in range(40):
                if not is_pid_alive(pid):
                    break
                time.sleep(0.05)
        updated = automation.update_job(
            job_id,
            job.get("output_dir") or settings["output_dir"],
            state=automation.STATE_CANCELLED,
            error=job.get("error") or "cancelled",
            finished_at=time.time(),
        )
        return {"job": _json_safe(automation.job_summary(updated))}
    raise ValueError(f"no job named {job_id}")


def handle_automation_job_retry_failed(params: dict[str, Any]) -> dict[str, Any]:
    """Run the prompts that do not have an image yet, in the same job directory."""
    job_id = str(params.get("id") or "")
    settings = automation.load_settings()
    for job in automation.reconcile_jobs(settings["output_dir"]):
        if str(job.get("id")) != job_id:
            continue
        if job.get("state") == automation.STATE_RUNNING and is_pid_alive(job.get("pid")):
            raise ValueError(f"{job_id} is still running")
        output_dir = job.get("output_dir") or settings["output_dir"]
        prompts = job.get("prompts") if isinstance(job.get("prompts"), list) else []
        if not any(isinstance(p, dict) and p.get("state") != automation.PROMPT_STATE_DONE for p in prompts):
            raise ValueError("every prompt of that job already produced its images")
        pid = _spawn_automation_job(job_id, output_dir, only_failed=True)
        updated = automation.update_job(
            job_id,
            output_dir,
            state=automation.STATE_RUNNING,
            pid=pid,
            error=None,
            finished_at=None,
        )
        return {
            "job": _json_safe(automation.job_summary(updated)),
            "log_path": str(automation.log_path(job_id, output_dir)),
        }
    raise ValueError(f"no job named {job_id}")


def _automation_job(job_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    """The one job a request names; `ValueError` when no job has that id."""
    if not job_id:
        raise ValueError("id is required")
    for job in automation.reconcile_jobs(settings["output_dir"]):
        if str(job.get("id")) == job_id:
            return job
    raise ValueError(f"no job named {job_id}")


def _automation_job_detail(job: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """The payload `automation_job_get` answers with, so a Gallery action needs one round trip."""
    job_id = str(job.get("id") or "")
    payload = dict(job)
    payload["summary"] = automation.job_summary(job)
    output_dir = job.get("output_dir") or settings["output_dir"]
    payload["log_tail"] = _automation_log_tail(automation.log_path(job_id, output_dir))
    return _json_safe(payload)


def _automation_idle_job(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """(job, settings, output_dir, job_id) for an action that must not touch a running job."""
    settings = automation.load_settings()
    job = _automation_job(str(params.get("id") or ""), settings)
    job_id = str(job.get("id"))
    if job.get("state") == automation.STATE_RUNNING and is_pid_alive(job.get("pid")):
        raise ValueError(f"{job_id} is still running")
    return job, settings, str(job.get("output_dir") or settings["output_dir"]), job_id


def _automation_image_path(job_id: str, output_dir: str, image: str) -> Path:
    """One image of a job, refusing a name that is not a bare file name inside its own directory."""
    root = automation.images_dir(job_id, output_dir)
    target = root / image
    if not image or target.parent != root:
        raise ValueError(f"not an image name: {image}")
    return target


def _automation_prompt_index(params: dict[str, Any]) -> int:
    raw = params.get("prompt_index")
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"prompt_index must be a number, not {raw!r}") from None


def handle_automation_image_delete(params: dict[str, Any]) -> dict[str, Any]:
    """Delete one image (and the sidecar beside it) from a job's directory and record.

    The record is the truth about which images a prompt has, so the name must be one it holds.
    Deleting the last image of a prompt drops that entry — there is nothing left to describe —
    and the remaining entries renumber so their `index` keeps matching their position.
    """
    job, settings, output_dir, job_id = _automation_idle_job(params)
    image = str(params.get("image") or "")
    prompts = job.get("prompts")
    if automation.prompt_entry_index(prompts, image) is None:
        raise ValueError(f"no prompt of {job_id} holds the image {image}")
    target = _automation_image_path(job_id, output_dir, image)
    for path in (target, target.with_suffix(".txt")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    updated = automation.update_job(job_id, output_dir, prompts=automation.drop_image(prompts, image))
    return _automation_job_detail(updated, settings)


def handle_automation_image_regenerate(params: dict[str, Any]) -> dict[str, Any]:
    """Draw one image again with a new random seed, writing over that image and its sidecar."""
    job, settings, output_dir, job_id = _automation_idle_job(params)
    image = str(params.get("image") or "")
    if automation.prompt_entry_index(job.get("prompts"), image) is None:
        raise ValueError(f"no prompt of {job_id} holds the image {image}")
    _automation_image_path(job_id, output_dir, image)
    pid = _spawn_automation_job(job_id, output_dir, image=image)
    updated = automation.update_job(
        job_id,
        output_dir,
        state=automation.STATE_RUNNING,
        pid=pid,
        error=None,
        finished_at=None,
    )
    return _automation_job_detail(updated, settings)


def _automation_append_count(params: dict[str, Any]) -> int:
    """How many images an append pass adds, refused outside the documented range."""
    count = params.get("count", 1)
    try:
        images = int(count)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"count must be a number, not {count!r}") from None
    if not automation.MIN_COUNT <= images <= automation.MAX_COUNT:
        raise ValueError(f"count must be {automation.MIN_COUNT}..{automation.MAX_COUNT}")
    return images


def handle_automation_prompt_extend(params: dict[str, Any]) -> dict[str, Any]:
    """Add `count` images to one prompt entry, each from its own new random seed."""
    job, settings, output_dir, job_id = _automation_idle_job(params)
    index = _automation_prompt_index(params)
    if automation.prompt_entry_at(job.get("prompts"), index) is None:
        raise ValueError(f"{job_id} has no prompt at index {index}")
    images = _automation_append_count(params)
    pid = _spawn_automation_job(job_id, output_dir, append=index, images=images)
    updated = automation.update_job(
        job_id,
        output_dir,
        state=automation.STATE_RUNNING,
        pid=pid,
        error=None,
        finished_at=None,
    )
    return _automation_job_detail(updated, settings)


def handle_automation_prompt_extend_all(params: dict[str, Any]) -> dict[str, Any]:
    """Add `count` images to every prompt entry, each from its own new random seed.

    The same pass the per-prompt action runs, for a set that wants more takes of everything:
    the runner walks the entries in order and appends to each one's own numbering.
    """
    job, settings, output_dir, job_id = _automation_idle_job(params)
    prompts = job.get("prompts")
    if not any(isinstance(entry, dict) for entry in prompts or []):
        raise ValueError(f"{job_id} has no prompt to extend")
    images = _automation_append_count(params)
    pid = _spawn_automation_job(job_id, output_dir, append_all=True, images=images)
    updated = automation.update_job(
        job_id,
        output_dir,
        state=automation.STATE_RUNNING,
        pid=pid,
        error=None,
        finished_at=None,
    )
    return _automation_job_detail(updated, settings)


def handle_automation_job_prompt_edit(params: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the text of one prompt entry.

    Only the record changes: the images already rendered keep their sidecars, which are the
    record of what was actually sent, and the next regeneration uses the new text. Editing is
    allowed while the job runs — the runner never writes `text` back, and a record update
    re-reads the file, so neither side can drop the other's fields.
    """
    settings = automation.load_settings()
    job = _automation_job(str(params.get("id") or ""), settings)
    job_id = str(job.get("id"))
    output_dir = str(job.get("output_dir") or settings["output_dir"])
    index = _automation_prompt_index(params)
    prompts = job.get("prompts")
    entry = automation.prompt_entry_at(prompts, index)
    if entry is None:
        raise ValueError(f"{job_id} has no prompt at index {index}")
    text = str(params.get("text") or "").strip()
    if not text:
        raise ValueError("text is empty")
    if len(text) > automation.MAX_PROMPT_CHARS:
        raise ValueError(f"text must be at most {automation.MAX_PROMPT_CHARS} characters")
    updated_prompts = list(prompts)
    updated_prompts[index] = {**entry, "text": text}
    updated = automation.update_job(job_id, output_dir, prompts=updated_prompts)
    return _automation_job_detail(updated, settings)


def handle_automation_job_delete(params: dict[str, Any]) -> dict[str, Any]:
    job_id = str(params.get("id") or "")
    settings = automation.load_settings()
    for job in automation.reconcile_jobs(settings["output_dir"]):
        if str(job.get("id")) != job_id:
            continue
        if job.get("state") == automation.STATE_RUNNING and is_pid_alive(job.get("pid")):
            raise ValueError(f"{job_id} is still running; cancel it first")
        directory = automation.job_dir(job_id, job.get("output_dir") or settings["output_dir"]).resolve()
        root = Path(job.get("output_dir") or settings["output_dir"]).expanduser().resolve()
        if directory.parent != root:
            raise ValueError(f"refusing to delete outside the job root: {directory}")
        shutil.rmtree(directory, ignore_errors=False)
        return {"id": job_id}
    raise ValueError(f"no job named {job_id}")


_HANDLERS = {
    "ping": handle_ping,
    "dashboard": handle_dashboard,
    "list_runs": handle_list_runs,
    "list_samples": handle_list_samples,
    "list_checkpoints": handle_list_checkpoints,
    "checkpoint_pins": handle_checkpoint_pins,
    "checkpoint_pin_set": handle_checkpoint_pin_set,
    "train_status": handle_train_status,
    "train_start": handle_train_start,
    "train_pause": handle_train_pause,
    "train_resume": handle_train_resume,
    "train_stop": handle_train_stop,
    "train_settings": handle_train_settings,
    "train_reset": handle_train_reset,
    "sample_prompts": handle_sample_prompts,
    "sample_prompts_set": handle_sample_prompts_set,
    "dataset_tag": handle_dataset_tag,
    "dataset_counts": handle_dataset_counts,
    "tagger_info": handle_tagger_info,
    "hardware_status": handle_hardware_status,
    "generate_sample": handle_generate_sample,
    "generate_checkpoint_samples": handle_generate_checkpoint_samples,
    "generate_checkpoint_samples_batch": handle_generate_checkpoint_samples_batch,
    "clear_checkpoint_samples": handle_clear_checkpoint_samples,
    "evaluate_checkpoint": handle_evaluate_checkpoint,
    "evaluation_prompts": handle_evaluation_prompts,
    "cancel_generation": handle_cancel_generation,
    "list_generated_samples": handle_list_generated_samples,
    "config_get": handle_config_get,
    "config_save": handle_config_save,
    "profile_list": handle_profile_list,
    "profile_get": handle_profile_get,
    "profile_save": handle_profile_save,
    "profile_delete": handle_profile_delete,
    "prompt_matrix": handle_prompt_matrix,
    "prompt_profile_list": handle_prompt_profile_list,
    "prompt_profile_get": handle_prompt_profile_get,
    "prompt_profile_save": handle_prompt_profile_save,
    "prompt_profile_delete": handle_prompt_profile_delete,
    "automation_config_get": handle_automation_config_get,
    "automation_config_save": handle_automation_config_save,
    "automation_discover": handle_automation_discover,
    "automation_workflow_list": handle_automation_workflow_list,
    "automation_workflow_validate": handle_automation_workflow_validate,
    "automation_workflow_save": handle_automation_workflow_save,
    "automation_workflow_delete": handle_automation_workflow_delete,
    "automation_prompt_list": handle_automation_prompt_list,
    "automation_prompt_get": handle_automation_prompt_get,
    "automation_prompt_save": handle_automation_prompt_save,
    "automation_prompt_delete": handle_automation_prompt_delete,
    "automation_job_start": handle_automation_job_start,
    "automation_job_list": handle_automation_job_list,
    "automation_job_get": handle_automation_job_get,
    "automation_job_cancel": handle_automation_job_cancel,
    "automation_job_retry_failed": handle_automation_job_retry_failed,
    "automation_job_delete": handle_automation_job_delete,
    "automation_image_delete": handle_automation_image_delete,
    "automation_image_regenerate": handle_automation_image_regenerate,
    "automation_prompt_extend": handle_automation_prompt_extend,
    "automation_prompt_extend_all": handle_automation_prompt_extend_all,
    "automation_job_prompt_edit": handle_automation_job_prompt_edit,
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

_OWNER_GRACE_SECONDS = 5.0
_MAX_SESSION_CONNECTIONS = 32

# --- one client at a time ---------------------------------------------------
#
# api.py does no locking of its own: it serves a single client session and leaves ordering to
# that client (`IpcResources` in Ranko). What it does enforce is that there *is* only one: the
# first client to say hello owns the helper, and any other instance is refused at once rather
# than served interleaved. An instance that never disconnects cleanly is replaced after
# `_OWNER_GRACE_SECONDS` without a live connection.


class ClientSession:
    """Who owns this helper, and how many connections they hold it with."""

    def __init__(
        self,
        grace: float = _OWNER_GRACE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.grace = grace
        self.clock = clock
        self._lock = threading.Lock()
        self.instance = ""
        self.client = ""
        self.since = 0.0
        self.connections = 0
        self.last_activity = 0.0

    def _live(self, now: float) -> bool:
        return self.connections > 0 or (now - self.last_activity) <= self.grace

    def hello(self, client: str, instance: str) -> tuple[bool, dict[str, Any]]:
        """Admit this connection, or say who has the helper.

        The same instance may open as many connections as it likes (one per lane); an unknown
        instance is refused while the owner is live, and takes over silently once it is not. A
        connection that identifies itself as nothing is the anonymous client, and shares that
        session with other unnamed ones - scripts and probes keep working exactly as before.
        """
        client = str(client or "").strip()[:80]
        instance = str(instance or "").strip()[:80]
        with self._lock:
            now = self.clock()
            if instance != self.instance:
                if self._live(now):
                    return False, self._refusal(now)
                self.instance = instance
                self.client = client
                self.since = now
                self.connections = 0
            if self.connections >= _MAX_SESSION_CONNECTIONS:
                return False, {
                    "code": "CLIENT_BUSY",
                    "error": f"{self.client or 'this client'} already holds {self.connections} connections",
                }
            self.connections += 1
            self.last_activity = now
            return True, {
                "owner": True,
                "client": self.client,
                "instance": self.instance,
                "since": self.since,
                "connections": self.connections,
            }

    def released(self) -> None:
        """One of the owner's connections went away."""
        with self._lock:
            self.connections = max(0, self.connections - 1)
            self.last_activity = self.clock()

    def info(self) -> dict[str, Any]:
        """What `hello` answers, for tests and for a client that asks again on a live connection."""
        with self._lock:
            now = self.clock()
            return {
                "owner": True,
                "client": self.client,
                "instance": self.instance,
                "since": self.since,
                "connections": self.connections,
                "live": self._live(now),
            }

    def _refusal(self, now: float) -> dict[str, Any]:
        held = _format_clock(self.since) if self.since else "?"
        who = self.client or self.instance or "an unnamed client"
        return {
            "code": "CLIENT_BUSY",
            "error": f"{who} owns the training helper (since {held}); close it and retry",
            "holder": {
                "client": self.client,
                "instance": self.instance,
                "since": self.since,
                "connections": self.connections,
            },
        }


def _format_clock(epoch: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(epoch))


_SESSION = ClientSession()
_WS_MAX_SIZE = 32 * 1024 * 1024
_WS_DEFAULT_PORT = 18765
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}


def dispatch(method: str, params: Optional[dict[str, Any]] = None) -> Any:
    """Serve one request.

    No locking, on purpose: the helper serves a single client session and that client (`IpcResources`
    in Ranko) is what keeps two calls that fight over the same thing off the wire at once. Two
    clients are prevented a level up, in the connection handler (`ClientSession`).
    """
    handler = _HANDLERS.get(method)
    if handler is None:
        raise ValueError(f"unknown method: {method}")
    return handler(params or {})


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


def _ws_handler(
    connection: Any,
    *,
    networks: list[ipaddress._BaseNetwork],
    session: Optional[ClientSession] = None,
) -> None:
    """Serve one connection.

    A connection is admitted by its first request: `hello {client, instance, lane}` says who it is,
    and a request that arrives without one claims a *free* session (the path a script or an older
    client takes). Both open the door; a second client finds it locked (`CLIENT_BUSY`, with who owns
    the helper and since when) and is closed out, because the helper does no locking of its own and
    one session is what makes that safe.
    """
    from websockets.exceptions import ConnectionClosed

    active = session if session is not None else _SESSION
    admitted = False
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
                method = req.get("method")
                if method == "hello":
                    if admitted:
                        payload = active.info()
                        ok = True
                    else:
                        params = req.get("params") or {}
                        if not isinstance(params, dict):
                            raise ValueError("params must be an object")
                        ok, payload = active.hello(
                            str(params.get("client") or ""),
                            str(params.get("instance") or ""),
                        )
                    if not ok:
                        print(f"api.py refused a second client: {payload['error']}", file=sys.stderr)
                        connection.send(
                            json.dumps(
                                {"id": req_id, "ok": False, **payload},
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                        )
                        return
                    admitted = True
                    connection.send(
                        json.dumps(
                            {"id": req_id, "ok": True, "result": _json_safe(payload)},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    )
                    continue
                if not admitted:
                    ok, payload = active.hello("", "")
                    if not ok:
                        print(f"api.py refused a second client: {payload['error']}", file=sys.stderr)
                        connection.send(
                            json.dumps(
                                {"id": req_id, "ok": False, **payload},
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                        )
                        return
                    admitted = True
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
    finally:
        if admitted:
            active.released()


def run_ws_loop(host: str, port: int, allow_networks: Optional[list[ipaddress._BaseNetwork]] = None) -> None:
    # Keep stdout unused for JSON: Ranko talks over the socket. Logs go to stderr.
    sys.stdout = sys.stderr
    _quiet_websockets_log()
    networks = list(allow_networks or [])
    from websockets.sync.server import serve

    handler = functools.partial(_ws_handler, networks=networks)
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

