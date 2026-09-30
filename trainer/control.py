from __future__ import annotations

import atexit
import fcntl
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from orphans import PROC, is_running
except ImportError:
    from trainer.orphans import PROC, is_running

SCHEMA = 1

STATUS_IDLE = "idle"
STATUS_STARTING = "starting"
STATUS_ENCODING = "encoding"
STATUS_TRAINING = "training"
STATUS_SAMPLING = "sampling"
STATUS_PAUSING = "pausing"
STATUS_PAUSED = "paused"
STATUS_RESUMING = "resuming"
STATUS_STOPPING = "stopping"
STATUS_FINISHED = "finished"
STATUS_ERROR = "error"

LIVE_STATUSES = frozenset(
    {
        STATUS_STARTING,
        STATUS_ENCODING,
        STATUS_TRAINING,
        STATUS_SAMPLING,
        STATUS_PAUSING,
        STATUS_PAUSED,
        STATUS_RESUMING,
        STATUS_STOPPING,
    }
)

PHASE_STATUSES = frozenset({STATUS_ENCODING, STATUS_TRAINING, STATUS_SAMPLING})

_WRITE_INTERVAL = 0.2

_state: dict[str, Any] = {}
_last_write_mono = 0.0
_last_cmd_seq = 0
_lock_fd: Any = None
_atexit_registered = False
_ended = False
_io_lock = threading.RLock()


def runtime_dir() -> Path:
    override = os.environ.get("AXL_RUNTIME_DIR")
    if override:
        path = Path(override)
    else:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg:
            path = Path(xdg) / "axltrainer"
        else:
            path = Path(f"/tmp/axltrainer-{os.getuid()}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_path() -> Path:
    return runtime_dir() / "state.json"


def command_path() -> Path:
    return runtime_dir() / "command.json"


def settings_path() -> Path:
    """The trainer's live cadence / sampling switch, written by api.py and read every step."""
    return runtime_dir() / "settings.json"


def lock_path() -> Path:
    return runtime_dir() / "train.lock"


def log_path() -> Path:
    return runtime_dir() / "train.log"


def default_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "pid": None,
        "started_at": None,
        "updated_at": time.time(),
        "status": STATUS_IDLE,
        "paused_from": None,
        "output_name": None,
        "run_id": None,
        "resume": None,
        "encoding": {"current": 0, "total": 0, "done": False},
        "training": {
            "step": 0,
            "total_steps": 0,
            "epoch": 0,
            "epochs": 0,
            "loss": None,
            "avg_loss": None,
        },
        "sampling": {
            "active": False,
            "repeat": 0,
            "repeats": 0,
            "denoise_step": 0,
            "denoise_steps": 0,
            "global_step": 0,
        },
        "swap": None,
        "settings": {"save_every_n_steps": 0, "sampling_enabled": True, "next_save_step": 0},
        "error": None,
        "detail": None,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _ensure_state() -> dict[str, Any]:
    global _state
    if not _state:
        _state = default_state()
    return _state


def read_state() -> dict[str, Any]:
    path = state_path()
    if not path.is_file():
        return default_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_state()
    if not isinstance(raw, dict):
        return default_state()
    merged = default_state()
    merged.update(raw)
    for key in ("encoding", "training", "sampling", "settings"):
        base = default_state()[key]
        value = raw.get(key)
        if isinstance(value, dict):
            base.update(value)
        merged[key] = base
    return merged


def write_state(updates: Optional[dict[str, Any]] = None, *, force: bool = False) -> dict[str, Any]:
    global _last_write_mono
    with _io_lock:
        state = _ensure_state()
        if updates:
            for key, value in updates.items():
                if key in ("encoding", "training", "sampling", "settings") and isinstance(value, dict):
                    current = state.get(key)
                    if not isinstance(current, dict):
                        current = default_state()[key]
                    current = dict(current)
                    current.update(value)
                    state[key] = current
                else:
                    state[key] = value
        now = time.time()
        state["updated_at"] = now
        state["schema"] = SCHEMA
        mono = time.monotonic()
        if not force and (mono - _last_write_mono) < _WRITE_INTERVAL:
            return dict(state)
        _atomic_write(state_path(), dict(state))
        _last_write_mono = mono
        return dict(state)


def is_pid_alive(pid: Optional[int]) -> bool:
    """True while the process is running.

    `os.kill(pid, 0)` alone calls a *zombie* alive: a child nobody has waited on keeps its `/proc`
    entry, answers signals, and holds nothing. `api.py` never `wait()`s the trainer — or a generator
    — it spawns, so a finished run's trainer sits in exactly that state, and the dashboard (which
    reads `alive` from `state.json`) would keep believing training is still on the GPU until api.py
    itself exits, which only happens when Ranko closes: the checkpoint panel's "Generate sample"
    stayed disabled in a session where the run had already finished. The process state is the
    arbiter (`orphans.is_running`: `Z` means gone), with the signal check as the fallback for a
    machine without `/proc`.
    """
    if pid is None:
        return False
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_i <= 0:
        return False
    if PROC.is_dir():
        return is_running(pid_i)
    try:
        os.kill(pid_i, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def reconcile(state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    current = dict(state or read_state())
    pid = current.get("pid")
    alive = is_pid_alive(pid)
    status = current.get("status") or STATUS_IDLE
    if status in LIVE_STATUSES and not alive:
        if status == STATUS_STARTING:
            try:
                age = time.time() - float(current.get("updated_at") or 0)
            except (TypeError, ValueError):
                age = 1e9
            if age < 20:
                return current
        current["status"] = STATUS_ERROR
        current["error"] = current.get("error") or "training process is no longer running"
        current["swap"] = None
        current["sampling"] = {**default_state()["sampling"], **(current.get("sampling") or {})}
        current["sampling"]["active"] = False
        _atomic_write(state_path(), current)
        global _state
        _state = current
    return current


def status_payload() -> dict[str, Any]:
    current = reconcile()
    current["alive"] = is_pid_alive(current.get("pid"))
    current["log_path"] = str(log_path())
    return current


def mark_starting(pid: int, output_name: str) -> dict[str, Any]:
    global _state, _ended
    _ended = False
    _state = default_state()
    return write_state(
        {
            "pid": int(pid),
            "started_at": time.time(),
            "status": STATUS_STARTING,
            "output_name": output_name,
            "error": None,
            "detail": None,
            "swap": None,
        },
        force=True,
    )


def _on_exit() -> None:
    if _ended:
        return
    state = _ensure_state()
    if state.get("status") in LIVE_STATUSES:
        end_run(STATUS_ERROR, error="process exited")


def begin_run(pid: int, output_name: str, run_id: Optional[str] = None) -> None:
    global _atexit_registered, _ended, _last_cmd_seq, _state
    if not try_acquire_lock():
        raise RuntimeError("another training run holds the lock")
    _ended = False
    _last_cmd_seq = 0
    existing = read_state()
    started_at = existing.get("started_at") if existing.get("pid") == pid else None
    _state = default_state()
    write_state(
        {
            "pid": int(pid),
            "started_at": started_at or time.time(),
            "status": STATUS_STARTING,
            "output_name": output_name,
            "run_id": run_id,
            "error": None,
            "detail": None,
            "swap": None,
        },
        force=True,
    )
    if not _atexit_registered:
        atexit.register(_on_exit)
        _atexit_registered = True


def set_resume(info: Optional[dict[str, Any]]) -> None:
    """Publish which checkpoint this run was seeded from (None = fresh run)."""
    write_state({"resume": info or None}, force=True)



def reset_to_idle() -> dict[str, Any]:
    global _state, _last_cmd_seq, _ended
    release_lock()
    _ended = True
    _last_cmd_seq = 0
    _state = default_state()
    command = command_path()
    if command.exists():
        try:
            command.unlink()
        except OSError:
            pass
    # A request left over from the finished run must not shape the next one; `train_start`
    # writes the new run's own values.
    clear_settings()
    return write_state(_state, force=True)


def end_run(status: str = STATUS_FINISHED, error: Optional[str] = None, detail: Optional[str] = None) -> None:
    global _ended
    _ended = True
    state = _ensure_state()
    sampling = dict(state.get("sampling") or default_state()["sampling"])
    sampling["active"] = False
    write_state(
        {
            "status": status,
            "error": error,
            "detail": detail,
            "swap": None,
            "paused_from": None,
            "sampling": sampling,
        },
        force=True,
    )
    release_lock()


def set_status(status: str, *, force: bool = True, **updates: Any) -> dict[str, Any]:
    payload = {"status": status}
    payload.update(updates)
    return write_state(payload, force=force)


def set_encoding(*, current: int, total: int, done: bool = False) -> None:
    write_state(
        {
            "status": STATUS_ENCODING,
            "encoding": {"current": int(current), "total": int(total), "done": bool(done)},
        },
        force=done,
    )


def set_training(
    *,
    step: int,
    total_steps: int,
    epoch: int,
    epochs: int,
    loss: Optional[float] = None,
    avg_loss: Optional[float] = None,
) -> None:
    write_state(
        {
            "status": STATUS_TRAINING,
            "training": {
                "step": int(step),
                "total_steps": int(total_steps),
                "epoch": int(epoch),
                "epochs": int(epochs),
                "loss": loss,
                "avg_loss": avg_loss,
            },
            "encoding": {**(_ensure_state().get("encoding") or {}), "done": True},
        }
    )


def set_sampling(
    *,
    active: bool,
    repeat: int = 0,
    repeats: int = 0,
    denoise_step: int = 0,
    denoise_steps: int = 0,
    global_step: int = 0,
    prompt_set: int = 0,
    prompt_sets: int = 0,
) -> None:
    current_status = _ensure_state().get("status")
    if active:
        status = STATUS_SAMPLING
    elif current_status in (STATUS_STOPPING, STATUS_PAUSED, STATUS_PAUSING, STATUS_RESUMING):
        status = current_status
    else:
        status = STATUS_TRAINING
    write_state(
        {
            "status": status,
            "sampling": {
                "active": bool(active),
                "repeat": int(repeat),
                "repeats": int(repeats),
                "denoise_step": int(denoise_step),
                "denoise_steps": int(denoise_steps),
                "global_step": int(global_step),
                "prompt_set": int(prompt_set),
                "prompt_sets": int(prompt_sets),
            },
        },
        force=not active,
    )


def set_swap(stage: str, detail: str, current: int, total: int) -> None:
    write_state(
        {
            "swap": {
                "stage": stage,
                "detail": detail,
                "current": int(current),
                "total": int(total),
            }
        },
        force=True,
    )


def clear_swap() -> None:
    write_state({"swap": None}, force=True)


@dataclass
class LiveSettings:
    """The checkpoint cadence and sampling switch a run is running with.

    `next_save_step` is the global step that writes the next checkpoint, not a modulo of the
    cadence: adopting a new cadence at step S makes the next checkpoint `S + N`, so "every N
    steps" always means "N steps from the change" and a changed cadence is never silently
    skipped. Untouched, the sequence stays N, 2N, 3N…, i.e. what the modulo rule produced.
    """

    save_every_n_steps: int = 0
    sampling_enabled: bool = True
    next_save_step: int = 0

    @classmethod
    def from_config(cls, cfg: Any) -> "LiveSettings":
        steps = max(0, int(getattr(cfg, "save_every_n_steps", 0) or 0))
        return cls(
            save_every_n_steps=steps,
            sampling_enabled=bool(getattr(cfg, "sampling_enabled", True)),
            next_save_step=steps,
        )

    def adopt(self, requested: Optional[dict[str, Any]], global_step: int) -> "LiveSettings":
        """`self`, or the settings `requested` asks for when they differ (None = nothing asked).

        Only a cadence change moves the schedule: flipping the sampling switch leaves the step
        of the next checkpoint where it was.
        """
        if not isinstance(requested, dict):
            return self
        try:
            steps = max(0, int(requested.get("save_every_n_steps", self.save_every_n_steps)))
        except (TypeError, ValueError):
            steps = self.save_every_n_steps
        enabled = bool(requested.get("sampling_enabled", self.sampling_enabled))
        if steps == self.save_every_n_steps and enabled == self.sampling_enabled:
            return self
        if steps == self.save_every_n_steps:
            next_save_step = self.next_save_step
        else:
            next_save_step = (global_step + steps) if steps > 0 else 0
        return LiveSettings(
            save_every_n_steps=steps,
            sampling_enabled=enabled,
            next_save_step=next_save_step,
        )

    def due(self, global_step: int) -> bool:
        return self.save_every_n_steps > 0 and global_step >= self.next_save_step

    def mark_saved(self, global_step: int) -> None:
        self.next_save_step = (
            global_step + self.save_every_n_steps if self.save_every_n_steps > 0 else 0
        )


def _read_settings_file() -> Optional[dict[str, Any]]:
    path = settings_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def read_settings() -> Optional[dict[str, Any]]:
    """The requested cadence / switch, which the trainer adopts at its next optimizer step.

    None when nothing has been asked for (no file, or an unreadable one): the run then keeps
    whatever `config.toml` (and the last adopted change) gave it.
    """
    raw = _read_settings_file()
    if raw is None:
        return None
    try:
        steps = max(0, int(raw.get("save_every_n_steps")))
    except (TypeError, ValueError):
        return None
    return {"save_every_n_steps": steps, "sampling_enabled": bool(raw.get("sampling_enabled", True))}


def request_settings(
    *,
    save_every_n_steps: Optional[int] = None,
    sampling_enabled: Optional[bool] = None,
    baseline: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Record what the trainer should run with (api.py side). Unset fields keep their value.

    "Their value" is the file's, which api.py keeps complete: a field the request does not name is
    read back from `settings.json`. A field *the file* does not carry — no file at all, because the
    run was started by hand with `bash start_train.sh` or the runtime dir was wiped — falls back to
    `baseline`, the settings the run is actually using. Without it a lone switch flip on such a run
    would also write the default cadence `0`, i.e. silently turn checkpoints off.
    """
    payload: dict[str, Any] = {}
    current = _read_settings_file()
    if isinstance(current, dict):
        payload.update(current)
    fallback = baseline if isinstance(baseline, dict) else {}
    if "save_every_n_steps" not in payload and fallback.get("save_every_n_steps") is not None:
        try:
            payload["save_every_n_steps"] = max(0, int(fallback["save_every_n_steps"]))
        except (TypeError, ValueError):
            pass
    if "sampling_enabled" not in payload and fallback.get("sampling_enabled") is not None:
        payload["sampling_enabled"] = bool(fallback["sampling_enabled"])
    if save_every_n_steps is not None:
        payload["save_every_n_steps"] = max(0, int(save_every_n_steps))
    if sampling_enabled is not None:
        payload["sampling_enabled"] = bool(sampling_enabled)
    payload.setdefault("save_every_n_steps", 0)
    payload.setdefault("sampling_enabled", True)
    _atomic_write(settings_path(), payload)
    return payload


def publish_settings(settings: "LiveSettings") -> None:
    """Publish the settings the trainer is actually running with (the dashboard reads these)."""
    write_state(
        {
            "settings": {
                "save_every_n_steps": int(settings.save_every_n_steps),
                "sampling_enabled": bool(settings.sampling_enabled),
                "next_save_step": int(settings.next_save_step),
            }
        },
        force=True,
    )


def clear_settings() -> None:
    try:
        settings_path().unlink()
    except OSError:
        pass


def _read_command_file() -> Optional[dict[str, Any]]:
    path = command_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def peek_command() -> Optional[str]:
    raw = _read_command_file()
    if raw is None:
        return None
    try:
        seq = int(raw.get("seq") or 0)
    except (TypeError, ValueError):
        return None
    if seq <= _last_cmd_seq:
        return None
    op = raw.get("op")
    if op in ("pause", "resume", "stop"):
        return str(op)
    return None


def poll_command() -> Optional[str]:
    global _last_cmd_seq
    raw = _read_command_file()
    if raw is None:
        return None
    try:
        seq = int(raw.get("seq") or 0)
    except (TypeError, ValueError):
        return None
    if seq <= _last_cmd_seq:
        return None
    op = raw.get("op")
    if op not in ("pause", "resume", "stop"):
        return None
    _last_cmd_seq = seq
    return str(op)


def request(op: str) -> dict[str, Any]:
    if op not in ("pause", "resume", "stop"):
        raise ValueError(f"unknown train command: {op}")
    raw = _read_command_file() or {}
    try:
        seq = int(raw.get("seq") or 0) + 1
    except (TypeError, ValueError):
        seq = 1
    payload = {"seq": seq, "op": op}
    _atomic_write(command_path(), payload)
    return payload


def try_acquire_lock() -> bool:
    global _lock_fd
    if _lock_fd is not None:
        return True
    fd = open(lock_path(), "a+", encoding="utf-8")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        return False
    fd.seek(0)
    fd.truncate()
    fd.write(str(os.getpid()))
    fd.flush()
    _lock_fd = fd
    return True


def release_lock() -> None:
    global _lock_fd
    fd = _lock_fd
    _lock_fd = None
    if fd is None:
        return
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        fd.close()
    except OSError:
        pass


def should_stop() -> bool:
    return _ensure_state().get("status") == STATUS_STOPPING
