from __future__ import annotations

import atexit
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

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
    for key in ("encoding", "training", "sampling"):
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
                if key in ("encoding", "training", "sampling") and isinstance(value, dict):
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
    if pid is None:
        return False
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_i <= 0:
        return False
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


def begin_run(pid: int, output_name: str) -> None:
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
            "error": None,
            "detail": None,
            "swap": None,
        },
        force=True,
    )
    if not _atexit_registered:
        atexit.register(_on_exit)
        _atexit_registered = True


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
