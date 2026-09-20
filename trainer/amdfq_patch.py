"""Resolve `[environment].amdfq` for the training launcher.

`none` / `tail` / `vmm`. The trainer does not import this during the loop; `start_train.sh`
and `api.py` `train_start` do, so a missing `.so` fails before GPU work. torch-free.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Optional

CHOICES = ("none", "tail", "vmm")

_SO_RELATIVE = {
    "tail": "amdfq/amdfq-tail-rs/target/release/libamdfq_tail_rs.so",
    "vmm": "amdfq/amdfq-vmm-rs/target/release/libamdfq_vmm_rs.so",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def normalize_amdfq(raw: Any) -> str:
    choice = str(raw or "none").strip().lower()
    if choice not in CHOICES:
        allowed = ", ".join(CHOICES)
        raise ValueError(f"amdfq must be one of {allowed}, not {raw!r}")
    return choice


def read_amdfq(config_path: str | Path | None = None) -> str:
    path = Path(config_path) if config_path is not None else repo_root() / "config.toml"
    flat: dict[str, Any] = {}
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
    except FileNotFoundError:
        pass
    return normalize_amdfq(flat.get("amdfq", "none"))


def resolve_preload(root: str | Path | None = None) -> dict[str, Optional[str]]:
    """Return `{choice, so, va_status}`. `so` is set for tail/vmm; missing file raises."""
    base = Path(root) if root is not None else repo_root()
    choice = read_amdfq(base / "config.toml")
    so: Optional[str] = None
    va_status: Optional[str] = None
    relative = _SO_RELATIVE.get(choice)
    if relative is not None:
        so_path = base / relative
        if not so_path.is_file():
            manifest = so_path.parent.parent / "Cargo.toml"
            raise FileNotFoundError(
                f"{choice} patch library missing: {so_path}. "
                f"Build it with: cargo build --release --manifest-path {manifest}"
            )
        so = str(so_path)
    if choice == "vmm":
        try:
            from trainer.control import runtime_dir
        except ImportError:
            from control import runtime_dir
        va_status = str(runtime_dir() / "amdfq_vmm_va")
    return {"choice": choice, "so": so, "va_status": va_status}


def launch_env_line(root: str | Path | None = None) -> str:
    """`choice|so|va_status` for `start_train.sh` (empty fields stay empty)."""
    info = resolve_preload(root)
    return f"{info['choice']}|{info['so'] or ''}|{info['va_status'] or ''}"
