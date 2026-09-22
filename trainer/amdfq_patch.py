"""Resolve `[environment].amdfq` for the training launcher.

`none` / `tail` / `vmm`. The trainer does not import this during the loop; `start_train.sh`
and `api.py` `train_start` do, so a missing `.so` fails before GPU work. torch-free.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any, Optional

CHOICES = ("none", "tail", "vmm")

_SO_RELATIVE = {
    "tail": "amdfq/amdfq-tail-rs/target/release/libamdfq_tail_rs.so",
    "vmm": "amdfq/amdfq-vmm-rs/target/release/libamdfq_vmm_rs.so",
}

GIB = 1 << 30
DEFAULT_VRAM_RESERVE_GIB = 0.0


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def normalize_amdfq(raw: Any) -> str:
    choice = str(raw or "none").strip().lower()
    if choice not in CHOICES:
        allowed = ", ".join(CHOICES)
        raise ValueError(f"amdfq must be one of {allowed}, not {raw!r}")
    return choice


def normalize_vram_reserve_gib(raw: Any) -> float:
    if raw is None or raw == "":
        return DEFAULT_VRAM_RESERVE_GIB
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"amdfq_vram_reserve_gib must be a number, not {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"amdfq_vram_reserve_gib must be >= 0, not {raw!r}")
    return value


def vram_reserve_bytes(gib: float) -> int:
    return int(gib * GIB)


def _read_flat(config_path: str | Path) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    try:
        with open(config_path, "rb") as handle:
            raw = tomllib.load(handle)
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
    except FileNotFoundError:
        pass
    return flat


def read_amdfq(config_path: str | Path | None = None) -> str:
    path = Path(config_path) if config_path is not None else repo_root() / "config.toml"
    return normalize_amdfq(_read_flat(path).get("amdfq", "none"))


def read_vram_reserve_gib(config_path: str | Path | None = None) -> float:
    path = Path(config_path) if config_path is not None else repo_root() / "config.toml"
    return normalize_vram_reserve_gib(_read_flat(path).get("amdfq_vram_reserve_gib"))


def normalize_va_never_reuse(raw: Any) -> bool:
    if raw is None or raw == "":
        return False
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError(f"amdfq_va_never_reuse must be true or false, not {raw!r}")


def read_va_never_reuse(config_path: str | Path | None = None) -> bool:
    path = Path(config_path) if config_path is not None else repo_root() / "config.toml"
    return normalize_va_never_reuse(_read_flat(path).get("amdfq_va_never_reuse"))


def resolve_preload(root: str | Path | None = None) -> dict[str, Optional[str]]:
    """Return `{choice, so, va_status, vram_reserve, va_never_reuse}`. `so` is set for tail/vmm; a
    missing file raises."""
    base = Path(root) if root is not None else repo_root()
    config_path = base / "config.toml"
    choice = read_amdfq(config_path)
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
    reserve = str(vram_reserve_bytes(read_vram_reserve_gib(config_path)))
    never_reuse = "1" if read_va_never_reuse(config_path) else "0"
    return {
        "choice": choice,
        "so": so,
        "va_status": va_status,
        "vram_reserve": reserve,
        "va_never_reuse": never_reuse,
    }


def launch_env_line(root: str | Path | None = None) -> str:
    """`choice|so|va_status|vram_reserve|va_never_reuse` for `start_train.sh` (empty fields stay
    empty)."""
    info = resolve_preload(root)
    return (
        f"{info['choice']}|{info['so'] or ''}|{info['va_status'] or ''}|{info['vram_reserve'] or ''}"
        f"|{info['va_never_reuse'] or ''}"
    )
