"""LoRA checkpoint discovery and metadata for resume + the Ranko picker."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Union

from safetensors import safe_open

try:
    from runs import run_id_re, safe_name
except ImportError:
    from trainer.runs import run_id_re, safe_name

_STEP_DIR_RE = re.compile(r"_s(\d{6})$")
_EPOCH_DIR_RE = re.compile(r"_e(\d{3})_s(\d{6})$")

# (absolute path, mtime_ns, size) -> metadata. Bounded; cleared wholesale when it fills, because a
# run's own checkpoints are what gets read over and over.
_METADATA_CACHE: dict[tuple[str, int, int], dict[str, str]] = {}
_METADATA_CACHE_LIMIT = 512


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def resolve_resume_path(raw: Union[str, Path]) -> Path:
    """Accept a `.safetensors` file or a checkpoint directory holding exactly one."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("resume_lora_path is empty")
    path = Path(text).expanduser()
    if path.is_dir():
        found = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() == ".safetensors"
        )
        if not found:
            raise ValueError(f"no .safetensors file in resume directory: {path}")
        if len(found) > 1:
            names = ", ".join(item.name for item in found[:4])
            raise ValueError(
                f"resume directory holds {len(found)} .safetensors files ({names}); "
                f"point resume_lora_path at the file: {path}"
            )
        return found[0]
    if not path.is_file():
        raise ValueError(f"resume checkpoint not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"resume checkpoint must be a .safetensors file: {path}")
    return path


def read_lora_metadata(path: Union[str, Path]) -> dict[str, str]:
    target = Path(path)
    key: Optional[tuple[str, int, int]] = None
    try:
        stat = target.stat()
        key = (str(target), int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        key = None
    if key is not None:
        cached = _METADATA_CACHE.get(key)
        if cached is not None:
            return dict(cached)
    try:
        with safe_open(str(target), framework="pt") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:  # noqa: BLE001 - surfaced as a config error
        raise ValueError(f"failed to read safetensors metadata from {target}: {exc}") from exc
    parsed = {str(key_name): str(value) for key_name, value in metadata.items()}
    if key is not None:
        # The dashboard lists a run's checkpoints with every poll, and each header read is a file
        # open: keyed by (path, mtime, size), a checkpoint written once is read once.
        if len(_METADATA_CACHE) >= _METADATA_CACHE_LIMIT:
            _METADATA_CACHE.clear()
        _METADATA_CACHE[key] = parsed
    return dict(parsed)


def parse_network_args(raw: Any) -> dict[str, str]:
    """`ss_network_args`: kohya JSON object, or `conv_dim=N conv_alpha=M`."""
    text = str(raw or "").strip()
    if not text:
        return {}
    if text[:1] in "{[":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
    out: dict[str, str] = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def infer_network_type(metadata: Optional[dict[str, Any]]) -> str:
    """Resume discriminator. Missing `ss_network_type` is standard unless conv_dim > 0."""
    data = metadata or {}
    raw = str(data.get("ss_network_type") or "").strip().lower()
    if raw:
        return raw
    conv_dim = _int_or_none(parse_network_args(data.get("ss_network_args")).get("conv_dim"))
    if conv_dim is not None and conv_dim > 0:
        return "locon"
    return "standard"


def conv_dim_alpha_from_metadata(metadata: Optional[dict[str, Any]]) -> tuple[int, int]:
    args = parse_network_args((metadata or {}).get("ss_network_args"))
    return _int_or_none(args.get("conv_dim")) or 0, _int_or_none(args.get("conv_alpha")) or 0


def require_resume_network_type(cfg: Any, metadata: dict[str, Any], source: Union[str, Path]) -> None:
    file_type = infer_network_type(metadata)
    cfg_type = str(getattr(cfg, "network_type", "standard") or "standard").strip().lower()
    if file_type != cfg_type:
        raise ValueError(
            f"checkpoint {source} is a {file_type} LoRA but this config is {cfg_type}; "
            f"set network_type to {file_type!r} to resume it"
        )


def parse_checkpoint_dir(dir_name: str, output_name: str) -> dict[str, Any]:
    """Infer step/epoch/final from an artifact directory name."""
    names = [name for name in (str(output_name or ""), safe_name(output_name)) if name]
    rest = dir_name
    for name in names:
        if dir_name.startswith(name):
            rest = dir_name[len(name):]
            break
    if rest == "_final":
        return {"final": True, "step": None, "epoch": None}
    match = _EPOCH_DIR_RE.search(rest)
    if match:
        return {"final": False, "step": int(match.group(2)), "epoch": int(match.group(1))}
    match = _STEP_DIR_RE.search(rest)
    if match:
        return {"final": False, "step": int(match.group(1)), "epoch": None}
    return {"final": False, "step": None, "epoch": None}


def discover_checkpoints(
    output_dir: Union[str, Path],
    output_name: str,
) -> list[dict[str, Any]]:
    """LoRA files inside this output_name's run directories, newest step first."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    pattern = run_id_re(output_name)
    samples_name = f"{output_name}_samples"

    items: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not pattern.match(run_dir.name):
            continue
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir() or child.name == samples_name:
                continue
            files = sorted(child.glob("*.safetensors"))
            if not files:
                continue
            parsed = parse_checkpoint_dir(child.name, output_name)
            target = files[0]
            try:
                metadata = read_lora_metadata(target)
            except ValueError:
                metadata = {}
            if parsed["step"] is None:
                parsed["step"] = _int_or_none(metadata.get("ss_steps"))
            if parsed["epoch"] is None:
                parsed["epoch"] = _int_or_none(metadata.get("ss_epoch"))
            try:
                stat = target.stat()
            except OSError:
                continue
            items.append(
                {
                    "path": str(target.resolve()),
                    "run_id": run_dir.name,
                    "dir": child.name,
                    "filename": target.name,
                    "step": parsed["step"],
                    "epoch": parsed["epoch"],
                    "final": bool(parsed["final"]),
                    "size_bytes": int(stat.st_size),
                    "modified": float(stat.st_mtime),
                    "network_dim": _int_or_none(metadata.get("ss_network_dim")),
                    "network_alpha": _int_or_none(metadata.get("ss_network_alpha")),
                    "output_name": metadata.get("ss_output_name") or output_name,
                }
            )

    items.sort(
        key=lambda item: (
            item["step"] if item["step"] is not None else -1,
            item["modified"],
        ),
        reverse=True,
    )
    return items
