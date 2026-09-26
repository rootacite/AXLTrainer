"""Filesystem operations Ranko used to do itself. Torch-free.

Handlers here are called from `api.dispatch`. Paths are allowlisted against
`config.toml` (train-data folders, repo `config.toml` / `configs/`, `output_dir`).
"""

from __future__ import annotations

import os
import random
import shutil
import tomllib
import uuid
from pathlib import Path
from typing import Any, Optional

from PIL import Image

try:
    from config import resolve_train_data_entries
except ImportError:
    from trainer.config import resolve_train_data_entries

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_SIDECAR_SUFFIX = ".mask.png"
TRASH_DEFAULT = Path("/tmp/axlranko/trash")
PROFILE_DIR_NAME = "configs"
PROFILE_EXT = ".toml"
PROFILE_MAX_NAME = 64
PROFILE_ILLEGAL = set('/\\:*?"<>|')
SHUFFLE_TEMP_PREFIX = "axl-shuffle-"
SHUFFLE_STEM_PAD = 4
SHUFFLE_FIRST = "0001"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    return _repo_root() / "config.toml"


def profile_dir() -> Path:
    return _repo_root() / PROFILE_DIR_NAME


def trash_dir() -> Path:
    raw = os.environ.get("AXL_TRASH_DIR")
    if raw:
        return Path(raw)
    return TRASH_DEFAULT


def is_mask_sidecar(name: str) -> bool:
    return name.lower().endswith(MASK_SIDECAR_SUFFIX)


def mask_path_for(image: Path) -> Path:
    return image.with_name(image.stem + MASK_SIDECAR_SUFFIX)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def train_data_roots(cfg: dict[str, Any]) -> list[Path]:
    entries = resolve_train_data_entries(cfg)
    return [Path(entry.path).expanduser().resolve() for entry in entries]


def output_root(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("output_dir") or "./output")).expanduser().resolve()


def require_dataset_dir(directory: Any, cfg: dict[str, Any]) -> Path:
    if not directory:
        raise ValueError("missing directory")
    path = Path(str(directory)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"not a directory: {path}")
    roots = train_data_roots(cfg)
    if path not in roots:
        raise ValueError(f"not a configured train_data directory: {path}")
    return path


def require_stem(stem: Any) -> str:
    value = str(stem or "").strip()
    if not value:
        raise ValueError("missing stem")
    if "/" in value or "\\" in value or value in {".", ".."} or ".." in value:
        raise ValueError(f"invalid stem: {stem}")
    return value


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def blob_path_allowed(path: Path, cfg: dict[str, Any]) -> bool:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return False
    if is_mask_sidecar(resolved.name):
        return False
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    if resolved.parent in train_data_roots(cfg):
        return True
    return _is_under(resolved, output_root(cfg))


def checkpoint_source_allowed(path: Path, cfg: dict[str, Any]) -> bool:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return False
    if resolved.suffix.lower() != ".safetensors":
        return False
    return _is_under(resolved, output_root(cfg))


def _image_files(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file()]
    images = [
        p
        for p in files
        if p.suffix.lower() in IMAGE_EXTENSIONS and not is_mask_sidecar(p.name)
    ]
    return sorted(images, key=lambda p: p.name.lower())


def _caption_files(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.name.lower(),
    )


def _probe_image(path: Path) -> tuple[int, int, bool]:
    with Image.open(path) as img:
        width, height = img.size
        has_alpha = img.mode in {"RGBA", "LA", "PA"} or (
            img.mode == "P" and "transparency" in img.info
        )
        return int(width), int(height), bool(has_alpha)


def _parse_tags(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def dataset_list(directory: Path) -> dict[str, Any]:
    images = _image_files(directory)
    captions = _caption_files(directory)
    image_stems = {p.stem for p in images}
    orphans = [p.name for p in captions if p.stem not in image_stems]
    items: list[dict[str, Any]] = []
    for image in images:
        txt = directory / f"{image.stem}.txt"
        mask = mask_path_for(image)
        tags: list[str] = []
        if txt.is_file():
            tags = _parse_tags(txt.read_text(encoding="utf-8"))
        try:
            width, height, has_alpha = _probe_image(image)
        except OSError:
            width, height, has_alpha = 0, 0, False
        items.append(
            {
                "stem": image.stem,
                "image": str(image),
                "txt": str(txt),
                "mask": str(mask),
                "width": width,
                "height": height,
                "tags": tags,
                "has_sidecar_mask": mask.is_file(),
                "has_alpha": has_alpha,
            }
        )
    return {"items": items, "orphans": orphans}


def caption_write(directory: Path, stem: str, text: str) -> dict[str, Any]:
    path = directory / f"{stem}.txt"
    atomic_write_text(path, text)
    return {}


def dataset_drop(
    directory: Path,
    rate: float,
    seed: Optional[int] = None,
    stems: Optional[list[str]] = None,
) -> dict[str, Any]:
    if not (0.0 < rate <= 1.0):
        raise ValueError("rate must be in (0, 1]")
    rng = random.Random(seed)
    dest = trash_dir()
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    listing = dataset_list(directory)
    if listing["orphans"]:
        raise ValueError(
            "Refusing to drop: isolated tag file(s) without a matching image: "
            + ", ".join(listing["orphans"])
        )
    wanted = set(stems) if stems is not None else None
    for item in listing["items"]:
        if wanted is not None and item["stem"] not in wanted:
            continue
        if not Path(item["txt"]).is_file():
            continue
        if rng.random() > rate:
            continue
        for key in ("image", "txt", "mask"):
            src = Path(item[key])
            if key == "mask" and not src.is_file():
                continue
            if not src.is_file():
                continue
            shutil.move(str(src), str(dest / src.name))
        moved += 1
    return {"moved": moved}


def _plan_groups(directory: Path) -> list[tuple[str, list[Path]]]:
    files = [p for p in directory.iterdir() if p.is_file()]
    images = [
        p
        for p in files
        if p.suffix.lower() in IMAGE_EXTENSIONS and not is_mask_sidecar(p.name)
    ]
    images = sorted(images, key=lambda p: p.name)
    captions = sorted(
        [p for p in files if p.suffix.lower() == ".txt"],
        key=lambda p: p.name,
    )
    masks = sorted(
        [p for p in files if is_mask_sidecar(p.name)],
        key=lambda p: p.name,
    )
    stems = sorted({p.stem for p in images})
    caption_stems = {p.stem for p in captions}
    orphans = sorted(caption_stems - set(stems))
    if orphans:
        names = ", ".join(f"{stem}.txt" for stem in orphans)
        raise ValueError(
            "Refusing to shuffle: isolated tag file(s) without a matching image: " + names
        )
    groups: list[tuple[str, list[Path]]] = []
    for stem in stems:
        members = [p for p in images if p.stem == stem]
        members += [p for p in captions if p.stem == stem]
        members += [p for p in masks if p.name[: -len(MASK_SIDECAR_SUFFIX)] == stem]
        groups.append((stem, members))
    return groups


def _suffix(stem: str, path: Path) -> str:
    return path.name[len(stem) :]


def dataset_shuffle(directory: Path, seed: Optional[int] = None) -> dict[str, Any]:
    groups = _plan_groups(directory)
    if not groups:
        return {"groups": 0, "renamed_files": 0, "first_stem": "", "last_stem": ""}
    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    applied: list[tuple[Path, Path]] = []

    def move(src: Path, dest: Path) -> None:
        try:
            os.rename(src, dest)
        except OSError as exc:
            raise OSError(f"could not rename {src.name} to {dest.name}: {exc}") from exc
        applied.append((src, dest))

    try:
        staged: list[tuple[str, tuple[str, list[Path]]]] = []
        for stem, members in order:
            temp = SHUFFLE_TEMP_PREFIX + uuid.uuid4().hex[:12]
            for member in members:
                move(member, directory / f"{temp}{_suffix(stem, member)}")
            staged.append((temp, (stem, members)))
        for index, (temp, (stem, members)) in enumerate(staged):
            base = str(index + 1).zfill(SHUFFLE_STEM_PAD)
            for member in members:
                suffix = _suffix(stem, member)
                move(directory / f"{temp}{suffix}", directory / f"{base}{suffix}")
    except Exception:
        for src, dest in reversed(applied):
            try:
                os.rename(dest, src)
            except OSError:
                pass
        raise

    renamed = sum(len(members) for _, members in order)
    last = str(len(order)).zfill(SHUFFLE_STEM_PAD)
    return {
        "groups": len(order),
        "renamed_files": renamed,
        "first_stem": SHUFFLE_FIRST,
        "last_stem": last,
    }


def mask_file(directory: Path, stem: str) -> Path:
    return directory / f"{stem}{MASK_SIDECAR_SUFFIX}"


def mask_get(directory: Path, stem: str) -> dict[str, Any]:
    import base64

    path = mask_file(directory, stem)
    if not path.is_file():
        return {"png_base64": ""}
    return {"png_base64": base64.b64encode(path.read_bytes()).decode("ascii")}


def mask_write(directory: Path, stem: str, png_base64: str) -> dict[str, Any]:
    import base64

    if not png_base64:
        raise ValueError("missing png_base64")
    try:
        data = base64.b64decode(png_base64, validate=False)
    except Exception as exc:
        raise ValueError("png_base64 is not valid base64") from exc
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("mask must be a PNG")
    atomic_write_bytes(mask_file(directory, stem), data)
    return {}


def mask_delete(directory: Path, stem: str) -> dict[str, Any]:
    path = mask_file(directory, stem)
    if path.is_file():
        path.unlink()
    return {}


def config_get() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise ValueError(f"Could not locate config.toml at {path}")
    return {"path": str(path), "text": path.read_text(encoding="utf-8")}


def config_save(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("config text is empty")
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"config.toml would not parse: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config.toml must be a table")
    path = config_path()
    if not path.is_file():
        raise ValueError(f"Config file does not exist: {path}")
    atomic_write_text(path, text)
    return {"path": str(path)}


def validate_profile_name(name: str) -> Optional[str]:
    trimmed = name.strip()
    if not trimmed:
        return "Enter a profile name"
    if len(trimmed) > PROFILE_MAX_NAME:
        return f"Use at most {PROFILE_MAX_NAME} characters"
    if trimmed.startswith("."):
        return "A profile name cannot start with a dot"
    if any(ch in PROFILE_ILLEGAL or ord(ch) < 32 for ch in trimmed):
        return 'A profile name cannot contain / \\ : * ? " < > |'
    return None


def _profile_path(name: str) -> Path:
    err = validate_profile_name(name)
    if err:
        raise ValueError(err)
    return profile_dir() / f"{name.strip()}{PROFILE_EXT}"


def profile_list() -> dict[str, Any]:
    directory = profile_dir()
    if not directory.is_dir():
        return {"profiles": []}
    profiles = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != PROFILE_EXT:
            continue
        stat = path.stat()
        profiles.append(
            {
                "name": path.stem,
                "modified": int(stat.st_mtime * 1000),
                "size": int(stat.st_size),
            }
        )
    return {"profiles": profiles}


def profile_get(name: str) -> dict[str, Any]:
    path = _profile_path(name)
    if not path.is_file():
        raise ValueError(f"No profile named {name.strip()}")
    return {"name": path.stem, "text": path.read_text(encoding="utf-8")}


def profile_save(name: str, text: str, overwrite: bool) -> dict[str, Any]:
    err = validate_profile_name(name)
    if err:
        raise ValueError(err)
    trimmed = name.strip()
    directory = profile_dir()
    directory.mkdir(parents=True, exist_ok=True)
    existing = None
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == PROFILE_EXT and path.stem.lower() == trimmed.lower():
            existing = path
            break
    if existing is not None and not overwrite:
        raise ValueError(f'A profile named "{trimmed}" already exists')
    target = existing if existing is not None else directory / f"{trimmed}{PROFILE_EXT}"
    atomic_write_text(target, text)
    return {"name": target.stem}


def profile_delete(name: str) -> dict[str, Any]:
    path = _profile_path(name)
    if not path.is_file():
        raise ValueError(f"The profile is already gone: {name}")
    if path.resolve().parent != profile_dir().resolve():
        raise ValueError(f"Not a profile file: {path}")
    path.unlink()
    return {}


def tag_lexicon() -> dict[str, Any]:
    path = _repo_root() / "tagger" / "selected_tags.csv"
    if not path.is_file():
        return {"text": ""}
    return {"text": path.read_text(encoding="utf-8")}


def checkpoint_export(source: Path, dest: Path) -> dict[str, Any]:
    src = source.expanduser().resolve()
    dst = dest.expanduser().resolve()
    if not src.is_file():
        raise ValueError(f"not a file: {src}")
    if dst == src:
        raise ValueError(f"destination is the same file as the source: {dst}")
    if dst.suffix.lower() != ".safetensors":
        raise ValueError("destination must end in .safetensors")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {"bytes": dst.stat().st_size}


def logging_root(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("logging_dir") or "./logs")).expanduser().resolve()


def fs_listdir(path: Any) -> dict[str, Any]:
    """List one directory. No file bytes. `..` is canonicalized via resolve()."""
    if not path:
        raise ValueError("missing path")
    resolved = Path(str(path)).expanduser().resolve()
    if resolved.is_file():
        raise ValueError(f"not a directory: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"not a directory: {resolved}")
    parent = resolved.parent
    parent_s = str(parent) if parent != resolved else None
    entries: list[dict[str, Any]] = []
    try:
        children = list(resolved.iterdir())
    except OSError as exc:
        raise ValueError(f"cannot list {resolved}: {exc}") from exc
    children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    for child in children:
        try:
            st = child.stat()
            is_dir = child.is_dir()
        except OSError:
            continue
        try:
            child_path = str(child.resolve()) if is_dir else str(child)
        except OSError:
            child_path = str(child)
        entries.append(
            {
                "name": child.name,
                "path": child_path,
                "is_dir": is_dir,
                "size": 0 if is_dir else int(st.st_size),
                "mtime_ms": int(st.st_mtime * 1000),
            }
        )
    return {"path": str(resolved), "parent": parent_s, "entries": entries}


def fs_roots(cfg: dict[str, Any]) -> dict[str, Any]:
    roots: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(name: str, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if not resolved.is_dir():
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        roots.append({"name": name, "path": key})

    add("Home", Path.home())
    add("Repo", _repo_root())
    for i, folder in enumerate(train_data_roots(cfg)):
        add("Train data" if i == 0 else f"Train data {i + 1}", folder)
    add("Output", output_root(cfg))
    add("Logs", logging_root(cfg))
    return {"roots": roots}
