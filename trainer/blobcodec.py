"""Resize and re-encode image blobs for Chromatrix IPC.

Torch-free on purpose: spawn workers import this module and must not import
`api.py` or torch (ROCm does not survive a fork/spawn of a process that already
initialised HIP).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FORMATS = {"jpeg", "webp", "png"}
MAX_EDGE_RANGE = (32, 4096)
QUALITY_RANGE = (1, 100)
DEFAULT_QUALITY = 80
DEFAULT_FORMAT = "jpeg"
# Bump when encode_one's bytes change for the same source+params, so /tmp
# cache entries from an older codec are not reused.
CODEC_VERSION = "flatten-v1"
MIME = {
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "png": "image/png",
}

_CACHE_DIR_DEFAULT = Path("/tmp/axlranko/blob-cache")
_LOCK = threading.Lock()
_cache: Optional["BlobCache"] = None
_executor: Optional[ProcessPoolExecutor] = None
_executor_workers: Optional[int] = None


def host_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 1 << 30


def cache_cap_bytes() -> int:
    raw = os.environ.get("AXL_BLOB_CACHE_BYTES")
    if raw:
        return max(0, int(raw))
    return host_ram_bytes() // 4


def cache_dir() -> Path:
    raw = os.environ.get("AXL_BLOB_CACHE_DIR")
    if raw:
        return Path(raw)
    return _CACHE_DIR_DEFAULT


def worker_count() -> int:
    raw = os.environ.get("AXL_BLOB_WORKERS")
    if raw is not None and raw != "":
        return max(0, int(raw))
    n = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
    return max(1, int(n or 1))


def parse_encode_params(params: dict[str, Any]) -> tuple[int, int, str]:
    if "max_edge" not in params or params.get("max_edge") is None:
        raise ValueError("max_edge is required")
    try:
        max_edge = int(params["max_edge"])
    except (TypeError, ValueError) as exc:
        raise ValueError("max_edge must be an integer") from exc
    lo, hi = MAX_EDGE_RANGE
    if not (lo <= max_edge <= hi):
        raise ValueError(f"max_edge must be in {lo}..{hi}")
    quality = params.get("quality", DEFAULT_QUALITY)
    if quality is None:
        quality = DEFAULT_QUALITY
    try:
        quality = int(quality)
    except (TypeError, ValueError) as exc:
        raise ValueError("quality must be an integer") from exc
    qlo, qhi = QUALITY_RANGE
    if not (qlo <= quality <= qhi):
        raise ValueError(f"quality must be in {qlo}..{qhi}")
    fmt = str(params.get("format") or DEFAULT_FORMAT).lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {sorted(FORMATS)}")
    return max_edge, quality, fmt


def cache_key(path: str, mtime_ns: int, size: int, max_edge: int, quality: int, fmt: str) -> str:
    payload = f"{path}\0{mtime_ns}\0{size}\0{max_edge}\0{quality}\0{fmt}\0{CODEC_VERSION}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in {"RGBA", "LA", "PA"} or (img.mode == "P" and "transparency" in img.info)


def flatten_rgb(img: Image.Image) -> Image.Image:
    """RGB for JPEG/WebP. Straight-alpha leftovers stay hidden (composite on black)."""
    if img.mode == "RGB":
        return img
    if not _has_alpha(img):
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def encode_one(args: tuple[str, int, int, str]) -> dict[str, Any]:
    """Worker entry: path + encode params → encoded bytes or an error."""
    path, max_edge, quality, fmt = args
    try:
        source = Path(path)
        with Image.open(source) as img:
            img.load()
            if fmt == "png":
                work = img.convert("RGBA") if img.mode not in {"RGB", "RGBA", "L"} else img
            else:
                work = flatten_rgb(img)
            width, height = work.size
            longest = max(width, height)
            if longest > max_edge:
                scale = max_edge / float(longest)
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                work = work.resize(new_size, Image.Resampling.LANCZOS)
            out_w, out_h = work.size
            buf = BytesIO()
            save_kw: dict[str, Any] = {}
            if fmt == "jpeg":
                save_kw["quality"] = quality
                save_kw["optimize"] = True
            elif fmt == "webp":
                save_kw["quality"] = quality
            work.save(buf, format=fmt.upper(), **save_kw)
            data = buf.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        return {
            "path": path,
            "ok": True,
            "hash": digest,
            "mime": MIME[fmt],
            "width": out_w,
            "height": out_h,
            "data": data,
        }
    except Exception as exc:
        return {"path": path, "ok": False, "error": str(exc)}


class BlobCache:
    def __init__(self, directory: Path, cap_bytes: int):
        self.directory = directory
        self.cap_bytes = cap_bytes
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self.directory.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        for sidecar in self.directory.glob("*.json"):
            key = sidecar.stem
            body = sidecar.with_suffix(".bin")
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                nbytes = int(meta.get("nbytes") or body.stat().st_size)
                accessed = float(meta.get("accessed") or sidecar.stat().st_mtime)
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                continue
            if not body.is_file():
                continue
            self._entries[key] = {
                "nbytes": nbytes,
                "accessed": accessed,
                "hash": meta.get("hash"),
                "mime": meta.get("mime"),
                "width": meta.get("width"),
                "height": meta.get("height"),
            }

    def _total(self) -> int:
        return sum(int(e["nbytes"]) for e in self._entries.values())

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            body = self.directory / f"{key}.bin"
            sidecar = self.directory / f"{key}.json"
            try:
                data = body.read_bytes()
            except OSError:
                self._entries.pop(key, None)
                return None
            now = time.time()
            entry["accessed"] = now
            try:
                meta = {
                    "hash": entry["hash"],
                    "mime": entry["mime"],
                    "width": entry["width"],
                    "height": entry["height"],
                    "nbytes": entry["nbytes"],
                    "accessed": now,
                }
                sidecar.write_text(json.dumps(meta), encoding="utf-8")
            except OSError:
                pass
            return {
                "hash": entry["hash"],
                "mime": entry["mime"],
                "width": entry["width"],
                "height": entry["height"],
                "nbytes": entry["nbytes"],
                "data": data,
                "cache": "hit",
            }

    def put(self, key: str, encoded: dict[str, Any]) -> None:
        data: bytes = encoded["data"]
        nbytes = len(data)
        with self._lock:
            if nbytes > self.cap_bytes:
                return
            self._evict(nbytes)
            body = self.directory / f"{key}.bin"
            sidecar = self.directory / f"{key}.json"
            tmp = body.with_suffix(".bin.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, body)
            now = time.time()
            meta = {
                "hash": encoded["hash"],
                "mime": encoded["mime"],
                "width": encoded["width"],
                "height": encoded["height"],
                "nbytes": nbytes,
                "accessed": now,
            }
            sidecar.write_text(json.dumps(meta), encoding="utf-8")
            self._entries[key] = {
                "nbytes": nbytes,
                "accessed": now,
                "hash": encoded["hash"],
                "mime": encoded["mime"],
                "width": encoded["width"],
                "height": encoded["height"],
            }

    def _evict(self, incoming: int) -> None:
        while self._entries and self._total() + incoming > self.cap_bytes:
            oldest = min(self._entries.items(), key=lambda kv: kv[1]["accessed"])
            key = oldest[0]
            self._entries.pop(key, None)
            for suffix in (".bin", ".json"):
                try:
                    (self.directory / f"{key}{suffix}").unlink()
                except OSError:
                    pass


def get_cache() -> BlobCache:
    global _cache
    with _LOCK:
        if _cache is None:
            _cache = BlobCache(cache_dir(), cache_cap_bytes())
        return _cache


def _get_executor(workers: int) -> ProcessPoolExecutor:
    global _executor, _executor_workers
    if workers <= 0:
        raise RuntimeError("inline encode does not use a pool")
    if _executor is None or _executor_workers != workers:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
        import multiprocessing as mp

        _executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )
        _executor_workers = workers
    return _executor


def reset_for_tests() -> None:
    """Drop the process pool and cache singleton so a test can change env."""
    global _cache, _executor, _executor_workers
    with _LOCK:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
        _executor_workers = None
        _cache = None


def _source_identity(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    canonical = str(path.resolve())
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
    return canonical, int(mtime_ns), int(stat.st_size)


def _run_encode(jobs: list[tuple[str, int, int, str]]) -> list[dict[str, Any]]:
    if not jobs:
        return []
    workers = worker_count()
    if workers <= 0:
        return [encode_one(job) for job in jobs]
    executor = _get_executor(workers)
    return list(executor.map(encode_one, jobs))


def resolve_blobs(
    paths: list[str],
    max_edge: int,
    quality: int,
    fmt: str,
    include_payload: bool,
) -> list[dict[str, Any]]:
    """Look up / encode each path. Failed items carry `error` and no payload."""
    cache = get_cache()
    items: list[Optional[dict[str, Any]]] = [None] * len(paths)
    misses: list[tuple[int, tuple[str, int, int, str], str]] = []

    for index, raw in enumerate(paths):
        path = Path(str(raw))
        try:
            canonical, mtime_ns, size = _source_identity(path)
        except OSError as exc:
            items[index] = {"path": str(raw), "error": str(exc)}
            continue
        key = cache_key(canonical, mtime_ns, size, max_edge, quality, fmt)
        hit = cache.get(key)
        if hit is not None:
            item = {
                "path": str(raw),
                "hash": hit["hash"],
                "width": hit["width"],
                "height": hit["height"],
                "bytes": hit["nbytes"],
                "mime": hit["mime"],
                "cache": "hit",
            }
            if include_payload:
                item["base64"] = base64.b64encode(hit["data"]).decode("ascii")
            items[index] = item
            continue
        misses.append((index, (str(path), max_edge, quality, fmt), key))

    encoded = _run_encode([job for _, job, _ in misses])

    for (index, _job, key), result in zip(misses, encoded):
        if not result.get("ok"):
            items[index] = {"path": paths[index], "error": result.get("error") or "encode failed"}
            continue
        cache.put(key, result)
        item = {
            "path": paths[index],
            "hash": result["hash"],
            "width": result["width"],
            "height": result["height"],
            "bytes": len(result["data"]),
            "mime": result["mime"],
            "cache": "miss",
        }
        if include_payload:
            item["base64"] = base64.b64encode(result["data"]).decode("ascii")
        items[index] = item

    return [item if item is not None else {"path": paths[i], "error": "internal"} for i, item in enumerate(items)]
