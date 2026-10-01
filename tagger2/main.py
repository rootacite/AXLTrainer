#!/usr/bin/env python3
"""Pixai tagger v1 captioner (ViTDet, 30 877 Danbooru tags) on the AMD GPU.

Writes a `", "`-joined caption next to each image in a folder, the way `tagger/main.py` (the
legacy WD14 ONNX captioner) does. Paths resolve relative to this file, so the working directory
can stay the repo root: `python tagger2/main.py /data --threshold 0.35`.

Two things this module pins on purpose; both were measured (RX 9070 XT, ROCm 7.2) rather than
assumed:

* **The Hub is never asked.** `pipeline()` resolves a tokenizer, a processor, a feature extractor
  and a video processor that this pipeline can live without, and each attempt is a Hub request
  that stalls where no proxy is configured. Resolving the snapshot ourselves (cache-only) and
  handing the pipeline the directory it points at keeps every one of those lookups local: 10.9 s
  over three images and zero outbound connections, against 16.80 s and a connection for the
  call that started from the repository id. `--download` is the one-time escape hatch for a
  machine that has never fetched the model.
* **MIOpen's find-db and kernel cache live in `tagger2/miopen_cache/`**, created and registered
  before torch is imported, so the first convolutions are not re-searched on every start (a cold
  directory costs about 0.8 s of the first image; a warm one keeps it).

The model is used exactly as published: fp32, its own per-category calibrated thresholds, and
`min_threshold` for the caller's floor.

`--only-tags` is the other half of the job: a *partial* pass that adds the named tags to the
captions already on disk and leaves every other tag alone, so an existing sidecar is corrected
rather than replaced. It hands the caller's threshold to the pipeline as `threshold` (an override,
not a floor under the calibrated values) because a tag the caller asked for must not be hidden by a
calibrated value that sits above the request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

_TAGGER_DIR = Path(__file__).resolve().parent

MODEL_ID = "pixai-labs/pixai-tagger-v1.0"
ENGINE = "pixai-tagger-v1.0"
DEFAULT_THRESHOLD = 0.35
# What a caption takes unless the caller says otherwise. `character`, `copyright`, `style` and
# `meta` describe where the picture comes from rather than who is in it, so they stay off.
DEFAULT_CATEGORIES = ("general",)
DEFAULT_BATCH_SIZE = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def setup_miopen_cache(cache_dir: str | Path | None = None) -> Path:
    """Create and register the MIOpen cache directory; call before torch is imported.

    `setdefault` on purpose: a shell that already pinned these (start_train.sh does) keeps its own
    directory, and `AXL_TAGGER_CACHE_DIR` overrides ours.
    """
    target = Path(cache_dir or os.environ.get("AXL_TAGGER_CACHE_DIR") or (_TAGGER_DIR / "miopen_cache"))
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(target))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(target))
    return target


MIOPEN_CACHE_DIR = setup_miopen_cache()


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def allow_download(args: argparse.Namespace) -> bool:
    """`--download` or `AXL_TAGGER_ALLOW_DOWNLOAD=1`; everything else stays on the local cache."""
    env = os.environ.get("AXL_TAGGER_ALLOW_DOWNLOAD", "").strip().lower()
    return bool(getattr(args, "download", False)) or env in {"1", "true", "yes", "on"}


def list_images(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.lower().endswith(".mask.png")
    ]
    files.sort(key=lambda p: p.name.lower())
    return files


def resolve_model_dir(model: str, *, local_files_only: bool = True) -> Path:
    """The model's local directory: the path itself, or the cached Hugging Face snapshot."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=local_files_only)).resolve()


def load_model_settings(model: str = MODEL_ID, *, local_files_only: bool = True) -> dict[str, Any]:
    """What the model declares about itself, read from its own `config.json` (no torch, no GPU).

    Never raises for a model that is not local yet: `available` says whether it is, and `reason`
    carries the one thing the caller has to do about it.
    """
    payload: dict[str, Any] = {
        "available": False,
        "engine": ENGINE,
        "model": model,
        "model_path": "",
        "cache_dir": str(MIOPEN_CACHE_DIR),
        "categories": [],
        "default_categories": list(DEFAULT_CATEGORIES),
        "reason": "",
    }
    try:
        model_dir = resolve_model_dir(model, local_files_only=local_files_only)
    except Exception as exc:  # noqa: BLE001 - every resolution failure is the same answer here
        payload["reason"] = (
            f"{type(exc).__name__}: {exc} - pass --download (or set AXL_TAGGER_ALLOW_DOWNLOAD=1) "
            "to fetch the model once (~1.9 GB)"
        )
        return payload

    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        payload["reason"] = f"unreadable config.json in {model_dir}: {exc}"
        return payload

    split = config.get("tags_split")
    thresholds = config.get("category_best_threshold") or {}
    if not isinstance(split, list) or not split:
        payload["reason"] = f"config.json in {model_dir} has no tags_split"
        return payload

    payload["available"] = True
    payload["model_path"] = str(model_dir)
    payload["categories"] = [
        {
            "key": str(entry[0]),
            "count": int(entry[1]),
            "calibrated": float(thresholds.get(entry[0], DEFAULT_THRESHOLD)),
        }
        for entry in split
        if isinstance(entry, (list, tuple)) and len(entry) >= 2
    ]
    return payload


def requested_categories(requested: Iterable[str] | str | None) -> list[str]:
    """A comma string or a sequence to a clean list; empty means the default."""
    if requested is None:
        return list(DEFAULT_CATEGORIES)
    items = requested.split(",") if isinstance(requested, str) else list(requested)
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return cleaned or list(DEFAULT_CATEGORIES)


def resolve_categories(requested: Iterable[str] | str | None, settings: dict[str, Any]) -> list[str]:
    """Keep the requested categories, in the model's own order, and refuse unknown keys."""
    wanted = requested_categories(requested)
    known = [str(item["key"]) for item in settings.get("categories") or []]
    if not known:
        return wanted
    unknown = [item for item in wanted if item not in known]
    if unknown:
        raise ValueError(
            f"unknown tagger categories: {', '.join(unknown)} (known: {', '.join(known)})"
        )
    return [key for key in known if key in wanted]


def effective_thresholds(
    settings: dict[str, Any], categories: Sequence[str], threshold: float
) -> dict[str, float]:
    """Per category, `max(calibrated, the caller's floor)` - what the pipeline itself applies."""
    by_key = {str(item["key"]): float(item["calibrated"]) for item in settings.get("categories") or []}
    return {key: max(by_key.get(key, float(threshold)), float(threshold)) for key in categories}


def build_pipeline(model_dir: Path, *, cpu: bool = False) -> Any:
    """Load the published model with its own pipeline class. Torch is imported here, not above."""
    from transformers import pipeline

    kwargs: dict[str, Any] = {}
    if cpu:
        kwargs["device"] = "cpu"
    # No `device` otherwise: transformers takes device 0 when the platform offers one, which on
    # this repo is the GPU. The task class is the model's own `TaggerPipeline`.
    return pipeline(
        model=str(model_dir),
        image_processor=str(model_dir),
        trust_remote_code=True,
        **kwargs,
    )


def tag_text(tag: str) -> str:
    """`hair_between_eyes` / `masking_tape_(medium)` -> the space-separated form captions use."""
    return tag.replace("_", " ")


# `(anal:1.2)` after its parentheses are stripped: a weight, not part of the tag. Mirrors
# `evaluation.normalize_tag` (trainer/evaluation.py), which compares the same two sides in the
# Dashboard's checkpoint evaluation; kept local because this script is standalone.
_WEIGHT_SUFFIX = re.compile(r"^(.*?)\s*:\s*[-+]?(?:\d+\.?\d*|\.\d+)$")
_WRAPPERS = {"(": ")", "[": "]", "{": "}"}


def normalize_tag(raw: str) -> str:
    """One caption tag in the comparable form: no wrapper, no weight, spaces, lowercase.

    `(anal:1.2)` -> `anal`, `[long hair]` -> `long hair`, `Masking_Tape_(Medium)` ->
    `masking tape (medium)`. Partial tagging uses it both to look a requested tag up in the model's
    own output (whose keys carry underscores) and to tell whether a caption already holds it.
    """
    text = str(raw or "").strip()
    while len(text) >= 2 and _WRAPPERS.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    match = _WEIGHT_SUFFIX.match(text)
    if match:
        text = match.group(1).strip()
    return " ".join(text.replace("_", " ").lower().split())


def requested_only_tags(raw: Iterable[str] | str | None) -> list[str]:
    """The `--only-tags` list as clean, ordered, de-duplicated display tags.

    Accepts a comma string (which is what the CLI hands over) or a sequence; the first spelling of
    a tag wins, and the comparison key is `normalize_tag`, so `anal` and `Anal` are one request.
    """
    if raw is None:
        return []
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        key = normalize_tag(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def tag_list(result: dict[str, Any], categories: Sequence[str]) -> list[str]:
    """The tags of the selected categories as a list, most confident first."""
    scored: list[tuple[str, float]] = []
    for category in categories:
        for tag, probability in (result.get(category) or {}).items():
            scored.append((str(tag), float(probability)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [tag_text(tag) for tag, _ in scored]


def caption_for(result: dict[str, Any], categories: Sequence[str]) -> str:
    """One caption from the selected categories, most confident first: `tag_list` joined."""
    return ", ".join(tag_list(result, categories))


def _results_of(item: Any) -> dict[str, Any]:
    """The pipeline answers `{"results": {...}}`; a bare dict is accepted too."""
    if isinstance(item, dict) and "results" in item:
        return item["results"]
    return item


def _run_batch(
    tagger: Any,
    paths: Sequence[Path],
    *,
    batch_size: int,
    threshold: float,
    override_threshold: bool = False,
) -> list[dict[str, Any]]:
    """One forward pass over `paths`.

    `override_threshold` hands the model's own `threshold` instead of `min_threshold`: the pipeline
    treats the first as the whole floor (the calibrated per-category value is replaced) and the
    second as a floor *under* it. Partial tagging needs the first, because the tags it looks for are
    the user's choice and must not be hidden by a calibrated value that sits above the request.
    """
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in paths]
    kwargs: dict[str, Any] = {"batch_size": max(1, int(batch_size))}
    if override_threshold:
        kwargs["threshold"] = float(threshold)
    else:
        kwargs["min_threshold"] = float(threshold)
    outputs = tagger(images, **kwargs)
    if not isinstance(outputs, list):
        outputs = [outputs]
    return [_results_of(item) for item in outputs]


def _check_threshold(threshold: Any) -> float:
    """The tagger floor as a float in 0..1; the message a caller can show as-is."""
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be between 0.0 and 1.0")
    return float(threshold)


def _resolve_tagger(
    tagger: Any,
    settings: Optional[dict[str, Any]],
    *,
    model: str,
    categories: Iterable[str] | str,
    threshold: float,
    cpu: bool,
    download: bool,
) -> tuple[Any, list[str], dict[str, float], str]:
    """The pipeline to tag with, plus what it resolved: the categories it writes, their effective
    floors, and its device. A caller that hands over its own `tagger` skips the load and the log."""
    _check_threshold(threshold)
    if tagger is None:
        _log(f"Resolving {model} from the local cache" + (" (download allowed)" if download else ""))
        settings = load_model_settings(model, local_files_only=not download)
        if not settings["available"]:
            raise RuntimeError(settings["reason"] or f"model not available: {model}")
        _log(f"Loading pipeline (MIOpen cache {MIOPEN_CACHE_DIR})")
        tagger = build_pipeline(Path(settings["model_path"]), cpu=cpu)
    settings = settings or {}
    chosen = (
        resolve_categories(categories, settings)
        if settings.get("categories")
        else requested_categories(categories)
    )
    floors = effective_thresholds(settings, chosen, float(threshold)) if settings.get("categories") else {}
    return tagger, chosen, floors, str(getattr(tagger, "device", "cpu"))


def _failed_entry(path: Path, error: Any) -> dict[str, Any]:
    return {"path": str(path), "name": path.name, "tags": [], "error": str(error)}


def _tagged_entry(path: Path, result: dict[str, Any], categories: Sequence[str]) -> dict[str, Any]:
    try:
        tags = tag_list(result, categories)
    except Exception as exc:  # noqa: BLE001 - one odd result must not end the folder
        return _failed_entry(path, exc)
    return {"path": str(path), "name": path.name, "tags": tags, "error": None}


def _tag_all(
    tagger: Any,
    paths: Sequence[Path],
    *,
    categories: Sequence[str],
    batch_size: int,
    threshold: float,
    on_entry: Optional[Callable[[dict[str, Any]], None]] = None,
    override_threshold: bool = False,
    build: Optional[Callable[[Path, dict[str, Any]], dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Tag every path, in order, one entry each: tags or the error that stopped it.

    Batched, and a batch that cannot be read falls back to one image at a time, so a single bad
    file costs one entry instead of the whole chunk. Progress goes to stderr as it goes. `on_entry`
    sees every entry right after it is built — a caller that persists them one by one (an
    evaluation's job record) can also raise from it to end the pass at that image.

    `build` turns one image's raw result into its entry; the default is the whole caption of
    `categories`, and partial tagging hands over a matcher that picks the requested tags out of the
    model's own output instead.
    """
    entries: list[dict[str, Any]] = []
    current_batch = max(1, int(batch_size))
    processed = 0

    def record(entry: dict[str, Any]) -> None:
        nonlocal processed
        entries.append(entry)
        if entry["error"]:
            _log(f"[Error] {entry['name']}: {entry['error']}")
        else:
            processed += 1
            detail = entry.get("summary") or f"{len(entry.get('tags') or [])} tags"
            _log(f"[{processed}/{len(paths)}] {entry['name']} -> {detail}")
        if on_entry is not None:
            on_entry(entry)

    index = 0
    while index < len(paths):
        chunk = list(paths[index : index + current_batch])
        try:
            results = _run_batch(
                tagger,
                chunk,
                batch_size=current_batch,
                threshold=float(threshold),
                override_threshold=override_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - one bad image must not end the folder
            if len(chunk) > 1:
                _log(f"Batch of {len(chunk)} failed ({exc}); retrying one at a time")
                current_batch = 1
                continue
            record(_failed_entry(chunk[0], exc))
            index += 1
            continue

        if len(results) != len(chunk):
            message = f"expected {len(chunk)} results, got {len(results)}"
            for path in chunk:
                record(_failed_entry(path, message))
            index += len(chunk)
            continue

        for path, result in zip(chunk, results):
            if build is None:
                record(_tagged_entry(path, result, categories))
            else:
                try:
                    record(build(path, result))
                except Exception as exc:  # noqa: BLE001 - same rule as a caption that would not build
                    record(_failed_entry(path, exc))
        index += len(chunk)
    return entries


def tag_paths(
    paths: Sequence[str | Path],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    categories: Iterable[str] | str = DEFAULT_CATEGORIES,
    tagger: Any = None,
    settings: Optional[dict[str, Any]] = None,
    model: str = MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cpu: bool = False,
    download: bool = False,
    on_entry: Optional[Callable[[dict[str, Any]], None]] = None,
) -> list[dict[str, Any]]:
    """Tag the images in `paths` and answer with their tags; nothing is written to disk.

    Same pipeline, categories, calibrated floors and error tolerance as `tag_directory`, for a
    caller that wants the labels themselves - the Dashboard's checkpoint evaluation. Each entry is
    `{"path", "name", "tags": [...], "error"}` in the input order; an image that could not be read
    carries its message and no tags rather than raising out of the pass. `on_entry` sees each entry
    as soon as it is built: the evaluation writes its progress from there, and ends the pass at that
    image by raising (a cancel, which leaves the entries already reported in place).
    """
    if not paths:
        # Nothing to tag: a caller with no images must not pay for the model load. The threshold is
        # still checked, so a bad request is answered rather than quietly accepted.
        _check_threshold(threshold)
        return []
    tagger, chosen, _floors, device = _resolve_tagger(
        tagger,
        settings,
        model=model,
        categories=categories,
        threshold=threshold,
        cpu=cpu,
        download=download,
    )
    targets = [Path(str(path)) for path in paths]
    _log(
        f"Tagging {len(targets)} images (threshold={float(threshold):g}, batch={max(1, int(batch_size))}, "
        f"device={device}, categories={', '.join(chosen)})"
    )
    return _tag_all(
        tagger,
        targets,
        categories=chosen,
        batch_size=batch_size,
        threshold=threshold,
        on_entry=on_entry,
    )


def tag_directory(
    directory: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    categories: Iterable[str] | str = DEFAULT_CATEGORIES,
    tagger: Any = None,
    settings: dict[str, Any] | None = None,
    model: str = MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cpu: bool = False,
    download: bool = False,
) -> dict[str, Any]:
    """Caption every image in `directory` (non-recursive), overwriting sidecar `.txt` files."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    current_batch = max(1, int(batch_size))
    tagger, chosen, floors, device = _resolve_tagger(
        tagger,
        settings,
        model=model,
        categories=categories,
        threshold=threshold,
        cpu=cpu,
        download=download,
    )

    image_files = list_images(directory)
    errors: list[dict[str, str]] = []
    processed = 0
    start = time.time()
    _log(
        f"Tagging {len(image_files)} images in {directory} "
        f"(threshold={float(threshold):g}, batch={current_batch}, device={device}, "
        f"categories={', '.join(chosen)})"
    )

    for entry in _tag_all(
        tagger,
        image_files,
        categories=chosen,
        batch_size=current_batch,
        threshold=threshold,
    ):
        if entry["error"]:
            errors.append({"file": entry["name"], "error": entry["error"]})
            continue
        try:
            Path(entry["path"]).with_suffix(".txt").write_text(", ".join(entry["tags"]), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - an unwritable sidecar is this file's failure
            errors.append({"file": entry["name"], "error": str(exc)})
            _log(f"[Error] {entry['name']}: {exc}")
            continue
        processed += 1

    elapsed = time.time() - start
    _log(f"Completed {processed}/{len(image_files)} in {elapsed:.2f}s ({len(errors)} failed)")

    return {
        "directory": str(directory),
        "engine": ENGINE,
        "mode": "full",
        "categories": chosen,
        "threshold": float(threshold),
        "thresholds": floors,
        "provider": f"{ENGINE} on {device}",
        "device": device,
        "total": len(image_files),
        "processed": processed,
        "failed": len(errors),
        "seconds": round(elapsed, 3),
        "errors": errors[:32],
    }


def matched_only_tags(result: dict[str, Any], wanted: Sequence[tuple[str, str]]) -> list[str]:
    """The requested tags the model reported for one image, in the request's own order.

    `wanted` is `(normalized key, the form a caption uses)` pairs. The lookup runs over every
    category the model returned, which is why partial tagging ignores `--categories`: a tag is
    found wherever the model files it.
    """
    observed: set[str] = set()
    for value in (result or {}).values():
        if isinstance(value, dict):
            observed.update(normalize_tag(tag) for tag in value)
    return [display for key, display in wanted if key in observed]


def _read_caption(path: Path) -> str:
    """An existing sidecar caption, or `""` when there is none (or it cannot be read)."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def merge_only_tags(text: str, matched: Sequence[str]) -> tuple[str, list[str]]:
    """Add the tags `text` does not already hold; answer `(new caption, added tags)`.

    Add-only on purpose: a caption the tagger or the user already wrote keeps every tag it has, its
    spelling included, and only gains the requested ones the model found. A tag already there in any
    comparable form (underscored, weighted, bracketed) counts as present, so it is never duplicated.
    """
    existing = str(text or "")
    present = {key for key in (normalize_tag(part) for part in existing.split(",")) if key}
    added = [tag for tag in matched if normalize_tag(tag) not in present]
    if not added:
        return existing, []
    base = existing.strip()
    joined = ", ".join(added)
    return (f"{base}, {joined}" if base else joined), added


def add_only_tags(
    directory: str | Path,
    tags: Iterable[str] | str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    tagger: Any = None,
    settings: dict[str, Any] | None = None,
    model: str = MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cpu: bool = False,
    download: bool = False,
) -> dict[str, Any]:
    """Add the requested tags to the captions that show them, and change nothing else.

    The counterweight to `tag_directory`: instead of overwriting a sidecar with the whole caption,
    this looks only for the tags the caller named, and appends the ones the model reports above
    `threshold`. `threshold` is authoritative here — it goes to the pipeline as `threshold`, so the
    model's own calibrated per-category value does not sit underneath it and hide a tag the caller
    asked for. Every other tag in an existing caption is left exactly as it is, and a caption that
    would gain nothing is not written at all (so an image with no sidecar and no match gains none).

    Returns the same summary shape `tag_directory` does, with `mode: "partial"`, `only_tags`,
    `added` (how many images gained each requested tag) and `unmatched` (tags that matched no image
    at all — usually a spelling the model does not use).
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    wanted_list = requested_only_tags(tags)
    if not wanted_list:
        raise ValueError("only_tags must name at least one tag")
    _check_threshold(threshold)

    current_batch = max(1, int(batch_size))
    tagger, _chosen, _floors, device = _resolve_tagger(
        tagger,
        settings,
        model=model,
        categories=DEFAULT_CATEGORIES,
        threshold=threshold,
        cpu=cpu,
        download=download,
    )
    wanted = [(normalize_tag(tag), tag_text(tag)) for tag in wanted_list]
    image_files = list_images(directory)
    errors: list[dict[str, str]] = []
    added_by_tag: dict[str, int] = {display: 0 for _key, display in wanted}
    changed = 0
    start = time.time()
    _log(
        f"Partial tagging {len(image_files)} images in {directory} "
        f"(threshold={float(threshold):g}, batch={current_batch}, device={device}, "
        f"tags={', '.join(display for _key, display in wanted)})"
    )

    def build(path: Path, result: dict[str, Any]) -> dict[str, Any]:
        matched = matched_only_tags(result, wanted)
        return {
            "path": str(path),
            "name": path.name,
            "tags": matched,
            "error": None,
            "summary": ", ".join(matched) or "no requested tag",
        }

    for entry in _tag_all(
        tagger,
        image_files,
        categories=DEFAULT_CATEGORIES,
        batch_size=current_batch,
        threshold=threshold,
        override_threshold=True,
        build=build,
    ):
        if entry["error"]:
            errors.append({"file": entry["name"], "error": entry["error"]})
            continue
        matched = [str(tag) for tag in entry.get("tags") or []]
        if not matched:
            continue
        caption_path = Path(entry["path"]).with_suffix(".txt")
        try:
            text, added = merge_only_tags(_read_caption(caption_path), matched)
            if added:
                caption_path.write_text(text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - an unwritable sidecar is this file's failure
            errors.append({"file": entry["name"], "error": str(exc)})
            _log(f"[Error] {entry['name']}: {exc}")
            continue
        if added:
            changed += 1
            for tag in added:
                added_by_tag[tag] = added_by_tag.get(tag, 0) + 1

    elapsed = time.time() - start
    _log(
        f"Completed {changed}/{len(image_files)} in {elapsed:.2f}s ({len(errors)} failed); "
        f"added {', '.join(f'{tag} x{count}' for tag, count in added_by_tag.items() if count) or 'nothing'}"
    )
    unmatched = [tag for tag, count in added_by_tag.items() if count == 0]
    for tag in unmatched:
        _log(f"[Warn] no image showed '{tag}'; it was not added anywhere")

    return {
        "directory": str(directory),
        "engine": ENGINE,
        "mode": "partial",
        "only_tags": [display for _key, display in wanted],
        "categories": [],
        "threshold": float(threshold),
        "thresholds": {},
        "provider": f"{ENGINE} on {device}",
        "device": device,
        "total": len(image_files),
        "processed": changed,
        "failed": len(errors),
        "seconds": round(elapsed, 3),
        "errors": errors[:32],
        "added": {tag: count for tag, count in added_by_tag.items() if count},
        "unmatched": unmatched,
    }


def _prompt_interactive(defaults: argparse.Namespace) -> None:
    try:
        import glob as _glob
        import readline

        def path_completer(text: str, state: int) -> str | None:
            target = os.path.expanduser(text)
            matches = _glob.glob((target or ".") + "*")
            matches = [m + os.sep if os.path.isdir(m) else m for m in matches]
            if state < len(matches):
                return matches[state]
            return None

        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(path_completer)
    except ImportError:
        _log("[Warning] readline not available; path auto-completion disabled.")

    settings = load_model_settings(defaults.model, local_files_only=not allow_download(defaults))
    if not settings["available"]:
        raise RuntimeError(settings["reason"] or f"model not available: {defaults.model}")
    categories = resolve_categories(defaults.categories, settings)
    _log("Loading pipeline...")
    tagger = build_pipeline(Path(settings["model_path"]), cpu=defaults.cpu)
    _log(f"Ready ({tagger.device}, categories: {', '.join(categories)})")

    while True:
        dir_input = input("Enter DIRECTORY path (or 'exit' to quit): ").strip()
        if dir_input.lower() in {"exit", "quit", "q"}:
            break
        if not os.path.isdir(dir_input):
            _log(f"Error: '{dir_input}' is not a valid directory.")
            continue
        raw = input(f"Enter confidence threshold [0.0 - 1.0] (default {DEFAULT_THRESHOLD}): ").strip()
        try:
            threshold = DEFAULT_THRESHOLD if raw == "" else float(raw)
            if not (0.0 <= threshold <= 1.0):
                _log("Value out of range. Using default.")
                threshold = DEFAULT_THRESHOLD
        except ValueError:
            _log("Invalid input. Using default.")
            threshold = DEFAULT_THRESHOLD
        tag_directory(
            dir_input,
            threshold=threshold,
            categories=categories,
            tagger=tagger,
            settings=settings,
            batch_size=defaults.batch_size,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write Pixai-tagger captions (comma-separated tags) next to each image.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        help="Image folder (non-recursive). Omit for an interactive prompt.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Lowest confidence a tag may keep; the model's own per-category calibrated threshold "
            f"is the floor underneath it (default {DEFAULT_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Images per forward pass (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help=f"Comma-separated categories to write (default {','.join(DEFAULT_CATEGORIES)})",
    )
    parser.add_argument(
        "--only-tags",
        default="",
        help=(
            "Partial tagging: a comma-separated list of tags to add to the captions that show them, "
            "leaving every other tag alone. The categories are ignored (the tag is looked up in all "
            "of them) and --threshold becomes the absolute confidence floor for these tags."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON result object to stdout (progress stays on stderr).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on the CPU instead of the platform's default device.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"Hugging Face id or local directory (default {MODEL_ID})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow the one-time Hub fetch of the model and its configs (off by default).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print the model's categories and calibrated thresholds as JSON, then exit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    download = allow_download(args)
    if args.info:
        payload = load_model_settings(args.model, local_files_only=not download)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return 0
    if args.directory is None:
        if args.only_tags:
            # Partial tagging is a batch mode only: the interactive loop has no third question.
            _log("error: --only-tags requires a directory argument")
            return 2
        if args.json:
            _log("error: --json requires a directory argument")
            return 2
        try:
            _prompt_interactive(args)
        except Exception as exc:  # noqa: BLE001
            _log(str(exc))
            return 1
        return 0
    try:
        if args.only_tags:
            result = add_only_tags(
                args.directory,
                args.only_tags,
                threshold=args.threshold,
                model=args.model,
                batch_size=args.batch_size,
                cpu=args.cpu,
                download=download,
            )
        else:
            result = tag_directory(
                args.directory,
                threshold=args.threshold,
                categories=args.categories,
                model=args.model,
                batch_size=args.batch_size,
                cpu=args.cpu,
                download=download,
            )
    except Exception as exc:  # noqa: BLE001
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        _log(str(exc))
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
