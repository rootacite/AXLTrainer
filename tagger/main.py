#!/usr/bin/env python3
"""WD-style ONNX captioner. Prefers MIGraphX (AMD GPU) with CPU fallback.

Paths are resolved relative to this file so the working directory can be the
repo root (`python tagger/main.py /data --threshold 0.35`).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_TAGGER_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _TAGGER_DIR / "migraphx_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ORT_MIGRAPHX_MODEL_CACHE_PATH", str(_CACHE_DIR))
os.environ.setdefault("ORT_MIGRAPHX_CACHE_PATH", str(_CACHE_DIR))
os.environ.setdefault("ORT_MIGRAPHX_FP16_ENABLE", "1")

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_SIZE = 448
DEFAULT_THRESHOLD = 0.35
DEFAULT_BATCH_SIZE = 1
DEFAULT_MODEL = _TAGGER_DIR / "model.onnx"
DEFAULT_LABELS = _TAGGER_DIR / "selected_tags.csv"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_labels(label_csv_path: str | Path = DEFAULT_LABELS) -> list[str]:
    path = Path(label_csv_path)
    tags: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty label csv: {path}")
        name_idx = 1
        if "name" in header:
            name_idx = header.index("name")
        for row in reader:
            if len(row) <= name_idx:
                continue
            tags.append(row[name_idx].replace("_", " "))
    if not tags:
        raise ValueError(f"no tags in {path}")
    return tags


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


def process_image(image_path: str | Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((MODEL_SIZE, MODEL_SIZE), resample=Image.Resampling.BICUBIC)
    image_data = np.asarray(img, dtype=np.float32)
    image_data = image_data[:, :, ::-1]
    return image_data


def create_session(
    model_path: str | Path = DEFAULT_MODEL,
    *,
    cpu: bool = False,
) -> ort.InferenceSession:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"missing ONNX model: {model_path}")

    available = set(ort.get_available_providers())
    if cpu:
        providers: list[Any] = ["CPUExecutionProvider"]
    else:
        providers = []
        if "MIGraphXExecutionProvider" in available:
            providers.append(("MIGraphXExecutionProvider", {"device_id": 0}))
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), providers=providers)
    return session


def session_io(session: ort.InferenceSession) -> tuple[str, str]:
    return session.get_inputs()[0].name, session.get_outputs()[0].name


def warmup_session(session: ort.InferenceSession, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    input_name, output_name = session_io(session)
    n = max(1, int(batch_size))
    dummy = np.zeros((n, MODEL_SIZE, MODEL_SIZE, 3), dtype=np.float32)
    session.run([output_name], {input_name: dummy})


def scores_to_caption(confidences: np.ndarray, tag_names: Sequence[str], threshold: float) -> str:
    n = min(len(tag_names), int(confidences.shape[0]))
    scored = [
        (tag_names[idx], float(confidences[idx]))
        for idx in range(n)
        if float(confidences[idx]) >= threshold
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(tag for tag, _ in scored)


def _run_batch(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    batch_paths: Sequence[Path],
    padded_size: int,
) -> np.ndarray:
    pixels = [process_image(path) for path in batch_paths]
    # MIGraphX compiles once per input shape; pad so every run uses padded_size.
    while len(pixels) < padded_size:
        pixels.append(pixels[-1])
    stacked = np.stack(pixels, axis=0)
    scores = session.run([output_name], {input_name: stacked})[0]
    return scores[: len(batch_paths)]


def tag_directory(
    directory: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    session: ort.InferenceSession | None = None,
    tag_names: Sequence[str] | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_path: str | Path = DEFAULT_MODEL,
    label_path: str | Path = DEFAULT_LABELS,
    cpu: bool = False,
    warmup: bool = True,
) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError("threshold must be between 0.0 and 1.0")

    current_batch = max(1, int(batch_size))
    owns_session = session is None
    if session is None:
        _log("Initializing ONNX session…")
        session = create_session(model_path, cpu=cpu)
        if warmup:
            _log(f"Warming up (batch={current_batch})…")
            warmup_session(session, current_batch)
    if tag_names is None:
        tag_names = load_labels(label_path)

    input_name, output_name = session_io(session)
    provider = session.get_providers()[0]
    image_files = list_images(directory)
    errors: list[dict[str, str]] = []
    processed = 0
    start = time.time()

    _log(
        f"Tagging {len(image_files)} images in {directory} "
        f"(threshold={threshold:g}, batch={current_batch}, provider={provider})"
    )

    index = 0
    while index < len(image_files):
        chunk = image_files[index : index + current_batch]
        try:
            scores = _run_batch(
                session, input_name, output_name, chunk, padded_size=current_batch
            )
        except Exception as exc:
            if len(chunk) > 1:
                _log(f"Batch of {len(chunk)} failed ({exc}); retrying one at a time")
                current_batch = 1
                continue
            path = chunk[0]
            errors.append({"file": path.name, "error": str(exc)})
            _log(f"[Error] {path.name}: {exc}")
            index += 1
            continue

        for offset, path in enumerate(chunk):
            try:
                caption = scores_to_caption(scores[offset], tag_names, threshold)
                path.with_suffix(".txt").write_text(caption, encoding="utf-8")
                processed += 1
                n_tags = len([t for t in caption.split(",") if t.strip()]) if caption else 0
                _log(f"[{processed}/{len(image_files)}] {path.name} -> {n_tags} tags")
            except Exception as exc:
                errors.append({"file": path.name, "error": str(exc)})
                _log(f"[Error] {path.name}: {exc}")
        index += len(chunk)

    elapsed = time.time() - start
    _log(f"Completed {processed}/{len(image_files)} in {elapsed:.2f}s ({len(errors)} failed)")
    if owns_session:
        del session

    return {
        "directory": str(directory),
        "threshold": float(threshold),
        "provider": provider,
        "total": len(image_files),
        "processed": processed,
        "failed": len(errors),
        "seconds": round(elapsed, 3),
        "errors": errors[:32],
    }


def _prompt_interactive() -> None:
    try:
        import readline
        import glob as _glob

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

    session = create_session()
    _log(f"Warming up (batch={DEFAULT_BATCH_SIZE})…")
    warmup_session(session, DEFAULT_BATCH_SIZE)
    tags = load_labels()
    _log(f"Session ready ({session.get_providers()[0]}, {len(tags)} labels)")

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
        tag_directory(dir_input, threshold, session=session, tag_names=tags, warmup=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write WD-tagger captions (comma-separated tags) next to each image.",
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
        help=f"Minimum tag confidence (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"ONNX batch size (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON result object to stdout (progress stays on stderr).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPUExecutionProvider (skip MIGraphX/CUDA).",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to model.onnx",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS),
        help="Path to selected_tags.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.directory is None:
        if args.json:
            _log("error: --json requires a directory argument")
            return 2
        _prompt_interactive()
        return 0
    try:
        result = tag_directory(
            args.directory,
            threshold=args.threshold,
            batch_size=args.batch_size,
            model_path=args.model,
            label_path=args.labels,
            cpu=args.cpu,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        _log(str(exc))
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
