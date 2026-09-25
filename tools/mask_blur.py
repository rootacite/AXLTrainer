#!/usr/bin/env python3
"""Batch-soften alpha-derived loss masks into `{stem}.mask.png` sidecars.

A character illustration ("stand") PNG usually carries a hard 0/255 alpha, and
with no sidecar next to it the trainer takes that alpha as the loss mask
(`load_loss_mask` in `trainer/utils.py`). A binary weight across the silhouette
edge shows up as jaggies in the trained output. This tool blurs the alpha with a
Gaussian and writes the result next to the image as `{stem}.mask.png`; the
sidecar wins over the alpha, so the softened mask takes effect with no config
change, and the training image itself is never touched.

The sidecar is written at the training image's own size on purpose: the loader
resizes a sidecar NEAREST back to the source size before its LANCZOS fit, so a
sidecar of a different size would return as nearest-neighbour steps instead of
the blur.

Constants and rules mirror `trainer/utils.py` (`MASK_SIDECAR_SUFFIX`, the image
extension set, the alpha rule). That module imports torch, so this helper keeps
its own copy rather than importing the training stack.

A sidecar written by this tool carries an `axl_mask_blur` PNG text chunk holding
the radius. An existing sidecar without that chunk - a mask painted in Ranko, or
anything else - is left untouched unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo

# Mirrors trainer/utils.py: MASK_SIDECAR_SUFFIX, list_images' extension set.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_SIDECAR_SUFFIX = ".mask.png"

# PNG text chunk marking a sidecar as this tool's output, and the radius it was built with.
MARKER_KEY = "axl_mask_blur"

DEFAULT_RADIUS = 16.0
# The range asked for in practice; a value outside it is accepted with a note.
RADIUS_HINT_RANGE = (8.0, 32.0)
MAX_RADIUS = 256.0
MAX_LISTED = 20

WRITTEN = "written"
PROTECTED = "protected"
NO_ALPHA = "no-alpha"
OPAQUE_ALPHA = "opaque-alpha"
EMPTY_ALPHA = "empty-alpha"
FAILED = "failed"

# Report order and wording. `written` counts files, the rest count images.
KIND_ORDER = (WRITTEN, PROTECTED, NO_ALPHA, OPAQUE_ALPHA, EMPTY_ALPHA, FAILED)
KIND_LABELS = {
    WRITTEN: "sidecars written",
    PROTECTED: "protected",
    NO_ALPHA: "no alpha channel",
    OPAQUE_ALPHA: "alpha uniformly opaque",
    EMPTY_ALPHA: "alpha uniformly transparent",
    FAILED: "failed",
}
# Kinds whose file names are worth printing: they either need a decision or are
# usually a sign of a broken sample. `no-alpha` is normal (every jpg).
NAMED_KINDS = (PROTECTED, OPAQUE_ALPHA, EMPTY_ALPHA, FAILED)


class Outcome(NamedTuple):
    image: Path
    sidecar: Path
    kind: str
    detail: str = ""


def is_mask_sidecar(path: Path) -> bool:
    """True for optional loss-mask files named `{stem}.mask.png`."""
    return path.name.lower().endswith(MASK_SIDECAR_SUFFIX)


def mask_path_for(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + MASK_SIDECAR_SUFFIX)


def image_has_alpha(img: Image.Image) -> bool:
    if img.mode in {"RGBA", "LA", "PA"}:
        return True
    return img.mode == "P" and "transparency" in img.info


def alpha_channel_as_l(img: Image.Image) -> Optional[Image.Image]:
    """The alpha channel as mode L, or None if the image is opaque."""
    if not image_has_alpha(img):
        return None
    if img.mode == "RGBA":
        return img.getchannel("A")
    if img.mode == "LA":
        return img.getchannel("A")
    return img.convert("RGBA").getchannel("A")


def format_radius(radius: float) -> str:
    return f"{radius:g}"


def sidecar_marker(sidecar: Path) -> Optional[str]:
    """The sidecar's blur marker, or None when it carries none (or cannot be read)."""
    try:
        with Image.open(sidecar) as img:
            return img.info.get(MARKER_KEY)
    except Exception:
        return None


def alpha_mask(image_path: Path) -> Tuple[Optional[Image.Image], str]:
    """`(alpha as L, "")`, or `(None, skip kind)` when there is nothing worth blurring."""
    with Image.open(image_path) as img:
        alpha = alpha_channel_as_l(img)
        if alpha is None:
            return None, NO_ALPHA
        alpha.load()  # the file handle closes with the `with`; the mask is used afterwards
        lo, hi = alpha.getextrema()
    if lo == 255:
        # An all-opaque alpha is a mask of ones: it weights everything, i.e. no mask at all.
        return None, OPAQUE_ALPHA
    if hi == 0:
        return None, EMPTY_ALPHA
    return alpha, ""


def write_sidecar(image_path: Path, alpha: Image.Image, radius: float) -> Path:
    """Blur `alpha` and write it beside `image_path`, atomically."""
    sidecar = mask_path_for(image_path)
    info = PngInfo()
    info.add_text(MARKER_KEY, format_radius(radius))
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    try:
        alpha.filter(ImageFilter.GaussianBlur(radius)).save(tmp, format="PNG", pnginfo=info)
        os.replace(tmp, sidecar)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return sidecar


def process_image(image_path: Path, radius: float, overwrite: bool, dry_run: bool) -> Outcome:
    sidecar = mask_path_for(image_path)
    marker = sidecar_marker(sidecar) if sidecar.is_file() else None
    if sidecar.is_file() and marker is None and not overwrite:
        return Outcome(image_path, sidecar, PROTECTED, "existing sidecar without a blur marker")

    try:
        alpha, reason = alpha_mask(image_path)
    except Exception as exc:  # one unreadable image must not abort the batch
        return Outcome(image_path, sidecar, FAILED, f"{type(exc).__name__}: {exc}")
    if alpha is None:
        return Outcome(image_path, sidecar, reason)

    detail = f"replaces marker radius {marker}" if marker else ""
    if dry_run:
        return Outcome(image_path, sidecar, WRITTEN, detail)
    try:
        write_sidecar(image_path, alpha, radius)
    except Exception as exc:
        return Outcome(image_path, sidecar, FAILED, f"{type(exc).__name__}: {exc}")
    return Outcome(image_path, sidecar, WRITTEN, detail)


def list_images(root: Path, recursive: bool) -> List[Path]:
    """Images of one dataset folder, sidecars excluded. Non-recursive by default."""
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        p
        for p in candidates
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_mask_sidecar(p)
    )


def process_folders(
    roots: Sequence[Path],
    radius: float,
    overwrite: bool,
    recursive: bool,
    dry_run: bool,
    jobs: int = 1,
) -> List[Outcome]:
    images = [p for root in roots for p in list_images(root, recursive=recursive)]
    return process_all(images, radius, overwrite, dry_run, jobs)


def default_jobs() -> int:
    return os.cpu_count() or 1


def _process_task(task: Tuple[Path, float, bool, bool]) -> Tuple[str, str, str, str]:
    """Worker entry for one image.

    Module-level and returning plain strings: the pool may run under the `forkserver`
    start method, where a class defined in the `__main__` module (this file, run as a
    script) would not unpickle in the parent.
    """
    image_path, radius, overwrite, dry_run = task
    outcome = process_image(image_path, radius, overwrite, dry_run)
    return (str(outcome.image), str(outcome.sidecar), outcome.kind, outcome.detail)


def process_all(
    images: Sequence[Path], radius: float, overwrite: bool, dry_run: bool, jobs: int
) -> List[Outcome]:
    """One outcome per image, in the order given. `jobs <= 1` runs in this process."""
    if jobs <= 1 or len(images) <= 1:
        return [process_image(p, radius, overwrite, dry_run) for p in images]

    tasks = [(p, radius, overwrite, dry_run) for p in images]
    with ProcessPoolExecutor(max_workers=min(jobs, len(images))) as pool:
        # map keeps the order, so the report matches a sequential run line for line.
        rows = list(pool.map(_process_task, tasks))
    return [Outcome(Path(image), Path(sidecar), kind, detail) for image, sidecar, kind, detail in rows]


def report(
    outcomes: Iterable[Outcome],
    radius: float,
    folder_count: int,
    dry_run: bool,
    verbose: bool,
    jobs: int = 1,
) -> None:
    outcomes = list(outcomes)
    counts = Counter(o.kind for o in outcomes)
    if verbose:
        for outcome in outcomes:
            suffix = f"  ({outcome.detail})" if outcome.detail else ""
            print(f"  {outcome.kind:<16} {outcome.image}{suffix}")

    prefix = "would write" if dry_run else "wrote"
    print(f"\nmask_blur: radius {format_radius(radius)} px (source pixels), "
          f"{len(outcomes)} image(s) in {folder_count} folder(s), {jobs} worker(s)"
          + (", dry run - nothing written" if dry_run else ""))
    print(f"  {prefix} {counts[WRITTEN]} sidecar(s)")
    for kind in KIND_ORDER[1:]:
        if counts[kind]:
            print(f"  {counts[kind]} {KIND_LABELS[kind]}")

    if counts[WRITTEN]:
        print("\n  note: the blur is applied at the source resolution. The loader scales the mask into\n"
              "        the training bucket, so the softening that reaches the loss is radius x fit/source.")

    for kind in NAMED_KINDS:
        # A protected entry names the sidecar (the file the decision is about); the rest
        # name the image.
        named = [o.sidecar if kind == PROTECTED else o.image for o in outcomes if o.kind == kind]
        if not named:
            continue
        heading = KIND_LABELS[kind]
        if kind == PROTECTED:
            heading += "  (left as they are; --overwrite replaces them)"
        print(f"\n{heading}:")
        for path in named[:MAX_LISTED]:
            print(f"  {path}")
        if len(named) > MAX_LISTED:
            print(f"  ... and {len(named) - MAX_LISTED} more")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mask_blur.py",
        description="Blur the alpha channel of dataset images into `{stem}.mask.png` loss-mask "
                    "sidecars, softening the silhouette edge. The training images are never "
                    "modified. Existing sidecars that this tool did not write are skipped.",
    )
    parser.add_argument("directories", nargs="+", metavar="DIR",
                        help="dataset folder(s) holding the images and their .txt captions")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, metavar="PX",
                        help=f"Gaussian standard deviation in source-image pixels, "
                             f"0 < PX <= {MAX_RADIUS:g} (default {DEFAULT_RADIUS:g}; the "
                             f"intended range is {RADIUS_HINT_RANGE[0]:g}-{RADIUS_HINT_RANGE[1]:g})")
    parser.add_argument("--recursive", action="store_true",
                        help="walk subfolders too (default: the folder only, like the desktop app)")
    parser.add_argument("--overwrite", action="store_true",
                        help="also replace sidecars without this tool's blur marker (hand-painted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written without changing any file")
    parser.add_argument("-j", "--jobs", type=int, default=0, metavar="N",
                        help="worker processes, 0 = one per CPU core (default 0). Each worker "
                             "holds one decoded image, so lower it if RAM is tight")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every image with its outcome")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not (0.0 < args.radius <= MAX_RADIUS):
        print(f"Error: --radius must be > 0 and <= {MAX_RADIUS:g}", file=sys.stderr)
        return 2
    if args.jobs < 0:
        print("Error: --jobs must be >= 0 (0 means one worker per CPU core)", file=sys.stderr)
        return 2
    low, high = RADIUS_HINT_RANGE
    if not (low <= args.radius <= high):
        print(f"Note: --radius {format_radius(args.radius)} is outside the usual "
              f"{low:g}-{high:g} px range; continuing.", file=sys.stderr)

    roots = [Path(d).expanduser() for d in args.directories]
    for root in roots:
        if not root.is_dir():
            print(f"Error: not a directory: {root}", file=sys.stderr)
            return 1

    jobs = args.jobs or default_jobs()
    outcomes = process_folders(roots, args.radius, args.overwrite, args.recursive, args.dry_run, jobs)
    report(outcomes, args.radius, len(roots), args.dry_run, args.verbose, jobs)
    return 1 if any(o.kind == FAILED for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
