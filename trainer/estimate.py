"""How many steps a run will take: the figure the Utils → Training section shows while you edit.

Torch-free on purpose (like `runs.py`, `genjob.py` and `evaluation.py`), because api.py serves it
and the helper process must not pay for a torch import to answer a form question.

The count is the number of images the trainer's dataset would draw in one epoch, and it mirrors
`trainer/utils.list_images` for that: the same extensions, the same `*.mask.png` exclusion and the
same recursive walk over each `[[environment.train_data]]` folder — a hand-written mirror, not an
import, because `utils.py` pulls torch in. The step arithmetic lives on the Ranko side
(`model/StepEstimate.kt`), which is what makes epoch / batch / GA edits cost no IPC at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# Same set `trainer/utils.list_images` walks, and the same mask exclusion.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASK_SUFFIX = ".mask.png"


def is_mask_sidecar(path: Path) -> bool:
    return path.name.lower().endswith(MASK_SUFFIX)


def count_images(root: Path) -> int:
    """Images under `root`, recursively: what `LoraImageDataset` would take from that folder."""
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not is_mask_sidecar(path)
    )


def _entry(entry: Any) -> tuple[str, int]:
    """One requested folder as `(path, repeat)`; a `TrainDataEntry` or a plain mapping."""
    if isinstance(entry, Mapping):
        path = str(entry.get("path") or "").strip()
        raw_repeat = entry.get("repeat", 1)
    else:
        path = str(getattr(entry, "path", "") or "").strip()
        raw_repeat = getattr(entry, "repeat", 1)
    try:
        repeat = int(raw_repeat)
    except (TypeError, ValueError):
        repeat = 1
    return path, max(1, repeat)


def count_train_images(entries: Iterable[Any]) -> dict[str, Any]:
    """Image counts per training folder, plus the per-epoch totals a step estimate is built from.

    Each entry answers `{path, repeat, images, error}`; a folder that is missing or unreadable
    carries its reason and counts as zero, so one bad path cannot take the whole estimate down.
    `samples` is the per-epoch figure with repeats applied — what `LoraImageDataset.total_samples`
    holds at run time.
    """
    rows: list[dict[str, Any]] = []
    images_total = 0
    samples_total = 0
    for raw in entries:
        path, repeat = _entry(raw)
        images = 0
        error: Optional[str] = None
        if not path:
            error = "no path"
        else:
            root = Path(path).expanduser()
            if not root.is_dir():
                error = "not a directory"
            else:
                try:
                    images = count_images(root)
                except OSError as exc:  # an unreadable tree is this folder's answer, not the form's
                    error = str(exc)
        rows.append(
            {
                "path": path,
                "repeat": int(repeat),
                "images": int(images),
                "error": error,
            }
        )
        images_total += images
        samples_total += images * repeat
    return {
        "entries": rows,
        "images": int(images_total),
        "samples": int(samples_total),
    }
