#!/usr/bin/env python3
"""Copy a region painted by hand on one image into other images of the same size.

Some pictures in a dataset differ from each other in one area only - the same pose
with a different face, one background swapped, a prop moved. Redoing that edit by
hand in an image editor, once per file, is tedious; this tool copies it instead.
`mask_blur.py` softens an alpha-derived loss mask and never touches pixels; this
tool transfers pixels and was written for the other direction.

Usage:

    python tools/paint_transfer.py SOURCE TARGET [TARGET ...]

Every image - source and targets - must have exactly the same pixel size, and a
mismatch is refused before the window opens: the copy is a straight pixel
replacement with no scaling or alignment step, so different sizes have no
well-defined meaning here.

The window shows the source fitted to it (never enlarged past 1:1). Left drag
paints, right drag erases, and the painted area is drawn as a semi-transparent
black overlay (opacity `DISPLAY_ALPHA`) so the pixels under it stay readable. The
overlay shows a 0-255 coverage mask: the brush is full strength inside
`1 - FALLOFF_FRACTION` of its radius and ramps to zero over the rest, so a
transfer that stops short of an edge blends into it instead of cutting it. Wheel
or `[` / `]` resizes the brush, Ctrl+Z undoes a stroke, Clear empties the mask.
Apply composites every target as `target * (1 - mask) + source * mask`.

Nothing is written until every target is composited in memory, and the targets
are copied to /tmp/axlpaint-backup-<timestamp>/ before the first write, so a
failure while reading or compositing leaves the dataset untouched and a failure
while writing is reported per file (exit code 1) with the backup already in
place. Exit codes: 0 written (or cancelled), 1 failed, 2 usage.

Encoding: PNG is lossless. A JPEG target is re-encoded with its own quantization
tables, sampling and progressive flag (`quality="keep"`), so the areas outside
the mask stay within a level or two of the original instead of being requantized
at a fixed quality; WebP is re-encoded at quality 95, because Pillow does not
report the original encoding. EXIF and ICC profiles are carried over. A target
whose mode Pillow cannot composite in is written back as RGB (`P`) or RGBA
(`LA`), a greyscale `L` target stays greyscale.

Needs PyQt6, Pillow and numpy; the `axl` env has all three.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Share of the brush radius that is drawn already at full strength; the rest is
# the ramp to zero coverage.
FALLOFF_FRACTION = 0.35
# Opacity of the painted preview (0-255). Low enough to read the pixels under it.
DISPLAY_ALPHA = 118
# Distance between two brush stamps along one stroke, as a fraction of the radius.
STAMP_SPACING = 0.25
MIN_RADIUS = 1
DEFAULT_RADIUS = 48
UNDO_LIMIT = 40
BACKUP_ROOT = Path("/tmp")
BACKUP_PREFIX = "axlpaint-backup-"
# Margin between the drawn image and the widget edge, in device pixels.
VIEW_MARGIN = 12


class ToolError(Exception):
    """A problem the user can fix: a missing file, a size mismatch, an unwritable target."""


# --------------------------------------------------------------------------- Qt


def pil_to_qimage(image: Image.Image) -> QImage:
    """RGBA8888 QImage that owns its own bytes: QImage does not copy a foreign buffer."""
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    return QImage(
        data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888
    ).copy()


class PaintCanvas(QWidget):
    """The source image plus the coverage mask painted over it.

    `mask` is the single source of truth (uint8, one byte per pixel, image-sized).
    `_preview` is a display copy of it: black with alpha = mask * DISPLAY_ALPHA / 255,
    which is what makes the painted area read as semi-transparent black. It is
    written through `_preview_rows`, a numpy view of the QImage's own buffer, so
    a stroke only touches the few rows and columns it covers.
    """

    maskChanged = pyqtSignal()
    radiusChanged = pyqtSignal(int)

    def __init__(self, image: Image.Image, radius: int = DEFAULT_RADIUS, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._source = pil_to_qimage(image)
        self._width, self._height = image.size
        self.mask = np.zeros((self._height, self._width), dtype=np.uint8)
        self._preview = QImage(
            self._width, self._height, QImage.Format.Format_ARGB32_Premultiplied
        )
        self._preview.fill(0)
        self._preview_rows = self._buffer_rows(self._preview)

        self._max_radius = max(MIN_RADIUS, max(image.size))
        self._radius = self.clamp_radius(radius)
        self._undo: List[List[Tuple[Tuple[int, int, int, int], np.ndarray]]] = []
        self._stroke: Optional[List[Tuple[Tuple[int, int, int, int], np.ndarray]]] = None
        self._erase = False
        self._last_point: Optional[Tuple[float, float]] = None
        self._cursor: Optional[QPointF] = None
        self._scale = 1.0
        self._origin = QPointF(0.0, 0.0)
        self._layout_view()

    # -- geometry ----------------------------------------------------------

    @staticmethod
    def _buffer_rows(image: QImage) -> np.ndarray:
        """Numpy view over a QImage's pixels, one row per scanline (stride-padded)."""
        pointer = image.bits()
        pointer.setsize(image.sizeInBytes())
        return np.frombuffer(pointer, dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine()
        )

    def _layout_view(self) -> None:
        available_w = max(1.0, self.width() - 2 * VIEW_MARGIN)
        available_h = max(1.0, self.height() - 2 * VIEW_MARGIN)
        self._scale = min(available_w / self._width, available_h / self._height, 1.0)
        self._origin = QPointF(
            (self.width() - self._width * self._scale) / 2.0,
            (self.height() - self._height * self._scale) / 2.0,
        )

    def _image_rect(self) -> QRectF:
        return QRectF(
            self._origin.x(),
            self._origin.y(),
            self._width * self._scale,
            self._height * self._scale,
        )

    def _to_image(self, position: QPointF) -> Tuple[float, float]:
        """Widget point to image point, clamped to one brush radius outside the image."""
        x = (position.x() - self._origin.x()) / self._scale
        y = (position.y() - self._origin.y()) / self._scale
        limit = float(self._radius)
        return (
            min(max(x, -limit), self._width + limit),
            min(max(y, -limit), self._height + limit),
        )

    # -- brush -------------------------------------------------------------

    def radius(self) -> int:
        return self._radius

    def max_radius(self) -> int:
        return self._max_radius

    def clamp_radius(self, radius: int) -> int:
        return int(max(MIN_RADIUS, min(int(radius), self._max_radius)))

    def set_radius(self, radius: int) -> None:
        radius = self.clamp_radius(radius)
        if radius == self._radius:
            return
        self._radius = radius
        self.radiusChanged.emit(radius)
        self.update()

    def painted_fraction(self) -> float:
        return float(np.count_nonzero(self.mask)) / self.mask.size

    # -- mask edits --------------------------------------------------------

    def _paint_stamp(self, cx: float, cy: float, erase: bool) -> Optional[Tuple[int, int, int, int]]:
        """Apply one brush stamp; returns the image rectangle it changed, if any."""
        radius = float(self._radius)
        x0 = max(0, int(math.floor(cx - radius)))
        y0 = max(0, int(math.floor(cy - radius)))
        x1 = min(self._width, int(math.ceil(cx + radius)) + 1)
        y1 = min(self._height, int(math.ceil(cy + radius)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        xs = np.arange(x0, x1, dtype=np.float32) + 0.5 - cx
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5 - cy
        distance = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        coverage = np.clip(
            (radius - distance) / max(1.0, radius * FALLOFF_FRACTION), 0.0, 1.0
        )
        rows = np.flatnonzero(coverage.any(axis=1))
        if rows.size == 0:
            return None
        cols = np.flatnonzero(coverage.any(axis=0))
        if cols.size == 0:
            return None
        top, bottom = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
        left, right = x0 + int(cols[0]), x0 + int(cols[-1]) + 1
        coverage = coverage[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

        region = self.mask[top:bottom, left:right]
        previous = region.copy()
        if erase:
            faded = region.astype(np.float32) * (1.0 - coverage)
            region[:] = np.clip(faded, 0.0, 255.0).astype(np.uint8)
        else:
            np.maximum(region, (coverage * 255.0).astype(np.uint8), out=region)
        if np.array_equal(previous, region):
            return None
        if self._stroke is not None:
            self._stroke.append(((left, top, right, bottom), previous))
        return (left, top, right, bottom)

    def _stroke_to(self, x: float, y: float) -> Optional[Tuple[int, int, int, int]]:
        """Stamp from the previous point to (x, y) and return the union rectangle."""
        if self._last_point is None:
            return None
        px, py = self._last_point
        spacing = max(0.5, self._radius * STAMP_SPACING)
        steps = max(1, int(math.ceil(math.hypot(x - px, y - py) / spacing)))
        box: Optional[Tuple[int, int, int, int]] = None
        for step in range(1, steps + 1):
            t = step / steps
            hit = self._paint_stamp(px + (x - px) * t, py + (y - py) * t, self._erase)
            if hit is not None:
                box = hit if box is None else _union(box, hit)
        self._last_point = (x, y)
        return box

    def _refresh_preview(self, box: Tuple[int, int, int, int]) -> None:
        left, top, right, bottom = box
        block = self.mask[top:bottom, left:right].astype(np.uint16)
        block = (block * DISPLAY_ALPHA // 255).astype(np.uint8)
        self._preview_rows[top:bottom, left * 4 + 3 : right * 4 : 4] = block

    def undo(self) -> None:
        if self._stroke is not None or not self._undo:
            return
        boxes = []
        for box, previous in reversed(self._undo.pop()):
            left, top, right, bottom = box
            self.mask[top:bottom, left:right] = previous
            boxes.append(box)
        if not boxes:
            return
        box = boxes[0]
        for extra in boxes[1:]:
            box = _union(box, extra)
        self._refresh_preview(box)
        self.maskChanged.emit()
        self.update()

    def clear_mask(self) -> None:
        if not self.mask.any():
            return
        self._undo.append([((0, 0, self._width, self._height), self.mask.copy())])
        self.mask[:] = 0
        self._refresh_preview((0, 0, self._width, self._height))
        self.maskChanged.emit()
        self.update()

    # -- events ------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        self._layout_view()
        super().resizeEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._erase = event.button() == Qt.MouseButton.RightButton
        self._stroke = []
        self._undo.append(self._stroke)
        del self._undo[:-UNDO_LIMIT]
        x, y = self._to_image(event.position())
        box = self._paint_stamp(x, y, self._erase)
        self._last_point = (x, y)
        self._cursor = event.position()
        if box is not None:
            self._refresh_preview(box)
        self.maskChanged.emit()
        self.update()

    def mouseMoveEvent(self, event) -> None:
        self._cursor = event.position()
        if self._last_point is not None:
            x, y = self._to_image(event.position())
            box = self._stroke_to(x, y)
            if box is not None:
                self._refresh_preview(box)
                self.maskChanged.emit()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        if self._stroke is not None and not self._stroke:
            self._undo.pop()
        self._stroke = None
        self._last_point = None
        self.maskChanged.emit()
        self.update()

    def leaveEvent(self, event) -> None:
        self._cursor = None
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.set_radius(self._radius + int(math.copysign(max(1, self._radius // 6), steps)))
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 28))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = self._image_rect()
        painter.drawImage(rect, self._source)
        painter.drawImage(rect, self._preview)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1))
        painter.drawRect(rect.adjusted(-0.5, -0.5, -0.5, -0.5))
        if self._cursor is not None:
            radius = self._radius * self._scale
            painter.setPen(QPen(QColor(0, 0, 0, 190), 3))
            painter.drawEllipse(self._cursor, radius, radius)
            painter.setPen(QPen(QColor(255, 255, 255, 230), 1))
            painter.drawEllipse(self._cursor, radius, radius)
        painter.end()


def _union(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


class PaintTransferDialog(QDialog):
    """Source preview, brush controls and the target list. `canvas.mask` is the result."""

    def __init__(
        self,
        source_path: Path,
        source_image: Image.Image,
        targets: Sequence[Path],
        radius: int = DEFAULT_RADIUS,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Paint region transfer — {source_path.name}")
        self.canvas = PaintCanvas(source_image, radius, self)
        self._syncing = False
        self._build_ui(source_path, source_image, targets)
        self.canvas.maskChanged.connect(self._sync_state)
        self.canvas.radiusChanged.connect(self._sync_radius)
        self._sync_radius(self.canvas.radius())
        self._sync_state()
        width = min(1500, max(900, source_image.width + 360))
        height = min(1000, max(620, source_image.height + 80))
        self.resize(width, height)

    def _build_ui(
        self, source_path: Path, source_image: Image.Image, targets: Sequence[Path]
    ) -> None:
        panel = QWidget(self)
        panel.setFixedWidth(300)
        column = QVBoxLayout(panel)
        column.setSpacing(8)

        heading = QLabel("Source")
        heading.setStyleSheet("font-weight: bold;")
        column.addWidget(heading)
        about = QLabel(
            f"{source_path.name}\n{source_image.width} × {source_image.height}\n{source_path.parent}"
        )
        about.setWordWrap(True)
        about.setToolTip(str(source_path))
        column.addWidget(about)

        column.addWidget(QLabel(f"Targets ({len(targets)})"))
        listing = QListWidget(self)
        listing.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        listing.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for path in targets:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            listing.addItem(item)
        listing.setMaximumHeight(170)
        column.addWidget(listing)

        column.addWidget(QLabel("Brush radius (px)"))
        brush_row = QHBoxLayout()
        self._radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._radius_slider.setRange(0, 1000)
        self._radius_slider.valueChanged.connect(self._on_slider)
        self._radius_spin = QSpinBox()
        self._radius_spin.setRange(MIN_RADIUS, self.canvas.max_radius())
        self._radius_spin.valueChanged.connect(self._on_spin)
        brush_row.addWidget(self._radius_slider, 1)
        brush_row.addWidget(self._radius_spin)
        column.addLayout(brush_row)

        edit_row = QHBoxLayout()
        undo = QPushButton("Undo")
        undo.clicked.connect(self.canvas.undo)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.canvas.clear_mask)
        edit_row.addWidget(undo)
        edit_row.addWidget(clear)
        column.addLayout(edit_row)

        self._status = QLabel("Painted: 0.00%")
        column.addWidget(self._status)

        hint = QLabel(
            "Left drag paints, right drag erases.\n"
            "Wheel or [ ] resizes the brush, Ctrl+Z undoes a stroke.\n"
            "The painted area is copied from the source into every target; the "
            "originals are backed up to /tmp first."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a8a94;")
        column.addWidget(hint)
        column.addStretch(1)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self._apply = QPushButton(
            f"Apply to {len(targets)} target{'s' if len(targets) != 1 else ''}"
        )
        self._apply.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self._apply)
        column.addLayout(buttons)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.canvas, 1)
        row.addWidget(panel)

    # -- control sync ------------------------------------------------------

    def _slider_to_radius(self, value: int) -> int:
        span = self.canvas.max_radius() - MIN_RADIUS
        return MIN_RADIUS + int(round(span * (value / 1000.0) ** 2))

    def _radius_to_slider(self, radius: int) -> int:
        span = self.canvas.max_radius() - MIN_RADIUS
        if span <= 0:
            return 0
        return int(round(1000.0 * math.sqrt(max(0.0, (radius - MIN_RADIUS) / span))))

    def _on_slider(self, value: int) -> None:
        if not self._syncing:
            self.canvas.set_radius(self._slider_to_radius(value))

    def _on_spin(self, value: int) -> None:
        if not self._syncing:
            self.canvas.set_radius(value)

    def _sync_radius(self, radius: int) -> None:
        self._syncing = True
        try:
            self._radius_spin.setValue(radius)
            self._radius_slider.setValue(self._radius_to_slider(radius))
        finally:
            self._syncing = False

    def _sync_state(self) -> None:
        painted = self.canvas.painted_fraction()
        self._status.setText(f"Painted: {painted * 100:.2f}%")
        self._apply.setEnabled(painted > 0)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.canvas.undo()
            return
        if key == Qt.Key.Key_BracketLeft:
            self.canvas.set_radius(self.canvas.radius() - max(1, self.canvas.radius() // 8))
            return
        if key == Qt.Key.Key_BracketRight:
            self.canvas.set_radius(self.canvas.radius() + max(1, self.canvas.radius() // 8))
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------- transfer


def open_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    if path.suffix.lower() not in Image.registered_extensions():
        raise ToolError(f"unsupported image type: {path}")
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise ToolError(f"cannot read {path}: {exc}") from exc
    return image


def load_images(
    source: Path, targets: Sequence[Path]
) -> Tuple[Image.Image, List[Tuple[Path, Image.Image]]]:
    """Open the source and every target; all sizes must match exactly."""
    source_image = open_image(source)
    size = source_image.size
    loaded: List[Tuple[Path, Image.Image]] = []
    mismatched: List[Tuple[Path, Tuple[int, int]]] = []
    for path in targets:
        image = open_image(path)
        loaded.append((path, image))
        if image.size != size:
            mismatched.append((path, image.size))
    if mismatched:
        lines = [
            f"every image must have the same size; {source} is {size[0]}x{size[1]}"
        ]
        lines += [f"  {path}: {w}x{h}" for path, (w, h) in mismatched]
        raise ToolError("\n".join(lines))
    return source_image, loaded


def work_mode(image: Image.Image) -> str:
    bands = image.getbands()
    if "A" in bands:
        return "RGBA"
    if bands == ("L",):
        return "L"
    return "RGB"


def compose_target(
    source_image: Image.Image, target_image: Image.Image, mask_image: Image.Image
) -> Image.Image:
    """`target * (1 - mask) + source * mask`, in a mode both images support."""
    mode = work_mode(target_image)
    return Image.composite(source_image.convert(mode), target_image.convert(mode), mask_image)


def save_image(image: Image.Image, path: Path, template: Image.Image) -> None:
    fmt = Image.registered_extensions().get(path.suffix.lower())
    if fmt is None:
        raise ToolError(f"unsupported image type: {path}")
    kwargs: Dict[str, object] = {}
    if template.info.get("icc_profile"):
        kwargs["icc_profile"] = template.info["icc_profile"]
    if template.info.get("exif"):
        kwargs["exif"] = template.info["exif"]
    if fmt == "JPEG":
        tables = getattr(template, "quantization", None)
        if tables:
            # Pillow's quality="keep" reads the tables and sampling off the image
            # being saved and refuses unless it carries format="JPEG".
            image.format = "JPEG"
            image.quantization = tables
            layer = getattr(template, "layer", None)
            if layer:
                image.layer = layer
            kwargs["quality"] = "keep"
        else:
            kwargs["quality"] = 95
        if "progressive" in template.info:
            kwargs["progressive"] = bool(template.info["progressive"])
    elif fmt == "WEBP":
        kwargs["quality"] = 95
    image.save(path, format=fmt, **kwargs)


def backup_targets(paths: Iterable[Path], stamp: str) -> Tuple[Path, List[Path]]:
    """Copy every target into /tmp/<prefix><stamp>/, keeping names unique."""
    root = BACKUP_ROOT / f"{BACKUP_PREFIX}{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    seen: Counter = Counter()
    written: List[Path] = []
    for path in paths:
        seen[path.name] += 1
        name = path.name
        if seen[path.name] > 1:
            name = f"{path.stem}~{seen[path.name]}{path.suffix}"
        destination = root / name
        shutil.copy2(path, destination)
        written.append(destination)
    return root, written


# -------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paint_transfer.py",
        description=(
            "Paint a region on SOURCE and copy it into images of the same size "
            "(originals backed up to /tmp)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  python tools/paint_transfer.py face_a.png face_b.png face_c.jpg\n"
        ),
    )
    parser.add_argument("source", metavar="SOURCE", type=Path, help="image to paint on")
    parser.add_argument(
        "targets",
        metavar="TARGET",
        nargs="+",
        type=Path,
        help="images that receive the painted region (same size as SOURCE)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_RADIUS,
        help=f"initial brush radius in image pixels (default {DEFAULT_RADIUS})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser()
    targets = [path.expanduser() for path in args.targets]

    try:
        source_image, loaded = load_images(source, targets)
    except ToolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.radius < MIN_RADIUS:
        print(f"Error: --radius must be >= {MIN_RADIUS}", file=sys.stderr)
        return 2

    app = QApplication.instance() or QApplication([sys.argv[0]])
    dialog = PaintTransferDialog(source, source_image, targets, args.radius)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        print("Cancelled; nothing written.")
        return 0

    mask = dialog.canvas.mask
    covered = float(np.count_nonzero(mask)) / mask.size
    if covered == 0.0:
        print("Nothing painted; nothing written.", file=sys.stderr)
        return 1

    mask_image = Image.fromarray(mask)
    try:
        composed = [
            (path, compose_target(source_image, image, mask_image), image)
            for path, image in loaded
        ]
    except Exception as exc:
        print(f"Error: could not composite the targets: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        backup_dir, backups = backup_targets(targets, stamp)
    except OSError as exc:
        print(f"Error: could not back up the targets to {BACKUP_ROOT}: {exc}", file=sys.stderr)
        return 1

    failures: List[Tuple[Path, Exception]] = []
    written: List[Path] = []
    for path, image, template in composed:
        try:
            save_image(image, path, template)
        except Exception as exc:
            failures.append((path, exc))
        else:
            written.append(path)

    print(f"Source : {source} ({source_image.width}x{source_image.height})")
    print(f"Painted: {covered * 100:.2f}% of the image")
    print(f"Backup : {backup_dir} ({len(backups)} files)")
    for path in written:
        print(f"Wrote  : {path}")
    for path, exc in failures:
        print(f"Error: could not write {path}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
