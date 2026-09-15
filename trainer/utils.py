import hashlib
import math
import random
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple
import numpy as np
import torch
from PIL import Image

def parse_kv_args(arg_string: str) -> Dict[str, Any]:
    """Parses custom command-line string arguments into typed dictionaries."""
    out: Dict[str, Any] = {}
    for item in arg_string.split():
        item = item.strip().strip('"').strip("'")
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        
        if v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
            continue
        try:
            if "," in v and not v.startswith("[") and not v.startswith("{"):
                parts = [p.strip() for p in v.split(",")]
                parsed = [float(p) if "." in p or "e" in p.lower() else int(p) for p in parts]
                out[k] = tuple(parsed)
                continue
            
            out[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            out[k] = v
    return out

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

MASK_SIDECAR_SUFFIX = ".mask.png"

# Letterbox fill. Mid grey maps to ~0 after image_to_tensor's *2-1 scaling, so the pad is a
# neutral input rather than a black or white edge; the loss ignores it either way.
FIT_PAD_VALUE = 127


def is_mask_sidecar(path: Path) -> bool:
    """True for optional loss-mask files named `{stem}.mask.png`."""
    return path.name.lower().endswith(MASK_SIDECAR_SUFFIX)


def mask_path_for(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + MASK_SIDECAR_SUFFIX)


def list_images(root: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.suffix.lower() in extensions and not is_mask_sidecar(p)
        ]
    )


def image_has_alpha(img: Image.Image) -> bool:
    if img.mode in {"RGBA", "LA", "PA"}:
        return True
    return img.mode == "P" and "transparency" in img.info


def alpha_channel_as_l(img: Image.Image) -> Image.Image | None:
    """Return the alpha channel as mode L, or None if the image is opaque."""
    if not image_has_alpha(img):
        return None
    if img.mode == "RGBA":
        return img.getchannel("A")
    if img.mode == "LA":
        return img.getchannel("A")
    return img.convert("RGBA").getchannel("A")


def _mask_tensor_from_l(
    img: Image.Image,
    geom: "Geometry",
    src_w: int,
    src_h: int,
) -> torch.Tensor:
    """Resize a content mask onto the bucket canvas; the letterbox pad stays exactly 0.

    The pad is pasted *after* the resize on purpose: pre-padding the source and then resampling
    lets LANCZOS ringing leak weight into the pad rows.
    """
    mask = img.convert("L")
    if mask.size != (int(src_w), int(src_h)):
        mask = mask.resize((int(src_w), int(src_h)), Image.Resampling.NEAREST)
    if mask.size != (geom.fit_w, geom.fit_h):
        mask = mask.resize((geom.fit_w, geom.fit_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (geom.bucket_w, geom.bucket_h), 0)
    canvas.paste(mask, (geom.left, geom.top))
    arr = torch.from_numpy(np.array(canvas)).float() / 255.0
    return arr.unsqueeze(0)


def _pad_only_mask(geom: "Geometry") -> torch.Tensor:
    """All-ones content area, hard-zero pad: the mask of a sample with no content mask."""
    arr = torch.ones((1, geom.bucket_h, geom.bucket_w), dtype=torch.float32)
    if geom.fit_h < geom.bucket_h:
        arr[:, : geom.top, :] = 0.0
        arr[:, geom.top + geom.fit_h :, :] = 0.0
    if geom.fit_w < geom.bucket_w:
        arr[:, :, : geom.left] = 0.0
        arr[:, :, geom.left + geom.fit_w :] = 0.0
    return arr


def load_loss_mask(
    image_path: Path,
    bucket_w: int,
    bucket_h: int,
    src_w: int,
    src_h: int,
    geom: "Geometry | None" = None,
) -> torch.Tensor:
    """Loss weights `[1, bucket_h, bucket_w]` in `[0, 1]`.

    Content source order: `{stem}.mask.png` sidecar, else the training image's alpha channel,
    else all ones. Every sample carries the letterbox pad at weight 0, so the returned tensor is
    always a full-bucket mask; the content is resized exactly like the RGB image.
    """
    if geom is None:
        geom = fit_geometry(src_w, src_h, bucket_w, bucket_h)

    sidecar = mask_path_for(image_path)
    if sidecar.is_file():
        with Image.open(sidecar) as img:
            return _mask_tensor_from_l(img, geom, src_w, src_h)

    with Image.open(image_path) as img:
        alpha = alpha_channel_as_l(img)
    if alpha is None:
        return _pad_only_mask(geom)
    return _mask_tensor_from_l(alpha, geom, src_w, src_h)


def apply_loss_mask(loss: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Weight a reduction='none' spatial loss by a 0–1 mask.

    `loss` is `[B, C, h, w]`. `mask` is `[B, 1, H, W]` or `[B, H, W]` at pixel
    or latent size; area-interpolated to `loss`'s spatial size (VAE is 8×).
    """
    if mask is None:
        return loss
    weight = mask
    if weight.dim() == 3:
        weight = weight.unsqueeze(1)
    weight = weight.to(device=loss.device, dtype=loss.dtype)
    if weight.shape[-2:] != loss.shape[-2:]:
        weight = torch.nn.functional.interpolate(
            weight, size=loss.shape[-2:], mode="area"
        )
    return loss * weight

def read_caption(image_path: Path, caption_extension: str = ".txt") -> str:
    caption_file = image_path.with_suffix(caption_extension)
    if caption_file.exists():
        return caption_file.read_text(encoding="utf-8", errors="ignore").strip()
    return image_path.stem.replace("_", " ")

def shuffle_caption(text: str, keep_tokens: int, rng: random.Random) -> str:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) <= keep_tokens:
        return ", ".join(parts)
    head = parts[:keep_tokens]
    tail = parts[keep_tokens:]
    rng.shuffle(tail)
    return ", ".join(head + tail)

def round_to_step(value: int, step: int) -> int:
    return max(step, (value // step) * step)


def _nearest_step(value: float, step: int) -> int:
    return max(step, int(round(value / step)) * step)


class Geometry(NamedTuple):
    """Where the (scaled) image content lands inside its bucket."""

    bucket_w: int
    bucket_h: int
    fit_w: int
    fit_h: int
    left: int
    top: int

    @property
    def pad_area(self) -> float:
        return 1.0 - (self.fit_w * self.fit_h) / float(self.bucket_w * self.bucket_h)


def fit_geometry(src_w: int, src_h: int, bucket_w: int, bucket_h: int) -> Geometry:
    """Scale the whole image into the bucket (contain, not cover) and centre it.

    Pure function of the source and bucket sizes: `pick_bucket_size` already refuses to build a
    bucket axis larger than the source (`bucket_no_upscale`), so the fit scale is <= 1 except for
    sources thinner than one step, which that rule floors at one step.
    """
    src_w, src_h = int(src_w), int(src_h)
    bucket_w, bucket_h = int(bucket_w), int(bucket_h)
    if src_w <= 0 or src_h <= 0 or bucket_w <= 0 or bucket_h <= 0:
        return Geometry(bucket_w, bucket_h, bucket_w, bucket_h, 0, 0)
    scale = min(bucket_w / src_w, bucket_h / src_h)
    fit_w = min(bucket_w, max(1, int(round(src_w * scale))))
    fit_h = min(bucket_h, max(1, int(round(src_h * scale))))
    return Geometry(
        bucket_w=bucket_w,
        bucket_h=bucket_h,
        fit_w=fit_w,
        fit_h=fit_h,
        left=(bucket_w - fit_w) // 2,
        top=(bucket_h - fit_h) // 2,
    )


def fit_to_bucket(image: Image.Image, geom: Geometry, fill: int = FIT_PAD_VALUE) -> Image.Image:
    """Resize `image` into `geom`'s content area and pad the rest with `fill`.

    Only `RGB` and `L` are accepted: the dataset feeds the RGB image and the mask through this
    same helper so both land on identical pixels.
    """
    if image.mode not in ("RGB", "L"):
        raise ValueError(f"fit_to_bucket expects RGB or L, got {image.mode}")
    resized = (
        image
        if image.size == (geom.fit_w, geom.fit_h)
        else image.resize((geom.fit_w, geom.fit_h), Image.Resampling.LANCZOS)
    )
    if resized.size == (geom.bucket_w, geom.bucket_h):
        return resized
    color = (fill, fill, fill) if image.mode == "RGB" else fill
    canvas = Image.new(image.mode, (geom.bucket_w, geom.bucket_h), color)
    canvas.paste(resized, (geom.left, geom.top))
    return canvas


def pick_bucket_size(
    w: int,
    h: int,
    min_reso: int,
    max_reso: int,
    step: int,
    no_upscale: bool,
    area: int | None = None,
) -> Tuple[int, int]:
    """Bucket whose aspect follows the image and whose pixel count aims at `area`.

    An area budget (`train_resolution ** 2`) is what lets tall images get tall buckets: pinning
    the short side to `min_reso` instead forced every portrait into one short bucket, where the
    long side hit `max_reso` and the overflow was cropped away. `min_reso`/`max_reso` are axis
    clamps here, and `no_upscale` never builds an axis larger than the source's own (sources
    thinner than one `step` are the exception, floored at one step).
    """
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return min_reso, min_reso
    if not area:
        # Callers normally pass `train_resolution ** 2`; this fallback is within ~2 % of the
        # shipped 1024² for the shipped clamps.
        area = int(min_reso) * int(max_reso)

    ar = w / h
    bucket_w = _nearest_step(math.sqrt(area * ar), step)
    bucket_h = _nearest_step(math.sqrt(area / ar), step)

    bucket_w = max(int(min_reso), min(bucket_w, int(max_reso)))
    bucket_h = max(int(min_reso), min(bucket_h, int(max_reso)))

    if no_upscale:
        bucket_w = min(bucket_w, round_to_step(w, step))
        bucket_h = min(bucket_h, round_to_step(h, step))

    return int(bucket_w), int(bucket_h)

def resize_and_center_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Cover-scale + centre crop. Not used by the training path (that is `fit_to_bucket`); kept
    for the mask-verification harness's independent implementation and `fixes/` probes."""
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return image.crop((left, top, left + target_w, top + target_h))

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(image)).float() / 255.0
    if arr.ndim == 2:
        arr = arr.unsqueeze(-1)
    arr = arr.permute(2, 0, 1)
    return arr * 2.0 - 1.0

def build_time_ids(
    original_size: Tuple[int, int],
    crop_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vals = list(original_size) + list(crop_top_left) + list(target_size)
    return torch.tensor(vals, device=device, dtype=dtype)
