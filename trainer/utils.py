import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
    bucket_w: int,
    bucket_h: int,
    src_w: int,
    src_h: int,
) -> torch.Tensor:
    mask = img.convert("L")
    if mask.size != (int(src_w), int(src_h)):
        mask = mask.resize((int(src_w), int(src_h)), Image.Resampling.NEAREST)
    mask = resize_and_center_crop(mask, bucket_w, bucket_h)
    arr = torch.from_numpy(np.array(mask)).float() / 255.0
    return arr.unsqueeze(0)


def load_loss_mask(
    image_path: Path,
    bucket_w: int,
    bucket_h: int,
    src_w: int,
    src_h: int,
) -> torch.Tensor:
    """Loss weights `[1, bucket_h, bucket_w]` in `[0, 1]`.

    Order: `{stem}.mask.png` sidecar if present; else the training image's
    alpha channel; else all ones (unmasked MSE). Crop/resize matches the RGB
    image so the mask stays aligned after bucketing.
    """
    sidecar = mask_path_for(image_path)
    if sidecar.is_file():
        with Image.open(sidecar) as img:
            return _mask_tensor_from_l(img, bucket_w, bucket_h, src_w, src_h)

    with Image.open(image_path) as img:
        alpha = alpha_channel_as_l(img)
        if alpha is None:
            return torch.ones((1, int(bucket_h), int(bucket_w)), dtype=torch.float32)
        return _mask_tensor_from_l(alpha, bucket_w, bucket_h, src_w, src_h)


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

def pick_bucket_size(
    w: int, h: int, min_reso: int, max_reso: int, step: int, no_upscale: bool
) -> Tuple[int, int]:
    if w <= 0 or h <= 0:
        return min_reso, min_reso

    ar = w / h
    if ar >= 1.0:
        bucket_h = min_reso if not (no_upscale and h < min_reso) else round_to_step(h, step)
        bucket_w = int(round(bucket_h * ar))
    else:
        bucket_w = min_reso if not (no_upscale and w < min_reso) else round_to_step(w, step)
        bucket_h = int(round(bucket_w / ar))

    bucket_w = round_to_step(bucket_w, step)
    bucket_h = round_to_step(bucket_h, step)
    bucket_w = max(step, min(bucket_w, max_reso))
    bucket_h = max(step, min(bucket_h, max_reso))

    if (ar >= 1.0 and bucket_w < bucket_h) or (ar < 1.0 and bucket_h < bucket_w):
        bucket_w, bucket_h = bucket_h, bucket_w

    return bucket_w, bucket_h

def resize_and_center_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
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
