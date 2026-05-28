# upscaler.py

import math
from functools import lru_cache

import numpy as np
from PIL import Image
import torch

import config

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError:
    RRDBNet = None
    RealESRGANer = None


# ============================================================
# RealESRGAN Loader
# ============================================================

@lru_cache(maxsize=1)
def get_realesrgan_upsampler():
    """
    Lazily initialize and cache the RealESRGAN upsampler.
    Avoids repeated VRAM allocation and model reload overhead.
    """

    if RealESRGANer is None:
        return None

    print("Loading RealESRGAN_x4plus_anime_6B model...")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=4,
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=config.REALESRGAN_MODEL_PATH,
        model=model,
        tile=512,
        tile_pad=32,
        pre_pad=0,
        half=(config.TORCH_DTYPE == torch.bfloat16),
        device=torch.device(config.DEVICE),
    )

    return upsampler


# ============================================================
# Utility
# ============================================================

def pil_to_bgr_np(image: Image.Image):
    arr = np.array(image)
    return arr[:, :, ::-1]


def bgr_np_to_pil(arr: np.ndarray):
    rgb = arr[:, :, ::-1]
    return Image.fromarray(rgb)


def create_blend_mask(
    width,
    height,
    overlap,
    fade_left,
    fade_top,
    fade_right,
    fade_bottom,
):
    """
    Create a spatial weight mask for seamless tile blending.

    Important:
    - Outer image edges stay fully opaque.
    - Only internal tile borders are feathered.
    """

    mask = np.ones((height, width), dtype=np.float32)

    if overlap <= 0:
        return mask

    overlap = min(overlap, width // 2, height // 2)

    ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)

    if fade_left:
        mask[:, :overlap] *= ramp[None, :]

    if fade_right:
        mask[:, -overlap:] *= ramp[::-1][None, :]

    if fade_top:
        mask[:overlap, :] *= ramp[:, None]

    if fade_bottom:
        mask[-overlap:, :] *= ramp[::-1][:, None]

    return mask


def cleanup_torch():
    """
    Release cached GPU memory.
    Works for CUDA and ROCm.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Physical Upscale
# ============================================================

def physical_upscale_realesrgan(image, upscale_factor):
    """
    Perform neural upscale before diffusion refinement.
    """

    upsampler = get_realesrgan_upsampler()

    if upsampler is None:
        print("RealESRGAN unavailable. Falling back to LANCZOS.")

        target_width = int(image.width * upscale_factor)
        target_height = int(image.height * upscale_factor)

        return image.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS
        )

    print(f"Running RealESRGAN upscale ({upscale_factor}x)...")

    image_bgr = pil_to_bgr_np(image)

    output_bgr, _ = upsampler.enhance(
        image_bgr,
        outscale=upscale_factor
    )

    return bgr_np_to_pil(output_bgr)


# ============================================================
# Tile Coordinate Generator
# ============================================================

def generate_tile_coordinates(width, height, tile_size, overlap):
    """
    Generate stable tile coordinates.

    Ensures:
    - Full image coverage
    - Consistent overlap
    - No invalid edge tiles
    """

    step = tile_size - overlap

    if step <= 0:
        raise ValueError("tile_size must be larger than overlap")

    x_positions = []

    x = 0
    while True:
        if x + tile_size >= width:
            x = max(0, width - tile_size)
            x_positions.append(x)
            break

        x_positions.append(x)
        x += step

    y_positions = []

    y = 0
    while True:
        if y + tile_size >= height:
            y = max(0, height - tile_size)
            y_positions.append(y)
            break

        y_positions.append(y)
        y += step

    for top in y_positions:
        for left in x_positions:
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)

            yield left, top, right, bottom


# ============================================================
# Ultimate SD Upscale
# ============================================================

def ultimate_sd_upscale(
    image,
    img2img_pipe,
    upscale_factor,
    tile_size=1024,
    overlap=128,
    denoise_strength=0.25,
):
    """
    Ultimate SD Upscale implementation.

    Pipeline:
    1. Neural upscale using RealESRGAN
    2. Diffusion-based tiled refinement
    3. Weighted overlap blending
    """

    print(f"Starting Ultimate SD Upscale ({upscale_factor}x)")

    # --------------------------------------------------------
    # Step 1: Physical Upscale
    # --------------------------------------------------------

    base_upscaled = physical_upscale_realesrgan(
        image,
        upscale_factor
    )

    width = base_upscaled.width
    height = base_upscaled.height

    print(f"Upscaled base resolution: {width}x{height}")

    # --------------------------------------------------------
    # Step 2: Prepare accumulation buffers
    # --------------------------------------------------------

    accumulation = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    coords = list(
        generate_tile_coordinates(
            width,
            height,
            tile_size,
            overlap
        )
    )

    total_tiles = len(coords)

    generator = torch.Generator(
        device=config.DEVICE
    ).manual_seed(config.SEED)

    # --------------------------------------------------------
    # Step 3: Tile Refinement
    # --------------------------------------------------------

    for idx, (left, top, right, bottom) in enumerate(coords, start=1):

        print(f"Processing tile {idx}/{total_tiles}")

        tile = base_upscaled.crop((left, top, right, bottom))

        refined = img2img_pipe(
            prompt=config.POSITIVE_PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            image=tile,
            strength=denoise_strength,
            num_inference_steps=config.STEPS * 3,
            guidance_scale=config.CFG_SCALE,
            generator=generator,
        ).images[0]

        refined_np = np.array(refined).astype(np.float32)

        tile_w = right - left
        tile_h = bottom - top

        fade_left = left > 0
        fade_top = top > 0
        fade_right = right < width
        fade_bottom = bottom < height

        mask = create_blend_mask(
            tile_w,
            tile_h,
            overlap,
            fade_left,
            fade_top,
            fade_right,
            fade_bottom,
        )

        accumulation[top:bottom, left:right] += (
            refined_np * mask[:, :, None]
        )

        weight_sum[top:bottom, left:right] += mask

        cleanup_torch()

    # --------------------------------------------------------
    # Step 4: Normalize
    # --------------------------------------------------------

    weight_sum = np.clip(weight_sum, 1e-5, None)

    final = accumulation / weight_sum[:, :, None]

    final = np.clip(final, 0, 255).astype(np.uint8)

    return Image.fromarray(final)