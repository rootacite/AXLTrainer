# upscaler.py
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import config

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError:
    RRDBNet = None
    RealESRGANer = None

def create_seamless_mask(tile_size, overlap):
    """
    Creates a gradient alpha mask for seamless tile blending.
    The edges fade to transparent to hide seams between tiles.
    """
    mask = Image.new("L", (tile_size, tile_size), 255)
    if overlap <= 0:
        return mask
        
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, 0, tile_size - 1, tile_size - 1], outline=0, width=overlap)
    
    blur_radius = overlap / 2.0
    feathered_mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    return feathered_mask

def physical_upscale_realesrgan(image, upscale_factor):
    """Uses RealESRGAN for high-quality anime upscaling before UNet refinement."""
    if RealESRGANer is None:
        print("RealESRGAN not installed. Falling back to basic LANCZOS.")
        target_width = int(image.width * upscale_factor)
        target_height = int(image.height * upscale_factor)
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    print("Loading RealESRGAN_x4plus_anime_6B model into VRAM...")
    
    # The anime_6B model explicitly uses 6 blocks instead of the default 23
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    
    # Initialize upsampler with memory-safe parameters
    upsampler = RealESRGANer(
        scale=4,
        model_path=config.REALESRGAN_MODEL_PATH,
        model=model,
        tile=512,      # Crucial: Chunks the ESRGAN processing to prevent OOM
        tile_pad=10,
        pre_pad=0,
        half=(config.TORCH_DTYPE == torch.float16),
        device=torch.device(config.DEVICE)
    )
    
    # PIL uses RGB, but cv2/RealESRGAN expects BGR. Convert using numpy slicing.
    img_array_rgb = np.array(image)
    img_array_bgr = img_array_rgb[:, :, ::-1]
    
    print(f"Executing RealESRGAN neural upscale to {upscale_factor}x...")
    output_bgr, _ = upsampler.enhance(img_array_bgr, outscale=upscale_factor)
    
    # Convert back to RGB for PIL
    output_rgb = output_bgr[:, :, ::-1]
    result_image = Image.fromarray(output_rgb)
    
    # Force VRAM cleanup to make room for UNet processing
    print("Unloading RealESRGAN from VRAM...")
    del upsampler
    del model
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        
    return result_image

def ultimate_sd_upscale(
    image, 
    img2img_pipe, 
    upscale_factor, 
    tile_size=1024, 
    overlap=64, 
    denoise_strength=0.25
):
    """
    Replicates the ComfyUI 'Ultimate SD Upscale' node logic.
    Step 1: Neural Upscale (RealESRGAN).
    Step 2: Tiled Detail Enhancement (Img2Img).
    """
    print(f"Starting Ultimate SD Upscale (Factor: {upscale_factor}x)")
    
    # 1. Base physical upscale using RealESRGAN (Anime optimized)
    base_upscaled = physical_upscale_realesrgan(image, upscale_factor)
    
    target_width = base_upscaled.width
    target_height = base_upscaled.height
    final_canvas = base_upscaled.copy()
    
    # Calculate grid dimensions
    step = tile_size - overlap
    x_tiles = math.ceil((target_width - overlap) / step)
    y_tiles = math.ceil((target_height - overlap) / step)
    
    total_tiles = x_tiles * y_tiles
    current_tile = 1
    
    blend_mask = create_seamless_mask(tile_size, overlap)
    generator = torch.Generator(device=config.DEVICE).manual_seed(config.SEED)

    # 2. Iterate through the grid and process each tile via UNet
    for y in range(y_tiles):
        for x in range(x_tiles):
            print(f"Processing Img2Img tile {current_tile}/{total_tiles}...")
            
            left = x * step
            top = y * step
            right = min(left + tile_size, target_width)
            bottom = min(top + tile_size, target_height)
            
            if right - left < tile_size:
                left = max(0, right - tile_size)
            if bottom - top < tile_size:
                top = max(0, bottom - tile_size)
                
            crop_box = (left, top, right, bottom)
            tile_image = base_upscaled.crop(crop_box)
            
            # Enhance tile details using shared UNet
            refined_tile = img2img_pipe(
                prompt=config.POSITIVE_PROMPT,
                negative_prompt=config.NEGATIVE_PROMPT,
                image=tile_image,
                strength=denoise_strength,
                num_inference_steps=config.STEPS,
                guidance_scale=config.CFG_SCALE,
                generator=generator
            ).images[0]
            
            final_canvas.paste(refined_tile, (left, top), blend_mask)
            current_tile += 1
            
    return final_canvas