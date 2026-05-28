# detailer.py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import config

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def generate_feather_mask(size, border_pixels=12):
    """Generates a smooth alpha blend mask to eliminate harsh bounding box borders."""
    mask = Image.new("L", size, 255)
    draw = ImageDraw.Draw(mask)
    for i in range(border_pixels):
        alpha = int(255 * (i / border_pixels))
        draw.rectangle(
            [i, i, size[0] - i, size[1] - i],
            outline=alpha
        )
    return mask.filter(ImageFilter.GaussianBlur(border_pixels / 2))

def enhance_bounding_box(image, bounding_box, inpaint_pipe, denoise_strength, guide_resolution):
    """Performs localized regional crop, upscale, inpaint refinement, and compositing."""
    canvas_w, canvas_h = image.size
    x1, y1, x2, y2 = map(int, bounding_box)
    
    # Include safety margin padding around the feature box
    pad_w = int((x2 - x1) * 0.15)
    pad_h = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(canvas_w, x2 + pad_w)
    y2 = min(canvas_h, y2 + pad_h)
    
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return image
        
    original_crop_box = (x1, y1, x2, y2)
    cropped_patch = image.crop(original_crop_box)
    original_patch_size = cropped_patch.size
    
    # Scale patch to configured guide size to capture fine details
    resized_patch = cropped_patch.resize((guide_resolution, guide_resolution), Image.Resampling.LANCZOS)
    full_inpaint_mask = Image.new("L", (guide_resolution, guide_resolution), 255)
    
    # Run targeted regional inpainting
    generator = torch.Generator(device=config.DEVICE).manual_seed(config.SEED)
    inpainted_patch = inpaint_pipe(
        prompt=config.POSITIVE_PROMPT,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=resized_patch,
        mask_image=full_inpaint_mask,
        strength=denoise_strength,
        guidance_scale=config.CFG_SCALE,
        num_inference_steps=config.STEPS * 4,
        generator=generator
    ).images[0]
    
    # Re-scale enhanced patch back to structural layout size
    restored_patch = inpainted_patch.resize(original_patch_size, Image.Resampling.LANCZOS)
    blend_mask = generate_feather_mask(original_patch_size, border_pixels=max(4, int(min(original_patch_size) * 0.1)))
    
    # Overlay onto source image
    image.paste(restored_patch, (x1, y1), blend_mask)
    return image

def run_detailer_pipeline(image, inpaint_pipe):
    """Sequentially scans the image with configured YOLO models and fixes details."""
    if YOLO is None:
        print("Ultralytics package is missing. Bypassing detail enhancement stages.")
        return image
        
    working_image = image.copy()
    
    for layer in config.REFINEMENT_PASSES:
        print(f"Executing: {layer['name']} restoration via model {layer['model']}...")
        try:
            detector = YOLO(layer['model'])
            detection_results = detector(working_image, verbose=False)
            
            for inference in detection_results:
                detected_boxes = inference.boxes.xyxy.cpu().numpy()
                for box in detected_boxes:
                    working_image = enhance_bounding_box(
                        image=working_image,
                        bounding_box=box,
                        inpaint_pipe=inpaint_pipe,
                        denoise_strength=layer['denoise'],
                        guide_resolution=layer['guide_size']
                    )
        except Exception as error_log:
            print(f"Skipping refinement for [{layer['name']}]: {error_log}")
            
    return working_image