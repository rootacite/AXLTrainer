# main.py
import os
import config
import model_loader
import pipeline
import detailer
import upscaler  # Import our new Ultimate SD Upscale module

def main():
    print("Initializing environment pipelines...")
    
    base_generation_pipeline = model_loader.load_base_pipeline()
    detail_inpainting_pipeline = model_loader.load_inpaint_pipeline_from_base(base_generation_pipeline)
    img2img_upscale_pipeline = model_loader.load_img2img_pipeline_from_base(base_generation_pipeline)
    
    # --- Phase 1: Base Generation ---
    print("Phase 1: Generating standard Base Canvas image...")
    raw_base_image = pipeline.generate_base_image(base_generation_pipeline)
    
    base_save_path = f"{config.OUTPUT_FILENAME_PREFIX}_1_base.png"
    raw_base_image.save(base_save_path)
    print(f"Base canvas rendering saved to: {base_save_path}")
    
    # --- Phase 2: Ultimate SD Upscale ---
    print("Phase 2: Executing Tiled Ultimate Upscale...")
    upscaled_image = upscaler.ultimate_sd_upscale(
        image=raw_base_image,
        img2img_pipe=img2img_upscale_pipeline,
        upscale_factor=2.0,      # Scale by 1.5x
        tile_size=640,          # SDXL optimal tile size
        overlap=64,              # Seam blending overlap
        denoise_strength=0.25    # Detail enhancement intensity
    )
    
    upscale_save_path = f"{config.OUTPUT_FILENAME_PREFIX}_2_upscaled.png"
    upscaled_image.save(upscale_save_path)
    print(f"Upscaled rendering saved to: {upscale_save_path}")
    
    # --- Phase 3: Regional Detailer (Face/Hands) ---
    print("Phase 3: Processing target regions through Detailer loops...")
    final_polished_image = detailer.run_detailer_pipeline(upscaled_image, detail_inpainting_pipeline)
    
    final_save_path = f"{config.OUTPUT_FILENAME_PREFIX}_3_final.png"
    final_polished_image.save(final_save_path)
    print(f"Processing complete. Enhanced output saved to: {final_save_path}")

if __name__ == "__main__":
    main()