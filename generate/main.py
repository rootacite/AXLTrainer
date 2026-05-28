import argparse
import importlib
import os
import sys
from typing import Tuple

import config
import model_loader
import pipeline
import detailer
import upscaler


def reload_project_modules():
    """
    Reload config and project modules so changes on disk take effect
    before the next generation round.
    """
    global config, model_loader, pipeline, detailer, upscaler

    config = importlib.reload(config)
    model_loader = importlib.reload(model_loader)
    pipeline = importlib.reload(pipeline)
    detailer = importlib.reload(detailer)
    upscaler = importlib.reload(upscaler)


def parse_args():
    parser = argparse.ArgumentParser(description="3-phase generation pipeline")
    parser.add_argument(
        "-s",
        "--stages",
        type=int,
        default=3,
        help="How many phases to run: 1 = base only, 2 = base + upscale, 3 = full pipeline (default: 3)",
    )
    parser.add_argument(
        "-c",
        "--continuous",
        action="store_true",
        help="Continuous generation mode. After each round, ask y/n; if y, reload config.py and run again.",
    )
    return parser.parse_args()


def run_once(run_index: int, max_phases: int):
    print("Initializing environment pipelines...")

    base_generation_pipeline = model_loader.load_base_pipeline()
    detail_inpainting_pipeline = model_loader.load_inpaint_pipeline_from_base(base_generation_pipeline)
    img2img_upscale_pipeline = model_loader.load_img2img_pipeline_from_base(base_generation_pipeline)

    run_prefix = f"{config.OUTPUT_FILENAME_PREFIX}_run{run_index:03d}"

    # Phase 1
    print("Phase 1: Generating standard Base Canvas image...")
    raw_base_image = pipeline.generate_base_image(base_generation_pipeline)

    base_save_path = f"{run_prefix}_1_base.png"
    raw_base_image.save(base_save_path)
    print(f"Base canvas rendering saved to: {base_save_path}")

    if max_phases < 2:
        print("Stopped at phase 1 as requested.")
        return

    # Phase 2
    print("Phase 2: Executing Tiled Ultimate Upscale...")
    upscaled_image = upscaler.ultimate_sd_upscale(
        image=raw_base_image,
        img2img_pipe=img2img_upscale_pipeline,
        upscale_factor=2.0,
        tile_size=640,
        overlap=64,
        denoise_strength=0.12,
    )

    upscale_save_path = f"{run_prefix}_2_upscaled.png"
    upscaled_image.save(upscale_save_path)
    print(f"Upscaled rendering saved to: {upscale_save_path}")

    if max_phases < 3:
        print("Stopped at phase 2 as requested.")
        return

    # Phase 3
    print("Phase 3: Processing target regions through Detailer loops...")
    final_polished_image = detailer.run_detailer_pipeline(upscaled_image, detail_inpainting_pipeline)

    final_save_path = f"{run_prefix}_3_final.png"
    final_polished_image.save(final_save_path)
    print(f"Processing complete. Enhanced output saved to: {final_save_path}")


def main():
    args = parse_args()
    max_phases = max(1, min(args.stages, 3))

    run_index = 1
    while True:
        run_once(run_index, max_phases)

        if not args.continuous:
            break

        try:
            choice = input("\nRun again? Enter y to reload config.py and continue, n to exit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break

        if choice != "y":
            print("Exit.")
            break

        print("\nReloading config.py and project modules...")
        reload_project_modules()
        run_index += 1


if __name__ == "__main__":
    main()