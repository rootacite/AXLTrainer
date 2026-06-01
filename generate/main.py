import argparse
import importlib
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from PIL import Image, ImageOps

import config
import detailer
import model_loader
import pipeline
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
    parser.add_argument(
        "-qu",
        "--quick",
        type=int,
        default=None,
        metavar="N",
        help="Quick mode: generate N random-seed images, only run phase 1, keep the base model resident in VRAM.",
    )
    parser.add_argument(
        "-se",
        "--seed",
        type=int,
        default=None,
        help="Override config.SEED for this run.",
    )
    parser.add_argument(
        "-st",
        "--steps",
        type=int,
        default=None,
        help="Override config.STEPS for this run.",
    )
    parser.add_argument(
        "-cfg",
        "--cfg-scale",
        dest="cfg_scale",
        type=float,
        default=None,
        help="Override config.CFG_SCALE for this run.",
    )
    return parser.parse_args()


def is_image_capable_terminal() -> bool:
    """
    Detect terminals that are likely able to render inline images.
    Currently tuned for kitty, with a safe fallback to False.
    """
    if not sys.stdout.isatty():
        return False

    term = os.environ.get("TERM", "").lower()
    if term == "xterm-kitty":
        return True

    if os.environ.get("KITTY_WINDOW_ID"):
        return True

    return False


def show_image_in_terminal(image, title: Optional[str] = None) -> bool:
    """
    Try to display an image directly in the terminal.
    Returns True on success, False on fallback.
    """
    if not is_image_capable_terminal():
        return False

    kitty_bin = shutil.which("kitty")
    if not kitty_bin:
        return False

    tmp_path = None
    try:
        preview_image = image.copy()

        target_width = 2560
        if preview_image.width != target_width:
            target_height = round(preview_image.height * target_width / preview_image.width)
            preview_image = preview_image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        preview_image.save(tmp_path)

        cmd = [kitty_bin, "+kitten", "icat", tmp_path]
        if title:
            print(f"\n[{title}] inline preview:")
        subprocess.run(cmd, check=False)
        return True
    except Exception as exc:
        print(f"[Terminal preview skipped] {exc}")
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_and_preview(image, save_path: str, stage_name: str, seed: Optional[int] = None):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    image.save(save_path)

    if seed is not None:
        print(f"{stage_name} saved to: {save_path} (seed={seed})")
    else:
        print(f"{stage_name} saved to: {save_path}")

    try:
        with Image.open(save_path) as preview_image:
            preview_image = ImageOps.exif_transpose(preview_image).convert("RGB")

            if show_image_in_terminal(preview_image, title=stage_name):
                print(f"[{stage_name}] displayed in terminal.")
            else:
                print(f"[{stage_name}] terminal preview not available.")
    except Exception as exc:
        print(f"[{stage_name}] preview failed: {exc}")


def generate_unique_random_seeds(count: int):
    seeds = []
    seen = set()

    while len(seeds) < count:
        seed = random.randrange(0, 2**63)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)

    return seeds


def run_quick_mode(
    num_images: int,
    steps_override: Optional[int],
    cfg_scale_override: Optional[float],
):
    print("Initializing quick mode pipeline (base stage only, VRAM-resident)...")
    base_generation_pipeline = model_loader.load_base_pipeline(keep_in_vram=True)

    seeds = generate_unique_random_seeds(num_images)

    for idx, seed in enumerate(seeds, start=1):
        print(f"\nQuick generation {idx}/{num_images}")
        generated_image, actual_seed = pipeline.generate_base_image(
            base_generation_pipeline,
            seed_override=seed,
            steps_override=steps_override,
            cfg_scale_override=cfg_scale_override,
            return_seed=True,
        )

        save_path = f"{config.OUTPUT_FILENAME_PREFIX}_quick_{idx:03d}_seed{actual_seed}.png"
        save_and_preview(
            generated_image,
            save_path,
            stage_name=f"Quick generation {idx}/{num_images}",
            seed=actual_seed,
        )
        print(f"Completed {idx}/{num_images}: seed={actual_seed}, file={save_path}")

    print("\nQuick generation complete.")


def run_once(
    run_index: int,
    max_phases: int,
    seed_override: Optional[int] = None,
    steps_override: Optional[int] = None,
    cfg_scale_override: Optional[float] = None,
):
    print("Initializing environment pipelines...")

    base_generation_pipeline = model_loader.load_base_pipeline()
    detail_inpainting_pipeline = model_loader.load_inpaint_pipeline_from_base(base_generation_pipeline)
    img2img_upscale_pipeline = model_loader.load_img2img_pipeline_from_base(base_generation_pipeline)

    run_prefix = f"{config.OUTPUT_FILENAME_PREFIX}_run{run_index:03d}"

    # Phase 1
    print("Phase 1: Generating standard Base Canvas image...")
    raw_base_image, actual_seed = pipeline.generate_base_image(
        base_generation_pipeline,
        seed_override=seed_override,
        steps_override=steps_override,
        cfg_scale_override=cfg_scale_override,
        return_seed=True,
    )

    base_save_path = f"{run_prefix}_1_base.png"
    save_and_preview(raw_base_image, base_save_path, "Base canvas rendering", seed=actual_seed)

    if max_phases < 2:
        print("Stopped at phase 1 as requested.")
        return

    # Phase 2
    print("Phase 2: Executing Tiled Ultimate Upscale...")
    upscaled_image = upscaler.ultimate_sd_upscale(
        image=raw_base_image,
        img2img_pipe=img2img_upscale_pipeline,
        upscale_factor=2.0,
        tile_size=480,
        overlap=64,
        denoise_strength=0.1,
        seed_override=actual_seed,
        steps_override=steps_override,
        cfg_scale_override=cfg_scale_override,
    )

    upscale_save_path = f"{run_prefix}_2_upscaled.png"
    save_and_preview(upscaled_image, upscale_save_path, "Upscaled rendering", seed=actual_seed)

    if max_phases < 3:
        print("Stopped at phase 2 as requested.")
        return

    # Phase 3
    print("Phase 3: Processing target regions through Detailer loops...")
    final_polished_image = detailer.run_detailer_pipeline(
        upscaled_image,
        detail_inpainting_pipeline,
        seed_override=actual_seed,
        steps_override=steps_override,
        cfg_scale_override=cfg_scale_override,
    )

    final_save_path = f"{run_prefix}_3_final.png"
    save_and_preview(final_polished_image, final_save_path, "Final output", seed=actual_seed)

    print(f"Processing complete. Enhanced output saved to: {final_save_path}")


def main():
    args = parse_args()

    if args.quick is not None:
        if args.quick <= 0:
            raise SystemExit("-qu/--quick must be a positive integer.")

        if args.continuous:
            print("Quick mode ignores continuous mode and runs once.")

        if args.seed is not None:
            print("Quick mode uses random seeds per image; -se is ignored here.")

        run_quick_mode(
            num_images=args.quick,
            steps_override=args.steps,
            cfg_scale_override=args.cfg_scale,
        )
        return

    max_phases = max(1, min(args.stages, 3))

    run_index = 1
    while True:
        run_once(
            run_index,
            max_phases,
            seed_override=args.seed,
            steps_override=args.steps,
            cfg_scale_override=args.cfg_scale,
        )

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