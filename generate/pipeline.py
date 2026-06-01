# pipeline.py
import torch
import config


def generate_base_image(
    pipe,
    seed_override=None,
    steps_override=None,
    cfg_scale_override=None,
    return_seed: bool = False,
):
    """Generates the initial base canvas using specific seed and text configurations."""
    actual_seed = config.resolve_seed(seed_override)
    steps = config.STEPS if steps_override is None else int(steps_override)
    cfg_scale = config.CFG_SCALE if cfg_scale_override is None else float(cfg_scale_override)

    generator = torch.Generator(device=config.DEVICE).manual_seed(actual_seed)

    print(f"Using seed: {actual_seed}")

    with torch.inference_mode():
        output = pipe(
            prompt=config.POSITIVE_PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            width=config.WIDTH,
            height=config.HEIGHT,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator
        )

    image = output.images[0]
    if return_seed:
        return image, actual_seed
    return image


def generate_upscaled_image(
    pipe,
    image,
    seed_override=None,
    steps_override=None,
    cfg_scale_override=None,
):
    actual_seed = config.resolve_seed(seed_override)
    steps = config.STEPS if steps_override is None else int(steps_override)
    cfg_scale = config.CFG_SCALE if cfg_scale_override is None else float(cfg_scale_override)

    generator = torch.Generator(device=config.DEVICE).manual_seed(actual_seed)

    with torch.inference_mode():
        upscaled_image = pipe(
            prompt=config.POSITIVE_PROMPT,
            negative_prompt=config.NEGATIVE_PROMPT,
            image=image,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            strength=0.25,
        ).images[0]

    return upscaled_image