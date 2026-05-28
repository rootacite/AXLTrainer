# pipeline.py
import torch
import config

def generate_base_image(pipe):
    """Generates the initial base canvas using specific seed and text configurations."""
    
    if config.SEED == 0:
        actual_seed = torch.seed()
    else:
        actual_seed = config.SEED

    generator = torch.Generator(
        device=config.DEVICE
    ).manual_seed(actual_seed)

    print(f"Using seed: {actual_seed}")
    
    output = pipe(
        prompt=config.POSITIVE_PROMPT,
        negative_prompt=config.NEGATIVE_PROMPT,
        width=config.WIDTH,
        height=config.HEIGHT,
        num_inference_steps=config.STEPS,
        guidance_scale=config.CFG_SCALE,
        generator=generator
    )
    
    return output.images[0]

def generate_upscaled_image(pipe, image):
    upscaled_image = pipe(
        prompt=config.POSITIVE_PROMPT,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=image,
        num_inference_steps=60,
        strength=0.25, 
    ).images[0]

    return upscaled_image