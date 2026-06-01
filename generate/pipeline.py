# pipeline.py
import torch
import config
from prompt_utils import encode_prompt_batch

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

    prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(
        prompts=[config.POSITIVE_PROMPT],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=config.clip_skip,
        max_token_length=config.max_token_length,
        device=torch.device(config.DEVICE),
        dtype=config.TORCH_DTYPE,
    )
    negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_batch(
        prompts=[config.NEGATIVE_PROMPT],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=config.clip_skip,
        max_token_length=config.max_token_length,
        device=torch.device(config.DEVICE),
        dtype=config.TORCH_DTYPE,
    )

    print(f"Using seed: {actual_seed}")

    with torch.inference_mode():
        output = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
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

    prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(
        prompts=[config.POSITIVE_PROMPT],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=config.clip_skip,
        max_token_length=config.max_token_length,
        device=torch.device(config.DEVICE),
        dtype=config.TORCH_DTYPE,
    )

    negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_batch(
        prompts=[config.NEGATIVE_PROMPT],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=config.clip_skip,
        max_token_length=config.max_token_length,
        device=torch.device(config.DEVICE),
        dtype=config.TORCH_DTYPE,
    )

    with torch.inference_mode():
        upscaled_image = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=image,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            strength=0.25,
        ).images[0]

    return upscaled_image