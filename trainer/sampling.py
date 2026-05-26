from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image

from config import TrainConfig
from text_processing import encode_prompt_batch
from trainer.env import flush_memory


@torch.inference_mode()
def generate_sample_image(
    *,
    accelerator: Accelerator,
    pipe,
    trained_unet: torch.nn.Module,
    trained_te1: torch.nn.Module,
    trained_te2: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    output_dir_base: Path,
) -> None:
    """Generate and save sample images from the current LoRA weights."""
    if not accelerator.is_main_process:
        return

    flush_memory(device)

    pipe = pipe.to(device)
    
    sampler_kwargs = {"timestep_spacing": "linspace"}
    if cfg.is_vpred:
        sampler_kwargs["prediction_type"] = "v_prediction"
        sampler_kwargs["rescale_betas_zero_snr"] = True
    else:
        sampler_kwargs["prediction_type"] = "epsilon"

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        **sampler_kwargs
    )

    pipe.unet = trained_unet
    pipe.text_encoder = trained_te1
    pipe.text_encoder_2 = trained_te2

    pipe.unet.eval()
    pipe.text_encoder.to(device=device, dtype=dtype).eval()
    pipe.text_encoder_2.to(device=device, dtype=dtype).eval()

    vae_dtype = torch.bfloat16
    pipe.vae.to(device=device, dtype=vae_dtype).eval()

    if hasattr(pipe.vae.config, "force_upcast"):
        pipe.vae.config.force_upcast = False

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(
        prompts=[cfg.sample_prompts],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=cfg.clip_skip,
        max_token_length=cfg.max_token_length,
        device=device,
        dtype=dtype,
    )
    negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_batch(
        prompts=[cfg.sample_negative],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=cfg.clip_skip,
        max_token_length=cfg.max_token_length,
        device=device,
        dtype=dtype,
    )

    sample_dir = output_dir_base / f"{cfg.output_name}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        for repeat_idx in range(max(1, cfg.sample_repeat)):
            generator = torch.Generator(device=device)
            if cfg.sample_seed == 0:
                current_seed = int(torch.randint(0, 2**32, (1,)).item())
                generator.manual_seed(current_seed)
                print(f"[Sample {repeat_idx}] Using random seed: {current_seed}")
            else:
                current_seed = cfg.sample_seed + repeat_idx
                generator.manual_seed(current_seed)

            latent_result = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=cfg.sample_width,
                height=cfg.sample_height,
                num_inference_steps=cfg.sample_steps,
                guidance_scale=cfg.guidance_scale,
                generator=generator,
                output_type="latent",
            )

            latents = latent_result.images.to(device=device, dtype=vae_dtype)
            latents = latents / pipe.vae.config.scaling_factor

            decoded = pipe.vae.decode(latents, return_dict=False)[0]
            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image[0].permute(1, 2, 0).detach().float().cpu().numpy()
            image = (image * 255).round().astype("uint8")

            out_filename = f"{cfg.output_name}_{global_step:06d}_{repeat_idx}.png"
            out_path = sample_dir / out_filename
            Image.fromarray(image).save(out_path)
    finally:
        pipe.vae.to("cpu")
        flush_memory(device)