from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image

import os
import sys

base_dir = os.getcwd()
if base_dir not in sys.path:
    sys.path.append(base_dir)

from text_processing import encode_prompt_batch

try:
    from config import TrainConfig
    from env import flush_memory
    import control
    from device_swap import SwapContext, at_safe_point
except ImportError:
    from trainer.config import TrainConfig
    from trainer.env import flush_memory
    from trainer import control
    from trainer.device_swap import SwapContext, at_safe_point


def _offload_module(module: torch.nn.Module) -> None:
    """Move a module back to CPU to release GPU memory."""
    module.to("cpu")


def _move_module_to_device(module: torch.nn.Module, device: torch.device) -> None:
    """Move a module without changing parameter dtypes.

    LoRA adapters stay fp32 while the frozen base is bf16. Casting the whole
    module to weight_dtype breaks Schedule-Free AdamW (z is fp32) and doubles
    VRAM during the copy.
    """
    module.to(device)


def _reify_autograd_tensors(module: torch.nn.Module) -> None:
    """Clone inference-mode parameters so training backward can save them."""
    with torch.inference_mode(False):
        for param in module.parameters():
            if torch.is_inference(param):
                param.data = param.data.clone()
        for buf in module.buffers():
            if torch.is_inference(buf):
                buf.data = buf.data.clone()


def _offload_text_encoders(*modules: torch.nn.Module | None) -> None:
    for module in modules:
        if module is not None:
            _offload_module(module)


def _prepare_denoise_device(
    unet: torch.nn.Module,
    device: torch.device,
    *text_encoders: torch.nn.Module | None,
) -> None:
    """UNet on the train device; TEs off GPU (resume sampling puts them back)."""
    _offload_text_encoders(*text_encoders)
    _move_module_to_device(unet, device)
    flush_memory(device)


def _prepare_decode_devices(
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    device: torch.device,
) -> None:
    """VAE decode must not overlap the denoise network on GPU."""
    _offload_module(unet)
    flush_memory(device)
    _move_module_to_device(vae, device)


def _restore_train_modules(
    *,
    unet: torch.nn.Module,
    te1: torch.nn.Module,
    te2: torch.nn.Module,
    vae: torch.nn.Module,
    device: torch.device,
) -> None:
    """Put UNet + TEs back where the training loop expects them."""
    _offload_module(vae)
    flush_memory(device)
    _move_module_to_device(unet, device)
    _move_module_to_device(te1, device)
    _move_module_to_device(te2, device)
    for module in (unet, te1, te2):
        _reify_autograd_tensors(module)


@torch.no_grad()
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
    swap_ctx: SwapContext | None = None,
) -> None:
    """Generate and save sample images with aggressive module offloading."""
    if not accelerator.is_main_process:
        return
    
    prev_unet_training = trained_unet.training
    prev_te1_training = trained_te1.training
    prev_te2_training = trained_te2.training

    flush_memory(device)

    _offload_module(pipe.vae)

    trained_unet.eval()
    trained_te1.eval()
    trained_te2.eval()

    pipe.unet = trained_unet
    pipe.text_encoder = trained_te1
    pipe.text_encoder_2 = trained_te2

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
    )

    num_inference_steps = cfg.sample_steps
    sigmas = np.linspace(pipe.scheduler.config.num_train_timesteps - 1, 0, num_inference_steps)
    sigmas = np.append(sigmas, 0.0).astype(np.float32)
    pipe.scheduler.sigmas = torch.from_numpy(sigmas).to(device)
    pipe.scheduler.num_inference_steps = num_inference_steps

    prompt_embeds, pooled_prompt_embeds, npu = encode_prompt_batch(
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
    negative_prompt_embeds, negative_pooled_prompt_embeds, _ = encode_prompt_batch(
        prompts=[cfg.sample_negative],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=cfg.clip_skip,
        max_token_length=cfg.max_token_length,
        device=device,
        dtype=dtype,
        target_num_chunks=npu
    )

    _offload_text_encoders(trained_te1, trained_te2)
    flush_memory(device)

    sample_dir = output_dir_base / f"{cfg.output_name}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    vae_dtype = torch.bfloat16
    repeats = max(1, cfg.sample_repeat)

    try:
        control.set_sampling(
            active=True,
            repeat=0,
            repeats=repeats,
            denoise_step=0,
            denoise_steps=cfg.sample_steps,
            global_step=global_step,
        )
        repeat_idx = 0
        while repeat_idx < repeats:
            if not at_safe_point("sampling", swap_ctx):
                return

            _prepare_denoise_device(trained_unet, device, trained_te1, trained_te2)

            interrupted = {"value": False}

            def _on_step_end(pipeline, step_index, _timestep, callback_kwargs):
                control.set_sampling(
                    active=True,
                    repeat=repeat_idx,
                    repeats=repeats,
                    denoise_step=int(step_index) + 1,
                    denoise_steps=cfg.sample_steps,
                    global_step=global_step,
                )
                pending = control.peek_command()
                if pending in ("pause", "stop"):
                    pipeline._interrupt = True
                    interrupted["value"] = True
                return callback_kwargs

            generator = torch.Generator(device="cpu")
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
                callback_on_step_end=_on_step_end,
            )

            if interrupted["value"] or getattr(pipe, "_interrupt", False):
                pipe._interrupt = False
                if not at_safe_point("sampling", swap_ctx):
                    return
                continue

            latents = latent_result.images.to(device=device, dtype=vae_dtype)
            latents = latents / pipe.vae.config.scaling_factor

            _prepare_decode_devices(trained_unet, pipe.vae, device)
            if hasattr(pipe.vae.config, "force_upcast"):
                pipe.vae.config.force_upcast = False
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

            decoded = pipe.vae.decode(latents, return_dict=False)[0]
            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image[0].permute(1, 2, 0).detach().float().cpu().numpy()
            image = (image * 255).round().astype("uint8")

            out_filename = f"{cfg.output_name}_{global_step:06d}_{repeat_idx}.png"
            Image.fromarray(image).save(sample_dir / out_filename)

            _offload_module(pipe.vae)
            repeat_idx += 1

    finally:
        _restore_train_modules(
            unet=trained_unet,
            te1=trained_te1,
            te2=trained_te2,
            vae=pipe.vae,
            device=device,
        )

        if prev_unet_training:
            trained_unet.train()
        else:
            trained_unet.eval()

        if prev_te1_training:
            trained_te1.train()
        else:
            trained_te1.eval()

        if prev_te2_training:
            trained_te2.train()
        else:
            trained_te2.eval()

        control.set_sampling(
            active=False,
            repeat=repeats,
            repeats=repeats,
            denoise_step=0,
            denoise_steps=cfg.sample_steps,
            global_step=global_step,
        )
        flush_memory(device)