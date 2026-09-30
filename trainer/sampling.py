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
    from config import TrainConfig, resolve_sample_sets
    from env import flush_memory
    import control
    from device_swap import SwapContext, at_safe_point
    from models import sample_scheduler_kwargs
except ImportError:
    from trainer.config import TrainConfig, resolve_sample_sets
    from trainer.env import flush_memory
    from trainer import control
    from trainer.device_swap import SwapContext, at_safe_point
    from trainer.models import sample_scheduler_kwargs


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


def _prepare_encode_device(
    te1: torch.nn.Module,
    te2: torch.nn.Module,
    device: torch.device,
) -> None:
    """Text encoders on the train device for a prompt pass (the denoise step puts them back off)."""
    _move_module_to_device(te1, device)
    _move_module_to_device(te2, device)
    flush_memory(device)


def _configure_scheduler(pipe, steps: int, device: torch.device, scheduler_kwargs: dict) -> None:
    """Linspace sigma schedule for one set's step count (sets may differ).

    `scheduler_kwargs` carries the prediction type this base/LoRA was trained with
    (`models.sample_scheduler_kwargs`): rendering a v-pred model with epsilons is noise.
    """
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
        **scheduler_kwargs,
    )
    sigmas = np.linspace(pipe.scheduler.config.num_train_timesteps - 1, 0, steps)
    sigmas = np.append(sigmas, 0.0).astype(np.float32)
    pipe.scheduler.sigmas = torch.from_numpy(sigmas).to(device)
    pipe.scheduler.num_inference_steps = steps


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
    """Generate and save sample images with aggressive module offloading.

    One image per repeat for every `[[validation.samples]]` set. The file name carries
    the set index (`{output_name}_{step:06d}_p{set}_{repeat}.png`) and `control` reports a
    global image counter plus the set this pass is on.
    """
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

    sets = resolve_sample_sets(cfg)
    total_images = sum(sample_set.repeat for sample_set in sets)
    scheduler_kwargs = sample_scheduler_kwargs(cfg, pipe.scheduler.config)

    sample_dir = output_dir_base / f"{cfg.output_name}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    vae_dtype = torch.bfloat16
    completed = 0

    try:
        control.set_sampling(
            active=True,
            repeat=0,
            repeats=total_images,
            denoise_step=0,
            denoise_steps=sets[0].steps,
            global_step=global_step,
            prompt_set=1,
            prompt_sets=len(sets),
        )
        for set_index, sample_set in enumerate(sets, start=1):
            _configure_scheduler(pipe, sample_set.steps, device, scheduler_kwargs)
            # Every set encodes on its own, so the chunk padding of one prompt never depends
            # on how long another set's prompt is.
            _prepare_encode_device(trained_te1, trained_te2, device)
            set_prompt_embeds, set_pooled_prompt_embeds, npu = encode_prompt_batch(
                prompts=[sample_set.prompt],
                tokenizer_1=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                text_encoder_1=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                clip_skip=cfg.clip_skip,
                max_token_length=cfg.max_token_length,
                device=device,
                dtype=dtype,
            )
            set_negative_embeds, set_negative_pooled_embeds, _ = encode_prompt_batch(
                prompts=[sample_set.negative],
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
            print(
                f"[Sample set {set_index}/{len(sets)}] {sample_set.name}: "
                f"{sample_set.repeat} image(s), {sample_set.width}x{sample_set.height}, "
                f"{sample_set.steps} steps, cfg {sample_set.guidance_scale}, seed {sample_set.seed}"
            )

            repeat_idx = 0
            while repeat_idx < sample_set.repeat:
                if not at_safe_point("sampling", swap_ctx):
                    return

                _prepare_denoise_device(trained_unet, device, trained_te1, trained_te2)

                interrupted = {"value": False}

                def _on_step_end(pipeline, step_index, _timestep, callback_kwargs):
                    control.set_sampling(
                        active=True,
                        repeat=completed,
                        repeats=total_images,
                        denoise_step=int(step_index) + 1,
                        denoise_steps=sample_set.steps,
                        global_step=global_step,
                        prompt_set=set_index,
                        prompt_sets=len(sets),
                    )
                    pending = control.peek_command()
                    if pending in ("pause", "stop"):
                        pipeline._interrupt = True
                        interrupted["value"] = True
                    return callback_kwargs

                generator = torch.Generator(device="cpu")
                if sample_set.seed == 0:
                    current_seed = int(torch.randint(0, 2**32, (1,)).item())
                    generator.manual_seed(current_seed)
                    print(f"[Sample {set_index}.{repeat_idx}] Using random seed: {current_seed}")
                else:
                    current_seed = sample_set.seed + repeat_idx
                    generator.manual_seed(current_seed)

                latent_result = pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=set_prompt_embeds,
                    negative_prompt_embeds=set_negative_embeds,
                    pooled_prompt_embeds=set_pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=set_negative_pooled_embeds,
                    width=sample_set.width,
                    height=sample_set.height,
                    num_inference_steps=sample_set.steps,
                    guidance_scale=sample_set.guidance_scale,
                    guidance_rescale=sample_set.guidance_rescale,
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

                out_filename = f"{cfg.output_name}_{global_step:06d}_p{set_index - 1}_{repeat_idx}.png"
                Image.fromarray(image).save(sample_dir / out_filename)

                _offload_module(pipe.vae)
                repeat_idx += 1
                completed += 1

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
            repeat=completed,
            repeats=total_images,
            denoise_step=0,
            denoise_steps=sets[0].steps,
            global_step=global_step,
            prompt_set=len(sets),
            prompt_sets=len(sets),
        )
        flush_memory(device)