from __future__ import annotations

import gc
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TrainConfig
from dataset import SDXLLoraDataset, make_collate_fn
from models import (
    build_optimizer,
    build_scheduler,
    enable_flash_attention,
    load_sdxl_pipeline,
    save_lora_checkpoint,
)
from text_processing import encode_prompt_batch
from utils import build_time_ids


# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
cache_dir = os.path.join(os.getcwd(), "migraphx_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = cache_dir
os.environ["ORT_MIGRAPHX_CACHE_PATH"] = cache_dir


def flush_memory(device: torch.device) -> None:
    """Release Python and GPU-side cached memory as much as possible."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def warm_latent_cache(
    dataset: SDXLLoraDataset,
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Pre-encode image latents to disk if they do not already exist."""
    if not (cfg.cache_latents and cfg.cache_latents_to_disk):
        return

    vae.eval()
    vae.to(device=device, dtype=dtype)

    pbar = tqdm(total=len(dataset), desc="Encoding Latents")
    for idx in range(len(dataset)):
        item = dataset[idx]
        cache_path = Path(item["cache_path"])

        if cache_path.exists():
            pbar.update(1)
            continue

        if item["img_type"] != "pixel":
            pbar.update(1)
            continue

        pixel_values = item["img_data"].unsqueeze(0).to(device=device, dtype=dtype)
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        torch.save(latent.squeeze(0).detach().cpu(), cache_path)
        pbar.update(1)

    pbar.close()
    vae.to("cpu")
    flush_memory(device)

@torch.inference_mode()
def generate_sample_image(
    *,
    accelerator: Accelerator,
    pipe,
    trained_unet: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,          # 修改：直接传入全局步数，方便命名
    output_dir_base: Path,     # 修改：传入输出根路径
) -> None:
    if not accelerator.is_main_process:
        return

    flush_memory(device)

    pipe = pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
    )

    import numpy as np
    num_inference_steps = cfg.sample_steps
    
    sigmas = np.linspace(pipe.scheduler.config.num_train_timesteps - 1, 0, num_inference_steps)
    sigmas = np.append(sigmas, 0.0).astype(np.float32)
    
    pipe.scheduler.sigmas = torch.from_numpy(sigmas).to(device)
    pipe.scheduler.num_inference_steps = num_inference_steps

    pipe.unet = trained_unet
    pipe.unet.eval()

    vae_dtype = torch.bfloat16
    pipe.vae.to(device=device, dtype=vae_dtype).eval()

    if hasattr(pipe.vae.config, "force_upcast"):
        pipe.vae.config.force_upcast = False

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.text_encoder.to(device=device, dtype=dtype).eval()
    pipe.text_encoder_2.to(device=device, dtype=dtype).eval()

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

    # =================【核心修改：支持循环重复生成与新命名】=================
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

            latents = latent_result.images
            latents = latents.to(device=device, dtype=vae_dtype)
            latents = latents / pipe.vae.config.scaling_factor

            decoded = pipe.vae.decode(latents, return_dict=False)[0]

            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image[0].permute(1, 2, 0).detach().float().cpu().numpy()
            image = (image * 255).round().astype("uint8")

            out_filename = f"{cfg.output_name}_{global_step:06d}_{repeat_idx}.png"
            out_path = sample_dir / out_filename
            
            from PIL import Image
            Image.fromarray(image).save(out_path)
            
    finally:
        pipe.vae.to("cpu")
        flush_memory(device)
    # =======================================================================


def maybe_enable_amp_backends() -> None:
    """Enable PyTorch attention backends when available."""
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def build_train_objects(cfg: TrainConfig):
    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=cfg.logging_dir,
    )
    device = accelerator.device

    maybe_enable_amp_backends()

    pipe = load_sdxl_pipeline(cfg.pretrained_model_name_or_path, weight_dtype)
    vae = pipe.vae
    unet = pipe.unet
    tokenizer_1, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_1, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2
    noise_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Allow SDPA / FlashAttention-style paths when supported.
    enable_flash_attention(unet)

    # 修改这里的 LoraConfig
    lora_config = LoraConfig(
        r=cfg.network_dim,
        lora_alpha=cfg.network_alpha,
        lora_dropout=cfg.network_dropout, 
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.enable_gradient_checkpointing()
    unet.print_trainable_parameters()

    vae.to(device=device, dtype=weight_dtype).eval()
    text_encoder_1.to(device=device, dtype=weight_dtype).eval()
    text_encoder_2.to(device=device, dtype=weight_dtype).eval()

    train_dataset = SDXLLoraDataset(cfg)
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.max_data_loader_n_workers,
        pin_memory=True,
        persistent_workers=cfg.persistent_workers,
        collate_fn=make_collate_fn(),
        drop_last=True,
    )

    optimizer = build_optimizer(cfg, unet.parameters())
    steps_per_epoch = max(1, len(dataloader) // cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.epoch
    lr_scheduler = build_scheduler(optimizer, total_steps, cfg)

    return (
        accelerator,
        device,
        weight_dtype,
        pipe,
        vae,
        unet,
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        noise_scheduler,
        train_dataset,
        dataloader,
        optimizer,
        lr_scheduler,
    )


def main() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    set_seed(cfg.seed)

    (
        accelerator,
        device,
        weight_dtype,
        pipe,
        vae,
        unet,
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        noise_scheduler,
        train_dataset,
        dataloader,
        optimizer,
        lr_scheduler,
    ) = build_train_objects(cfg)

    # Cache latents once, then release the VAE from GPU.
    if cfg.cache_latents and cfg.cache_latents_to_disk:
        if accelerator.is_main_process:
            print("Checking/Generating latents cache...")
            warm_latent_cache(train_dataset, vae, cfg, device, weight_dtype)
        accelerator.wait_for_everyone()

    vae.to("cpu")
    flush_memory(device)

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.output_name,
            config=vars(cfg),
        )

    global_step = 0
    progress = tqdm(range(max(1, len(dataloader) // cfg.gradient_accumulation_steps * cfg.epoch)), disable=not accelerator.is_local_main_process)

    for epoch in range(cfg.epoch):
        train_dataset.set_epoch(epoch)
        unet.train()

        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Group items by bucket so tensor shapes stay consistent inside a group.
                groups = defaultdict(list)
                for i in range(len(batch["caption"])):
                    bw = int(batch["bucket_w"][i].item())
                    bh = int(batch["bucket_h"][i].item())
                    groups[(bw, bh)].append(i)

                batch_loss_sum = 0.0
                batch_item_count = 0

                for (bucket_w, bucket_h), indices in groups.items():
                    prompts = [batch["caption"][i] for i in indices]
                    latents_list: List[torch.Tensor] = []
                    time_ids_list: List[torch.Tensor] = []

                    for i in indices:
                        img_type = batch["img_type"][i]
                        src_w = int(batch["src_w"][i].item())
                        src_h = int(batch["src_h"][i].item())
                        cache_path = Path(batch["cache_path"][i])
                        img_data = batch["img_data"][i]

                        if img_type == "latent":
                            latent = img_data.to(device=device, dtype=weight_dtype)
                        else:
                            pixel_values = img_data.unsqueeze(0).to(device=device, dtype=weight_dtype)
                            with torch.no_grad():
                                latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                            latent = latent.squeeze(0)

                            if cfg.cache_latents and cfg.cache_latents_to_disk:
                                torch.save(latent.detach().cpu(), cache_path)

                        latents_list.append(latent)
                        time_ids_list.append(
                            build_time_ids(
                                original_size=(src_h, src_w),
                                crop_top_left=(0, 0),
                                target_size=(bucket_h, bucket_w),
                                device=device,
                                dtype=weight_dtype,
                            )
                        )

                    latents = torch.stack(latents_list, dim=0).to(device=device, dtype=weight_dtype)
                    time_ids = torch.stack(time_ids_list, dim=0)

                    prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(
                        prompts=prompts,
                        tokenizer_1=tokenizer_1,
                        tokenizer_2=tokenizer_2,
                        text_encoder_1=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        clip_skip=cfg.clip_skip,
                        max_token_length=cfg.max_token_length,
                        device=device,
                        dtype=weight_dtype,
                    )

                    noise = torch.randn_like(latents)
                    if cfg.noise_offset > 0:
                        offset = cfg.noise_offset * torch.randn(
                            latents.shape[0], latents.shape[1], 1, 1,
                            device=device,
                            dtype=weight_dtype,
                        )
                        noise = noise + offset

                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": time_ids,
                        },
                        return_dict=False,
                    )[0]

                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    scaled_loss = loss * (len(indices) / len(batch["caption"]))
                    accelerator.backward(scaled_loss)

                    batch_loss_sum += loss.item() * len(indices)
                    batch_item_count += len(indices)

                    del latents, time_ids, prompt_embeds, pooled_prompt_embeds
                    del noise, timesteps, noisy_latents, model_pred

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                avg_loss = batch_loss_sum / max(1, batch_item_count)
                progress.set_description(f"epoch={epoch + 1}/{cfg.epoch} loss={avg_loss:.4f}")

                if accelerator.is_main_process:
                    current_base_lr = lr_scheduler.get_last_lr()[0]
                    prodigy_d = optimizer.param_groups[0].get("d", 1.0)
                    effective_lr = current_base_lr * prodigy_d

                    accelerator.log(
                        {
                            "Train/Loss": avg_loss,
                            "LR/Base_Scheduled": current_base_lr,
                            "LR/Prodigy_D_Factor": prodigy_d,
                            "LR/Effective_Actual_LR": effective_lr,
                        },
                        step=global_step,
                    )

                    if cfg.save_every_n_steps > 0 and global_step % cfg.save_every_n_steps == 0:
                        save_lora_checkpoint(accelerator, unet, cfg, global_step)
                        generate_sample_image(
                            accelerator=accelerator,
                            pipe=pipe,
                            trained_unet=accelerator.unwrap_model(unet),
                            cfg=cfg,
                            device=device,
                            dtype=weight_dtype,
                            global_step=global_step, 
                            output_dir_base=Path(cfg.output_dir),
                        )

    save_lora_checkpoint(accelerator, unet, cfg, global_step, final=True)
    generate_sample_image(
        accelerator=accelerator,
        pipe=pipe,
        trained_unet=accelerator.unwrap_model(unet),
        cfg=cfg,
        device=device,
        dtype=weight_dtype,
        global_step=global_step,   
        output_dir_base=Path(cfg.output_dir),
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
