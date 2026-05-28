from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from config import TrainConfig
from models import save_lora_checkpoint
from text_processing import encode_prompt_batch
from trainer.sampling import generate_sample_image
from utils import build_time_ids


def group_indices_by_bucket(batch: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """Group batch items by spatial bucket to keep tensor shapes consistent."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(len(batch["caption"])):
        bw = int(batch["bucket_w"][idx].item())
        bh = int(batch["bucket_h"][idx].item())
        groups[(bw, bh)].append(idx)
    return groups


def encode_latent_for_item(
    *,
    item_index: int,
    batch: dict[str, Any],
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Load a latent directly or encode a pixel image on demand."""
    img_type = batch["img_type"][item_index]
    cache_path = Path(batch["cache_path"][item_index])
    img_data = batch["img_data"][item_index]

    if img_type == "latent":
        return img_data.to(device=device, dtype=weight_dtype)

    pixel_values = img_data.unsqueeze(0).to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
    latent = latent.squeeze(0)

    if cfg.cache_latents and cfg.cache_latents_to_disk:
        torch.save(latent.detach().cpu(), cache_path)

    return latent


def build_group_inputs(
    *,
    indices: list[int],
    batch: dict[str, Any],
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Build prompts, latents, and time IDs for one bucket group."""
    prompts = [batch["caption"][i] for i in indices]
    latents_list: list[torch.Tensor] = []
    time_ids_list: list[torch.Tensor] = []

    for i in indices:
        src_w = int(batch["src_w"][i].item())
        src_h = int(batch["src_h"][i].item())
        bucket_w = int(batch["bucket_w"][i].item())
        bucket_h = int(batch["bucket_h"][i].item())

        latent = encode_latent_for_item(
            item_index=i,
            batch=batch,
            vae=vae,
            cfg=cfg,
            device=device,
            weight_dtype=weight_dtype,
        )
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
    return prompts, latents, time_ids


def compute_bucket_loss(
    *,
    prompts: list[str],
    latents: torch.Tensor,
    time_ids: torch.Tensor,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    unet,
    noise_scheduler,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the forward pass for one bucket group and return the loss."""
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
            latents.shape[0],
            latents.shape[1],
            1,
            1,
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

    return F.mse_loss(model_pred.float(), noise.float(), reduction="mean")


def _maybe_log_and_sample(
    *,
    accelerator: Accelerator,
    cfg: TrainConfig,
    pipe,
    unet,
    text_encoder_1,
    text_encoder_2,
    unet_optimizer,
    te_optimizer,
    te_scheduler,
    device: torch.device,
    weight_dtype: torch.dtype,
    global_step: int,
) -> None:
    """Save checkpoints and generate samples on step boundaries."""
    if accelerator.is_main_process:
        unet_effective_lr = unet_optimizer.param_groups[0].get("scheduled_lr",
                                                      unet_optimizer.param_groups[0]["lr"])
        te_base_lr = te_scheduler.get_last_lr()[0]

        accelerator.log(
            {
                "Train/Loss": _maybe_log_and_sample.last_loss,
                "UNet/LR/Effective_Actual_LR": unet_effective_lr,
                "TE/LR/Base_Scheduled": te_base_lr,
                "TE/LR/Effective_Actual_LR": te_base_lr,
            },
            step=global_step,
        )

        if cfg.save_every_n_steps > 0 and global_step % cfg.save_every_n_steps == 0:
            if hasattr(unet_optimizer, "eval"):
                unet_optimizer.eval()
            try:
                save_lora_checkpoint(accelerator, unet, text_encoder_1, text_encoder_2, cfg, global_step)
                generate_sample_image(
                    accelerator=accelerator,
                    pipe=pipe,
                    trained_unet=accelerator.unwrap_model(unet),
                    trained_te1=accelerator.unwrap_model(text_encoder_1),
                    trained_te2=accelerator.unwrap_model(text_encoder_2),
                    cfg=cfg,
                    device=device,
                    dtype=weight_dtype,
                    global_step=global_step,
                    output_dir_base=Path(cfg.output_dir),
                )
            finally:
                if hasattr(unet_optimizer, "train"):
                    unet_optimizer.train()

_maybe_log_and_sample.last_loss = 0.0


def train_one_epoch(
    *,
    accelerator: Accelerator,
    cfg: TrainConfig,
    pipe,
    vae,
    unet,
    text_encoder_1,
    text_encoder_2,
    dataloader,
    noise_scheduler,
    unet_optimizer,
    te_optimizer,
    te_scheduler,
    device: torch.device,
    weight_dtype: torch.dtype,
    global_step: int,
    progress,
) -> int:
    """Train one epoch and keep all step-based actions aligned with optimizer steps."""
    unet.train()
    if hasattr(unet_optimizer, "train"):
        unet_optimizer.train()
    text_encoder_1.train()
    text_encoder_2.train()

    for batch in dataloader:
        with accelerator.accumulate(unet, text_encoder_1, text_encoder_2):
            groups = group_indices_by_bucket(batch)

            batch_loss_sum = 0.0
            batch_item_count = 0

            for _, indices in groups.items():
                prompts, latents, time_ids = build_group_inputs(
                    indices=indices,
                    batch=batch,
                    vae=vae,
                    cfg=cfg,
                    device=device,
                    weight_dtype=weight_dtype,
                )

                loss = compute_bucket_loss(
                    prompts=prompts,
                    latents=latents,
                    time_ids=time_ids,
                    tokenizer_1=pipe.tokenizer,
                    tokenizer_2=pipe.tokenizer_2,
                    text_encoder_1=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    cfg=cfg,
                    device=device,
                    weight_dtype=weight_dtype,
                )

                scaled_loss = loss * (len(indices) / len(batch["caption"]))
                accelerator.backward(scaled_loss)

                batch_loss_sum += loss.item() * len(indices)
                batch_item_count += len(indices)

            if accelerator.sync_gradients:
                unet_clip_params = [p for p in unet.parameters() if p.requires_grad]
                te_clip_params = (
                    [p for p in text_encoder_1.parameters() if p.requires_grad]
                    + [p for p in text_encoder_2.parameters() if p.requires_grad]
                )

                accelerator.clip_grad_norm_(unet_clip_params, cfg.max_grad_norm)
                accelerator.clip_grad_norm_(te_clip_params, cfg.te_max_grad_norm)

            unet_optimizer.step()
            te_optimizer.step()
            te_scheduler.step()
            unet_optimizer.zero_grad(set_to_none=True)
            te_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            avg_loss = batch_loss_sum / max(1, batch_item_count)
            _maybe_log_and_sample.last_loss = avg_loss

            if progress is not None:
                progress.update(1)
                progress.set_description(f"epoch={cfg._current_epoch}/{cfg.epoch} step={global_step} loss={avg_loss:.4f}")

            _maybe_log_and_sample(
                accelerator=accelerator,
                cfg=cfg,
                pipe=pipe,
                unet=unet,
                text_encoder_1=text_encoder_1,
                text_encoder_2=text_encoder_2,
                unet_optimizer=unet_optimizer,
                te_optimizer=te_optimizer,
                te_scheduler=te_scheduler,
                device=device,
                weight_dtype=weight_dtype,
                global_step=global_step,
            )

    return global_step