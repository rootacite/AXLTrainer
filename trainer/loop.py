from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from config import TrainConfig
    from loss_log import LossRecorder
    from env import flush_memory
    import control
    from control import LiveSettings
    from device_swap import SwapContext, at_safe_point
    from family import FamilyModules, ModelFamily
    from models import artifact_root
    from setup import TrainArtifacts
except ImportError:
    from trainer.config import TrainConfig
    from trainer.loss_log import LossRecorder
    from trainer.env import flush_memory
    from trainer import control
    from trainer.control import LiveSettings
    from trainer.device_swap import SwapContext, at_safe_point
    from trainer.family import FamilyModules, ModelFamily
    from trainer.models import artifact_root
    from trainer.setup import TrainArtifacts

_loss_recorder = LossRecorder()


def _batch_int(values: Any, idx: int) -> int:
    item = values[idx]
    if torch.is_tensor(item):
        return int(item.item())
    return int(item)


def group_indices_by_bucket(batch: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """Group batch items by spatial bucket to keep tensor shapes consistent."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(len(batch["caption"])):
        bw = _batch_int(batch["bucket_w"], idx)
        bh = _batch_int(batch["bucket_h"], idx)
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
        return img_data.to(device=device, dtype=weight_dtype, non_blocking=True)

    pixel_values = img_data.unsqueeze(0).to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
    latent = latent.squeeze(0)

    if cfg.cache_latents and cfg.cache_latents_to_disk:
        torch.save(latent.detach().cpu(), cache_path)

    return latent


def _stack_extra(extras: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not extras or not extras[0]:
        return {}
    return {key: torch.stack([item[key] for item in extras], dim=0) for key in extras[0]}


def build_group_inputs(
    *,
    indices: list[int],
    batch: dict[str, Any],
    family: ModelFamily,
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, dict[str, torch.Tensor]]:
    """Build prompts, latents, and family-specific extra cond for one bucket group."""
    prompts = [batch["caption"][i] for i in indices]
    extras: list[dict[str, torch.Tensor]] = []
    for i in indices:
        extras.append(
            family.extra_cond(
                src_wh=(_batch_int(batch["src_w"], i), _batch_int(batch["src_h"], i)),
                bucket_wh=(_batch_int(batch["bucket_w"], i), _batch_int(batch["bucket_h"], i)),
                device=device,
                dtype=weight_dtype,
            )
        )

    stacked = _stack_extra(extras)
    batch_masks = batch.get("loss_mask")
    if batch_masks is not None:
        if torch.is_tensor(batch_masks):
            stacked["loss_mask"] = batch_masks[indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        else:
            stacked["loss_mask"] = torch.stack(
                [batch_masks[i] for i in indices], dim=0
            ).to(device=device, dtype=torch.float32)

    img_data = batch["img_data"]
    if torch.is_tensor(img_data):
        latents = img_data[indices].to(device=device, dtype=weight_dtype, non_blocking=True)
        return prompts, latents, stacked

    latents_list: list[torch.Tensor] = []
    for i in indices:
        latents_list.append(
            encode_latent_for_item(
                item_index=i,
                batch=batch,
                vae=vae,
                cfg=cfg,
                device=device,
                weight_dtype=weight_dtype,
            )
        )
    latents = torch.stack(latents_list, dim=0).to(device=device, dtype=weight_dtype)
    return prompts, latents, stacked


def _maybe_log_and_sample(
    *,
    artifacts: TrainArtifacts,
    cfg: TrainConfig,
    global_step: int,
    swap_ctx: SwapContext | None = None,
) -> None:
    """Log the step, then save a checkpoint (and, when the switch is on, its samples).

    Sampling has no cadence of its own: a sample always belongs to the checkpoint written in the
    same step, so turning sampling off makes a save point checkpoint-only.
    """
    accelerator = artifacts.accelerator
    if accelerator.is_main_process:
        denoise_optimizer = artifacts.denoise_optimizer
        te_scheduler = artifacts.te_scheduler
        unet_effective_lr = denoise_optimizer.param_groups[0].get(
            "scheduled_lr", denoise_optimizer.param_groups[0]["lr"]
        )
        te_base_lr = te_scheduler.get_last_lr()[0]

        accelerator.log(
            {
                "Train/Loss": _maybe_log_and_sample.last_loss,
                "Train/Avg_Loss": _maybe_log_and_sample.last_avg_loss,
                "UNet/LR/Effective_Actual_LR": unet_effective_lr,
                "TE/LR/Base_Scheduled": te_base_lr,
                "TE/LR/Effective_Actual_LR": te_base_lr,
            },
            step=global_step,
        )

        settings = artifacts.settings
        if settings.due(global_step):
            if hasattr(denoise_optimizer, "eval"):
                denoise_optimizer.eval()
            try:
                artifacts.family.save_lora(
                    accelerator, artifacts.modules, cfg, global_step
                )
                if settings.sampling_enabled:
                    artifacts.family.generate_sample(
                        accelerator=accelerator,
                        modules=artifacts.modules,
                        cfg=cfg,
                        device=artifacts.device,
                        dtype=artifacts.weight_dtype,
                        global_step=global_step,
                        output_dir_base=artifact_root(cfg),
                        swap_ctx=swap_ctx,
                    )
            finally:
                if hasattr(denoise_optimizer, "train"):
                    denoise_optimizer.train()
            settings.mark_saved(global_step)
            control.publish_settings(settings)

_maybe_log_and_sample.last_loss = 0.0
_maybe_log_and_sample.last_avg_loss = 0.0


def adopt_live_settings(artifacts: TrainArtifacts, global_step: int) -> None:
    """Take over a cadence / sampling switch requested while the run was going.

    Called once per optimizer step, after `at_safe_point` so a change made while paused applies
    to the step the run resumes with.
    """
    adopted = artifacts.settings.adopt(control.read_settings(), global_step)
    if adopted is artifacts.settings:
        return
    artifacts.settings = adopted
    control.publish_settings(adopted)


def train_one_epoch(
    *,
    artifacts: TrainArtifacts,
    cfg: TrainConfig,
    global_step: int,
    progress,
    total_train_steps: int,
    swap_ctx: SwapContext | None = None,
) -> int:
    """Train one epoch and keep all step-based actions aligned with optimizer steps."""
    accelerator = artifacts.accelerator
    modules: FamilyModules = artifacts.modules
    family = artifacts.family
    denoise = modules.denoise
    text_encoders = modules.text_encoders
    denoise_optimizer = artifacts.denoise_optimizer
    te_optimizer = artifacts.te_optimizer
    te_scheduler = artifacts.te_scheduler
    device = artifacts.device
    weight_dtype = artifacts.weight_dtype

    denoise.train()
    if hasattr(denoise_optimizer, "train"):
        denoise_optimizer.train()
    for te in text_encoders:
        te.train()

    denoise_clip_params = [p for p in denoise.parameters() if p.requires_grad]
    te_clip_params = [
        p
        for te in text_encoders
        for p in te.parameters()
        if p.requires_grad
    ]

    epoch_step = 0
    epoch_index = max(0, int(cfg._current_epoch) - 1)

    for batch in artifacts.dataloader:
        with accelerator.accumulate(denoise, *text_encoders):
            groups = group_indices_by_bucket(batch)

            batch_loss_sum: torch.Tensor | None = None
            batch_item_count = 0
            n_caption = len(batch["caption"])

            for _, indices in groups.items():
                prompts, latents, extra = build_group_inputs(
                    indices=indices,
                    batch=batch,
                    family=family,
                    vae=modules.vae,
                    cfg=cfg,
                    device=device,
                    weight_dtype=weight_dtype,
                )

                loss = family.compute_loss(
                    prompts=prompts,
                    latents=latents,
                    extra=extra,
                    modules=modules,
                    cfg=cfg,
                    device=device,
                    dtype=weight_dtype,
                )

                scaled_loss = loss * (len(indices) / n_caption)
                accelerator.backward(scaled_loss)
                weighted = loss.detach() * len(indices)
                batch_loss_sum = weighted if batch_loss_sum is None else batch_loss_sum + weighted
                batch_item_count += len(indices)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(denoise_clip_params, cfg.max_grad_norm)
                accelerator.clip_grad_norm_(te_clip_params, cfg.te_max_grad_norm)
                denoise_optimizer.step()
                te_optimizer.step()
                te_scheduler.step()
                denoise_optimizer.zero_grad(set_to_none=True)
                te_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            if batch_loss_sum is None:
                avg_loss = 0.0
            else:
                avg_loss = float((batch_loss_sum / max(1, batch_item_count)).item())
            _loss_recorder.add(epoch=epoch_index, step=epoch_step, loss=avg_loss)
            _maybe_log_and_sample.last_loss = avg_loss
            _maybe_log_and_sample.last_avg_loss = _loss_recorder.moving_average
            epoch_step += 1

            if progress is not None:
                progress.update(1)
                progress.set_description(
                    f"epoch={cfg._current_epoch}/{cfg.epoch} step={global_step} loss={avg_loss:.4f}"
                )

            control.set_training(
                step=global_step,
                total_steps=total_train_steps,
                epoch=int(cfg._current_epoch),
                epochs=cfg.epoch,
                loss=float(avg_loss),
                avg_loss=float(_loss_recorder.moving_average),
            )
            if not at_safe_point("training", swap_ctx):
                return global_step

            adopt_live_settings(artifacts, global_step)
            _maybe_log_and_sample(
                artifacts=artifacts,
                cfg=cfg,
                global_step=global_step,
                swap_ctx=swap_ctx,
            )
            if control.should_stop():
                return global_step

        if cfg.flush_memory_every_step:
            flush_memory(device)

    return global_step
