from __future__ import annotations

import os
from pathlib import Path

from accelerate.utils import set_seed
from tqdm.auto import tqdm

from config import TrainConfig
from models import lora_checkpoint_file, save_lora_checkpoint
from cache import warm_latent_cache
from env import flush_memory
from loop import train_one_epoch
from sampling import generate_sample_image
from setup import build_train_objects

try:
    import control
    from device_swap import SwapContext
except ImportError:
    from trainer import control
    from trainer.device_swap import SwapContext


def main() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    control.begin_run(os.getpid(), cfg.output_name)
    set_seed(cfg.seed)

    artifacts = None
    swap_ctx: SwapContext | None = None
    global_step = 0
    stopped_during = None

    try:
        artifacts = build_train_objects(cfg)
        accelerator = artifacts.accelerator
        device = artifacts.device
        weight_dtype = artifacts.weight_dtype
        swap_ctx = SwapContext(
            device=device,
            vae=artifacts.vae,
            unet=artifacts.unet,
            text_encoder_1=artifacts.text_encoder_1,
            text_encoder_2=artifacts.text_encoder_2,
            unet_optimizer=artifacts.unet_optimizer,
            te_optimizer=artifacts.te_optimizer,
        )

        if cfg.cache_latents and cfg.cache_latents_to_disk:
            if accelerator.is_main_process:
                print("Checking/Generating latents cache...")
                finished = warm_latent_cache(
                    artifacts.train_dataset,
                    artifacts.vae,
                    cfg,
                    device,
                    weight_dtype,
                    swap_ctx=swap_ctx,
                )
                if not finished:
                    stopped_during = "encoding"
            accelerator.wait_for_everyone()
            if stopped_during == "encoding" or control.should_stop():
                control.end_run(
                    control.STATUS_FINISHED,
                    detail="stopped_during_encoding",
                )
                return

        artifacts.vae.to("cpu")
        flush_memory(device)

        (
            artifacts.unet,
            artifacts.text_encoder_1,
            artifacts.text_encoder_2,
            artifacts.unet_optimizer,
            artifacts.te_optimizer,
            artifacts.dataloader,
            artifacts.te_scheduler,
        ) = accelerator.prepare(
            artifacts.unet,
            artifacts.text_encoder_1,
            artifacts.text_encoder_2,
            artifacts.unet_optimizer,
            artifacts.te_optimizer,
            artifacts.dataloader,
            artifacts.te_scheduler,
        )
        swap_ctx.unet = artifacts.unet
        swap_ctx.text_encoder_1 = artifacts.text_encoder_1
        swap_ctx.text_encoder_2 = artifacts.text_encoder_2
        swap_ctx.unet_optimizer = artifacts.unet_optimizer
        swap_ctx.te_optimizer = artifacts.te_optimizer

        if accelerator.is_main_process:
            accelerator.init_trackers(
                project_name=cfg.output_name,
                config=vars(cfg),
            )

        steps_per_epoch = max(
            1,
            (len(artifacts.dataloader) + cfg.gradient_accumulation_steps - 1) // cfg.gradient_accumulation_steps,
        )
        total_train_steps = steps_per_epoch * cfg.epoch
        control.set_training(
            step=0,
            total_steps=total_train_steps,
            epoch=0,
            epochs=cfg.epoch,
        )

        progress = tqdm(
            total=total_train_steps,
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(cfg.epoch):
            artifacts.train_dataset.set_epoch(epoch)
            cfg._current_epoch = epoch + 1

            global_step = train_one_epoch(
                accelerator=accelerator,
                cfg=cfg,
                pipe=artifacts.pipe,
                vae=artifacts.vae,
                unet=artifacts.unet,
                text_encoder_1=artifacts.text_encoder_1,
                text_encoder_2=artifacts.text_encoder_2,
                dataloader=artifacts.dataloader,
                noise_scheduler=artifacts.noise_scheduler,
                unet_optimizer=artifacts.unet_optimizer,
                te_optimizer=artifacts.te_optimizer,
                te_scheduler=artifacts.te_scheduler,
                device=device,
                weight_dtype=weight_dtype,
                global_step=global_step,
                progress=progress,
                total_train_steps=total_train_steps,
                swap_ctx=swap_ctx,
            )
            if control.should_stop():
                stopped_during = "training"
                break

        if stopped_during == "training":
            if global_step > 0 and not lora_checkpoint_file(cfg, global_step).is_file():
                save_lora_checkpoint(
                    accelerator,
                    artifacts.unet,
                    artifacts.text_encoder_1,
                    artifacts.text_encoder_2,
                    cfg,
                    global_step,
                )
            progress.close()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.end_training()
            control.end_run(control.STATUS_FINISHED, detail="stopped_during_training")
            return

        if hasattr(artifacts.unet_optimizer, "eval"):
            artifacts.unet_optimizer.eval()
        try:
            save_lora_checkpoint(
                accelerator,
                artifacts.unet,
                artifacts.text_encoder_1,
                artifacts.text_encoder_2,
                cfg,
                global_step,
                final=True,
            )
            generate_sample_image(
                accelerator=accelerator,
                pipe=artifacts.pipe,
                trained_unet=accelerator.unwrap_model(artifacts.unet),
                trained_te1=accelerator.unwrap_model(artifacts.text_encoder_1),
                trained_te2=accelerator.unwrap_model(artifacts.text_encoder_2),
                cfg=cfg,
                device=device,
                dtype=weight_dtype,
                global_step=global_step,
                output_dir_base=Path(cfg.output_dir),
                swap_ctx=swap_ctx,
            )
        finally:
            if hasattr(artifacts.unet_optimizer, "train"):
                artifacts.unet_optimizer.train()

        if control.should_stop():
            progress.close()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.end_training()
            control.end_run(control.STATUS_FINISHED, detail="stopped_during_sampling")
            return

        progress.close()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            accelerator.end_training()
        control.end_run(control.STATUS_FINISHED)
    except Exception as exc:
        control.end_run(control.STATUS_ERROR, error=str(exc))
        raise


if __name__ == "__main__":
    main()
