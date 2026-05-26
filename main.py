from __future__ import annotations

import os
from pathlib import Path

from accelerate.utils import set_seed
from tqdm.auto import tqdm

from config import TrainConfig
from models import save_lora_checkpoint
from trainer.cache import warm_latent_cache
from trainer.env import flush_memory
from trainer.loop import train_one_epoch
from trainer.sampling import generate_sample_image
from trainer.setup import build_train_objects


def main() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    set_seed(cfg.seed)

    artifacts = build_train_objects(cfg)
    accelerator = artifacts.accelerator
    device = artifacts.device
    weight_dtype = artifacts.weight_dtype

    if cfg.cache_latents and cfg.cache_latents_to_disk:
        if accelerator.is_main_process:
            print("Checking/Generating latents cache...")
            warm_latent_cache(
                artifacts.train_dataset,
                artifacts.vae,
                cfg,
                device,
                weight_dtype,
            )
        accelerator.wait_for_everyone()

    artifacts.vae.to("cpu")
    flush_memory(device)

    (
        artifacts.unet,
        artifacts.text_encoder_1,
        artifacts.text_encoder_2,
        artifacts.unet_optimizer,
        artifacts.te_optimizer,
        artifacts.dataloader,
        artifacts.unet_scheduler,
        artifacts.te_scheduler,
    ) = accelerator.prepare(
        artifacts.unet,
        artifacts.text_encoder_1,
        artifacts.text_encoder_2,
        artifacts.unet_optimizer,
        artifacts.te_optimizer,
        artifacts.dataloader,
        artifacts.unet_scheduler,
        artifacts.te_scheduler,
    )

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

    progress = tqdm(
        total=total_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
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
            unet_scheduler=artifacts.unet_scheduler,
            te_scheduler=artifacts.te_scheduler,
            device=device,
            weight_dtype=weight_dtype,
            global_step=global_step,
            progress=progress,
        )

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
    )

    progress.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()