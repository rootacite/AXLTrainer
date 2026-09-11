from __future__ import annotations

import os
from pathlib import Path

from accelerate.utils import set_seed
from tqdm.auto import tqdm

from config import TrainConfig
from models import artifact_root, lora_checkpoint_file
from cache import warm_latent_cache
from env import flush_memory
from loop import train_one_epoch
from runs import create_run_dirs
from setup import build_train_objects

try:
    import control
    from device_swap import SwapContext
except ImportError:
    from trainer import control
    from trainer.device_swap import SwapContext


def _prepare_artifacts(artifacts) -> None:
    n_te = len(artifacts.modules.text_encoders)
    prepared = artifacts.accelerator.prepare(
        artifacts.modules.denoise,
        *artifacts.modules.text_encoders,
        artifacts.denoise_optimizer,
        artifacts.te_optimizer,
        artifacts.dataloader,
        artifacts.te_scheduler,
    )
    artifacts.modules.denoise = prepared[0]
    artifacts.modules.text_encoders = list(prepared[1 : 1 + n_te])
    artifacts.denoise_optimizer = prepared[1 + n_te]
    artifacts.te_optimizer = prepared[2 + n_te]
    artifacts.dataloader = prepared[3 + n_te]
    artifacts.te_scheduler = prepared[4 + n_te]


def main() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    # Every run gets its own timestamped output/log directory so artifacts from an
    # earlier run (including step names restarting at 0) are never overwritten.
    run_id = create_run_dirs(cfg.output_dir, cfg.logging_dir, cfg.output_name)
    cfg.run_dir = str(Path(cfg.output_dir) / run_id)

    control.begin_run(os.getpid(), cfg.output_name, run_id=run_id)
    set_seed(cfg.seed)

    artifacts = None
    swap_ctx: SwapContext | None = None
    global_step = 0
    stopped_during = None

    try:
        artifacts = build_train_objects(cfg)
        control.set_resume(artifacts.resume)
        accelerator = artifacts.accelerator
        device = artifacts.device
        weight_dtype = artifacts.weight_dtype
        swap_ctx = SwapContext(
            device=device,
            vae=artifacts.modules.vae,
            denoise=artifacts.modules.denoise,
            text_encoders=list(artifacts.modules.text_encoders),
            denoise_optimizer=artifacts.denoise_optimizer,
            te_optimizer=artifacts.te_optimizer,
        )

        if cfg.cache_latents and cfg.cache_latents_to_disk:
            if accelerator.is_main_process:
                print("Checking/Generating latents cache...")
                finished = warm_latent_cache(
                    artifacts.train_dataset,
                    artifacts.modules.vae,
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

        artifacts.modules.vae.to("cpu")
        flush_memory(device)

        _prepare_artifacts(artifacts)
        swap_ctx.denoise = artifacts.modules.denoise
        swap_ctx.text_encoders = list(artifacts.modules.text_encoders)
        swap_ctx.denoise_optimizer = artifacts.denoise_optimizer
        swap_ctx.te_optimizer = artifacts.te_optimizer

        if accelerator.is_main_process:
            accelerator.init_trackers(
                project_name=run_id,
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
                artifacts=artifacts,
                cfg=cfg,
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
                artifacts.family.save_lora(
                    accelerator, artifacts.modules, cfg, global_step
                )
            progress.close()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.end_training()
            control.end_run(control.STATUS_FINISHED, detail="stopped_during_training")
            return

        if hasattr(artifacts.denoise_optimizer, "eval"):
            artifacts.denoise_optimizer.eval()
        try:
            artifacts.family.save_lora(
                accelerator,
                artifacts.modules,
                cfg,
                global_step,
                final=True,
            )
            artifacts.family.generate_sample(
                accelerator=accelerator,
                modules=artifacts.modules,
                cfg=cfg,
                device=device,
                dtype=weight_dtype,
                global_step=global_step,
                output_dir_base=artifact_root(cfg),
                swap_ctx=swap_ctx,
            )
        finally:
            if hasattr(artifacts.denoise_optimizer, "train"):
                artifacts.denoise_optimizer.train()

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
