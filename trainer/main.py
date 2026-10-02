from __future__ import annotations

import os
from pathlib import Path

from accelerate.utils import set_seed
from tqdm.auto import tqdm

from config import (
    TrainConfig,
    active_sample_sets,
    resolve_train_data_entries,
    save_run_config,
    tracker_hparams,
)
from models import artifact_root, lora_checkpoint_file
from cache import warm_latent_cache
from env import flush_memory
from loop import train_one_epoch
from runs import create_run_dirs
from setup import build_train_objects

try:
    import control
    from control import LiveSettings
    from device_swap import SwapContext
except ImportError:
    from trainer import control
    from trainer.control import LiveSettings
    from trainer.device_swap import SwapContext


def _prepare_artifacts(artifacts) -> None:
    # Do not prepare the dataloader: Accelerate would device-place metadata
    # tensors and may wrap/replace the bucket batch sampler. This trainer is
    # single-process; latents move to GPU in the train loop.
    n_te = len(artifacts.modules.text_encoders)
    prepared = artifacts.accelerator.prepare(
        artifacts.modules.denoise,
        *artifacts.modules.text_encoders,
        artifacts.denoise_optimizer,
        artifacts.te_optimizer,
        artifacts.te_scheduler,
    )
    artifacts.modules.denoise = prepared[0]
    artifacts.modules.text_encoders = list(prepared[1 : 1 + n_te])
    artifacts.denoise_optimizer = prepared[1 + n_te]
    artifacts.te_optimizer = prepared[2 + n_te]
    artifacts.te_scheduler = prepared[3 + n_te]


def main() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    # Every run gets its own timestamped output/log directory so artifacts from an
    # earlier run (including step names restarting at 0) are never overwritten.
    run_id = create_run_dirs(cfg.output_dir, cfg.logging_dir, cfg.output_name)
    cfg.run_dir = str(Path(cfg.output_dir) / run_id)
    # Keep the file this run trained with beside its logs: the next run may edit config.toml, and a
    # later sample or evaluation of one of this run's checkpoints has to use its prompts, not theirs.
    save_run_config(cfg.logging_dir, run_id)

    control.begin_run(os.getpid(), cfg.output_name, run_id=run_id)
    set_seed(cfg.seed)

    # Publish what this run starts with, so the Dashboard shows the cadence and the sampling
    # switch while the model is still loading (`api.py` seeded settings.json from the same config).
    live = LiveSettings.from_config(cfg)
    control.publish_settings(live)

    artifacts = None
    swap_ctx: SwapContext | None = None
    global_step = 0
    stopped_during = None

    try:
        # Fail before the model load when a `[[validation.samples]]` entry is unusable or an
        # `[[environment.train_data]]` entry is malformed (the dataset resolves the latter again
        # when it is built). The sets are the ones this run will sample with (`active_sample_sets`),
        # so a run whose prompts were already edited reports the ones in force.
        sample_sets = active_sample_sets(cfg)
        resolve_train_data_entries(cfg)
        artifacts = build_train_objects(cfg, settings=live)
        control.set_resume(artifacts.resume)
        accelerator = artifacts.accelerator
        if accelerator.is_main_process:
            print(
                f"Validation: {len(sample_sets)} prompt set(s), "
                f"{sum(s.repeat for s in sample_sets)} image(s) per sample point: "
                + ", ".join(f"{s.name} x{s.repeat}" for s in sample_sets)
            )
        if accelerator.is_main_process:
            ds = artifacts.train_dataset
            n_masked = getattr(ds, "n_masked", 0)
            print(f"Loss masks: {n_masked}/{len(ds)} samples")
            # Silent only for the plain single-folder case, so a repeat is never implicit.
            if len(ds.entries) > 1 or any(entry.repeat != 1 for entry in ds.entries):
                folders = " + ".join(
                    f"{entry.path} x{entry.repeat} ({count} images)"
                    for entry, count in zip(ds.entries, ds.entry_image_counts)
                )
                print(f"Dataset: {folders} = {ds.total_samples} samples/epoch (repeats included)")
            n_padded = getattr(ds, "n_padded", 0)
            if n_padded:
                dims = sorted({(r["bucket_w"], r["bucket_h"]) for r in ds.records})
                print(
                    f"Letterbox: {n_padded}/{len(ds)} samples padded, mean "
                    f"{getattr(ds, 'mean_pad', 0.0) * 100:.1f}% of the bucket; "
                    f"buckets: {', '.join(f'{w}x{h}' for w, h in dims)}"
                )
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
                config=tracker_hparams(cfg),
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
            sampler = getattr(artifacts.dataloader, "batch_sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
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
            if artifacts.settings.sampling_enabled:
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
