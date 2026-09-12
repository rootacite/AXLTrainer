from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

try:
    from config import TrainConfig
    from dataset import BucketBatchSampler, LoraImageDataset, make_collate_fn
    from family import FamilyModules, ModelFamily, require_trainable, resolve_family
    from models import (
        build_scheduler,
        build_te_optimizer,
        build_unet_optimizer,
    )
    from env import setup_migraphx_cache
except ImportError:
    from trainer.config import TrainConfig
    from trainer.dataset import BucketBatchSampler, LoraImageDataset, make_collate_fn
    from trainer.family import FamilyModules, ModelFamily, require_trainable, resolve_family
    from trainer.models import (
        build_scheduler,
        build_te_optimizer,
        build_unet_optimizer,
    )
    from trainer.env import setup_migraphx_cache


@dataclass
class TrainArtifacts:
    family: ModelFamily
    modules: FamilyModules
    accelerator: Accelerator
    device: torch.device
    weight_dtype: torch.dtype
    train_dataset: LoraImageDataset
    dataloader: DataLoader
    denoise_optimizer: Any
    te_optimizer: Any
    te_scheduler: Any
    resume: dict[str, Any]


def maybe_enable_amp_backends() -> None:
    """Enable PyTorch attention backends when available."""
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def create_accelerator(cfg: TrainConfig) -> Accelerator:
    """Create the training accelerator."""
    return Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=cfg.logging_dir,
    )


def build_dataloader(cfg: TrainConfig) -> tuple[LoraImageDataset, DataLoader]:
    """Build dataset and a bucket-grouped dataloader.

    One batch is one aspect-ratio bucket so the train loop can stack latents
    at `train_batch_size` instead of splitting a shuffled mixed-reso batch.
    """
    train_dataset = LoraImageDataset(cfg)
    batch_sampler = BucketBatchSampler(
        train_dataset.buckets,
        batch_size=cfg.train_batch_size,
        seed=cfg.seed,
    )
    n_workers = max(0, int(cfg.max_data_loader_n_workers))
    dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=bool(cfg.persistent_workers) and n_workers > 0,
        collate_fn=make_collate_fn(),
    )
    return train_dataset, dataloader


def build_optimizers_and_schedulers(
    cfg: TrainConfig,
    modules: FamilyModules,
    dataloader: DataLoader,
):
    """Build optimizers for the denoise network and text encoders."""
    denoise_params = [p for p in modules.denoise.parameters() if p.requires_grad]
    te_params = [
        p
        for te in modules.text_encoders
        for p in te.parameters()
        if p.requires_grad
    ]

    denoise_optimizer = build_unet_optimizer(cfg, denoise_params)
    te_optimizer = build_te_optimizer(cfg, te_params)

    steps_per_epoch = max(1, math.ceil(len(dataloader) / cfg.gradient_accumulation_steps))
    total_steps = steps_per_epoch * cfg.epoch
    te_scheduler = build_scheduler(te_optimizer, total_steps, cfg)
    return denoise_optimizer, te_optimizer, te_scheduler


def build_train_objects(cfg: TrainConfig) -> TrainArtifacts:
    """Build everything required for training in a readable sequence."""
    family = resolve_family(cfg)
    require_trainable(family)

    setup_migraphx_cache()
    maybe_enable_amp_backends()

    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    accelerator = create_accelerator(cfg)
    device = accelerator.device

    pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, weight_dtype)
    modules = family.unpack(pipe)
    modules.noise_scheduler = family.build_noise_scheduler(pipe, cfg)

    modules.vae.requires_grad_(False)
    modules.denoise.requires_grad_(False)
    modules = family.apply_lora(cfg, modules)
    # Resume weights must land before accelerator.prepare wraps the modules.
    resume = family.load_lora(cfg, modules)

    # Encoding only needs the VAE on GPU. TEs stay on CPU until accelerator.prepare
    # after the latent cache (otherwise CLIP-G + VAE encode OOMs a 16GB card).
    modules.vae.to(device=device).eval()

    train_dataset, dataloader = build_dataloader(cfg)
    denoise_optimizer, te_optimizer, te_scheduler = build_optimizers_and_schedulers(
        cfg,
        modules,
        dataloader,
    )

    return TrainArtifacts(
        family=family,
        modules=modules,
        accelerator=accelerator,
        device=device,
        weight_dtype=weight_dtype,
        train_dataset=train_dataset,
        dataloader=dataloader,
        denoise_optimizer=denoise_optimizer,
        te_optimizer=te_optimizer,
        te_scheduler=te_scheduler,
        resume=resume,
    )
