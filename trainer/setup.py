from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from diffusers import DDIMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import SDXLLoraDataset, make_collate_fn
from models import (
    build_scheduler,
    build_te_optimizer,
    build_unet_optimizer,
    enable_flash_attention,
    load_sdxl_pipeline,
)
from trainer.env import setup_migraphx_cache


@dataclass
class TrainArtifacts:
    accelerator: Accelerator
    device: torch.device
    weight_dtype: torch.dtype
    pipe: Any
    vae: torch.nn.Module
    unet: torch.nn.Module
    tokenizer_1: Any
    tokenizer_2: Any
    text_encoder_1: torch.nn.Module
    text_encoder_2: torch.nn.Module
    noise_scheduler: DDIMScheduler
    train_dataset: SDXLLoraDataset
    dataloader: DataLoader
    unet_optimizer: Any
    te_optimizer: Any
    unet_scheduler: Any
    te_scheduler: Any


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


def apply_lora_modules(cfg: TrainConfig, unet, text_encoder_1, text_encoder_2):
    """Attach LoRA adapters to UNet and both text encoders."""
    te_lora_config = LoraConfig(
        r=cfg.network_dim,
        lora_alpha=cfg.network_alpha,
        lora_dropout=cfg.network_dropout,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    unet_lora_config = LoraConfig(
        r=cfg.network_dim,
        lora_alpha=cfg.network_alpha,
        lora_dropout=cfg.network_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    text_encoder_1 = get_peft_model(text_encoder_1, te_lora_config)
    text_encoder_2 = get_peft_model(text_encoder_2, te_lora_config)
    unet = get_peft_model(unet, unet_lora_config)
    unet.enable_gradient_checkpointing()

    return unet, text_encoder_1, text_encoder_2


def build_dataloader(cfg: TrainConfig) -> tuple[SDXLLoraDataset, DataLoader]:
    """Build dataset and dataloader."""
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
    return train_dataset, dataloader


def build_optimizers_and_schedulers(
    cfg: TrainConfig,
    unet,
    text_encoder_1,
    text_encoder_2,
    dataloader: DataLoader,
):
    """Build optimizers and schedulers for UNet and text encoders."""
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    te_params = [
        p for p in text_encoder_1.parameters() if p.requires_grad
    ] + [
        p for p in text_encoder_2.parameters() if p.requires_grad
    ]

    unet_optimizer = build_unet_optimizer(cfg, unet_params)
    te_optimizer = build_te_optimizer(cfg, te_params)

    steps_per_epoch = max(1, math.ceil(len(dataloader) / cfg.gradient_accumulation_steps))
    total_steps = steps_per_epoch * cfg.epoch

    unet_scheduler = build_scheduler(unet_optimizer, total_steps, cfg)
    te_scheduler = build_scheduler(te_optimizer, total_steps, cfg)

    return unet_optimizer, te_optimizer, unet_scheduler, te_scheduler


def build_train_objects(cfg: TrainConfig) -> TrainArtifacts:
    """Build everything required for training in a readable sequence."""
    setup_migraphx_cache()
    maybe_enable_amp_backends()

    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    accelerator = create_accelerator(cfg)
    device = accelerator.device

    pipe = load_sdxl_pipeline(cfg.pretrained_model_name_or_path, weight_dtype)
    vae = pipe.vae
    unet = pipe.unet
    tokenizer_1, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_1, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2
    
    # Adapter for NoobAI
    scheduler_kwargs = {}
    if cfg.is_vpred:
        scheduler_kwargs["prediction_type"] = "v_prediction"
        scheduler_kwargs["rescale_betas_zero_snr"] = True
    else:
        scheduler_kwargs["prediction_type"] = "epsilon"

    noise_scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, 
        **scheduler_kwargs
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    enable_flash_attention(unet)

    unet, text_encoder_1, text_encoder_2 = apply_lora_modules(
        cfg,
        unet,
        text_encoder_1,
        text_encoder_2,
    )

    vae.to(device=device, dtype=weight_dtype).eval()
    text_encoder_1.to(device=device, dtype=weight_dtype)
    text_encoder_2.to(device=device, dtype=weight_dtype)

    train_dataset, dataloader = build_dataloader(cfg)
    unet_optimizer, te_optimizer, unet_scheduler, te_scheduler = build_optimizers_and_schedulers(
        cfg,
        unet,
        text_encoder_1,
        text_encoder_2,
        dataloader,
    )

    return TrainArtifacts(
        accelerator=accelerator,
        device=device,
        weight_dtype=weight_dtype,
        pipe=pipe,
        vae=vae,
        unet=unet,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        noise_scheduler=noise_scheduler,
        train_dataset=train_dataset,
        dataloader=dataloader,
        unet_optimizer=unet_optimizer,
        te_optimizer=te_optimizer,
        unet_scheduler=unet_scheduler,
        te_scheduler=te_scheduler,
    )
