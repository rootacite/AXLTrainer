# models.py
import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0

from config import TrainConfig
from utils import parse_kv_args

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    cfg: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = max(0, min(cfg.lr_warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))

        if cfg.lr_scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if cfg.lr_scheduler == "linear":
            return 1.0 - progress
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def enable_flash_attention(unet: Any) -> None:
    if not hasattr(unet, "set_attn_processor"):
        logger.warning("UNet does not support set_attn_processor; flash attention not enabled.")
        return

    unet.set_attn_processor(AttnProcessor2_0())
    logger.info("UNet attention processor set to AttnProcessor2_0.")


def build_unet_optimizer(cfg: TrainConfig, params):
    try:
        from prodigyopt import Prodigy
    except ImportError as e:
        raise RuntimeError("Prodigy is required for UNet optimizer.") from e

    kwargs = parse_kv_args(cfg.unet_prodigy_args)
    return Prodigy(params, lr=cfg.unet_learning_rate, **kwargs)


def build_te_optimizer(cfg: TrainConfig, params):
    return torch.optim.AdamW(
        params,
        lr=cfg.te_learning_rate,
        betas=(cfg.te_betas_1, cfg.te_betas_2),
        weight_decay=cfg.te_weight_decay,
    )


def load_sdxl_pipeline(path: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    path_obj = Path(path)
    loader_func = (
        StableDiffusionXLPipeline.from_single_file
        if path_obj.is_file()
        else StableDiffusionXLPipeline.from_pretrained
    )

    return loader_func(
        str(path_obj),
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def _merge_lora_state_dicts(*state_dicts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Merge LoRA tensors into one flat safetensors payload."""
    merged: dict[str, torch.Tensor] = {}
    for state_dict in state_dicts:
        for key, value in state_dict.items():
            merged[key] = value.detach().cpu()
    return merged


def save_lora_checkpoint(
    accelerator: Accelerator,
    unet: Any,
    text_encoder_1: Any,
    text_encoder_2: Any,
    cfg: TrainConfig,
    global_step: int,
    epoch: Optional[int] = None,
    final: bool = False,
) -> None:
    """Export a ComfyUI-friendly LoRA safetensors file."""
    if not accelerator.is_main_process:
        return

    from peft import get_peft_model_state_dict

    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_te1 = accelerator.unwrap_model(text_encoder_1)
    unwrapped_te2 = accelerator.unwrap_model(text_encoder_2)

    unet_lora_state = get_peft_model_state_dict(unwrapped_unet)
    te1_lora_state = get_peft_model_state_dict(unwrapped_te1)
    te2_lora_state = get_peft_model_state_dict(unwrapped_te2)

    if final:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_final"
    elif epoch is not None:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_e{epoch:03d}_s{global_step:06d}"
    else:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_s{global_step:06d}"

    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "pytorch_lora_weights.safetensors"

    # Keep the export self-describing for downstream tools.
    metadata = {
        "format": "diffusers_lora",
        "base_model": str(cfg.pretrained_model_name_or_path),
        "architecture": "sdxl",
        "trained_components": "unet,text_encoder,text_encoder_2",
        "network_dim": str(cfg.network_dim),
        "network_alpha": str(cfg.network_alpha),
        "output_name": str(cfg.output_name),
        "global_step": str(global_step),
        "final": str(bool(final)),
    }

    merged_state = _merge_lora_state_dicts(
        unet_lora_state,
        te1_lora_state,
        te2_lora_state,
    )

    save_file(merged_state, str(out_file), metadata=metadata)
    logger.info("Saved LoRA checkpoint to %s", out_file)