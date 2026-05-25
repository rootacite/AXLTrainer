# models.py
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional

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
        # step 是 scheduler.step() 之后的全局更新步
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


def build_optimizer(cfg: TrainConfig, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if cfg.optimizer.lower() == "prodigy":
        try:
            from prodigyopt import Prodigy
            kwargs = parse_kv_args(cfg.optimizer_args)
            return Prodigy(params, lr=cfg.learning_rate, **kwargs)
        except ImportError:
            logger.warning("Prodigy optimizer package missing. Falling back cleanly to AdamW.")
            return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.03)

    return torch.optim.AdamW(params, lr=cfg.learning_rate)


def load_sdxl_pipeline(path: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    path_obj = Path(path)
    loader_func = StableDiffusionXLPipeline.from_single_file if path_obj.is_file() else StableDiffusionXLPipeline.from_pretrained
    
    return loader_func(
        str(path_obj),
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

def save_lora_checkpoint(
    accelerator: Accelerator,
    unet: Any,
    cfg: TrainConfig,
    global_step: int,
    epoch: Optional[int] = None,
    final: bool = False,
) -> None:
    if not accelerator.is_main_process:
        return

    from peft import get_peft_model_state_dict
    unwrapped = accelerator.unwrap_model(unet)
    lora_state = get_peft_model_state_dict(unwrapped)

    if final:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_final"
    elif epoch is not None:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_e{epoch:03d}_s{global_step:06d}"
    else:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_s{global_step:06d}"
        
    out_dir.mkdir(parents=True, exist_ok=True)

    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=lora_state,
        safe_serialization=True,
    )
    save_file(lora_state, str(out_dir / "pytorch_lora_weights.safetensors"))
