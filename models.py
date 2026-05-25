# models.py
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0   # 新增

from config import TrainConfig
from utils import parse_kv_args

logger = logging.getLogger(__name__)


def enable_flash_attention(unet: Any) -> None:
    """
    启用 PyTorch 2.x 的 SDPA 注意力实现。
    在支持的环境下会走 flash/memory-efficient backend。
    """
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
            return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.04)

    return torch.optim.AdamW(params, lr=cfg.learning_rate)

def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, cfg: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=0.0)
    if cfg.lr_scheduler == "linear":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / max(1, total_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

def load_sdxl_pipeline(path: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    path_obj = Path(path)
    loader_func = StableDiffusionXLPipeline.from_single_file if path_obj.is_file() else StableDiffusionXLPipeline.from_pretrained
    
    return loader_func(
        str(path_obj),
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
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
