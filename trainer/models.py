import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from safetensors.torch import save_file

from config import TrainConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDXL UNet: diffusers path  →  kohya / ComfyUI ldm path
#
# Attention blocks (transformer blocks inside ResNet stages)
# ---------------------------------------------------------------------------
_UNET_ATTN_MAP: dict[str, str] = {
    "down_blocks.1.attentions.0": "input_blocks.4.1",
    "down_blocks.1.attentions.1": "input_blocks.5.1",
    "down_blocks.2.attentions.0": "input_blocks.7.1",
    "down_blocks.2.attentions.1": "input_blocks.8.1",
    "mid_block.attentions.0":     "middle_block.1",
    "up_blocks.0.attentions.0":   "output_blocks.0.1",
    "up_blocks.0.attentions.1":   "output_blocks.1.1",
    "up_blocks.0.attentions.2":   "output_blocks.2.1",
    "up_blocks.1.attentions.0":   "output_blocks.3.1",
    "up_blocks.1.attentions.1":   "output_blocks.4.1",
    "up_blocks.1.attentions.2":   "output_blocks.5.1",
}

# ResNet shortcut convolutions (conv_shortcut → skip_connection)
_UNET_RESNET_MAP: dict[str, str] = {
    "down_blocks.0.resnets.0.conv_shortcut": "input_blocks.1.0.skip_connection",
    "down_blocks.0.resnets.1.conv_shortcut": "input_blocks.2.0.skip_connection",
    "down_blocks.1.resnets.1.conv_shortcut": "input_blocks.5.0.skip_connection",
    "down_blocks.2.resnets.1.conv_shortcut": "input_blocks.8.0.skip_connection",
    "mid_block.resnets.0.conv_shortcut":     "middle_block.0.skip_connection",
    "mid_block.resnets.1.conv_shortcut":     "middle_block.2.skip_connection",
    # up_blocks — no shortcut convs in standard SDXL, but kept for completeness
}

# Downsampler / upsampler convolutions
_UNET_SAMPLER_MAP: dict[str, str] = {
    "down_blocks.2.downsamplers.0.conv": "input_blocks.9.0.op",
    "up_blocks.2.upsamplers.0.conv":     "output_blocks.8.1.conv",
}

# Time / label embedding projections
_UNET_EMBED_MAP: dict[str, str] = {
    "class_embedding.linear_1": "label_emb.0.0",
    "class_embedding.linear_2": "label_emb.0.2",
    # add_embedding used in some SDXL variants
    "add_embedding.linear_1":   "label_emb.0.0",
    "add_embedding.linear_2":   "label_emb.0.2",
}

# Merged lookup: longest-match wins (applied in order, so put longer keys first)
_UNET_PATH_MAP: dict[str, str] = {
    **_UNET_RESNET_MAP,
    **_UNET_SAMPLER_MAP,
    **_UNET_EMBED_MAP,
    **_UNET_ATTN_MAP,
}

# ---------------------------------------------------------------------------
# CLIP / text-encoder path fragments that need renaming
# Applied as simple string replacements, longest-first.
# ---------------------------------------------------------------------------
_TE_PATH_REPLACEMENTS: list[tuple[str, str]] = [
    # Layer stack
    ("text_model.encoder.layers.", "text_model_encoder_layers_"),
    # Embeddings sub-module
    ("text_model.embeddings.", "text_model_embeddings_"),
    # Top-level projections and norms (te2 / OpenCLIP specific)
    ("text_projection.",        "text_projection_"),
    ("text_model.final_layer_norm.", "text_model_final_layer_norm_"),
    # te1 (OpenAI CLIP) top-level norm
    ("final_layer_norm.",       "final_layer_norm_"),
]


# ---------------------------------------------------------------------------
# Helper: remap a raw PEFT key to the kohya / ComfyUI naming convention
# ---------------------------------------------------------------------------
def _remap_unet_path(key: str) -> str:
    """Map a diffusers UNet sub-path to its ldm/kohya equivalent."""
    key = key.replace("unet.", "", 1)
    # Apply all substitutions — iterate so overlapping prefixes resolve correctly
    for src, dst in _UNET_PATH_MAP.items():
        if src in key:
            key = key.replace(src, dst, 1)
            break  # each layer matches exactly one rule
    return key


def _remap_te_path(key: str) -> str:
    """Map a diffusers text-encoder sub-path to its kohya equivalent."""
    for src, dst in _TE_PATH_REPLACEMENTS:
        key = key.replace(src, dst)
    return key


def _remap_core_path(raw_key: str, prefix: str) -> str:
    """
    Strip PEFT wrappers and apply the appropriate path remapping for
    *prefix* ∈ {"unet", "te1", "te2"}.

    Returns the dot-free, underscore-joined core path used in kohya keys.
    """
    key = raw_key.replace("base_model.model.", "")

    if prefix == "unet":
        key = _remap_unet_path(key)
    elif prefix in ("te1", "te2"):
        key = _remap_te_path(key)

    # Final normalisation: dots → underscores
    return key.replace(".", "_")


# ---------------------------------------------------------------------------
# PEFT state-dict → kohya safetensors conversion
# ---------------------------------------------------------------------------
def _convert_peft_to_kohya_bf16(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """
    Convert a PEFT LoRA state-dict to the kohya / ComfyUI safetensors format.

    Key schema:
        lora_{prefix}_{core_path}.lora_down.weight
        lora_{prefix}_{core_path}.lora_up.weight
        lora_{prefix}_{core_path}.alpha          (scalar)

    All tensors are cast to bfloat16 and moved to CPU.
    """
    converted: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        key = k.replace("base_model.model.", "")

        if ".lora_A." in key:
            suffix = "lora_down.weight"
            core_raw = key.split(".lora_A.")[0]
        elif ".lora_B." in key:
            suffix = "lora_up.weight"
            core_raw = key.split(".lora_B.")[0]
        else:
            # Pass-through weights (e.g. lora_magnitude_vector) — skip for now
            logger.debug("Skipping non-AB LoRA key: %s", k)
            continue

        core_path = _remap_core_path(core_raw, prefix)
        weight_key = f"lora_{prefix}_{core_path}.{suffix}"
        converted[weight_key] = v.detach().cpu().to(dtype=torch.bfloat16)

        alpha_key = f"lora_{prefix}_{core_path}.alpha"
        if alpha_key not in converted:
            converted[alpha_key] = torch.tensor(alpha, dtype=torch.bfloat16)

    return converted


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Attention / optimizer helpers
# ---------------------------------------------------------------------------
def enable_flash_attention(unet: Any) -> None:
    if not hasattr(unet, "set_attn_processor"):
        logger.warning("UNet does not support set_attn_processor; flash attention not enabled.")
        return
    unet.set_attn_processor(AttnProcessor2_0())
    logger.info("UNet attention processor set to AttnProcessor2_0.")


def build_unet_optimizer(cfg: TrainConfig, params):
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError as e:
        raise RuntimeError("schedulefree is required for UNet optimizer.") from e

    return AdamWScheduleFree(
        params,
        lr=cfg.unet_learning_rate,
        betas=(cfg.unet_betas_1, cfg.unet_betas_2),
        eps=cfg.unet_eps,
        weight_decay=cfg.unet_weight_decay,
        warmup_steps=cfg.unet_warmup_steps,
    )


def build_te_optimizer(cfg: TrainConfig, params):
    return torch.optim.AdamW(
        params,
        lr=cfg.te_learning_rate,
        betas=(cfg.te_betas_1, cfg.te_betas_2),
        weight_decay=cfg.te_weight_decay,
    )


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Kohya metadata builder
# ---------------------------------------------------------------------------
def _build_kohya_metadata(
    cfg: TrainConfig,
    global_step: int,
    epoch: Optional[int],
    final: bool,
) -> dict[str, str]:
    meta: dict[str, str] = {}

    def put(key: str, value: Any) -> None:
        if value is None:
            return
        meta[key] = str(value)

    put("modelspec.sai_model_spec",    cfg.modelspec_sai_model_spec)
    put("modelspec.implementation",    cfg.modelspec_implementation)
    put("modelspec.architecture",      cfg.modelspec_architecture)
    put("modelspec.prediction_type",   "epsilon")
    put("modelspec.title",             cfg.output_name)
    put("modelspec.date",              datetime.now(timezone.utc).isoformat(timespec="seconds"))

    put("ss_network_module",           "networks.lora")
    put("ss_network_dim",              cfg.network_dim)
    put("ss_network_alpha",            cfg.network_alpha)
    put("ss_output_name",              cfg.output_name)
    put("ss_seed",                     cfg.seed)
    put("ss_steps",                    global_step)
    put("ss_epoch",                    0 if epoch is None else epoch)
    put("ss_final",                    int(bool(final)))

    put("ss_learning_rate",            cfg.learning_rate)
    put("ss_unet_lr",                  cfg.unet_learning_rate)
    put("ss_text_encoder_lr",          cfg.te_learning_rate)
    put("ss_lr_scheduler",             cfg.lr_scheduler)
    put("ss_lr_warmup_steps",          cfg.lr_warmup_steps)
    put("ss_mixed_precision",          cfg.mixed_precision)
    put("ss_max_grad_norm",            cfg.max_grad_norm)
    put("ss_clip_skip",                cfg.clip_skip)
    put("ss_network_dropout",          cfg.network_dropout)
    put("ss_enable_bucket",            cfg.enable_bucket)
    put("ss_bucket_no_upscale",        cfg.bucket_no_upscale)
    put("ss_min_bucket_reso",          cfg.min_bucket_reso)
    put("ss_max_bucket_reso",          cfg.max_bucket_reso)
    put("ss_resolution",               cfg.train_resolution)
    put("ss_max_token_length",         cfg.max_token_length)
    put("ss_keep_tokens",              cfg.keep_tokens)
    put("ss_noise_offset",             cfg.noise_offset)
    put("ss_shuffle_caption",          cfg.shuffle_caption)
    put("ss_train_data_dir",           cfg.train_data_dir)
    put("ss_pretrained_model_name_or_path", cfg.pretrained_model_name_or_path)

    put("ss_session_id",               cfg.ss_session_id)
    put("ss_training_comment",         cfg.ss_training_comment)
    put("ss_sd_model_hash",            cfg.ss_sd_model_hash)
    put("ss_new_sd_model_hash",        cfg.ss_new_sd_model_hash)
    put("ss_dataset_dirs",             cfg.ss_dataset_dirs)
    put("ss_bucket_info",              cfg.ss_bucket_info)

    return meta


# ---------------------------------------------------------------------------
# Checkpoint saver
# ---------------------------------------------------------------------------
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
    """
    Save a kohya-style LoRA checkpoint compatible with ComfyUI.

    Key layout (all bfloat16):
        lora_unet_{path}.lora_down.weight / lora_up.weight / alpha
        lora_te1_{path}.lora_down.weight  / lora_up.weight / alpha
        lora_te2_{path}.lora_down.weight  / lora_up.weight / alpha
    """
    if not accelerator.is_main_process:
        return

    from peft import get_peft_model_state_dict

    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_te1  = accelerator.unwrap_model(text_encoder_1)
    unwrapped_te2  = accelerator.unwrap_model(text_encoder_2)

    unet_state = _convert_peft_to_kohya_bf16(
        get_peft_model_state_dict(unwrapped_unet), "unet", cfg.network_alpha
    )
    te1_state = _convert_peft_to_kohya_bf16(
        get_peft_model_state_dict(unwrapped_te1), "te1", cfg.network_alpha
    )
    te2_state = _convert_peft_to_kohya_bf16(
        get_peft_model_state_dict(unwrapped_te2), "te2", cfg.network_alpha
    )

    # Detect key collisions across components (should never happen in practice)
    all_keys = list(unet_state) + list(te1_state) + list(te2_state)
    if len(all_keys) != len(set(all_keys)):
        logger.warning("Duplicate keys detected when merging LoRA state dicts!")

    merged: dict[str, torch.Tensor] = {**unet_state, **te1_state, **te2_state}

    # Determine output path
    if final:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_final"
    elif epoch is not None:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_e{epoch:03d}_s{global_step:06d}"
    else:
        out_dir = Path(cfg.output_dir) / f"{cfg.output_name}_s{global_step:06d}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "pytorch_lora_weights.safetensors"

    metadata = _build_kohya_metadata(cfg, global_step, epoch, final)
    save_file(merged, str(out_file), metadata=metadata)

    unet_keys = len(unet_state) // 3  # each layer = down + up + alpha
    te1_keys  = len(te1_state)  // 3
    te2_keys  = len(te2_state)  // 3
    logger.info(
        "Saved LoRA checkpoint → %s  |  unet=%d  te1=%d  te2=%d layers  dtype=bf16",
        out_file, unet_keys, te1_keys, te2_keys,
    )
