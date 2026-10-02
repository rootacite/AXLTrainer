from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from diffusers.models.attention_processor import AttnProcessor2_0
from safetensors import safe_open

import logging

logger = logging.getLogger(__name__)

try:
    from config import TrainConfig
except ImportError:
    from trainer.config import TrainConfig


def sample_scheduler_kwargs(cfg: TrainConfig, base_scheduler_config: Any) -> dict[str, Any]:
    """The `prediction_type` / `rescale_betas_zero_snr` pair a sample pass has to render with.

    A model trained to predict velocities renders to noise when the sampler resolves epsilons
    (and the reverse), so the *sampling* scheduler has to agree with the base model — the same
    agreement `build_noise_scheduler` (family_sdxl) keeps on the training side, from the same two
    derived `TrainConfig` fields. The base pipeline's own scheduler config is the second source
    (a diffusers v-pred repo), and a single-file base's `v_pred` / `ztsnr` marker tensors are
    folded into `cfg` already. ComfyUI's equivalent pair is `ModelSamplingDiscrete = v_prediction`
    with `zsnr = true`.
    """
    v_prediction = str(getattr(cfg, "prediction_type", "epsilon")) == "v_prediction"
    zero_snr = bool(getattr(cfg, "zero_terminal_snr", False))
    base_prediction = str(getattr(base_scheduler_config, "prediction_type", "epsilon") or "")
    v_prediction = v_prediction or base_prediction == "v_prediction"
    zero_snr = zero_snr or bool(getattr(base_scheduler_config, "rescale_betas_zero_snr", False))
    return {
        "prediction_type": "v_prediction" if v_prediction else "epsilon",
        "rescale_betas_zero_snr": zero_snr,
    }


def enable_flash_attention(unet: Any) -> None:
    if not hasattr(unet, "set_attn_processor"):
        logger.warning("UNet does not support set_attn_processor; flash attention not enabled.")
        return
    unet.set_attn_processor(AttnProcessor2_0())
    logger.info("UNet attention processor set to AttnProcessor2_0.")


def _schedule_free_class():
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError as e:
        raise RuntimeError("schedulefree is required for the trainer's optimizers.") from e
    return AdamWScheduleFree


def build_unet_optimizer(cfg: TrainConfig, params):
    # `eps` keeps the library default (1e-8), like the TE optimizer's.
    AdamWScheduleFree = _schedule_free_class()
    return AdamWScheduleFree(
        params,
        lr=cfg.unet_learning_rate,
        betas=(cfg.unet_betas_1, cfg.unet_betas_2),
        weight_decay=cfg.unet_weight_decay,
        warmup_steps=cfg.unet_warmup_steps,
    )


def build_te_optimizer(cfg: TrainConfig, params):
    # Schedule-Free AdamW like the UNet, with its own warmup (`te_warmup_steps`); `eps` keeps the
    # library default (1e-8).
    AdamWScheduleFree = _schedule_free_class()
    return AdamWScheduleFree(
        params,
        lr=cfg.te_learning_rate,
        betas=(cfg.te_betas_1, cfg.te_betas_2),
        weight_decay=cfg.te_weight_decay,
        warmup_steps=cfg.te_warmup_steps,
    )


def build_kohya_metadata(
    cfg: TrainConfig,
    global_step: int,
    epoch: Optional[int],
    final: bool,
    *,
    prediction_type: str,
) -> dict[str, str]:
    meta: dict[str, str] = {}

    def put(key: str, value: Any) -> None:
        if value is None:
            return
        meta[key] = str(value)

    put("modelspec.sai_model_spec", cfg.modelspec_sai_model_spec)
    put("modelspec.implementation", cfg.modelspec_implementation)
    put("modelspec.architecture", cfg.modelspec_architecture)
    put("modelspec.prediction_type", prediction_type)
    put("modelspec.title", cfg.output_name)
    put("modelspec.date", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    put("ss_network_module", "networks.lora")
    network_type = str(getattr(cfg, "network_type", "standard") or "standard").strip().lower()
    put("ss_network_type", network_type)
    put("ss_network_dim", cfg.network_dim)
    put("ss_network_alpha", cfg.network_alpha)
    if network_type == "locon":
        put(
            "ss_network_args",
            f"conv_dim={int(cfg.conv_dim)} conv_alpha={int(cfg.conv_alpha)}",
        )
    put("ss_output_name", cfg.output_name)
    put("ss_seed", cfg.seed)
    put("ss_steps", global_step)
    put("ss_epoch", 0 if epoch is None else epoch)
    put("ss_final", int(bool(final)))
    put("ss_base_model_version", cfg.base_model_version)
    put("ss_v_pred", int(cfg.prediction_type == "v_prediction"))
    # kohya records its `--min_snr_gamma` argument; this records the value that actually ran, and
    # `TrainConfig` derives 0 for a v-prediction base, so a file asking for Min-SNR on one says so.
    put("ss_min_snr_gamma", cfg.min_snr_gamma)

    put("ss_learning_rate", cfg.learning_rate)
    put("ss_unet_lr", cfg.unet_learning_rate)
    put("ss_text_encoder_lr", cfg.te_learning_rate)
    put("ss_lr_warmup_steps", cfg.te_warmup_steps)
    put("ss_mixed_precision", cfg.mixed_precision)
    put("ss_max_grad_norm", cfg.unet_max_grad_norm)
    put("ss_clip_skip", cfg.clip_skip)
    put("ss_network_dropout", cfg.network_dropout)
    put("ss_enable_bucket", cfg.enable_bucket)
    put("ss_bucket_no_upscale", cfg.bucket_no_upscale)
    put("ss_min_bucket_reso", cfg.min_bucket_reso)
    put("ss_max_bucket_reso", cfg.max_bucket_reso)
    put("ss_resolution", cfg.train_resolution)
    put("ss_max_token_length", cfg.max_token_length)
    put("ss_keep_tokens", cfg.keep_tokens)
    put("ss_noise_offset", cfg.noise_offset)
    put("ss_shuffle_caption", cfg.shuffle_caption)
    put("ss_train_data_dir", cfg.train_data_dir)
    put("ss_pretrained_model_name_or_path", cfg.pretrained_model_name_or_path)

    put("ss_session_id", cfg.ss_session_id)
    put("ss_training_comment", cfg.ss_training_comment)
    put("ss_sd_model_hash", cfg.ss_sd_model_hash)
    put("ss_new_sd_model_hash", cfg.ss_new_sd_model_hash)
    put("ss_dataset_dirs", cfg.ss_dataset_dirs)
    put("ss_bucket_info", cfg.ss_bucket_info)

    return meta


def safe_output_name(name: str) -> str:
    text = (name or "").strip() or "lora"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return cleaned or "lora"


def artifact_root(cfg: TrainConfig) -> Path:
    """Run-scoped artifact directory (falls back to output_dir outside a run)."""
    run_dir = getattr(cfg, "run_dir", "")
    return Path(run_dir) if run_dir else Path(cfg.output_dir)


def lora_checkpoint_file(
    cfg: TrainConfig,
    global_step: int,
    epoch: Optional[int] = None,
    final: bool = False,
) -> Path:
    name = safe_output_name(cfg.output_name)
    if final:
        out_dir = artifact_root(cfg) / f"{cfg.output_name}_final"
    elif epoch is not None:
        out_dir = artifact_root(cfg) / f"{cfg.output_name}_e{epoch:03d}_s{global_step:06d}"
    else:
        out_dir = artifact_root(cfg) / f"{cfg.output_name}_s{global_step:06d}"
    return out_dir / f"{name}.safetensors"
