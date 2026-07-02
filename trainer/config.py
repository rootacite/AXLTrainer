import tomllib
from dataclasses import dataclass, field
from typing import Optional

def _load_toml_config(file_path: str = "config.toml") -> dict:
    try:
        with open(file_path, "rb") as f:
            raw_toml = tomllib.load(f)
            
        flat_config = {}
        for section in raw_toml.values():
            if isinstance(section, dict):
                flat_config.update(section)
        return flat_config
    except FileNotFoundError:
        print(f"[Warn] {file_path} not found. Using hardcoded defaults.")
        return {}

_CONFIG = _load_toml_config()

def get_val(key: str, default):
    return _CONFIG.get(key, default)

@dataclass
class TrainConfig:
    # Environment and Paths
    pretrained_model_name_or_path: str = get_val("pretrained_model_name_or_path", "/home/acite/LLM/models/diffusers/waillu_170")
    train_data_dir: str = get_val("train_data_dir", "/home/acite/LLM/Character/kanae")
    output_name: str = get_val("output_name", "kanae")
    output_dir: str = get_val("output_dir", "/home/acite/LLM/axltrainer/outputs")
    logging_dir: str = get_val("logging_dir", "/home/acite/LLM/axltrainer/logs")

    # Model / dataset spec
    base_model_version: str = get_val("base_model_version", "sdxl_base_v1-0")
    modelspec_architecture: str = get_val("modelspec_architecture", "stable-diffusion-xl-v1-base/lora")
    modelspec_implementation: str = get_val("modelspec_implementation", "https://github.com/Stability-AI/generative-models")
    modelspec_sai_model_spec: str = get_val("modelspec_sai_model_spec", "1.0.0")

    # Training mode
    is_vpred: bool = get_val("is_vpred", False)
    min_snr_gamma: float = get_val("min_snr_gamma", 5.0)

    # Core Hyperparameters
    seed: int = get_val("seed", 1145141919)
    mixed_precision: str = get_val("mixed_precision", "bf16")
    train_batch_size: int = get_val("train_batch_size", 3)
    gradient_accumulation_steps: int = get_val("gradient_accumulation_steps", 1)
    learning_rate: float = get_val("learning_rate", 1.0)
    lr_scheduler: str = get_val("lr_scheduler", "cosine")
    lr_warmup_steps: int = get_val("lr_warmup_steps", 100)
    max_grad_norm: float = get_val("max_grad_norm", 1.0)
    epoch: int = get_val("epoch", 60)
    save_every_n_epochs: int = get_val("save_every_n_epochs", 1)
    save_every_n_steps: int = get_val("save_every_n_steps", 100)

    # Network Dimensions
    network_dim: int = get_val("network_dim", 48)
    network_alpha: int = get_val("network_alpha", 24)
    network_dropout: float = get_val("network_dropout", 0.15)
    clip_skip: int = get_val("clip_skip", 1)
    max_token_length: int = get_val("max_token_length", 225)

    # Aspect Ratio Bucketing
    enable_bucket: bool = get_val("enable_bucket", True)
    bucket_no_upscale: bool = get_val("bucket_no_upscale", True)
    train_resolution: int = get_val("train_resolution", 1024)
    bucket_reso_steps: int = get_val("bucket_reso_steps", 128)
    min_bucket_reso: int = get_val("min_bucket_reso", 768)
    max_bucket_reso: int = get_val("max_bucket_reso", 1280)

    # Optimization Features
    cache_latents: bool = get_val("cache_latents", True)
    cache_latents_to_disk: bool = get_val("cache_latents_to_disk", True)
    shuffle_caption: bool = get_val("shuffle_caption", True)
    keep_tokens: int = get_val("keep_tokens", 2)
    caption_extension: str = get_val("caption_extension", ".txt")
    noise_offset: float = get_val("noise_offset", 0.05)

    # UNet optimizer (Schedule-Free AdamW)
    unet_learning_rate: float = get_val("unet_learning_rate", 6e-5)
    unet_weight_decay: float = get_val("unet_weight_decay", 0.01)
    unet_betas_1: float = get_val("unet_betas_1", 0.9)
    unet_betas_2: float = get_val("unet_betas_2", 0.99)
    unet_eps: float = get_val("unet_eps", 1e-8)
    unet_warmup_steps: int = get_val("unet_warmup_steps", 100)

    # TE optimizer (fixed AdamW)
    te_learning_rate: float = get_val("te_learning_rate", 6e-6)
    te_weight_decay: float = get_val("te_weight_decay", 0.01)
    te_betas_1: float = get_val("te_betas_1", 0.9)
    te_betas_2: float = get_val("te_betas_2", 0.99)
    te_max_grad_norm: float = get_val("te_max_grad_norm", 0.3)

    # Infrastructure
    max_data_loader_n_workers: int = get_val("max_data_loader_n_workers", 20)
    persistent_workers: bool = get_val("persistent_workers", True)

    # Inference Validation Samples
    sample_prompts: str = get_val("sample_prompts", "(kanae_style:1.2), masterpiece, best quality, amazing quality, newest, soft_shading, source_anime, solo, white thighhighs, 1girl, full body, from above")
    sample_negative: str = get_val("sample_negative", "bad quality, worst quality, worst detail, sketch, multi-person, group, gangbang, intercrural, internal, gore, guro, horror, non-human, monster, alien, zombie, fused fingers, distorted anatomy, bad composition, lowres")
    sample_width: int = get_val("sample_width", 1280)
    sample_height: int = get_val("sample_height", 720)
    sample_steps: int = get_val("sample_steps", 55)
    sample_seed: int = get_val("sample_seed", 0)
    sample_repeat: int = get_val("sample_repeat", 3)
    guidance_scale: float = get_val("guidance_scale", 6.0)

    # Optional kohya-like bookkeeping (Defaults to None)
    ss_session_id: Optional[int] = get_val("ss_session_id", None)
    ss_training_comment: Optional[str] = get_val("ss_training_comment", None)
    ss_sd_model_hash: Optional[str] = get_val("ss_sd_model_hash", None)
    ss_new_sd_model_hash: Optional[str] = get_val("ss_new_sd_model_hash", None)
    ss_dataset_dirs: Optional[str] = get_val("ss_dataset_dirs", None)
    ss_bucket_info: Optional[str] = get_val("ss_bucket_info", None)

    _current_epoch: int = field(default=0, init=False)
