from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Environment and Paths
    pretrained_model_name_or_path: str = "/home/acite/LLM/models/diffusers/waillu_170"
    train_data_dir: str = "/home/acite/Pictures/05_babara"
    output_name: str = "babara"

    # train_data_dir: str = "/home/acite/Pictures/05_miyako"
    # output_name: str = "miyako"
    output_dir: str = "/home/acite/LLM/axltrainer/outputs"
    logging_dir: str = "/home/acite/LLM/axltrainer/logs"

    # Model / dataset spec
    base_model_version: str = "sdxl_base_v1-0"
    modelspec_architecture: str = "stable-diffusion-xl-v1-base/lora"
    modelspec_implementation: str = "https://github.com/Stability-AI/generative-models"
    modelspec_sai_model_spec: str = "1.0.0"

    # Training mode
    is_vpred: bool = False
    min_snr_gamma: float = 5.0

    # Core Hyperparameters
    seed: int = 1145141919
    mixed_precision: str = "bf16"
    train_batch_size: int = 3
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 240
    max_grad_norm: float = 1.0
    epoch: int = 48
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 40

    # Network Dimensions
    network_dim: int = 64
    network_alpha: int = 32
    network_dropout: float = 0.20
    clip_skip: int = 1
    max_token_length: int = 225

    # Aspect Ratio Bucketing
    enable_bucket: bool = True
    bucket_no_upscale: bool = True
    train_resolution: int = 1024
    bucket_reso_steps: int = 128 # Ref: Fix Log 1


    min_bucket_reso: int = 768
    max_bucket_reso: int = 1280

    # Optimization Features
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    shuffle_caption: bool = True
    keep_tokens: int = 2
    caption_extension: str = ".txt"
    noise_offset: float = 0.05

     # UNet optimizer (Schedule-Free AdamW)
    unet_learning_rate: float = 2.5e-5
    unet_weight_decay: float = 0.01
    unet_betas_1: float = 0.9
    unet_betas_2: float = 0.99
    unet_eps: float = 1e-8
    unet_warmup_steps: int = 240

    # TE optimizer (fixed AdamW)
    te_learning_rate: float = 2.5e-6
    te_weight_decay: float = 0.01
    te_betas_1: float = 0.9
    te_betas_2: float = 0.99
    te_max_grad_norm: float = 0.3

    # Infrastructure
    max_data_loader_n_workers: int = 20
    persistent_workers: bool = True

    # Inference Validation Samples
    sample_prompts: str = ("barbara_style, masterpiece, newest, soft_shading, source_anime, solo, looking at viewer, closeup, lying on bed, on back, unbuttoned, open clothes, sleepwear, unbuttoned shirt, bare shoulders, large breasts, cleavage, nipples, panties on one leg, pussy, masturbation, skin focus, blushing, shy, cute, indoors, bedroom, bed, soft lighting, warm atmosphere, peaceful,")
    sample_negative: str = "bad quality,worst quality,worst detail,sketch,censor, censorship, multi-person, group, gangbang, intercrural, internal, gore, guro, horror, non-human, monster, alien, zombie, fused fingers, distorted anatomy, bad composition, lowres"
    sample_width: int = 1280
    sample_height: int = 720
    sample_steps: int = 36
    sample_seed: int = 0
    sample_repeat: int = 3
    guidance_scale: float = 6.0

    # Optional kohya-like bookkeeping
    ss_session_id: int | None = None
    ss_training_comment: str | None = None
    ss_sd_model_hash: str | None = None
    ss_new_sd_model_hash: str | None = None
    ss_dataset_dirs: str | None = None
    ss_bucket_info: str | None = None

    _current_epoch = 0
