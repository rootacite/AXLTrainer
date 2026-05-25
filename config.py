from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Environment and Paths
    pretrained_model_name_or_path: str = "/home/acite/LLM/ComfyUI/models/checkpoints/waiIllustriousSDXL_v160.safetensors"
    train_data_dir: str = "/home/acite/Pictures/03_tsukuyumi"
    output_dir: str = "/home/acite/LLM/kohya_ss/outputs"
    logging_dir: str = "/home/acite/LLM/kohya_ss/logs"
    output_name: str = "tsukuyumi"

    # Core Hyperparameters
    seed: int = 1145141919
    mixed_precision: str = "bf16"
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    max_grad_norm: float = 1.0
    epoch: int = 32
    save_every_n_epochs: int = 1
    save_every_n_steps: int = 20

    # Network Dimensions
    network_dim: int = 64
    network_alpha: int = 64
    network_dropout: float = 0.2
    clip_skip: int = 2
    max_token_length: int = 225

    # Aspect Ratio Bucketing
    enable_bucket: bool = True
    bucket_no_upscale: bool = True
    train_resolution: int = 1024
    bucket_reso_steps: int = 64
    min_bucket_reso: int = 768
    max_bucket_reso: int = 1280

    # Optimization Features
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    shuffle_caption: bool = True
    keep_tokens: int = 2
    caption_extension: str = ".txt"
    noise_offset: float = 0.05

    # Optimizer Configurations
    optimizer: str = "Prodigy"
    optimizer_args: str = (
        '"decouple=True" "weight_decay=0.03" "d_coef=1.0" '
        '"use_bias_correction=True" "safeguard_warmup=True" "betas=0.9,0.99" '
    )

    # Infrastructure
    max_data_loader_n_workers: int = 20
    persistent_workers: bool = True

    # Inference Validation Samples
    sample_prompts: str = (
        "tsukuyumi_style,cube_style,newest,1girl,1boy,visual novel,soft shading, "
        "low twintails, grey hair, looking at viewer, dress, twintails,large breasts,closed mouth,shy,smile,lying on bed,"
    )
    sample_negative: str = "lowres, bad anatomy, (mosaic censoring:1.3)"
    sample_width: int = 768
    sample_height: int = 768
    sample_steps: int = 35
    sample_seed: int = 0
    sample_repeat: int = 3
    guidance_scale: float = 6.0
