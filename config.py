from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Environment and Paths
    pretrained_model_name_or_path: str = "/home/acite/LLM/ComfyUI/models/checkpoints/waiIllustriousSDXL_v170.safetensors"
    train_data_dir: str = "/home/acite/Pictures/03_tsukuyumi"
    output_dir: str = "/tmp/axltrainer//outputs"
    logging_dir: str = "/tmp/axltrainer/logs"
    output_name: str = "tsukuyumi"

    is_vpred: bool = False    # For noobAI   
    min_snr_gamma: float = 5.0   

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
    network_dropout: float = 0.25
    clip_skip: int = 1
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

    # UNet optimizer (fixed Prodigy)
    unet_learning_rate: float = 1.0
    unet_prodigy_args: str = (
        '"decouple=True" "weight_decay=0.03" "d_coef=1.0" '
        '"use_bias_correction=True" "safeguard_warmup=True" "betas=0.9,0.99"'
    )

    # TE optimizer (fixed AdamW)
    te_learning_rate: float = 5e-5
    te_weight_decay: float = 0.01
    te_betas_1: float = 0.9
    te_betas_2: float = 0.99
    te_max_grad_norm: float = 0.3

    # Infrastructure
    max_data_loader_n_workers: int = 20
    persistent_workers: bool = True

    # Inference Validation Samples
    sample_prompts: str = (
        "tsukuyumi_style,cube_style,newest,1girl,1boy,visual novel,soft shading, "
        "low twintails, grey hair, looking at viewer, dress, twintails,large breasts,closed mouth,shy,smile,lying on bed,"
    )
    sample_negative: str = "lowres, bad anatomy, (mosaic censoring:1.3), (text:1.3)"
    sample_width: int = 1024
    sample_height: int = 1024
    sample_steps: int = 36
    sample_seed: int = 0
    sample_repeat: int = 3
    guidance_scale: float = 6.3
