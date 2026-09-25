import json
import tomllib
from dataclasses import dataclass, field
from typing import Any, Optional

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


# Ranges shared with the Ranko Validation form; a value outside them is rejected
# in the GUI and again here, so a hand-edited config.toml fails at startup with
# a message naming the offending entry.
SAMPLE_SIZE_RANGE = (64, 4096)
SAMPLE_STEPS_RANGE = (1, 150)
SAMPLE_CFG_RANGE = (0.0, 30.0)
SAMPLE_SEED_RANGE = (0, 2**32 - 1)
SAMPLE_REPEAT_RANGE = (1, 32)

# How many times one dataset folder is drawn inside a single epoch. Shared with the Ranko
# Environment form, which rejects a value outside it before saving.
TRAIN_DATA_REPEAT_RANGE = (1, 512)


@dataclass(frozen=True)
class SampleSet:
    """One `[[validation.samples]]` entry with every key resolved."""

    name: str
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    repeat: int


@dataclass(frozen=True)
class TrainDataEntry:
    """One `[[environment.train_data]]` entry: a dataset folder and its per-epoch repeat."""

    path: str
    repeat: int


def _set_int(value, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        raise ValueError("expected an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("expected an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected an integer") from exc


def _set_float(value, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        raise ValueError("expected a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected a number") from exc


def _set_str(value, default: str) -> str:
    if value is None:
        return str(default)
    return str(value)


def _check_range(label: str, value, bounds) -> None:
    low, high = bounds
    if not (low <= value <= high):
        raise ValueError(f"{label} must be between {low} and {high}")


def _default_set_name(prompt: str, index: int) -> str:
    tag = prompt.split(",")[0].strip()
    return tag or f"Set {index}"


def _scalar(cfg, key: str, hardcoded):
    """Flat `[validation]` scalar from a `TrainConfig` or the flattened TOML mapping."""
    value = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)
    if value is None:
        value = get_val(key, hardcoded)
    return value


def resolve_sample_sets(cfg) -> list[SampleSet]:
    """Resolve `[[validation.samples]]` into concrete sets.

    `cfg` is a `TrainConfig` or the flattened TOML mapping. Every key a set omits falls
    back to the flat `[validation]` scalar of the same shape, and a config with no
    `[[validation.samples]]` at all yields exactly one set built from those scalars -
    i.e. the single-prompt behaviour this replaced.
    """
    if isinstance(cfg, dict):
        raw = cfg.get("samples") or []
    else:
        raw = getattr(cfg, "samples", None) or []
    if not isinstance(raw, list):
        raise ValueError("validation.samples must be an array of tables")

    sets: list[SampleSet] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"validation.samples[{index}] must be a table")
        label = f"validation.samples[{index}]"
        try:
            prompt = _set_str(entry.get("prompt"), _scalar(cfg, "sample_prompts", ""))
            negative = _set_str(entry.get("negative"), _scalar(cfg, "sample_negative", ""))
            width = _set_int(entry.get("width"), _scalar(cfg, "sample_width", 1280))
            height = _set_int(entry.get("height"), _scalar(cfg, "sample_height", 720))
            steps = _set_int(entry.get("steps"), _scalar(cfg, "sample_steps", 55))
            guidance = _set_float(entry.get("guidance_scale"), _scalar(cfg, "guidance_scale", 6.0))
            seed = _set_int(entry.get("seed"), _scalar(cfg, "sample_seed", 0))
            repeat = _set_int(entry.get("repeat"), _scalar(cfg, "sample_repeat", 3))
            if not prompt.strip():
                raise ValueError("prompt must not be empty")
            _check_range("width", width, SAMPLE_SIZE_RANGE)
            _check_range("height", height, SAMPLE_SIZE_RANGE)
            _check_range("steps", steps, SAMPLE_STEPS_RANGE)
            _check_range("guidance_scale", guidance, SAMPLE_CFG_RANGE)
            _check_range("seed", seed, SAMPLE_SEED_RANGE)
            _check_range("repeat", repeat, SAMPLE_REPEAT_RANGE)
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc

        name = _set_str(entry.get("name"), "").strip() or _default_set_name(prompt, index)
        sets.append(
            SampleSet(
                name=name,
                prompt=prompt,
                negative=negative,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance,
                seed=seed,
                repeat=repeat,
            )
        )

    if sets:
        return sets

    prompt = str(_scalar(cfg, "sample_prompts", ""))
    if not prompt.strip():
        raise ValueError("validation.samples is empty and sample_prompts is blank")
    return [
        SampleSet(
            name=_default_set_name(prompt, 1),
            prompt=prompt,
            negative=str(_scalar(cfg, "sample_negative", "")),
            width=int(_scalar(cfg, "sample_width", 1280)),
            height=int(_scalar(cfg, "sample_height", 720)),
            steps=int(_scalar(cfg, "sample_steps", 55)),
            guidance_scale=float(_scalar(cfg, "guidance_scale", 6.0)),
            seed=int(_scalar(cfg, "sample_seed", 0)),
            repeat=int(_scalar(cfg, "sample_repeat", 3)),
        )
    ]


def resolve_train_data_entries(cfg) -> list[TrainDataEntry]:
    """Resolve `[[environment.train_data]]` into concrete dataset folders.

    `cfg` is a `TrainConfig` or the flattened TOML mapping. A config with no
    `[[environment.train_data]]` blocks yields exactly one entry built from the flat
    `train_data_dir` scalar with repeat 1 - the single-folder behaviour this replaced, so a
    hand-edited or older config trains the same as before.
    """
    if isinstance(cfg, dict):
        raw = cfg.get("train_data") or []
    else:
        raw = getattr(cfg, "train_data", None) or []
    if not isinstance(raw, list):
        raise ValueError("environment.train_data must be an array of tables")

    entries: list[TrainDataEntry] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"train_data[{index}] must be a table")
        label = f"train_data[{index}]"
        try:
            path = _set_str(entry.get("path"), "").strip()
            if not path:
                raise ValueError("path must not be empty")
            repeat = _set_int(entry.get("repeat"), 1)
            _check_range("repeat", repeat, TRAIN_DATA_REPEAT_RANGE)
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        entries.append(TrainDataEntry(path=path, repeat=repeat))

    if entries:
        return entries

    path = str(_scalar(cfg, "train_data_dir", "") or "").strip()
    if not path:
        raise ValueError("environment.train_data is empty and train_data_dir is blank")
    return [TrainDataEntry(path=path, repeat=1)]


def tracker_hparams(cfg) -> dict[str, Any]:
    """`vars(cfg)` reduced to what TensorBoard's hparams record accepts.

    `torch.utils.tensorboard.add_hparams` takes int/float/str/bool/torch.Tensor and skips `None`;
    anything else aborts `accelerator.init_trackers`, which runs after the pipeline is loaded and
    the latent cache is built. `[[validation.samples]]` is a list of tables, so it is recorded the
    way kohya records `ss_bucket_info`: as a JSON string, still readable in the HParams tab.

    `cfg` is a `TrainConfig` or the flattened TOML mapping.
    """
    source = cfg if isinstance(cfg, dict) else vars(cfg)
    params: dict[str, Any] = {}
    for key, value in source.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            params[key] = value
        else:
            params[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return params


@dataclass
class TrainConfig:
    # Environment and Paths
    pretrained_model_name_or_path: str = get_val("pretrained_model_name_or_path", "/home/acite/LLM/models/diffusers/waillu_170")
    # The single folder everything that expects one path reads (metadata, the tag button). It
    # mirrors the first `[[environment.train_data]]` entry; resolved by `resolve_train_data_entries`.
    train_data_dir: str = get_val("train_data_dir", "/home/acite/LLM/Character/kanae")
    # `[[environment.train_data]]`: raw tables, resolved by `resolve_train_data_entries`.
    # Empty means "one entry built from train_data_dir with repeat 1".
    train_data: list = field(default_factory=lambda: list(get_val("train_data", []) or []))
    output_name: str = get_val("output_name", "kanae")
    output_dir: str = get_val("output_dir", "/home/acite/LLM/axltrainer/outputs")
    logging_dir: str = get_val("logging_dir", "/home/acite/LLM/axltrainer/logs")
    # none | tail | vmm: LD_PRELOAD for start_train.sh (Ranko Utils → ROCm).
    amdfq: str = get_val("amdfq", "none")
    # GiB of HIP-reported free VRAM the VMM hook will not consume. 0 disables.
    amdfq_vram_reserve_gib: float = get_val("amdfq_vram_reserve_gib", 1.0)
    # MiB of one VMM-hook allocation pool (0 = off, else 16..512). Not read by the trainer loop:
    # it travels config.toml -> trainer/amdfq_patch.py -> AMDFQ_POOL_SIZE -> the hook.
    amdfq_pool_mib: int = get_val("amdfq_pool_mib", 64)

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

    # Resume: kohya LoRA .safetensors (or its directory) to load before training
    resume_lora_path: str = get_val("resume_lora_path", "")
    # Run-scoped artifact directory, filled in at runtime by main.py.
    # Do not set this in config.toml: every run would reuse the same directory.
    run_dir: str = get_val("run_dir", "")

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
    min_bucket_reso: int = get_val("min_bucket_reso", 384)
    max_bucket_reso: int = get_val("max_bucket_reso", 2688)

    # Optimization Features
    cache_latents: bool = get_val("cache_latents", True)
    cache_latents_to_disk: bool = get_val("cache_latents_to_disk", True)
    gradient_checkpointing_unet: bool = get_val("gradient_checkpointing_unet", True)
    gradient_checkpointing_te: bool = get_val("gradient_checkpointing_te", True)
    shuffle_caption: bool = get_val("shuffle_caption", True)
    keep_tokens: int = get_val("keep_tokens", 2)
    caption_extension: str = get_val("caption_extension", ".txt")
    noise_offset: float = get_val("noise_offset", 0.05)
    flush_memory_every_step: bool = get_val("flush_memory_every_step", True)

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
    # `[[validation.samples]]`: raw tables, resolved by `resolve_sample_sets`.
    # Empty means "one set built from the flat sample_* keys above".
    samples: list = field(default_factory=lambda: list(get_val("samples", []) or []))

    # Optional kohya-like bookkeeping (Defaults to None)
    ss_session_id: Optional[int] = get_val("ss_session_id", None)
    ss_training_comment: Optional[str] = get_val("ss_training_comment", None)
    ss_sd_model_hash: Optional[str] = get_val("ss_sd_model_hash", None)
    ss_new_sd_model_hash: Optional[str] = get_val("ss_new_sd_model_hash", None)
    ss_dataset_dirs: Optional[str] = get_val("ss_dataset_dirs", None)
    ss_bucket_info: Optional[str] = get_val("ss_bucket_info", None)

    _current_epoch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        try:
            from family import require_matching_spec
        except ImportError:
            from trainer.family import require_matching_spec
        require_matching_spec(
            self.base_model_version,
            self.modelspec_architecture,
            self.modelspec_implementation,
            self.modelspec_sai_model_spec,
        )
        if self.amdfq_vram_reserve_gib < 0:
            raise ValueError(
                f"amdfq_vram_reserve_gib must be >= 0, not {self.amdfq_vram_reserve_gib}"
            )
