# Project Dump

- Root: `/home/acite/LLM/Trainer`
- Generated: 2026-05-26 19:27:49

## Table of Contents

| # | Path | Size (bytes) | Modified | Status |
|---:|------|-------------:|----------|--------|
| 1 | `LICENSE` | 1067 | 2026-05-26 06:17:25 | skipped (ignored) |
| 2 | `README.md` | 2987 | 2026-05-26 06:25:06 | skipped (ignored) |
| 3 | `clean.py` | 3553 | 2026-05-26 06:17:25 | included |
| 4 | `config.py` | 2470 | 2026-05-26 19:19:09 | included |
| 5 | `dataset.py` | 3944 | 2026-05-26 06:17:25 | included |
| 6 | `main.py` | 4519 | 2026-05-26 19:20:28 | included |
| 7 | `models.py` | 3807 | 2026-05-26 08:27:56 | included |
| 8 | `requirements.txt` | 288 | 2026-05-26 06:17:25 | included |
| 9 | `start.sh` | 590 | 2026-05-26 19:20:46 | included |
| 10 | `text_processing.py` | 4327 | 2026-05-26 08:01:42 | included |
| 11 | `ui.py` | 11159 | 2026-05-26 08:38:59 | skipped (ignored) |
| 12 | `utils.py` | 4021 | 2026-05-26 06:17:25 | included |
| 13 | `__pycache__/config.cpython-312.pyc` | 3459 | 2026-05-26 08:40:06 | skipped (binary) |
| 14 | `__pycache__/config.cpython-314.pyc` | 4232 | 2026-05-26 08:40:11 | skipped (binary) |
| 15 | `__pycache__/dataset.cpython-312.pyc` | 6711 | 2026-05-26 06:27:32 | skipped (binary) |
| 16 | `__pycache__/dataset.cpython-314.pyc` | 8154 | 2026-05-26 06:17:25 | skipped (binary) |
| 17 | `__pycache__/models.cpython-312.pyc` | 5977 | 2026-05-26 08:40:35 | skipped (binary) |
| 18 | `__pycache__/models.cpython-314.pyc` | 6793 | 2026-05-26 06:17:25 | skipped (binary) |
| 19 | `__pycache__/text_processing.cpython-312.pyc` | 5461 | 2026-05-26 08:08:49 | skipped (binary) |
| 20 | `__pycache__/text_processing.cpython-314.pyc` | 6392 | 2026-05-26 06:17:25 | skipped (binary) |
| 21 | `__pycache__/utils.cpython-312.pyc` | 7514 | 2026-05-26 06:27:32 | skipped (binary) |
| 22 | `__pycache__/utils.cpython-314.pyc` | 9507 | 2026-05-26 06:17:25 | skipped (binary) |
| 23 | `trainer/cache.py` | 1266 | 2026-05-26 19:11:55 | included |
| 24 | `trainer/env.py` | 652 | 2026-05-26 19:10:26 | included |
| 25 | `trainer/loop.py` | 9202 | 2026-05-26 19:20:42 | included |
| 26 | `trainer/sampling.py` | 4470 | 2026-05-26 19:11:36 | included |
| 27 | `trainer/setup.py` | 5918 | 2026-05-26 19:20:42 | included |

---

## File Contents

### clean.py

- Size: 3553 bytes
- Modified: 2026-05-26 06:17:25

```text
import os
import shutil
from pathlib import Path

# Import the configuration from your project
from config import TrainConfig


def clean_project():
    print("=" * 50)
    print("      LoRA Training Directory Cleaner      ")
    print("=" * 50)

    # Initialize configuration
    try:
        cfg = TrainConfig()
    except Exception as e:
        print(f"[Error] Failed to load TrainConfig: {e}")
        return

    # Extract target paths and names from config
    output_dir = Path(cfg.output_dir)
    logging_dir = Path(cfg.logging_dir)
    output_name = cfg.output_name

    # Define specific targets based on config parameters
    samples_dir = output_dir / f"{output_name}_samples"
    tensorboard_log_dir = logging_dir / output_name

    print(f"Loaded configuration for project: '{output_name}'")
    print(f"Base Output Directory: {output_dir}")
    print(f"Base Logging Directory: {logging_dir}\n")

    # 1. Clean Generated Samples
    print("-" * 40)
    print("Step 1: Cleaning sample images...")
    if samples_dir.exists() and samples_dir.is_dir():
        try:
            shutil.rmtree(samples_dir)
            print(f"[Success] Removed samples directory: {samples_dir}")
        except Exception as e:
            print(f"[Warning] Could not remove samples directory: {e}")
    else:
        print(f"[Info] No samples directory found at: {samples_dir}")

    # 2. Clean TensorBoard Logs
    print("\n" + "-" * 40)
    print("Step 2: Cleaning TensorBoard logs...")
    if tensorboard_log_dir.exists() and tensorboard_log_dir.is_dir():
        try:
            shutil.rmtree(tensorboard_log_dir)
            print(f"[Success] Removed project logs: {tensorboard_log_dir}")
        except Exception as e:
            print(f"[Warning] Could not remove log directory: {e}")
    else:
        print(f"[Info] No log directory found at: {tensorboard_log_dir}")

    # 3. Optional: Clean Trained Weights (Safetensors)
    print("\n" + "-" * 40)
    print("Step 3: Checking for existing weights...")

    # Find directories matching output_name patterns
    weight_dirs = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_dir() and path.name.startswith(output_name):
                # Skip the samples directory as it was handled in step 1
                if path.name == f"{output_name}_samples":
                    continue
                weight_dirs.append(path)

    if weight_dirs:
        print(f"Found {len(weight_dirs)} checkpoint directory/directories:")
        for d in weight_dirs:
            print(f"  - {d.name}/")
        
        print("")  # Blank line for readability
        # Prompt user for confirmation
        confirmation = input("Do you want to delete these trained weight checkpoints? (y/N): ").strip().lower()
        
        if confirmation in ['y', 'yes']:
            print("Deleting weights...")
            for d in weight_dirs:
                try:
                    shutil.rmtree(d)
                    print(f"  [Deleted] {d.name}")
                except Exception as e:
                    print(f"  [Error] Failed to delete {d.name}: {e}")
            print("[Success] Selected weight checkpoints have been removed.")
        else:
            print("[Info] Skipped weights deletion. Safe-saving checkpoints.")
    else:
        print(f"[Info] No matching weight checkpoints found for '{output_name}' in {output_dir}")

    print("\n" + "=" * 50)
    print("Cleanup task completed.")
    print("=" * 50)


if __name__ == "__main__":
    clean_project()
```

### config.py

- Size: 2470 bytes
- Modified: 2026-05-26 19:19:09

```text
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
    network_dropout: float = 0.25
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
    sample_negative: str = "lowres, bad anatomy, (mosaic censoring:1.3)"
    sample_width: int = 768
    sample_height: int = 768
    sample_steps: int = 35
    sample_seed: int = 0
    sample_repeat: int = 5
    guidance_scale: float = 5.5

```

### dataset.py

- Size: 3944 bytes
- Modified: 2026-05-26 06:17:25

```text
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from config import TrainConfig
from utils import (
    image_to_tensor, list_images, pick_bucket_size, 
    read_caption, resize_and_center_crop, sha1_text, shuffle_caption
)

class SDXLLoraDataset(Dataset):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.root = Path(cfg.train_data_dir)
        self.images = list_images(self.root)
        if not self.images:
            raise RuntimeError(f"No usable images found in target training data route: {self.root}")
        
        self.epoch = 0
        self.latent_cache_dir = self.root / ".latents_cache"
        if cfg.cache_latents and cfg.cache_latents_to_disk:
            self.latent_cache_dir.mkdir(parents=True, exist_ok=True)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images)

    def _caption_for(self, image_path: Path) -> str:
        cap = read_caption(image_path, self.cfg.caption_extension)
        if self.cfg.shuffle_caption:
            seed_val = self.cfg.seed + self.epoch + int(sha1_text(str(image_path)), 16) % 10_000
            rng = random.Random(seed_val)
            cap = shuffle_caption(cap, self.cfg.keep_tokens, rng)
        return cap

    def _cache_path(self, image_path: Path, bucket_w: int, bucket_h: int) -> Path:
        key = f"{image_path.resolve()}::{bucket_w}x{bucket_h}"
        return self.latent_cache_dir / f"{sha1_text(key)}.pt"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.images[idx]
        with Image.open(image_path) as img:
            src_w, src_h = img.size

        if self.cfg.enable_bucket:
            bucket_w, bucket_h = pick_bucket_size(
                src_w, src_h,
                min_reso=self.cfg.min_bucket_reso,
                max_reso=self.cfg.max_bucket_reso,
                step=self.cfg.bucket_reso_steps,
                no_upscale=self.cfg.bucket_no_upscale,
            )
        else:
            bucket_w = bucket_h = self.cfg.train_resolution

        cache_path = self._cache_path(image_path, bucket_w, bucket_h)
        
        # Load from disk if cached, otherwise prepare raw pixel values
        if self.cfg.cache_latents and self.cfg.cache_latents_to_disk and cache_path.exists():
            img_type = "latent"
            img_data = torch.load(cache_path, map_location="cpu")
        else:
            img_type = "pixel"
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = resize_and_center_crop(img, bucket_w, bucket_h)
                img_data = image_to_tensor(img)

        return {
            "image_path": str(image_path),
            "caption": self._caption_for(image_path),
            "bucket_w": bucket_w,
            "bucket_h": bucket_h,
            "src_w": src_w,
            "src_h": src_h,
            "img_type": img_type,
            "img_data": img_data,
            "cache_path": str(cache_path),
        }

def make_collate_fn():
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "image_path": [ex["image_path"] for ex in examples],
            "caption": [ex["caption"] for ex in examples],
            "bucket_w": torch.tensor([ex["bucket_w"] for ex in examples], dtype=torch.long),
            "bucket_h": torch.tensor([ex["bucket_h"] for ex in examples], dtype=torch.long),
            "src_w": torch.tensor([ex["src_w"] for ex in examples], dtype=torch.long),
            "src_h": torch.tensor([ex["src_h"] for ex in examples], dtype=torch.long),
            "img_type": [ex["img_type"] for ex in examples],
            "img_data": [ex["img_data"] for ex in examples], 
            "cache_path": [ex["cache_path"] for ex in examples],
        }
    return collate_fn

```

### main.py

- Size: 4519 bytes
- Modified: 2026-05-26 19:20:28

```text
from __future__ import annotations

import os
from pathlib import Path

from accelerate.utils import set_seed
from tqdm.auto import tqdm

from config import TrainConfig
from trainer.cache import warm_latent_cache
from trainer.env import flush_memory
from trainer.loop import maybe_save_and_sample, train_one_epoch
from trainer.setup import build_train_objects
from models import save_lora_checkpoint


def main() -> None:
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    set_seed(cfg.seed)

    artifacts = build_train_objects(cfg)
    accelerator = artifacts.accelerator
    device = artifacts.device
    weight_dtype = artifacts.weight_dtype

    if cfg.cache_latents and cfg.cache_latents_to_disk:
        if accelerator.is_main_process:
            print("Checking/Generating latents cache...")
            warm_latent_cache(
                artifacts.train_dataset,
                artifacts.vae,
                cfg,
                device,
                weight_dtype,
            )
        accelerator.wait_for_everyone()

    artifacts.vae.to("cpu")
    flush_memory(device)

    (
        artifacts.unet,
        artifacts.text_encoder_1,
        artifacts.text_encoder_2,
        artifacts.unet_optimizer,
        artifacts.te_optimizer,
        artifacts.dataloader,
        artifacts.unet_scheduler,
        artifacts.te_scheduler,
    ) = accelerator.prepare(
        artifacts.unet,
        artifacts.text_encoder_1,
        artifacts.text_encoder_2,
        artifacts.unet_optimizer,
        artifacts.te_optimizer,
        artifacts.dataloader,
        artifacts.unet_scheduler,
        artifacts.te_scheduler,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.output_name,
            config=vars(cfg),
        )

    total_train_steps = max(
        1,
        ((len(artifacts.dataloader) + cfg.gradient_accumulation_steps - 1) // cfg.gradient_accumulation_steps)
        * cfg.epoch,
    )
    progress = tqdm(
        total=total_train_steps,
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    for epoch in range(cfg.epoch):
        artifacts.train_dataset.set_epoch(epoch)

        global_step = train_one_epoch(
            accelerator=accelerator,
            cfg=cfg,
            pipe=artifacts.pipe,
            vae=artifacts.vae,
            unet=artifacts.unet,
            text_encoder_1=artifacts.text_encoder_1,
            text_encoder_2=artifacts.text_encoder_2,
            dataloader=artifacts.dataloader,
            train_dataset=artifacts.train_dataset,
            noise_scheduler=artifacts.noise_scheduler,
            unet_optimizer=artifacts.unet_optimizer,
            te_optimizer=artifacts.te_optimizer,
            unet_scheduler=artifacts.unet_scheduler,
            te_scheduler=artifacts.te_scheduler,
            device=device,
            weight_dtype=weight_dtype,
            global_step=global_step,
        )

        if accelerator.sync_gradients:
            progress.update(1)
            if accelerator.is_main_process:
                progress.set_description(f"epoch={epoch + 1}/{cfg.epoch} step={global_step}")

                if cfg.save_every_n_steps > 0 and global_step % cfg.save_every_n_steps == 0:
                    maybe_save_and_sample(
                        accelerator=accelerator,
                        pipe=artifacts.pipe,
                        unet=artifacts.unet,
                        text_encoder_1=artifacts.text_encoder_1,
                        text_encoder_2=artifacts.text_encoder_2,
                        cfg=cfg,
                        device=device,
                        weight_dtype=weight_dtype,
                        global_step=global_step,
                    )

    save_lora_checkpoint(
        accelerator,
        artifacts.unet,
        artifacts.text_encoder_1,
        artifacts.text_encoder_2,
        cfg,
        global_step,
        final=True,
    )

    maybe_save_and_sample(
        accelerator=accelerator,
        pipe=artifacts.pipe,
        unet=artifacts.unet,
        text_encoder_1=artifacts.text_encoder_1,
        text_encoder_2=artifacts.text_encoder_2,
        cfg=cfg,
        device=device,
        weight_dtype=weight_dtype,
        global_step=global_step,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
```

### models.py

- Size: 3807 bytes
- Modified: 2026-05-26 08:27:56

```text
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
    text_encoder_1: Any, 
    text_encoder_2: Any, 
    cfg: TrainConfig,
    global_step: int,
    epoch: Optional[int] = None,
    final: bool = False,
) -> None:
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

    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=unet_lora_state,
        text_encoder_lora_layers=te1_lora_state,  
        text_encoder_2_lora_layers=te2_lora_state, 
        safe_serialization=True,
    )

```

### requirements.txt

- Size: 288 bytes
- Modified: 2026-05-26 06:17:25

```text
# --- 核心深度学习框架 (ROCm 专用) ---

# --- 矩阵计算与图像处理 ---
numpy<2.0.0
Pillow

# --- Hugging Face 家族工具链 ---
transformers
diffusers
accelerate
peft
safetensors

# --- 进度条与优化器 ---
tqdm
prodigyopt
tensorboard
plotly
streamlit-autorefresh

```

### start.sh

- Size: 590 bytes
- Modified: 2026-05-26 19:20:46

```text
#!/bin/bash
set -euo pipefail

export AMD_LOG_LEVEL=0
export CK_LOG_LEVEL=0

export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
export MIOPEN_LOG_LEVEL=1
export MIOPEN_LOG_BUFFER_SIZE=0

export MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS=0
export MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR=0
export MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK=0

export MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen"
export MIOPEN_USER_DB_PATH="$HOME/.config/miopen"

exec python -u main.py \
    1> >(grep -Ev "grid_desc|CandidateSelectionModel|metadata" >> /dev/null)

```

### text_processing.py

- Size: 4327 bytes
- Modified: 2026-05-26 08:01:42

```text
from typing import List, Sequence, Tuple

import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)


def _chunk_ids(
    token_ids: List[int],
    chunk_size: int,
    max_token_length: int,
) -> List[List[int]]:
    token_ids = token_ids[:max_token_length]

    if not token_ids:
        return [[]]

    return [
        token_ids[i : i + chunk_size]
        for i in range(0, len(token_ids), chunk_size)
    ]

def tokenize_long_prompt(
    text: str,
    tokenizer: CLIPTokenizer,
    max_token_length: int,
) -> torch.Tensor:
    chunk_size = tokenizer.model_max_length - 2

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ).input_ids

    chunks = _chunk_ids(
        token_ids,
        chunk_size,
        max_token_length,
    )

    max_chunks = max(1, max_token_length // chunk_size)
    while len(chunks) < max_chunks:
        chunks.append([])

    seqs = []

    for chunk in chunks:
        ids = (
            [tokenizer.bos_token_id]
            + chunk
            + [tokenizer.eos_token_id]
        )

        if len(ids) < tokenizer.model_max_length:
            ids = ids + [tokenizer.pad_token_id] * (
                tokenizer.model_max_length - len(ids)
            )
        else:
            ids = ids[: tokenizer.model_max_length]
            ids[-1] = tokenizer.eos_token_id

        seqs.append(torch.tensor(ids, dtype=torch.long))

    return torch.stack(seqs, dim=0)


def _get_pooled_output(output) -> torch.Tensor:
    """
    兼容不同 transformers / diffusers 版本
    """

    # 新版本 SDXL 常见
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds

    # 某些 CLIP 版本
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    # fallback
    return output.last_hidden_state[:, 0]


def encode_prompt_batch(
    prompts: Sequence[str],
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    clip_skip: int,
    max_token_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:

    prompt_embeds_out: List[torch.Tensor] = []
    pooled_out: List[torch.Tensor] = []

    for prompt in prompts:

        ids_1 = tokenize_long_prompt(
            prompt,
            tokenizer_1,
            max_token_length,
        ).to(device)

        ids_2 = tokenize_long_prompt(
            prompt,
            tokenizer_2,
            max_token_length,
        ).to(device)

        if len(ids_1) != len(ids_2):
            raise RuntimeError(
                f"Tokenizer chunk mismatch: "
                f"{len(ids_1)} vs {len(ids_2)}"
            )

        chunk_embeds_1 = []
        chunk_embeds_2 = []
        pooled_chunks = []

        for c1, c2 in zip(ids_1, ids_2):

            c1 = c1.unsqueeze(0)
            c2 = c2.unsqueeze(0)

            out1 = text_encoder_1(
                c1,
                output_hidden_states=True,
                return_dict=True,
            )

            out2 = text_encoder_2(
                c2,
                output_hidden_states=True,
                return_dict=True,
            )

            if clip_skip > 0:
                hs1 = out1.hidden_states[-(clip_skip + 1)]
                hs2 = out2.hidden_states[-(clip_skip + 1)]
            else:
                hs1 = out1.last_hidden_state
                hs2 = out2.last_hidden_state

            pooled = _get_pooled_output(out2)

            chunk_embeds_1.append(hs1)
            chunk_embeds_2.append(hs2)
            pooled_chunks.append(pooled)

        emb1 = torch.cat(chunk_embeds_1, dim=1)
        emb2 = torch.cat(chunk_embeds_2, dim=1)

        prompt_embeds = torch.cat(
            [emb1, emb2],
            dim=-1,
        )

        pooled_prompt_embeds = pooled_chunks[0]

        prompt_embeds_out.append(
            prompt_embeds.squeeze(0).to(dtype)
        )

        pooled_out.append(
            pooled_prompt_embeds.squeeze(0).to(dtype)
        )

    return (
        torch.stack(prompt_embeds_out, dim=0),
        torch.stack(pooled_out, dim=0),
    )

```

### utils.py

- Size: 4021 bytes
- Modified: 2026-05-26 06:17:25

```text
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image

def parse_kv_args(arg_string: str) -> Dict[str, Any]:
    """Parses custom command-line string arguments into typed dictionaries."""
    out: Dict[str, Any] = {}
    for item in arg_string.split():
        item = item.strip().strip('"').strip("'")
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        
        if v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
            continue
        try:
            if "," in v and not v.startswith("[") and not v.startswith("{"):
                parts = [p.strip() for p in v.split(",")]
                parsed = [float(p) if "." in p or "e" in p.lower() else int(p) for p in parts]
                out[k] = tuple(parsed)
                continue
            
            out[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except ValueError:
            out[k] = v
    return out

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

def list_images(root: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in extensions])

def read_caption(image_path: Path, caption_extension: str = ".txt") -> str:
    caption_file = image_path.with_suffix(caption_extension)
    if caption_file.exists():
        return caption_file.read_text(encoding="utf-8", errors="ignore").strip()
    return image_path.stem.replace("_", " ")

def shuffle_caption(text: str, keep_tokens: int, rng: random.Random) -> str:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) <= keep_tokens:
        return ", ".join(parts)
    head = parts[:keep_tokens]
    tail = parts[keep_tokens:]
    rng.shuffle(tail)
    return ", ".join(head + tail)

def round_to_step(value: int, step: int) -> int:
    return max(step, (value // step) * step)

def pick_bucket_size(
    w: int, h: int, min_reso: int, max_reso: int, step: int, no_upscale: bool
) -> Tuple[int, int]:
    if w <= 0 or h <= 0:
        return min_reso, min_reso

    ar = w / h
    if ar >= 1.0:
        bucket_h = min_reso if not (no_upscale and h < min_reso) else round_to_step(h, step)
        bucket_w = int(round(bucket_h * ar))
    else:
        bucket_w = min_reso if not (no_upscale and w < min_reso) else round_to_step(w, step)
        bucket_h = int(round(bucket_w / ar))

    bucket_w = round_to_step(bucket_w, step)
    bucket_h = round_to_step(bucket_h, step)
    bucket_w = max(step, min(bucket_w, max_reso))
    bucket_h = max(step, min(bucket_h, max_reso))

    if (ar >= 1.0 and bucket_w < bucket_h) or (ar < 1.0 and bucket_h < bucket_w):
        bucket_w, bucket_h = bucket_h, bucket_w

    return bucket_w, bucket_h

def resize_and_center_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return image.crop((left, top, left + target_w, top + target_h))

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(image)).float() / 255.0
    if arr.ndim == 2:
        arr = arr.unsqueeze(-1)
    arr = arr.permute(2, 0, 1)
    return arr * 2.0 - 1.0

def build_time_ids(
    original_size: Tuple[int, int],
    crop_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vals = list(original_size) + list(crop_top_left) + list(target_size)
    return torch.tensor(vals, device=device, dtype=dtype)

```

### trainer/cache.py

- Size: 1266 bytes
- Modified: 2026-05-26 19:11:55

```text
from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from config import TrainConfig
from dataset import SDXLLoraDataset
from env import flush_memory


@torch.no_grad()
def warm_latent_cache(
    dataset: SDXLLoraDataset,
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Pre-encode image latents to disk when disk caching is enabled."""
    if not (cfg.cache_latents and cfg.cache_latents_to_disk):
        return

    vae.eval()
    vae.to(device=device, dtype=dtype)

    pbar = tqdm(total=len(dataset), desc="Encoding Latents")
    try:
        for idx in range(len(dataset)):
            item = dataset[idx]
            cache_path = Path(item["cache_path"])

            if cache_path.exists() or item["img_type"] != "pixel":
                pbar.update(1)
                continue

            pixel_values = item["img_data"].unsqueeze(0).to(device=device, dtype=dtype)
            latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            torch.save(latent.squeeze(0).detach().cpu(), cache_path)
            pbar.update(1)
    finally:
        pbar.close()
        vae.to("cpu")
        flush_memory(device)
```

### trainer/env.py

- Size: 652 bytes
- Modified: 2026-05-26 19:10:26

```text
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch


def setup_migraphx_cache(cache_dir_name: str = "migraphx_cache") -> Path:
    """Create and register the MIGraphX cache directory."""
    cache_dir = Path.cwd() / cache_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = str(cache_dir)
    os.environ["ORT_MIGRAPHX_CACHE_PATH"] = str(cache_dir)
    return cache_dir


def flush_memory(device: torch.device) -> None:
    """Release Python and GPU-side cached memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
```

### trainer/loop.py

- Size: 9202 bytes
- Modified: 2026-05-26 19:20:42

```text
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from config import TrainConfig
from text_processing import encode_prompt_batch
from utils import build_time_ids
from sampling import generate_sample_image
from models import save_lora_checkpoint


def group_indices_by_bucket(batch: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    """Group batch items by spatial bucket to keep tensor shapes consistent."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(len(batch["caption"])):
        bw = int(batch["bucket_w"][idx].item())
        bh = int(batch["bucket_h"][idx].item())
        groups[(bw, bh)].append(idx)
    return groups


def encode_latent_for_item(
    *,
    item_index: int,
    batch: dict[str, Any],
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Load a latent directly or encode a pixel image on demand."""
    img_type = batch["img_type"][item_index]
    cache_path = Path(batch["cache_path"][item_index])
    img_data = batch["img_data"][item_index]

    if img_type == "latent":
        return img_data.to(device=device, dtype=weight_dtype)

    pixel_values = img_data.unsqueeze(0).to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
    latent = latent.squeeze(0)

    if cfg.cache_latents and cfg.cache_latents_to_disk:
        torch.save(latent.detach().cpu(), cache_path)

    return latent


def build_group_inputs(
    *,
    indices: list[int],
    batch: dict[str, Any],
    vae: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Build prompts, latents, and time IDs for one bucket group."""
    prompts = [batch["caption"][i] for i in indices]
    latents_list: list[torch.Tensor] = []
    time_ids_list: list[torch.Tensor] = []

    for i in indices:
        src_w = int(batch["src_w"][i].item())
        src_h = int(batch["src_h"][i].item())
        bucket_w = int(batch["bucket_w"][i].item())
        bucket_h = int(batch["bucket_h"][i].item())

        latent = encode_latent_for_item(
            item_index=i,
            batch=batch,
            vae=vae,
            cfg=cfg,
            device=device,
            weight_dtype=weight_dtype,
        )
        latents_list.append(latent)
        time_ids_list.append(
            build_time_ids(
                original_size=(src_h, src_w),
                crop_top_left=(0, 0),
                target_size=(bucket_h, bucket_w),
                device=device,
                dtype=weight_dtype,
            )
        )

    latents = torch.stack(latents_list, dim=0).to(device=device, dtype=weight_dtype)
    time_ids = torch.stack(time_ids_list, dim=0)
    return prompts, latents, time_ids


def compute_bucket_loss(
    *,
    prompts: list[str],
    latents: torch.Tensor,
    time_ids: torch.Tensor,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    unet,
    noise_scheduler,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the forward pass for one bucket group and return the loss."""
    prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(
        prompts=prompts,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        clip_skip=cfg.clip_skip,
        max_token_length=cfg.max_token_length,
        device=device,
        dtype=weight_dtype,
    )

    noise = torch.randn_like(latents)
    if cfg.noise_offset > 0:
        offset = cfg.noise_offset * torch.randn(
            latents.shape[0],
            latents.shape[1],
            1,
            1,
            device=device,
            dtype=weight_dtype,
        )
        noise = noise + offset

    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        },
        return_dict=False,
    )[0]

    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
    return loss


def train_one_epoch(
    *,
    accelerator: Accelerator,
    cfg: TrainConfig,
    pipe,
    vae,
    unet,
    text_encoder_1,
    text_encoder_2,
    dataloader,
    train_dataset,
    noise_scheduler,
    unet_optimizer,
    te_optimizer,
    unet_scheduler,
    te_scheduler,
    device: torch.device,
    weight_dtype: torch.dtype,
    global_step: int,
) -> int:
    """Train one full epoch and return the updated global step."""
    unet.train()
    text_encoder_1.train()
    text_encoder_2.train()

    for batch in dataloader:
        with accelerator.accumulate(unet, text_encoder_1, text_encoder_2):
            groups = group_indices_by_bucket(batch)

            batch_loss_sum = 0.0
            batch_item_count = 0

            for _, indices in groups.items():
                prompts, latents, time_ids = build_group_inputs(
                    indices=indices,
                    batch=batch,
                    vae=vae,
                    cfg=cfg,
                    device=device,
                    weight_dtype=weight_dtype,
                )

                loss = compute_bucket_loss(
                    prompts=prompts,
                    latents=latents,
                    time_ids=time_ids,
                    tokenizer_1=pipe.tokenizer,
                    tokenizer_2=pipe.tokenizer_2,
                    text_encoder_1=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    cfg=cfg,
                    device=device,
                    weight_dtype=weight_dtype,
                )

                scaled_loss = loss * (len(indices) / len(batch["caption"]))
                accelerator.backward(scaled_loss)

                batch_loss_sum += loss.item() * len(indices)
                batch_item_count += len(indices)

            if accelerator.sync_gradients:
                unet_clip_params = [p for p in unet.parameters() if p.requires_grad]
                te_clip_params = [
                    p for p in text_encoder_1.parameters() if p.requires_grad
                ] + [
                    p for p in text_encoder_2.parameters() if p.requires_grad
                ]

                accelerator.clip_grad_norm_(unet_clip_params, cfg.max_grad_norm)
                accelerator.clip_grad_norm_(te_clip_params, cfg.te_max_grad_norm)

                unet_optimizer.step()
                te_optimizer.step()
                unet_scheduler.step()
                te_scheduler.step()
                unet_optimizer.zero_grad(set_to_none=True)
                te_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            avg_loss = batch_loss_sum / max(1, batch_item_count)

            if accelerator.is_main_process:
                unet_base_lr = unet_scheduler.get_last_lr()[0]
                te_base_lr = te_scheduler.get_last_lr()[0]

                unet_prodigy_d = unet_optimizer.param_groups[0].get("d", 1.0)
                unet_effective_lr = unet_base_lr * unet_prodigy_d

                accelerator.log(
                    {
                        "Train/Loss": avg_loss,
                        "UNet/LR/Base_Scheduled": unet_base_lr,
                        "UNet/LR/Prodigy_D_Factor": unet_prodigy_d,
                        "UNet/LR/Effective_Actual_LR": unet_effective_lr,
                        "TE/LR/Base_Scheduled": te_base_lr,
                        "TE/LR/Effective_Actual_LR": te_base_lr,
                    },
                    step=global_step,
                )

    return global_step


def maybe_save_and_sample(
    *,
    accelerator: Accelerator,
    pipe,
    unet,
    text_encoder_1,
    text_encoder_2,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
    global_step: int,
) -> None:
    """Save LoRA weights and generate a sample image."""
    save_lora_checkpoint(accelerator, unet, text_encoder_1, text_encoder_2, cfg, global_step)

    generate_sample_image(
        accelerator=accelerator,
        pipe=pipe,
        trained_unet=accelerator.unwrap_model(unet),
        trained_te1=accelerator.unwrap_model(text_encoder_1),
        trained_te2=accelerator.unwrap_model(text_encoder_2),
        cfg=cfg,
        device=device,
        dtype=weight_dtype,
        global_step=global_step,
        output_dir_base=Path(cfg.output_dir),
    )
```

### trainer/sampling.py

- Size: 4470 bytes
- Modified: 2026-05-26 19:11:36

```text
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image

from config import TrainConfig
from text_processing import encode_prompt_batch
from env import flush_memory


@torch.inference_mode()
def generate_sample_image(
    *,
    accelerator: Accelerator,
    pipe,
    trained_unet: torch.nn.Module,
    trained_te1: torch.nn.Module,
    trained_te2: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    output_dir_base: Path,
) -> None:
    """Generate and save sample images from the current LoRA weights."""
    if not accelerator.is_main_process:
        return

    flush_memory(device)

    pipe = pipe.to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
    )

    num_inference_steps = cfg.sample_steps
    sigmas = np.linspace(pipe.scheduler.config.num_train_timesteps - 1, 0, num_inference_steps)
    sigmas = np.append(sigmas, 0.0).astype(np.float32)

    pipe.scheduler.sigmas = torch.from_numpy(sigmas).to(device)
    pipe.scheduler.num_inference_steps = num_inference_steps

    pipe.unet = trained_unet
    pipe.text_encoder = trained_te1
    pipe.text_encoder_2 = trained_te2

    pipe.unet.eval()
    pipe.text_encoder.to(device=device, dtype=dtype).eval()
    pipe.text_encoder_2.to(device=device, dtype=dtype).eval()

    vae_dtype = torch.bfloat16
    pipe.vae.to(device=device, dtype=vae_dtype).eval()

    if hasattr(pipe.vae.config, "force_upcast"):
        pipe.vae.config.force_upcast = False

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(
        prompts=[cfg.sample_prompts],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=cfg.clip_skip,
        max_token_length=cfg.max_token_length,
        device=device,
        dtype=dtype,
    )
    negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_batch(
        prompts=[cfg.sample_negative],
        tokenizer_1=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        text_encoder_1=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        clip_skip=cfg.clip_skip,
        max_token_length=cfg.max_token_length,
        device=device,
        dtype=dtype,
    )

    sample_dir = output_dir_base / f"{cfg.output_name}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        for repeat_idx in range(max(1, cfg.sample_repeat)):
            generator = torch.Generator(device=device)
            if cfg.sample_seed == 0:
                current_seed = int(torch.randint(0, 2**32, (1,)).item())
                generator.manual_seed(current_seed)
                print(f"[Sample {repeat_idx}] Using random seed: {current_seed}")
            else:
                current_seed = cfg.sample_seed + repeat_idx
                generator.manual_seed(current_seed)

            latent_result = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=cfg.sample_width,
                height=cfg.sample_height,
                num_inference_steps=cfg.sample_steps,
                guidance_scale=cfg.guidance_scale,
                generator=generator,
                output_type="latent",
            )

            latents = latent_result.images.to(device=device, dtype=vae_dtype)
            latents = latents / pipe.vae.config.scaling_factor

            decoded = pipe.vae.decode(latents, return_dict=False)[0]
            image = (decoded / 2 + 0.5).clamp(0, 1)
            image = image[0].permute(1, 2, 0).detach().float().cpu().numpy()
            image = (image * 255).round().astype("uint8")

            out_filename = f"{cfg.output_name}_{global_step:06d}_{repeat_idx}.png"
            out_path = sample_dir / out_filename
            Image.fromarray(image).save(out_path)
    finally:
        pipe.vae.to("cpu")
        flush_memory(device)
```

### trainer/setup.py

- Size: 5918 bytes
- Modified: 2026-05-26 19:20:42

```text
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from diffusers import DDIMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import SDXLLoraDataset, make_collate_fn
from models import (
    build_scheduler,
    build_te_optimizer,
    build_unet_optimizer,
    enable_flash_attention,
    load_sdxl_pipeline,
)
from env import setup_migraphx_cache


@dataclass
class TrainArtifacts:
    accelerator: Accelerator
    device: torch.device
    weight_dtype: torch.dtype
    pipe: Any
    vae: torch.nn.Module
    unet: torch.nn.Module
    tokenizer_1: Any
    tokenizer_2: Any
    text_encoder_1: torch.nn.Module
    text_encoder_2: torch.nn.Module
    noise_scheduler: DDIMScheduler
    train_dataset: SDXLLoraDataset
    dataloader: DataLoader
    unet_optimizer: Any
    te_optimizer: Any
    unet_scheduler: Any
    te_scheduler: Any


def maybe_enable_amp_backends() -> None:
    """Enable PyTorch attention backends when available."""
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def create_accelerator(cfg: TrainConfig) -> Accelerator:
    """Create the training accelerator."""
    return Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=cfg.logging_dir,
    )


def apply_lora_modules(cfg: TrainConfig, unet, text_encoder_1, text_encoder_2):
    """Attach LoRA adapters to UNet and both text encoders."""
    te_lora_config = LoraConfig(
        r=cfg.network_dim,
        lora_alpha=cfg.network_alpha,
        lora_dropout=cfg.network_dropout,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    unet_lora_config = LoraConfig(
        r=cfg.network_dim,
        lora_alpha=cfg.network_alpha,
        lora_dropout=cfg.network_dropout,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    text_encoder_1 = get_peft_model(text_encoder_1, te_lora_config)
    text_encoder_2 = get_peft_model(text_encoder_2, te_lora_config)
    unet = get_peft_model(unet, unet_lora_config)
    unet.enable_gradient_checkpointing()

    return unet, text_encoder_1, text_encoder_2


def build_dataloader(cfg: TrainConfig) -> tuple[SDXLLoraDataset, DataLoader]:
    """Build dataset and dataloader."""
    train_dataset = SDXLLoraDataset(cfg)
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.max_data_loader_n_workers,
        pin_memory=True,
        persistent_workers=cfg.persistent_workers,
        collate_fn=make_collate_fn(),
        drop_last=True,
    )
    return train_dataset, dataloader


def build_optimizers_and_schedulers(
    cfg: TrainConfig,
    unet,
    text_encoder_1,
    text_encoder_2,
    dataloader: DataLoader,
):
    """Build optimizers and schedulers for UNet and text encoders."""
    unet_params = [p for p in unet.parameters() if p.requires_grad]
    te_params = [
        p for p in text_encoder_1.parameters() if p.requires_grad
    ] + [
        p for p in text_encoder_2.parameters() if p.requires_grad
    ]

    unet_optimizer = build_unet_optimizer(cfg, unet_params)
    te_optimizer = build_te_optimizer(cfg, te_params)

    steps_per_epoch = max(1, math.ceil(len(dataloader) / cfg.gradient_accumulation_steps))
    total_steps = steps_per_epoch * cfg.epoch

    unet_scheduler = build_scheduler(unet_optimizer, total_steps, cfg)
    te_scheduler = build_scheduler(te_optimizer, total_steps, cfg)

    return unet_optimizer, te_optimizer, unet_scheduler, te_scheduler


def build_train_objects(cfg: TrainConfig) -> TrainArtifacts:
    """Build everything required for training in a readable sequence."""
    setup_migraphx_cache()
    maybe_enable_amp_backends()

    weight_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    accelerator = create_accelerator(cfg)
    device = accelerator.device

    pipe = load_sdxl_pipeline(cfg.pretrained_model_name_or_path, weight_dtype)
    vae = pipe.vae
    unet = pipe.unet
    tokenizer_1, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
    text_encoder_1, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2
    noise_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    enable_flash_attention(unet)

    unet, text_encoder_1, text_encoder_2 = apply_lora_modules(
        cfg,
        unet,
        text_encoder_1,
        text_encoder_2,
    )

    vae.to(device=device, dtype=weight_dtype).eval()
    text_encoder_1.to(device=device, dtype=weight_dtype)
    text_encoder_2.to(device=device, dtype=weight_dtype)

    train_dataset, dataloader = build_dataloader(cfg)
    unet_optimizer, te_optimizer, unet_scheduler, te_scheduler = build_optimizers_and_schedulers(
        cfg,
        unet,
        text_encoder_1,
        text_encoder_2,
        dataloader,
    )

    return TrainArtifacts(
        accelerator=accelerator,
        device=device,
        weight_dtype=weight_dtype,
        pipe=pipe,
        vae=vae,
        unet=unet,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        noise_scheduler=noise_scheduler,
        train_dataset=train_dataset,
        dataloader=dataloader,
        unet_optimizer=unet_optimizer,
        te_optimizer=te_optimizer,
        unet_scheduler=unet_scheduler,
        te_scheduler=te_scheduler,
    )

```


-----

## Summary

- Total files scanned: 27
- Included text files: 14
- Skipped binary files: 10
- Skipped ignored files: 3
- Unreadable files: 0
- Truncated files (per-file cap): 0
