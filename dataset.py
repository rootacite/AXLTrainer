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
