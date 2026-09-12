from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

try:
    from config import TrainConfig
    from utils import (
        image_to_tensor, list_images, pick_bucket_size,
        read_caption, resize_and_center_crop, sha1_text, shuffle_caption
    )
except ImportError:
    from trainer.config import TrainConfig
    from trainer.utils import (
        image_to_tensor, list_images, pick_bucket_size,
        read_caption, resize_and_center_crop, sha1_text, shuffle_caption
    )


class LoraImageDataset(Dataset):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.root = Path(cfg.train_data_dir)
        self.images = list_images(self.root)
        if not self.images:
            raise RuntimeError(f"No usable images found in target training data route: {self.root}")

        # Shared with persistent DataLoader workers so caption shuffle follows epoch.
        self._epoch = torch.zeros(1, dtype=torch.int32)
        try:
            self._epoch.share_memory_()
        except RuntimeError:
            pass

        self.latent_cache_dir = self.root / ".latents_cache"
        if cfg.cache_latents and cfg.cache_latents_to_disk:
            self.latent_cache_dir.mkdir(parents=True, exist_ok=True)

        self.records: list[dict[str, Any]] = []
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, image_path in enumerate(self.images):
            with Image.open(image_path) as img:
                src_w, src_h = img.size
            if cfg.enable_bucket:
                bucket_w, bucket_h = pick_bucket_size(
                    src_w, src_h,
                    min_reso=cfg.min_bucket_reso,
                    max_reso=cfg.max_bucket_reso,
                    step=cfg.bucket_reso_steps,
                    no_upscale=cfg.bucket_no_upscale,
                )
            else:
                bucket_w = bucket_h = cfg.train_resolution
            self.records.append(
                {
                    "path": image_path,
                    "src_w": int(src_w),
                    "src_h": int(src_h),
                    "bucket_w": int(bucket_w),
                    "bucket_h": int(bucket_h),
                }
            )
            self.buckets[(int(bucket_w), int(bucket_h))].append(index)

    @property
    def epoch(self) -> int:
        return int(self._epoch[0].item())

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch.fill_(int(value))

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
        record = self.records[idx]
        image_path: Path = record["path"]
        bucket_w = record["bucket_w"]
        bucket_h = record["bucket_h"]
        cache_path = self._cache_path(image_path, bucket_w, bucket_h)

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
            "src_w": record["src_w"],
            "src_h": record["src_h"],
            "img_type": img_type,
            "img_data": img_data,
            "cache_path": str(cache_path),
        }


class BucketBatchSampler(Sampler[list[int]]):
    """Yield index batches that all share one aspect-ratio bucket.

    Remainders smaller than `batch_size` are kept so small buckets still train.
    """

    def __init__(
        self,
        buckets: dict[tuple[int, int], Sequence[int]],
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.buckets = {key: list(indices) for key, indices in buckets.items() if indices}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self._length = self._count_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _count_batches(self) -> int:
        total = 0
        for indices in self.buckets.values():
            if not indices:
                continue
            total += int(math.ceil(len(indices) / self.batch_size))
        return total

    def _batches_for_epoch(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.buckets.values():
            order = list(indices)
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size]
                if chunk:
                    batches.append(chunk)
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches_for_epoch())

    def __len__(self) -> int:
        return self._length


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    img_items = [ex["img_data"] for ex in examples]
    img_data: Any = img_items
    if img_items and all(ex["img_type"] == "latent" for ex in examples) and all(
        torch.is_tensor(item) for item in img_items
    ):
        try:
            img_data = torch.stack(img_items, dim=0)
        except RuntimeError:
            img_data = img_items

    return {
        "image_path": [ex["image_path"] for ex in examples],
        "caption": [ex["caption"] for ex in examples],
        "bucket_w": [int(ex["bucket_w"]) for ex in examples],
        "bucket_h": [int(ex["bucket_h"]) for ex in examples],
        "src_w": [int(ex["src_w"]) for ex in examples],
        "src_h": [int(ex["src_h"]) for ex in examples],
        "img_type": [ex["img_type"] for ex in examples],
        "img_data": img_data,
        "cache_path": [ex["cache_path"] for ex in examples],
    }


def make_collate_fn():
    return collate_fn


SDXLLoraDataset = LoraImageDataset
