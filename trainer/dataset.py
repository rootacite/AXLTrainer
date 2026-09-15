from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

try:
    from config import TrainConfig
    from utils import (
        Geometry, fit_geometry, fit_to_bucket, image_has_alpha, image_to_tensor, list_images,
        load_loss_mask, mask_path_for, pick_bucket_size, read_caption, sha1_text, shuffle_caption,
    )
except ImportError:
    from trainer.config import TrainConfig
    from trainer.utils import (
        Geometry, fit_geometry, fit_to_bucket, image_has_alpha, image_to_tensor, list_images,
        load_loss_mask, mask_path_for, pick_bucket_size, read_caption, sha1_text, shuffle_caption,
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
        self.n_masked = 0
        self.n_padded = 0
        self._pad_total = 0.0
        self._check_bucket_settings()
        for index, image_path in enumerate(self.images):
            with Image.open(image_path) as img:
                src_w, src_h = img.size
                has_alpha = image_has_alpha(img)
            if cfg.enable_bucket:
                bucket_w, bucket_h = pick_bucket_size(
                    src_w, src_h,
                    min_reso=cfg.min_bucket_reso,
                    max_reso=cfg.max_bucket_reso,
                    step=cfg.bucket_reso_steps,
                    no_upscale=cfg.bucket_no_upscale,
                    area=cfg.train_resolution ** 2,
                )
            else:
                bucket_w = bucket_h = cfg.train_resolution
            geom = fit_geometry(src_w, src_h, bucket_w, bucket_h)
            if geom.pad_area > 0:
                self.n_padded += 1
            self._pad_total += geom.pad_area
            has_mask = mask_path_for(image_path).is_file() or has_alpha
            if has_mask:
                self.n_masked += 1
            self.records.append(
                {
                    "path": image_path,
                    "src_w": int(src_w),
                    "src_h": int(src_h),
                    "bucket_w": int(bucket_w),
                    "bucket_h": int(bucket_h),
                    "geom": geom,
                    "has_mask": has_mask,
                }
            )
            self.buckets[(int(bucket_w), int(bucket_h))].append(index)

        self.mean_pad = self._pad_total / len(self.records) if self.records else 0.0

    def _check_bucket_settings(self) -> None:
        """The area budget and the axis clamps must bracket each other, or every bucket is a clamp."""
        cfg = self.cfg
        if not cfg.enable_bucket:
            return
        reso = int(cfg.train_resolution)
        if not int(cfg.min_bucket_reso) <= reso <= int(cfg.max_bucket_reso):
            print(
                f"[Warn] min_bucket_reso={cfg.min_bucket_reso} / max_bucket_reso={cfg.max_bucket_reso} "
                f"do not bracket train_resolution={reso}: buckets will sit on a clamp and lose the "
                f"image's aspect ratio.",
                file=sys.stderr,
            )

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

    def _cache_path(self, image_path: Path, geom: Geometry) -> Path:
        """Keyed by the fit geometry, not just the bucket: a bucket-size coincidence must not
        serve a latent that was encoded from differently placed pixels."""
        key = (
            f"{image_path.resolve()}::{geom.bucket_w}x{geom.bucket_h}"
            f"::{geom.left},{geom.top},{geom.fit_w}x{geom.fit_h}"
        )
        return self.latent_cache_dir / f"{sha1_text(key)}.pt"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image_path: Path = record["path"]
        bucket_w = record["bucket_w"]
        bucket_h = record["bucket_h"]
        geom: Geometry = record["geom"]
        cache_path = self._cache_path(image_path, geom)

        if self.cfg.cache_latents and self.cfg.cache_latents_to_disk and cache_path.exists():
            img_type = "latent"
            img_data = torch.load(cache_path, map_location="cpu")
        else:
            img_type = "pixel"
            with Image.open(image_path) as img:
                img = fit_to_bucket(img.convert("RGB"), geom)
                img_data = image_to_tensor(img)

        loss_mask = load_loss_mask(
            image_path,
            bucket_w,
            bucket_h,
            record["src_w"],
            record["src_h"],
            geom=geom,
        )
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
            "loss_mask": loss_mask,
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

    mask_items = [ex["loss_mask"] for ex in examples]
    loss_mask: Any = mask_items
    if mask_items and all(torch.is_tensor(item) for item in mask_items):
        try:
            loss_mask = torch.stack(mask_items, dim=0)
        except RuntimeError:
            loss_mask = mask_items

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
        "loss_mask": loss_mask,
    }


def make_collate_fn():
    return collate_fn


SDXLLoraDataset = LoraImageDataset
