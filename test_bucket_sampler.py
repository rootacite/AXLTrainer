import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from trainer.config import TrainConfig
from trainer.dataset import BucketBatchSampler, LoraImageDataset, collate_fn
from trainer.loop import group_indices_by_bucket


def _write_images(root: Path, sizes: list[tuple[int, int]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i, (w, h) in enumerate(sizes):
        Image.new("RGB", (w, h), (i, 40, 80)).save(root / f"img_{i:03d}.png")
        (root / f"img_{i:03d}.txt").write_text(
            f"keep_a, keep_b, tag_{i}, extra_{i}, alt_{i}, more_{i}, last_{i}",
            encoding="utf-8",
        )


def _cfg(data_dir: Path, **overrides) -> TrainConfig:
    cfg = TrainConfig()
    cfg.train_data_dir = str(data_dir)
    cfg.enable_bucket = True
    cfg.min_bucket_reso = 768
    cfg.max_bucket_reso = 1280
    cfg.bucket_reso_steps = 128
    cfg.cache_latents = True
    cfg.cache_latents_to_disk = True
    cfg.shuffle_caption = True
    cfg.keep_tokens = 2
    cfg.seed = 123
    cfg.train_batch_size = 3
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class BucketBatchSamplerTest(unittest.TestCase):
    def test_same_bucket_per_batch_and_keeps_remainders(self):
        buckets = {
            (1024, 1024): [0, 1, 2, 3, 4],
            (1280, 768): [5, 6],
            (768, 1280): [7],
        }
        sampler = BucketBatchSampler(buckets, batch_size=3, seed=0)
        batches = list(sampler)
        self.assertEqual(len(sampler), 4)
        self.assertEqual(len(batches), 4)

        membership = {idx: key for key, ids in buckets.items() for idx in ids}
        seen: list[int] = []
        for batch in batches:
            keys = {membership[i] for i in batch}
            self.assertEqual(len(keys), 1, msg=batch)
            seen.extend(batch)
        self.assertCountEqual(seen, list(range(8)))
        sizes = sorted(len(batch) for batch in batches)
        self.assertEqual(sizes, [1, 2, 2, 3])

    def test_epoch_reshuffles(self):
        buckets = {(512, 512): list(range(12))}
        sampler = BucketBatchSampler(buckets, batch_size=4, seed=7)
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), len(second))


class DatasetBucketTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        # Distinct aspect ratios → distinct buckets at step 128.
        _write_images(
            self.root,
            [(1024, 1024), (1024, 1024), (1024, 1024), (1280, 768), (768, 1280)],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_records_sizes_without_pixels(self):
        cfg = _cfg(self.root)
        dataset = LoraImageDataset(cfg)
        self.assertEqual(len(dataset), 5)
        self.assertGreaterEqual(len(dataset.buckets), 2)
        for record in dataset.records:
            self.assertIn("src_w", record)
            self.assertIn("bucket_w", record)

    def test_cache_hit_skips_image_file(self):
        cfg = _cfg(self.root)
        dataset = LoraImageDataset(cfg)
        item = dataset[0]
        cache_path = Path(item["cache_path"])
        torch.save(torch.zeros(4, 8, 8), cache_path)
        missing = Path(item["image_path"])
        missing.unlink()
        cached = dataset[0]
        self.assertEqual(cached["img_type"], "latent")
        self.assertEqual(tuple(cached["img_data"].shape), (4, 8, 8))

    def test_collate_keeps_python_ints_and_stacks_latents(self):
        examples = [
            {
                "image_path": f"{i}.png",
                "caption": "a",
                "bucket_w": 1024,
                "bucket_h": 768,
                "src_w": 1280,
                "src_h": 800,
                "img_type": "latent",
                "img_data": torch.ones(4, 2, 3) * i,
                "cache_path": f"{i}.pt",
            }
            for i in range(3)
        ]
        batch = collate_fn(examples)
        self.assertEqual(batch["bucket_w"], [1024, 1024, 1024])
        self.assertIsInstance(batch["bucket_w"][0], int)
        self.assertTrue(torch.is_tensor(batch["img_data"]))
        self.assertEqual(tuple(batch["img_data"].shape), (3, 4, 2, 3))
        groups = group_indices_by_bucket(batch)
        self.assertEqual(list(groups.keys()), [(1024, 768)])
        self.assertEqual(groups[(1024, 768)], [0, 1, 2])

    def test_caption_epoch_via_shared_tensor(self):
        cfg = _cfg(self.root, shuffle_caption=True, keep_tokens=2)
        dataset = LoraImageDataset(cfg)
        dataset.set_epoch(0)
        cap0 = dataset[0]["caption"]
        dataset.set_epoch(1)
        cap1 = dataset[0]["caption"]
        self.assertNotEqual(cap0, cap1)


if __name__ == "__main__":
    unittest.main()
