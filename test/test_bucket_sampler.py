import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

import sys

# `python test/test_bucket_sampler.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import TrainConfig
from trainer.dataset import BucketBatchSampler, LoraImageDataset, collate_fn
from trainer.loop import group_indices_by_bucket
from trainer.utils import (
    fit_geometry,
    load_loss_mask,
    pick_bucket_size,
    round_to_step,
)

STEP = 128
MIN_RESO = 384
MAX_RESO = 2688
AREA = 1024 ** 2


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
    cfg.train_resolution = 1024
    cfg.min_bucket_reso = 384
    cfg.max_bucket_reso = 2688
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

    def test_cache_hit_serves_the_cached_latent(self):
        # The cache holds latents only; the mask is recomputed from the source image every epoch,
        # so a cache hit skips the pixel decode, not the file itself.
        cfg = _cfg(self.root)
        dataset = LoraImageDataset(cfg)
        item = dataset[0]
        cache_path = Path(item["cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(4, 8, 8), cache_path)
        cached = dataset[0]
        self.assertEqual(cached["img_type"], "latent")
        self.assertEqual(tuple(cached["img_data"].shape), (4, 8, 8))
        self.assertEqual(tuple(cached["loss_mask"].shape), (1, item["bucket_h"], item["bucket_w"]))

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
                "loss_mask": torch.ones(1, 768, 1024),
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


class BucketGeometryTest(unittest.TestCase):
    """Area-budgeted buckets + fit geometry: what the old rule cropped must survive intact."""

    SIZES = [
        (1024, 1024), (1280, 768), (768, 1280), (512, 1536), (896, 2176),
        (2048, 512), (3840, 2160), (300, 3000), (100, 400), (1200, 800),
    ]

    def _bucket(self, w, h, no_upscale=True):
        return pick_bucket_size(w, h, MIN_RESO, MAX_RESO, STEP, no_upscale, area=AREA)

    def test_bucket_is_step_aligned_and_within_clamps(self):
        for w, h in self.SIZES:
            bw, bh = self._bucket(w, h)
            # `no_upscale` may floor an axis below `min_bucket_reso` when the source itself is
            # smaller than one step; the clamps bound everything else.
            floors = (round_to_step(w, STEP), round_to_step(h, STEP))
            for side, axis_floor, src in ((bw, floors[0], w), (bh, floors[1], h)):
                # ROCm requires latent dims divisible by 16, i.e. bucket sides divisible by 128
                self.assertEqual(side % STEP, 0, msg=(w, h, bw, bh))
                self.assertGreaterEqual(side, min(MIN_RESO, axis_floor), msg=(w, h, bw, bh))
                self.assertLessEqual(side, MAX_RESO, msg=(w, h, bw, bh))
                self.assertLessEqual(side, max(STEP, axis_floor), msg=(w, h, bw, bh))

    def test_geometry_keeps_content_and_never_upscales(self):
        for w, h in self.SIZES:
            bw, bh = self._bucket(w, h)
            geom = fit_geometry(w, h, bw, bh)
            self.assertLessEqual(geom.fit_w, bw, msg=(w, h))
            self.assertLessEqual(geom.fit_h, bh, msg=(w, h))
            self.assertGreaterEqual(geom.left, 0, msg=(w, h))
            self.assertGreaterEqual(geom.top, 0, msg=(w, h))
            self.assertLessEqual(geom.fit_w / w, 1.0 + 1e-9, msg=(w, h))
            self.assertLessEqual(geom.fit_h / h, 1.0 + 1e-9, msg=(w, h))
            # the content is scaled, never cropped or distorted
            self.assertAlmostEqual(geom.fit_w / geom.fit_h, w / h, delta=0.02, msg=(w, h))
            self.assertLessEqual(geom.pad_area, 0.3, msg=(w, h))

    def test_bucket_aspect_follows_the_image_when_clamps_do_not_bind(self):
        for w, h in [(1024, 1024), (1280, 768), (768, 1280), (512, 1536), (2048, 512), (1200, 800)]:
            bw, bh = self._bucket(w, h)
            self.assertGreater(bw, MIN_RESO, msg=(w, h))
            self.assertLess(bh, MAX_RESO, msg=(w, h))
            self.assertAlmostEqual(bw / bh, w / h, delta=0.12, msg=(w, h))

    def test_no_upscale_never_builds_an_axis_bigger_than_the_source(self):
        for w, h in self.SIZES:
            bw, bh = self._bucket(w, h, no_upscale=True)
            self.assertLessEqual(bw, round_to_step(w, STEP) if w >= STEP else STEP, msg=(w, h))
            self.assertLessEqual(bh, round_to_step(h, STEP) if h >= STEP else STEP, msg=(w, h))

    def test_pad_is_exactly_zero_for_every_shape(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            shapes = [(64, 192), (192, 64), (100, 100), (200, 150), (96, 64)]
            for i, (w, h) in enumerate(shapes):
                path = root / f"img_{i}.png"
                Image.new("RGB", (w, h), (10, 20, 30)).save(path)
                bw, bh = self._bucket(w, h)
                geom = fit_geometry(w, h, bw, bh)
                mask = load_loss_mask(path, bw, bh, w, h, geom=geom)
                self.assertEqual(tuple(mask.shape), (1, bh, bw), msg=(w, h))
                pad = torch.ones(bh, bw, dtype=torch.bool)
                pad[geom.top : geom.top + geom.fit_h, geom.left : geom.left + geom.fit_w] = False
                if bool(pad.any()):
                    self.assertEqual(float(mask[0][pad].max()), 0.0, msg=(w, h))

    def test_cache_key_follows_the_geometry(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_images(root, [(512, 1536)])
            ds = LoraImageDataset(_cfg(root))
            path = ds.records[0]["path"]
            same = ds._cache_path(path, fit_geometry(512, 1536, 256, 896))
            different = ds._cache_path(path, fit_geometry(512, 1536, 256, 768))
            self.assertNotEqual(same, different)
            self.assertEqual(same, ds._cache_path(path, fit_geometry(512, 1536, 256, 896)))
            # the dataset's own lookup uses the record's geometry
            record = ds.records[0]
            self.assertEqual(ds._cache_path(path, record["geom"]), Path(ds[0]["cache_path"]))


if __name__ == "__main__":
    unittest.main()
