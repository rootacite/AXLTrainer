import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from trainer.config import TrainConfig
from trainer.dataset import LoraImageDataset, collate_fn
from trainer.utils import (
    apply_loss_mask,
    is_mask_sidecar,
    list_images,
    load_loss_mask,
    mask_path_for,
)


def _write_rgb(path: Path, size=(64, 64), color=(20, 40, 60)) -> None:
    Image.new("RGB", size, color=color).save(path)


def _write_mask(path: Path, size, paint) -> None:
    img = Image.new("L", size, color=0)
    paint(ImageDraw.Draw(img))
    img.save(path)


def _cfg(data_dir: str, resolution: int = 64) -> TrainConfig:
    return TrainConfig(
        train_data_dir=data_dir,
        enable_bucket=False,
        train_resolution=resolution,
        cache_latents=False,
        cache_latents_to_disk=False,
        shuffle_caption=False,
        max_data_loader_n_workers=0,
        persistent_workers=False,
    )


class MaskSidecarHelpersTest(unittest.TestCase):
    def test_name_rules(self):
        self.assertTrue(is_mask_sidecar(Path("x.mask.png")))
        self.assertTrue(is_mask_sidecar(Path("photo.MASK.PNG")))
        self.assertFalse(is_mask_sidecar(Path("x.png")))
        self.assertFalse(is_mask_sidecar(Path("mask.png")))
        self.assertEqual(mask_path_for(Path("/d/cat.jpg")), Path("/d/cat.mask.png"))

    def test_list_images_skips_sidecars(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_rgb(root / "a.png")
            _write_rgb(root / "a.mask.png", color=(255, 255, 255))
            _write_rgb(root / "b.jpg")
            names = [p.name for p in list_images(root)]
            self.assertEqual(names, ["a.png", "b.jpg"])


class ApplyLossMaskTest(unittest.TestCase):
    def test_ones_is_identity_mean(self):
        err = torch.ones(2, 4, 8, 8)
        masked = apply_loss_mask(err, torch.ones(2, 1, 8, 8))
        self.assertTrue(torch.allclose(masked, err))
        self.assertAlmostEqual(float(masked.mean()), 1.0)

    def test_zeros_is_zero(self):
        err = torch.ones(1, 4, 8, 8) * 3
        masked = apply_loss_mask(err, torch.zeros(1, 1, 8, 8))
        self.assertAlmostEqual(float(masked.mean()), 0.0)

    def test_half_gray_scales(self):
        err = torch.ones(1, 4, 4, 4)
        mask = torch.full((1, 1, 4, 4), 0.5)
        masked = apply_loss_mask(err, mask)
        self.assertAlmostEqual(float(masked.mean()), 0.5, places=5)

    def test_area_downsample_to_latent(self):
        err = torch.ones(1, 4, 2, 2)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, :8, :8] = 1.0
        masked = apply_loss_mask(err, mask)
        self.assertEqual(tuple(masked.shape), (1, 4, 2, 2))
        self.assertGreater(float(masked[0, 0, 0, 0]), 0.9)
        self.assertLess(float(masked[0, 0, 1, 1]), 0.1)

    def test_none_mask_passthrough(self):
        err = torch.randn(1, 4, 4, 4)
        self.assertTrue(torch.equal(apply_loss_mask(err, None), err))


class DatasetMaskTest(unittest.TestCase):
    def test_missing_mask_is_ones_and_sidecar_not_a_sample(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_rgb(root / "a.png", size=(64, 64))
            (root / "a.txt").write_text("solo", encoding="utf-8")
            _write_mask(
                root / "a.mask.png",
                (64, 64),
                lambda d: d.rectangle([0, 0, 31, 63], fill=255),
            )
            _write_rgb(root / "b.png", size=(64, 64), color=(80, 10, 10))
            (root / "b.txt").write_text("smile", encoding="utf-8")

            ds = LoraImageDataset(_cfg(str(root), resolution=64))
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds.n_masked, 1)
            names = {Path(r["path"]).name for r in ds.records}
            self.assertEqual(names, {"a.png", "b.png"})

            item_a = ds[0] if Path(ds.records[0]["path"]).name == "a.png" else ds[1]
            item_b = ds[0] if Path(ds.records[0]["path"]).name == "b.png" else ds[1]
            self.assertEqual(tuple(item_a["loss_mask"].shape), (1, 64, 64))
            self.assertGreater(float(item_a["loss_mask"][0, 32, 8].mean()), 0.9)
            self.assertLess(float(item_a["loss_mask"][0, 32, 56].mean()), 0.1)
            self.assertTrue(torch.allclose(item_b["loss_mask"], torch.ones(1, 64, 64)))

            batch = collate_fn([item_a, item_b])
            self.assertTrue(torch.is_tensor(batch["loss_mask"]))
            self.assertEqual(tuple(batch["loss_mask"].shape), (2, 1, 64, 64))

    def test_alpha_channel_used_when_no_sidecar(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            img = Image.new("RGBA", (64, 64), (10, 20, 30, 0))
            for y in range(64):
                for x in range(32):
                    img.putpixel((x, y), (10, 20, 30, 255))
            img.save(root / "a.png")
            (root / "a.txt").write_text("solo", encoding="utf-8")

            mask = load_loss_mask(root / "a.png", 64, 64, 64, 64)
            self.assertGreater(float(mask[0, 32, 8]), 0.9)
            self.assertLess(float(mask[0, 32, 56]), 0.1)

            ds = LoraImageDataset(_cfg(str(root), resolution=64))
            self.assertEqual(ds.n_masked, 1)
            item = ds[0]
            self.assertGreater(float(item["loss_mask"][0, 32, 8]), 0.9)
            self.assertLess(float(item["loss_mask"][0, 32, 56]), 0.1)

    def test_sidecar_overrides_image_alpha(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            Image.new("RGBA", (64, 64), (10, 20, 30, 0)).save(root / "a.png")
            _write_mask(
                root / "a.mask.png",
                (64, 64),
                lambda d: d.rectangle([0, 0, 63, 63], fill=255),
            )
            mask = load_loss_mask(root / "a.png", 64, 64, 64, 64)
            self.assertTrue(torch.allclose(mask, torch.ones(1, 64, 64)))

    def test_opaque_rgb_stays_unmasked(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_rgb(root / "a.png", size=(64, 64))
            mask = load_loss_mask(root / "a.png", 64, 64, 64, 64)
            self.assertTrue(torch.allclose(mask, torch.ones(1, 64, 64)))

    def test_load_loss_mask_matches_image_center_crop(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = (96, 64)
            img_path = root / "c.png"
            _write_rgb(img_path, size=src)
            # 96x64 → 64x64 center-crops x∈[16,80]. A blob at source (48,32)
            # lands at destination (32,32).
            _write_mask(
                root / "c.mask.png",
                src,
                lambda d: d.ellipse([40, 24, 56, 40], fill=255),
            )
            mask = load_loss_mask(img_path, 64, 64, src[0], src[1])
            self.assertEqual(tuple(mask.shape), (1, 64, 64))
            self.assertGreater(float(mask[0, 32, 32]), 0.9)
            self.assertLess(float(mask[0, 0, 0]), 0.1)
            self.assertLess(float(mask[0, 63, 63]), 0.1)


if __name__ == "__main__":
    unittest.main()
