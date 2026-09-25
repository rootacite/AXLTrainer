import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import sys

# `python test/test_masked_loss.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import TrainConfig
from trainer.dataset import LoraImageDataset, collate_fn
from trainer.utils import (
    FIT_PAD_VALUE,
    apply_loss_mask,
    fit_geometry,
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


def _cfg(data_dir: str, resolution: int = 64, **overrides) -> TrainConfig:
    cfg = TrainConfig(
        # The repo's `[[environment.train_data]]` blocks outrank `train_data_dir`: a test that
        # points the config at its own folder has to drop them or it reads another dataset.
        train_data=[],
        train_data_dir=data_dir,
        enable_bucket=False,
        train_resolution=resolution,
        cache_latents=False,
        cache_latents_to_disk=False,
        shuffle_caption=False,
        max_data_loader_n_workers=0,
        persistent_workers=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


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

    def test_load_loss_mask_follows_fit_geometry(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = (96, 64)
            img_path = root / "c.png"
            _write_rgb(img_path, size=src)
            # 96x64 fits into 64x64 at scale 2/3 -> content 64x43, centred: top rows are pad.
            # A blob at source (48,32) lands at content (32,21), i.e. destination (32,31).
            _write_mask(
                root / "c.mask.png",
                src,
                lambda d: d.ellipse([40, 24, 56, 40], fill=255),
            )
            geom = fit_geometry(src[0], src[1], 64, 64)
            self.assertEqual((geom.fit_w, geom.fit_h, geom.left, geom.top), (64, 43, 0, 10))

            mask = load_loss_mask(img_path, 64, 64, src[0], src[1])
            self.assertEqual(tuple(mask.shape), (1, 64, 64))
            self.assertGreater(float(mask[0, 31, 32]), 0.9)
            # pad rows above and below the content are exactly zero
            self.assertEqual(float(mask[0, : geom.top, :].max()), 0.0)
            self.assertEqual(float(mask[0, geom.top + geom.fit_h :, :].max()), 0.0)

    def test_unmasked_image_still_carries_zero_pad(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_rgb(root / "a.png", size=(96, 64))
            (root / "a.txt").write_text("solo", encoding="utf-8")

            mask = load_loss_mask(root / "a.png", 64, 64, 96, 64)
            geom = fit_geometry(96, 64, 64, 64)
            self.assertEqual(float(mask[0, : geom.top, :].max()), 0.0)
            self.assertGreater(float(mask[0, geom.top : geom.top + geom.fit_h, :].min()), 0.99)
            # content area is uncovered except for the mask's own resampling edge
            self.assertGreater(float(mask[:, geom.top : geom.top + geom.fit_h, :].mean()), 0.99)

            ds = LoraImageDataset(_cfg(str(root)))
            self.assertEqual(ds.n_masked, 0)
            self.assertEqual(ds.n_padded, 1)
            item = ds[0]
            self.assertEqual(float(item["loss_mask"][0, 0, 0]), 0.0)
            pixels = item["img_data"]
            pad = pixels[:, : geom.top, :]
            self.assertTrue(torch.allclose(pad, torch.full_like(pad, (FIT_PAD_VALUE / 255.0) * 2 - 1)))

    def test_tall_image_keeps_head_and_feet(self):
        """Today's crop rule dropped both end bands of a tall image; fit+pad must keep them."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src_w, src_h = 512, 1536
            band = src_h // 20
            img = Image.new("RGB", (src_w, src_h), (0, 0, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, src_w - 1, band - 1], fill=(255, 0, 0))
            draw.rectangle([0, src_h - band, src_w - 1, src_h - 1], fill=(0, 255, 0))
            img.save(root / "tall.png")
            (root / "tall.txt").write_text("tall", encoding="utf-8")
            # trainable exactly on the two end bands
            def paint(d):
                d.rectangle([0, 0, src_w - 1, band - 1], fill=255)
                d.rectangle([0, src_h - band, src_w - 1, src_h - 1], fill=255)

            _write_mask(root / "tall.mask.png", (src_w, src_h), paint)

            cfg = _cfg(
                str(root),
                resolution=512,
                enable_bucket=True,
                min_bucket_reso=128,
                max_bucket_reso=2048,
            )
            ds = LoraImageDataset(cfg)
            record = ds.records[0]
            self.assertEqual((record["bucket_w"], record["bucket_h"]), (256, 896))

            item = ds[0]
            pixels = item["img_data"]
            mask = item["loss_mask"][0]
            red = (pixels[0] + 1) / 2 > 0.8
            green = (pixels[1] + 1) / 2 > 0.8
            self.assertTrue(bool(red.any()), "head band is missing from the sample")
            self.assertTrue(bool(green.any()), "feet band is missing from the sample")
            self.assertGreater(float(mask[red].mean()), 0.99)
            self.assertGreater(float(mask[green].mean()), 0.99)
            # and the pad left over from fitting a 1:3 image into the bucket carries no weight
            geom = record["geom"]
            pad = torch.ones(geom.bucket_h, geom.bucket_w, dtype=torch.bool)
            pad[geom.top : geom.top + geom.fit_h, geom.left : geom.left + geom.fit_w] = False
            self.assertTrue(bool(pad.any()), "expected some letterbox pad in this bucket")
            self.assertEqual(float(mask[pad].max()), 0.0)

    def test_alpha_mask_carries_zero_pad(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            img = Image.new("RGBA", (96, 64), (255, 0, 0, 0))
            ImageDraw.Draw(img).rectangle([24, 16, 71, 47], fill=(0, 255, 0, 255))
            img.save(root / "b.png")
            (root / "b.txt").write_text("solo", encoding="utf-8")

            geom = fit_geometry(96, 64, 64, 64)
            mask = load_loss_mask(root / "b.png", 64, 64, 96, 64)
            content = mask[:, geom.top : geom.top + geom.fit_h, :]
            self.assertEqual(float(mask[:, : geom.top, :].max()), 0.0)
            self.assertGreater(float(content[0, 21, 32]), 0.9)
            self.assertLess(float(content[0, 2, 2]), 0.1)
            # the pad does not change the content's own coverage: the opaque rect is 48x32 of the
            # 96x64 source, i.e. 25 % of the (whole, uncropped) image
            self.assertAlmostEqual(float(content.mean()), 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
