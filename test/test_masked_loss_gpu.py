import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

import sys

# `python test/test_masked_loss_gpu.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import TrainConfig
from trainer.dataset import LoraImageDataset
from trainer.family import resolve_family
from trainer.utils import list_images, mask_path_for

_HAS_CUDA = torch.cuda.is_available()


def _copy_two_samples(src_dir: Path, dest: Path) -> list[Path]:
    images = list_images(src_dir)
    images = [p for p in images if p.parent == src_dir] or images
    if len(images) < 2:
        raise unittest.SkipTest(f"need at least 2 images in {src_dir}")
    copied = []
    for src in images[:2]:
        dest_img = dest / src.name
        shutil.copy2(src, dest_img)
        txt = src.with_suffix(".txt")
        if txt.is_file():
            shutil.copy2(txt, dest / txt.name)
        copied.append(dest_img)
    return copied


def _center_white_mask(image_path: Path) -> None:
    with Image.open(image_path) as img:
        w, h = img.size
    mask = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=255)
    mask.save(mask_path_for(image_path))


@unittest.skipUnless(_HAS_CUDA, "CUDA/ROCm GPU required")
class MaskedLossGpuTest(unittest.TestCase):
    def test_real_sdxl_masked_loss_lower_than_full(self):
        cfg = TrainConfig()
        model_path = Path(cfg.pretrained_model_name_or_path)
        data_dir = Path(cfg.train_data_dir)
        if not model_path.exists():
            self.skipTest(f"pretrained model missing: {model_path}")
        if not data_dir.is_dir():
            self.skipTest(f"train_data_dir missing: {data_dir}")

        with tempfile.TemporaryDirectory() as raw:
            dest = Path(raw)
            copied = _copy_two_samples(data_dir, dest)
            _center_white_mask(copied[0])

            cfg.train_data_dir = str(dest)
            cfg.cache_latents = False
            cfg.cache_latents_to_disk = False
            cfg.max_data_loader_n_workers = 0
            cfg.persistent_workers = False
            cfg.shuffle_caption = False
            cfg.enable_bucket = False
            cfg.train_resolution = 768

            ds = LoraImageDataset(cfg)
            self.assertEqual(len(ds), 2)
            self.assertGreaterEqual(ds.n_masked, 1)

            item = next(
                ds[i]
                for i in range(len(ds))
                if Path(ds.records[i]["path"]).name == copied[0].name
            )
            family = resolve_family(cfg)
            pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, torch.bfloat16)
            modules = family.unpack(pipe)
            modules.noise_scheduler = family.build_noise_scheduler(pipe, cfg)

            device = torch.device("cuda")
            dtype = torch.bfloat16
            modules.vae.to(device=device, dtype=dtype)
            modules.vae.eval()
            if hasattr(modules.vae.config, "force_upcast"):
                modules.vae.config.force_upcast = False
            for te in modules.text_encoders:
                te.to(device=device, dtype=dtype)
                te.eval()

            pixel = item["img_data"].unsqueeze(0).to(device=device, dtype=dtype)
            with torch.no_grad():
                latents = modules.vae.encode(pixel).latent_dist.sample()
                latents = latents * modules.vae.config.scaling_factor
                encoded = family.encode_prompts(
                    [item["caption"] or "test"], modules, cfg, device, dtype
                )

            modules.vae.to("cpu")
            for te in modules.text_encoders:
                te.to("cpu")
            del pixel
            torch.cuda.empty_cache()

            modules.denoise.to(device=device, dtype=dtype)
            modules.denoise.eval()

            extra = family.extra_cond(
                src_wh=(item["src_w"], item["src_h"]),
                bucket_wh=(item["bucket_w"], item["bucket_h"]),
                device=device,
                dtype=dtype,
            )
            extra_full = {
                **extra,
                "loss_mask": torch.ones_like(item["loss_mask"]).unsqueeze(0).to(device),
            }
            extra_masked = {
                **extra,
                "loss_mask": item["loss_mask"].unsqueeze(0).to(device),
            }

            with torch.no_grad():
                torch.manual_seed(0)
                loss_full = family.denoise_loss(
                    latents=latents,
                    encoded=encoded,
                    extra=extra_full,
                    modules=modules,
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                )
                torch.manual_seed(0)
                loss_masked = family.denoise_loss(
                    latents=latents,
                    encoded=encoded,
                    extra=extra_masked,
                    modules=modules,
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                )
            self.assertTrue(torch.isfinite(loss_full))
            self.assertTrue(torch.isfinite(loss_masked))
            self.assertLess(float(loss_masked), float(loss_full))

            if hasattr(modules.denoise, "enable_gradient_checkpointing"):
                modules.denoise.enable_gradient_checkpointing()
            latents_g = latents.detach().requires_grad_(True)
            loss_b = family.denoise_loss(
                latents=latents_g,
                encoded=encoded,
                extra=extra_masked,
                modules=modules,
                cfg=cfg,
                device=device,
                dtype=dtype,
            )
            loss_b.backward()
            self.assertIsNotNone(latents_g.grad)
            self.assertTrue(torch.isfinite(latents_g.grad).all())

    def test_alpha_channel_without_sidecar(self):
        cfg = TrainConfig()
        model_path = Path(cfg.pretrained_model_name_or_path)
        data_dir = Path(cfg.train_data_dir)
        if not model_path.exists():
            self.skipTest(f"pretrained model missing: {model_path}")
        if not data_dir.is_dir():
            self.skipTest(f"train_data_dir missing: {data_dir}")

        with tempfile.TemporaryDirectory() as raw:
            dest = Path(raw)
            copied = _copy_two_samples(data_dir, dest)
            png = dest / "alpha.png"
            with Image.open(copied[0]) as src:
                rgba = src.convert("RGBA")
                w, h = rgba.size
                pixels = rgba.load()
                for y in range(h):
                    for x in range(w):
                        r, g, b, _ = pixels[x, y]
                        a = 255 if (w // 4 <= x < 3 * w // 4 and h // 4 <= y < 3 * h // 4) else 0
                        pixels[x, y] = (r, g, b, a)
                rgba.save(png)
            src_txt = copied[0].with_suffix(".txt")
            if src_txt.is_file():
                shutil.copy2(src_txt, png.with_suffix(".txt"))
            copied[0].unlink(missing_ok=True)
            src_txt.unlink(missing_ok=True)

            cfg.train_data_dir = str(dest)
            cfg.cache_latents = False
            cfg.cache_latents_to_disk = False
            cfg.max_data_loader_n_workers = 0
            cfg.persistent_workers = False
            cfg.shuffle_caption = False
            cfg.enable_bucket = False
            cfg.train_resolution = 768

            ds = LoraImageDataset(cfg)
            png_name = png.name
            item = next(
                ds[i]
                for i in range(len(ds))
                if Path(ds.records[i]["path"]).name == png_name
            )
            self.assertGreater(ds.n_masked, 0)
            self.assertLess(float(item["loss_mask"].mean()), 0.95)
            self.assertGreater(float(item["loss_mask"].mean()), 0.05)

            family = resolve_family(cfg)
            pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, torch.bfloat16)
            modules = family.unpack(pipe)
            modules.noise_scheduler = family.build_noise_scheduler(pipe, cfg)
            device = torch.device("cuda")
            dtype = torch.bfloat16
            modules.vae.to(device=device, dtype=dtype)
            modules.vae.eval()
            if hasattr(modules.vae.config, "force_upcast"):
                modules.vae.config.force_upcast = False
            for te in modules.text_encoders:
                te.to(device=device, dtype=dtype)
                te.eval()
            pixel = item["img_data"].unsqueeze(0).to(device=device, dtype=dtype)
            with torch.no_grad():
                latents = modules.vae.encode(pixel).latent_dist.sample()
                latents = latents * modules.vae.config.scaling_factor
                encoded = family.encode_prompts(
                    [item["caption"] or "test"], modules, cfg, device, dtype
                )
            modules.vae.to("cpu")
            for te in modules.text_encoders:
                te.to("cpu")
            del pixel
            torch.cuda.empty_cache()
            modules.denoise.to(device=device, dtype=dtype)
            modules.denoise.eval()
            extra = family.extra_cond(
                src_wh=(item["src_w"], item["src_h"]),
                bucket_wh=(item["bucket_w"], item["bucket_h"]),
                device=device,
                dtype=dtype,
            )
            extra_full = {
                **extra,
                "loss_mask": torch.ones_like(item["loss_mask"]).unsqueeze(0).to(device),
            }
            extra_alpha = {
                **extra,
                "loss_mask": item["loss_mask"].unsqueeze(0).to(device),
            }
            with torch.no_grad():
                torch.manual_seed(1)
                loss_full = family.denoise_loss(
                    latents=latents,
                    encoded=encoded,
                    extra=extra_full,
                    modules=modules,
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                )
                torch.manual_seed(1)
                loss_alpha = family.denoise_loss(
                    latents=latents,
                    encoded=encoded,
                    extra=extra_alpha,
                    modules=modules,
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                )
            self.assertTrue(torch.isfinite(loss_full))
            self.assertTrue(torch.isfinite(loss_alpha))
            self.assertLess(float(loss_alpha), float(loss_full))
