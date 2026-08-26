#!/usr/bin/env python
"""Standalone verification for trainer/cache.py::warm_latent_cache.

Usage:
    python trainer/test_warm_latent_cache.py          # mock-VAE logic tests
    python trainer/test_warm_latent_cache.py --real   # + real SDXL VAE smoke test on CPU

The real smoke test reads a few images + captions from the configured
train_data_dir (read-only) and copies them into a temp dir; every cache
write lands in the temp dir, never in the real dataset.
"""

import argparse
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PIL import Image

from config import TrainConfig
from dataset import SDXLLoraDataset
from utils import pick_bucket_size


# ---------------------------------------------------------------- mock VAE

class _MockLatentDist:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return self.mean


class MockVAE(torch.nn.Module):
    """Deterministic stand-in: latent = input (x * 2.0 * scaling_factor 0.5)."""

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"scaling_factor": 0.5})()
        self.encode_calls = 0
        self.batch_sizes = []

    def encode(self, x):
        self.encode_calls += 1
        self.batch_sizes.append(x.shape[0])
        return type("Enc", (), {"latent_dist": _MockLatentDist(x * 2.0)})()


# ------------------------------------------------------------- test helpers

def make_dataset_dir(root: Path, n_images: int = 17) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sizes = [(768, 768), (1200, 800), (800, 1200), (640, 480), (1024, 1024)]
    for i in range(n_images):
        w, h = sizes[i % len(sizes)]
        color = ((i * 37) % 256, (i * 61) % 256, (i * 97) % 256)
        Image.new("RGB", (w, h), color).save(root / f"img_{i:03d}.png")
        (root / f"img_{i:03d}.txt").write_text(f"test caption {i}", encoding="utf-8")
    return root


def make_cfg(data_dir: Path, **overrides) -> TrainConfig:
    cfg = TrainConfig()
    cfg.train_data_dir = str(data_dir)
    cfg.enable_bucket = True
    cfg.min_bucket_reso = 768
    cfg.max_bucket_reso = 1280
    cfg.bucket_reso_steps = 128
    cfg.cache_latents = True
    cfg.cache_latents_to_disk = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def bucket_for(img: Path, cfg: TrainConfig):
    with Image.open(img) as im:
        w, h = im.size
    return pick_bucket_size(
        w, h,
        min_reso=cfg.min_bucket_reso,
        max_reso=cfg.max_bucket_reso,
        step=cfg.bucket_reso_steps,
        no_upscale=cfg.bucket_no_upscale,
    )


def collect_cached(root: Path) -> dict:
    """image stem -> (cache path, latent tensor) for every cached image."""
    cfg = make_cfg(root)
    dataset = SDXLLoraDataset(cfg)
    out = {}
    for img in sorted(root.glob("*.png")):
        bw, bh = bucket_for(img, cfg)
        path = dataset._cache_path(img, bw, bh)
        assert path.exists(), f"missing cache file for {img.name}"
        out[img.stem] = (path, torch.load(path, map_location="cpu"))
    return out


def run_serial(dataset, vae, cfg, device, dtype) -> None:
    """Reference: the original serial loop from before the pipeline rewrite."""
    vae.eval()
    vae.to(device=device, dtype=dtype)
    for idx in range(len(dataset)):
        item = dataset[idx]
        cache_path = Path(item["cache_path"])
        if cache_path.exists() or item["img_type"] != "pixel":
            continue
        pixel_values = item["img_data"].unsqueeze(0).to(device=device, dtype=dtype)
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        torch.save(latent.squeeze(0).detach().cpu(), cache_path)
    vae.to("cpu")


# ------------------------------------------------------------------- tests

def test_equivalence_with_serial() -> None:
    """New pipelined impl must produce identical latents to the old loop."""
    import cache

    with tempfile.TemporaryDirectory() as td:
        dir_a = Path(td) / "a"
        dir_b = Path(td) / "b"
        make_dataset_dir(dir_a, n_images=17)
        make_dataset_dir(dir_b, n_images=17)

        cfg_a, cfg_b = make_cfg(dir_a), make_cfg(dir_b)
        ds_a, ds_b = SDXLLoraDataset(cfg_a), SDXLLoraDataset(cfg_b)

        vae_serial = MockVAE()
        run_serial(ds_b, vae_serial, cfg_b, torch.device("cpu"), torch.float32)

        vae_new = MockVAE()
        cache.warm_latent_cache(
            ds_a, vae_new, cfg_a, torch.device("cpu"), torch.float32,
            prefetch_workers=4, encode_batch_size=4,
        )

        got_a = collect_cached(dir_a)
        got_b = collect_cached(dir_b)
        assert got_a.keys() == got_b.keys(), "cache file sets differ"
        for stem in got_a:
            assert torch.equal(got_a[stem][1], got_b[stem][1]), f"latent differs for {stem}"

        # batching sanity: batches grouped per bucket, all <= batch_size
        per_bucket = Counter(bucket_for(img, cfg_b) for img in sorted(dir_b.glob("*.png")))
        expected_calls = sum(-(-c // 4) for c in per_bucket.values())
        assert vae_new.encode_calls == expected_calls, (vae_new.encode_calls, expected_calls)
        assert all(b <= 4 for b in vae_new.batch_sizes)
        print(
            f"  [ok] equivalence: {len(got_a)} latents identical; "
            f"serial calls={vae_serial.encode_calls} -> batched calls={vae_new.encode_calls}"
        )


def test_skip_on_second_run() -> None:
    """Warm cache twice; the second run must re-encode nothing."""
    import cache

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_dataset_dir(root, n_images=9)
        cfg = make_cfg(root)

        ds1 = SDXLLoraDataset(cfg)
        vae1 = MockVAE()
        cache.warm_latent_cache(
            ds1, vae1, cfg, torch.device("cpu"), torch.float32,
            prefetch_workers=4, encode_batch_size=4,
        )
        first_files = sorted(p.name for p in (root / ".latents_cache").glob("*.pt"))

        ds2 = SDXLLoraDataset(cfg)
        vae2 = MockVAE()
        cache.warm_latent_cache(
            ds2, vae2, cfg, torch.device("cpu"), torch.float32,
            prefetch_workers=4, encode_batch_size=4,
        )
        assert vae2.encode_calls == 0, "second run should not encode anything"
        second_files = sorted(p.name for p in (root / ".latents_cache").glob("*.pt"))
        assert first_files == second_files
        print(f"  [ok] second run: 0 encodes (first run used {vae1.encode_calls} batched calls)")


def test_mixed_precached() -> None:
    """Pre-existing cache files are skipped; the rest are still encoded."""
    import cache

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_dataset_dir(root, n_images=8)
        cfg = make_cfg(root)

        ds0 = SDXLLoraDataset(cfg)
        item = ds0[0]
        cache_path = Path(item["cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(4, 8, 8), cache_path)

        ds1 = SDXLLoraDataset(cfg)
        vae = MockVAE()
        cache.warm_latent_cache(
            ds1, vae, cfg, torch.device("cpu"), torch.float32,
            prefetch_workers=4, encode_batch_size=4,
        )
        files = list((root / ".latents_cache").glob("*.pt"))
        assert len(files) == 8, f"expected 8 cache files, got {len(files)}"
        assert sum(vae.batch_sizes) == 7, vae.batch_sizes
        assert all(b <= 4 for b in vae.batch_sizes)
        print(f"  [ok] pre-cached 1 of 8 skipped; encoded batches={vae.batch_sizes}")


def test_gate_disabled() -> None:
    """cache_latents=False -> immediate no-op."""
    import cache

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_dataset_dir(root, n_images=5)
        cfg = make_cfg(root, cache_latents=False)

        ds = SDXLLoraDataset(cfg)
        vae = MockVAE()
        cache.warm_latent_cache(ds, vae, cfg, torch.device("cpu"), torch.float32)
        assert vae.encode_calls == 0
        assert not (root / ".latents_cache").exists()
        print("  [ok] cache_latents=False -> no-op")


def test_real_vae_smoke(model_root: Path, real_data_root: Path) -> None:
    """Real SDXL VAE with 3 images copied from the real dataset.

    Uses GPU/bf16 when available (matching the real training path), else CPU/fp32.
    """
    import cache
    from diffusers import AutoencoderKL

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    dtype = torch.bfloat16 if use_gpu else torch.float32
    where = "GPU/bf16" if use_gpu else "CPU/fp32"

    src_imgs = sorted(real_data_root.glob("*.png"))[:3]
    assert src_imgs, f"no images in {real_data_root}"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for p in src_imgs:
            shutil.copy2(p, root / p.name)
            txt = p.with_suffix(".txt")
            if txt.exists():
                shutil.copy2(txt, root / txt.name)

        cfg = make_cfg(root, enable_bucket=False, train_resolution=512)
        vae = AutoencoderKL.from_pretrained(
            str(model_root), subfolder="vae", torch_dtype=torch.float32
        )
        orig_encode = vae.encode
        calls = {"n": 0}

        def counting_encode(x, *a, **k):
            calls["n"] += 1
            return orig_encode(x, *a, **k)

        vae.encode = counting_encode

        t0 = time.time()
        ds = SDXLLoraDataset(cfg)
        cache.warm_latent_cache(
            ds, vae, cfg, device, dtype,
            prefetch_workers=2, encode_batch_size=2,
        )
        dt = time.time() - t0

        files = sorted((root / ".latents_cache").glob("*.pt"))
        assert len(files) == 3, f"expected 3 cache files, got {len(files)}"
        lat = torch.load(files[0], map_location="cpu")
        assert tuple(lat.shape) == (4, 64, 64), lat.shape
        assert lat.dtype == dtype, (lat.dtype, dtype)
        assert calls["n"] == 2, f"expected 2 batched encodes (2+1), got {calls['n']}"

        # second run: everything cached, zero encodes
        calls["n"] = 0
        ds2 = SDXLLoraDataset(cfg)
        cache.warm_latent_cache(
            ds2, vae, cfg, device, dtype,
            prefetch_workers=2, encode_batch_size=2,
        )
        assert calls["n"] == 0
        print(
            f"  [ok] real VAE ({where}): 3 images -> 3 latents {tuple(lat.shape)} "
            f"in {dt:.1f}s; second run 0 encodes"
        )


# ------------------------------------------------------------------- runner

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="also run the real-VAE smoke test on CPU")
    args = parser.parse_args()

    tests = [
        ("equivalence vs serial", test_equivalence_with_serial),
        ("skip on second run", test_skip_on_second_run),
        ("mixed pre-cached", test_mixed_precached),
        ("gate disabled", test_gate_disabled),
    ]
    if args.real:
        tests.append(("real VAE smoke (CPU)", test_real_vae_smoke))

    failed = 0
    for name, fn in tests:
        print(f"== {name}")
        try:
            if name.startswith("real VAE"):
                probe = TrainConfig()
                fn(Path(probe.pretrained_model_name_or_path), Path(probe.train_data_dir))
            else:
                fn()
        except Exception:
            failed += 1
            traceback.print_exc()

    print("PASS" if failed == 0 else f"FAIL ({failed} test(s) failed)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
