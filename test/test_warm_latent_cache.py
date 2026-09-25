#!/usr/bin/env python
"""Standalone verification for trainer/cache.py::warm_latent_cache.

Usage (cwd = repo root):
    python test/test_warm_latent_cache.py          # mock-VAE logic tests
    python test/test_warm_latent_cache.py --real   # + real SDXL VAE smoke test on CPU

The real smoke test reads a few images + captions from the configured
train_data_dir (read-only) and copies them into a temp dir; every cache
write lands in the temp dir, never in the real dataset.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
# Package-qualified imports below: putting `trainer/` itself on sys.path would hand the
# trainer's dual imports (`try: import control / except ImportError: from trainer import ...`)
# a second, top-level module identity, and `unittest discover` imports this file too.
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    # Isolate control.state.json from the live runtime dir. Script mode only: `unittest
    # discover` imports this module as well and must not rewrite the shared environment.
    _RUNTIME_DIR = tempfile.TemporaryDirectory(prefix="axl-cache-test-")
    os.environ["AXL_RUNTIME_DIR"] = _RUNTIME_DIR.name

from PIL import Image

from trainer.config import TrainConfig
from trainer.dataset import LoraImageDataset
from trainer.utils import fit_geometry, pick_bucket_size


# ---------------------------------------------------------------- mock VAE

class _MockLatentDist:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return self.mean


class MockVAE(torch.nn.Module):
    """Deterministic stand-in: latent = pooled pixels (x * 2.0 * scaling_factor 0.5).

    The 8x pooling and the 4 channels are what a real VAE produces, and they are what make the
    files this writes look like cache entries: the dataset serves a cached latent only when its
    shape is the bucket's.
    """

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"scaling_factor": 0.5})()
        self.encode_calls = 0
        self.batch_sizes = []

    def encode(self, x):
        self.encode_calls += 1
        self.batch_sizes.append(x.shape[0])
        pooled = x[:, :1, ::8, ::8].expand(-1, 4, -1, -1)
        return type("Enc", (), {"latent_dist": _MockLatentDist(pooled * 2.0)})()


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
    # The repo's `[[environment.train_data]]` blocks outrank `train_data_dir`, so a test that
    # points the config at its own folder has to drop them: leaving them in reads - and writes
    # latents into - the dataset `config.toml` names.
    cfg.train_data = []
    cfg.train_data_dir = str(data_dir)
    cfg.enable_bucket = True
    cfg.train_resolution = 1024
    cfg.min_bucket_reso = 384
    cfg.max_bucket_reso = 2688
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
        area=cfg.train_resolution ** 2,
    )


def collect_cached(root: Path) -> dict:
    """image stem -> (cache path, latent tensor) for every cached image."""
    cfg = make_cfg(root)
    dataset = LoraImageDataset(cfg)
    out = {}
    for record in dataset.records:
        bw, bh = bucket_for(record["path"], cfg)
        path = dataset._cache_path(
            record["path"],
            fit_geometry(record["src_w"], record["src_h"], bw, bh),
            dataset.latent_cache_dir,
        )
        assert path.exists(), f"missing cache file for {record['path'].name}"
        out[record["path"].stem] = (path, torch.load(path, map_location="cpu"))
    return out


def run_serial(dataset, vae, cfg, device, dtype) -> None:
    """Reference: the original serial loop from before the pipeline rewrite.

    Its skip rule is the pipelined one's - the dataset's `img_type`, which is "pixel" for
    anything the cache cannot answer with - so both sides encode the same set of images.
    """
    vae.eval()
    vae.to(device=device, dtype=dtype)
    for idx in range(len(dataset)):
        item = dataset[idx]
        cache_path = Path(item["cache_path"])
        if item["img_type"] != "pixel":
            continue
        pixel_values = item["img_data"].unsqueeze(0).to(device=device, dtype=dtype)
        latent = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        torch.save(latent.squeeze(0).detach().cpu(), cache_path)
    vae.to("cpu")


# ------------------------------------------------------------------- tests

def test_equivalence_with_serial() -> None:
    """New pipelined impl must produce identical latents to the old loop."""
    from trainer import cache

    with tempfile.TemporaryDirectory() as td:
        dir_a = Path(td) / "a"
        dir_b = Path(td) / "b"
        make_dataset_dir(dir_a, n_images=17)
        make_dataset_dir(dir_b, n_images=17)

        cfg_a, cfg_b = make_cfg(dir_a), make_cfg(dir_b)
        ds_a, ds_b = LoraImageDataset(cfg_a), LoraImageDataset(cfg_b)

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
    from trainer import cache

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_dataset_dir(root, n_images=9)
        cfg = make_cfg(root)

        ds1 = LoraImageDataset(cfg)
        vae1 = MockVAE()
        cache.warm_latent_cache(
            ds1, vae1, cfg, torch.device("cpu"), torch.float32,
            prefetch_workers=4, encode_batch_size=4,
        )
        first_files = sorted(p.name for p in (root / ".latents_cache").glob("*.pt"))

        ds2 = LoraImageDataset(cfg)
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
    from trainer import cache

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_dataset_dir(root, n_images=8)
        cfg = make_cfg(root)

        ds0 = LoraImageDataset(cfg)
        item = ds0[0]
        cache_path = Path(item["cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(4, item["bucket_h"] // 8, item["bucket_w"] // 8), cache_path)

        ds1 = LoraImageDataset(cfg)
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


def test_foreign_cache_file_is_re_encoded() -> None:
    """A file that does not hold its bucket's latent is re-encoded over, not trusted."""
    from trainer import cache

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_dataset_dir(root, n_images=8)
        cfg = make_cfg(root)

        ds0 = LoraImageDataset(cfg)
        item = ds0[0]
        cache_path = Path(item["cache_path"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(4, 8, 8), cache_path)

        ds1 = LoraImageDataset(cfg)
        vae = MockVAE()
        cache.warm_latent_cache(
            ds1, vae, cfg, torch.device("cpu"), torch.float32,
            prefetch_workers=4, encode_batch_size=4,
        )
        assert sum(vae.batch_sizes) == 8, vae.batch_sizes
        rewritten = torch.load(cache_path, map_location="cpu")
        expected = (4, item["bucket_h"] // 8, item["bucket_w"] // 8)
        assert tuple(rewritten.shape) == expected, (rewritten.shape, expected)
        print(f"  [ok] foreign cache file re-encoded as {tuple(rewritten.shape)}")


def test_gate_disabled() -> None:
    """cache_latents=False -> immediate no-op."""
    from trainer import cache

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_dataset_dir(root, n_images=5)
        cfg = make_cfg(root, cache_latents=False)

        ds = LoraImageDataset(cfg)
        vae = MockVAE()
        cache.warm_latent_cache(ds, vae, cfg, torch.device("cpu"), torch.float32)
        assert vae.encode_calls == 0
        assert not (root / ".latents_cache").exists()
        print("  [ok] cache_latents=False -> no-op")


def test_real_vae_smoke(model_root: Path, real_data_root: Path) -> None:
    """Real SDXL VAE with 3 images copied from the real dataset.

    Uses GPU/bf16 when available (matching the real training path), else CPU/fp32.
    """
    from trainer import cache
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
            str(model_root), subfolder="vae", dtype=dtype
        )
        orig_encode = vae.encode
        calls = {"n": 0}

        def counting_encode(x, *a, **k):
            calls["n"] += 1
            return orig_encode(x, *a, **k)

        vae.encode = counting_encode

        t0 = time.time()
        ds = LoraImageDataset(cfg)
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
        ds2 = LoraImageDataset(cfg)
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
        ("foreign cache file re-encoded", test_foreign_cache_file_is_re_encoded),
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
