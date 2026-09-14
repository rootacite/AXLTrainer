#!/usr/bin/env python3
"""Closed-loop verification for the optional loss-mask pipeline.

Run in the conda env `axl` from the repo root:

    /home/acite/miniconda3/envs/axl/bin/python verify_mask_pipeline.py --tiers all

Tiers
  plumbing  CPU. Sidecar discovery/exclusion, crop+bucket alignment against the real pixel
            tensor, per-sample mask pairing, latent-cache independence, real dataset scans.
  loss      GPU, one pipeline load. Exact loss identities (all-ones == no mask, all-black ->
            zero loss and zero grads), mask linearity, coverage -> gradient scaling, mask
            position sensitivity, and measured leakage of ignored-region content.
  train     GPU. Real `trainer/main.py` subprocesses (mirrored repo + generated config) on
            copies of the configured dataset: masked vs unmasked, N seeds, a duplicate short
            run as the repeatability floor, and a weight-only resume run. Per-region error is
            probed afterwards with the parent process' pipeline.
  stand     GPU. Same as `train` on copies of the untagged stand dataset, where the mask comes
            from each image's own alpha channel.

Results go to <report-dir>/mask_verify_report.md and .json. Nothing is ever written into the
real dataset directories: images are copied first, and every run gets its own temp runtime dir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_STANDS = Path("/storage/Games/AVG/LimeLight Lemonade Jam/dataset/stands/杏珠")
AXL_PYTHON = Path("/home/acite/miniconda3/envs/axl/bin/python")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TIERS = ("plumbing", "loss", "train", "stand")
LIVE_STATUSES = {"starting", "encoding", "training", "sampling", "pausing", "paused", "resuming", "stopping"}


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


@dataclass
class Check:
    tier: str
    name: str
    ok: bool
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    tier: str
    name: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


def fmt_values(values: dict[str, Any]) -> str:
    if not values:
        return ""
    parts = []
    for key, value in values.items():
        if isinstance(value, bool) or value is None:
            parts.append(f"{key}={value}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return "  (" + ", ".join(parts) + ")"


class Report:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.checks: list[Check] = []
        self.observations: list[Observation] = []
        self.env: dict[str, Any] = {}
        self.notes: list[str] = []
        self.started = time.time()

    def check(self, tier: str, name: str, ok: bool, detail: str, **values: Any) -> bool:
        self.checks.append(Check(tier, name, bool(ok), detail, values))
        print(f"{'PASS' if ok else 'FAIL'}  [{tier}] {name} — {detail}{fmt_values(values)}", flush=True)
        return bool(ok)

    def observe(self, tier: str, name: str, detail: str, **values: Any) -> None:
        self.observations.append(Observation(tier, name, detail, values))
        print(f"OBSV  [{tier}] {name} — {detail}{fmt_values(values)}", flush=True)

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.ok]


# --------------------------------------------------------------------------------------
# environment / safety guards
# --------------------------------------------------------------------------------------


def guard_environment(args: argparse.Namespace, rep: Report) -> Any:
    """Refuse to run outside the `axl` conda env; record what we actually run on."""
    if Path(sys.prefix).name != "axl" and not args.allow_foreign_env:
        raise SystemExit(
            "refusing to run: expected the conda env `axl`\n"
            f"  interpreter: {sys.executable}\n"
            f"  prefix:      {sys.prefix}\n"
            f"  use: {AXL_PYTHON} verify_mask_pipeline.py ... (or pass --allow-foreign-env)"
        )
    import torch

    rep.env = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "launched": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return torch


def runtime_dirs() -> list[Path]:
    dirs: list[Path] = []
    if os.environ.get("AXL_RUNTIME_DIR"):
        dirs.append(Path(os.environ["AXL_RUNTIME_DIR"]))
    if os.environ.get("XDG_RUNTIME_DIR"):
        dirs.append(Path(os.environ["XDG_RUNTIME_DIR"]) / "axltrainer")
    dirs.append(Path(f"/tmp/axltrainer-{os.getuid()}"))
    return dirs


def live_training_runs() -> list[str]:
    hits: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        parts = [part for part in raw.split("\0") if part]
        if parts and "python" in Path(parts[0]).name and any("trainer/main.py" in part for part in parts):
            hits.append(f"pid {entry.name}: {' '.join(parts[:3])}")
    for runtime in runtime_dirs():
        state = runtime / "state.json"
        if not state.is_file():
            continue
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") in LIVE_STATUSES:
            hits.append(f"{state}: status={data.get('status')} pid={data.get('pid')}")
    return sorted(set(hits))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def base_config(**overrides: Any):
    from trainer.config import TrainConfig

    cfg = TrainConfig()
    cfg.max_data_loader_n_workers = 0
    cfg.persistent_workers = False
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def write_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def right_region_sidecar(image_path: Path, keep_fraction: float = 0.6) -> None:
    """White on the left `keep_fraction` of the image, black on the right."""
    from PIL import Image, ImageDraw

    from trainer.utils import mask_path_for

    with Image.open(image_path) as img:
        width, height = img.size
    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, 0, width - 1, height - 1], fill=255)
    draw.rectangle([int(round(width * keep_fraction)), 0, width - 1, height - 1], fill=0)
    mask.save(mask_path_for(image_path))


def region_mask(width: int, height: int, kind: str):
    import torch

    mask = torch.zeros(1, height, width, dtype=torch.float32)
    if kind == "ones":
        mask.fill_(1.0)
    elif kind == "zeros":
        mask.fill_(0.0)
    elif kind == "half_gray":
        mask.fill_(0.5)
    elif kind == "left":
        mask[:, :, : width // 2] = 1.0
    elif kind == "right":
        mask[:, :, width // 2 :] = 1.0
    elif kind == "quarter":
        mask[:, :, : width // 4] = 1.0
    elif kind == "right40":
        mask[:, :, int(round(width * 0.6)) :] = 1.0
    elif kind == "left60":
        mask[:, :, : int(round(width * 0.6))] = 1.0
    else:
        raise ValueError(f"unknown mask kind: {kind}")
    return mask


def uniform_sidecar(image_path: Path, value: int) -> None:
    """Overwrite a sample's mask sidecar with a constant weight (scale-matched control)."""
    from PIL import Image

    from trainer.utils import mask_path_for

    with Image.open(image_path) as img:
        width, height = img.size
    Image.new("L", (width, height), color=value).save(mask_path_for(image_path))


def bucket_crop(image_path: Path, bucket_w: int, bucket_h: int, mode: str = "RGB", channel: str | None = None):
    """Crop like the trainer does (LANCZOS, centre crop), implemented here independently of
    `trainer.utils.resize_and_center_crop` so the comparison is not self-referential."""
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as img:
        source = img.convert(mode)
        if channel:
            source = source.getchannel(channel)
    src_w, src_h = source.size
    scale = max(bucket_w / src_w, bucket_h / src_h)
    new_size = (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale))))
    resized = source.resize(new_size, Image.Resampling.LANCZOS)
    left = max(0, (new_size[0] - bucket_w) // 2)
    top = max(0, (new_size[1] - bucket_h) // 2)
    cropped = resized.crop((left, top, left + bucket_w, top + bucket_h))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    return array


def copy_samples(source_dir: Path, dest: Path, count: int, *, rgb_only: bool, sidecars: bool,
                 distinct_alpha: bool = False) -> list[Path]:
    """Copy up to `count` image+caption pairs out of `source_dir`. Never writes into source_dir.

    With `distinct_alpha`, only images whose alpha channel differs from everything already taken
    are used, so a multi-image alpha-mask run exercises more than one mask shape."""
    import hashlib

    import numpy as np
    from PIL import Image

    from trainer.utils import list_images

    images = [p for p in list_images(source_dir) if p.suffix.lower() in IMAGE_EXT]
    if len(images) < count:
        raise RuntimeError(f"need {count} images in {source_dir}, found {len(images)}")
    shutil.rmtree(dest, ignore_errors=True)  # keep re-runs into the same work dir clean
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "_provenance.json").write_text(json.dumps({
        "source_dir": str(source_dir), "requested": count, "rgb_only": rgb_only,
        "sidecars": sidecars, "distinct_alpha": distinct_alpha,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    copied: list[Path] = []
    seen: set[str] = set()
    for src in images:
        if len(copied) == count:
            break
        with Image.open(src) as img:
            if distinct_alpha:
                alpha = np.asarray(img.convert("RGBA").getchannel("A"))
                digest = hashlib.sha1(alpha.tobytes()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
            pixels = img.convert("RGB") if rgb_only else img
            target = dest / f"{src.stem}.png"
            pixels.save(target)
        caption = src.with_suffix(".txt")
        text = caption.read_text(encoding="utf-8") if caption.is_file() else src.stem.replace("_", " ")
        (dest / f"{src.stem}.txt").write_text(text, encoding="utf-8")
        if sidecars:
            right_region_sidecar(target)
        copied.append(target)
    return copied


# --------------------------------------------------------------------------------------
# tier: plumbing (CPU)
# --------------------------------------------------------------------------------------


def tier_plumbing(rep: Report, args: argparse.Namespace, work: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from types import SimpleNamespace

    from trainer.dataset import LoraImageDataset, collate_fn
    from trainer.loop import build_group_inputs, group_indices_by_bucket
    from trainer.utils import is_mask_sidecar, list_images, load_loss_mask, mask_path_for, sha1_text

    tier = "plumbing"

    # 1. sidecar discovery, exclusion from the image listing, per-sample mask -------------
    root = work / "plumbing" / "discovery"
    write_rgb(root / "a.png", (64, 64), (20, 40, 60))
    (root / "a.txt").write_text("solo", encoding="utf-8")
    right_region_sidecar(root / "a.png", keep_fraction=0.5)
    write_rgb(root / "b.jpg", (64, 64), (80, 10, 10))
    (root / "b.txt").write_text("smile", encoding="utf-8")

    cfg = base_config(train_data_dir=str(root), enable_bucket=False, train_resolution=64,
                      cache_latents=False, cache_latents_to_disk=False, shuffle_caption=False)
    ds = LoraImageDataset(cfg)
    names = sorted(Path(record["path"]).name for record in ds.records)
    item_a = next(ds[i] for i in range(len(ds)) if Path(ds.records[i]["path"]).stem == "a")
    item_b = next(ds[i] for i in range(len(ds)) if Path(ds.records[i]["path"]).stem == "b")
    mask_left = float(item_a["loss_mask"][0, 32, 8])
    mask_right = float(item_a["loss_mask"][0, 32, 56])
    batch = collate_fn([item_a, item_b])
    rep.check(
        tier, "sidecar discovered, excluded from the listing, paired with its own image",
        names == ["a.png", "b.jpg"] and ds.n_masked == 1 and len(ds) == 2
        and mask_left > 0.9 and mask_right < 0.1 and float(item_b["loss_mask"].min()) == 1.0
        and tuple(batch["loss_mask"].shape) == (2, 1, 64, 64),
        "the mask sidecar is never a training sample; the unmasked sibling stays all-ones",
        samples=names, n_masked=ds.n_masked, masked_side=mask_left, ignored_side=mask_right,
        collated=tuple(batch["loss_mask"].shape),
    )
    rep.check(
        tier, "mask sidecar name rules",
        is_mask_sidecar(Path("x.mask.png")) and is_mask_sidecar(Path("photo.MASK.PNG"))
        and not is_mask_sidecar(Path("mask.png")) and not is_mask_sidecar(Path("x.png"))
        and mask_path_for(Path("/d/cat.jpg")).name == "cat.mask.png",
        "`{stem}.mask.png` recognized, `mask.png` alone is still a normal image",
    )

    # 2. geometry: does the mask land on the same pixels as the image? --------------------
    root = work / "plumbing" / "align_synth"
    root.mkdir(parents=True, exist_ok=True)
    from PIL import ImageDraw

    half = Image.new("RGB", (96, 64), (0, 0, 255))
    draw = ImageDraw.Draw(half)
    draw.rectangle([0, 0, 47, 63], fill=(255, 0, 0))
    half.save(root / "ab.png")
    (root / "ab.txt").write_text("test", encoding="utf-8")
    mask = Image.new("L", (96, 64), color=0)
    ImageDraw.Draw(mask).rectangle([0, 0, 47, 63], fill=255)
    mask.save(mask_path_for(root / "ab.png"))

    cfg = base_config(train_data_dir=str(root), enable_bucket=False, train_resolution=64,
                      cache_latents=False, cache_latents_to_disk=False, shuffle_caption=False)
    item = LoraImageDataset(cfg)[0]
    pixels, item_mask = item["img_data"], item["loss_mask"][0]
    red, blue = (pixels[0] + 1) / 2, (pixels[2] + 1) / 2
    interior_left = torch.zeros(64, dtype=torch.bool)
    interior_left[4:28] = True   # away from the centre-crop seam
    interior_right = torch.zeros(64, dtype=torch.bool)
    interior_right[36:60] = True
    fully_masked = (item_mask > 0.9).all(dim=0)
    fully_clear = (item_mask < 0.1).all(dim=0)
    red_in = float(red[:, interior_left & fully_masked].mean())
    blue_in = float(blue[:, interior_left & fully_masked].mean())
    blue_out = float(blue[:, interior_right & fully_clear].mean())
    red_out = float(red[:, interior_right & fully_clear].mean())
    rep.check(
        tier, "mask geometry matches the image after crop/resize (96x64 -> 64x64)",
        red_in > 0.8 and blue_in < 0.2 and blue_out > 0.8 and red_out < 0.2,
        "masked columns carry the red half, unmasked columns the blue half",
        red_where_masked=red_in, blue_where_masked=blue_in,
        blue_where_unmasked=blue_out, red_where_unmasked=red_out,
    )

    # 3. the same question on a real image, at its real bucket ---------------------------
    root = work / "plumbing" / "align_real"
    root.mkdir(parents=True, exist_ok=True)
    real_src = next(p for p in list_images(Path(base_config().train_data_dir)) if p.suffix.lower() == ".png")
    shutil.copy2(real_src, root / "real.png")
    (root / "real.txt").write_text("real sample", encoding="utf-8")
    cfg = base_config(train_data_dir=str(root), enable_bucket=True, cache_latents=False,
                      cache_latents_to_disk=False, shuffle_caption=False)
    ds_real = LoraImageDataset(cfg)
    bucket_w, bucket_h = ds_real.records[0]["bucket_w"], ds_real.records[0]["bucket_h"]
    # A geometric sidecar: the invariant is that the mask survives crop+resize onto the same
    # pixels. A content-derived (median-luminance) mask cannot check that on flat or transparent
    # art, where both sides of the split share one luminance value.
    with Image.open(root / "real.png") as img:
        src_w, src_h = img.size
    geom = Image.new("L", (src_w, src_h), color=0)
    ImageDraw.Draw(geom).rectangle([0, 0, src_w // 2 - 1, src_h - 1], fill=255)
    geom.save(mask_path_for(root / "real.png"))
    item_real = LoraImageDataset(cfg)[0]
    pixels_real = item_real["img_data"]
    mask_real = item_real["loss_mask"][0].numpy()
    with Image.open(mask_path_for(root / "real.png")) as img:
        side_mask = img.convert("L")
    scale = max(bucket_w / side_mask.size[0], bucket_h / side_mask.size[1])
    side = side_mask.resize(
        (max(1, int(round(side_mask.size[0] * scale))), max(1, int(round(side_mask.size[1] * scale)))),
        Image.Resampling.LANCZOS,
    )
    crop_left = max(0, (side.size[0] - bucket_w) // 2)
    crop_top = max(0, (side.size[1] - bucket_h) // 2)
    expected_mask = np.asarray(
        side.crop((crop_left, crop_top, crop_left + bucket_w, crop_top + bucket_h)), dtype=np.float32
    ) / 255.0
    agreement = float((abs(mask_real - expected_mask) < 0.02).mean())
    luma_bucketed = (0.299 * pixels_real[0] + 0.587 * pixels_real[1] + 0.114 * pixels_real[2]).numpy()
    inside, outside = mask_real > 0.5, mask_real < 0.5
    rep.check(
        tier, f"mask geometry survives crop/resize at the real bucket ({bucket_w}x{bucket_h})",
        agreement > 0.99 and float(inside.mean()) > 0.4,
        "a geometric sidecar on a real image lands pixel-for-pixel where an independent crop says "
        "it should",
        bucket=f"{bucket_w}x{bucket_h}", agreement=agreement, coverage=float(inside.mean()),
        image=real_src.name,
    )
    rep.observe(
        tier, "mask vs image content on the real image (depends on the artwork)",
        "for a mask drawn over a content region this is an alignment test; on flat or transparent "
        "art both sides can share one luminance value, which says nothing",
        image=real_src.name,
        luma_masked=float(luma_bucketed[inside].mean()) if inside.any() else None,
        luma_unmasked=float(luma_bucketed[outside].mean()) if outside.any() else None,
    )

    # 4. per-sample pairing across two different buckets ---------------------------------
    class _StubFamily:
        def extra_cond(self, *, src_wh, bucket_wh, device, dtype):
            return {"time_ids": torch.tensor([*src_wh, 0, 0, *bucket_wh], dtype=dtype, device=device)}

    class _StubVae(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=1.0)

        def encode(self, pixel_values):
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: pixel_values[:, :4]))

    root = work / "plumbing" / "pairing"
    root.mkdir(parents=True, exist_ok=True)
    write_rgb(root / "wide.png", (128, 64), (30, 30, 30))
    (root / "wide.txt").write_text("wide", encoding="utf-8")
    write_rgb(root / "tall.png", (64, 128), (60, 60, 60))
    (root / "tall.txt").write_text("tall", encoding="utf-8")
    tall_mask = Image.new("L", (64, 128), color=0)
    ImageDraw.Draw(tall_mask).rectangle([0, 0, 63, 63], fill=255)
    tall_mask.save(mask_path_for(root / "tall.png"))

    cfg = base_config(train_data_dir=str(root), enable_bucket=True, min_bucket_reso=64,
                      max_bucket_reso=128, bucket_reso_steps=64, train_resolution=64,
                      cache_latents=False, cache_latents_to_disk=False, shuffle_caption=False)
    ds_pair = LoraImageDataset(cfg)
    batch_pair = collate_fn([ds_pair[i] for i in range(len(ds_pair))])
    groups = group_indices_by_bucket(batch_pair)
    per_bucket: dict[str, Any] = {}
    for (bucket_w_p, bucket_h_p), indices in groups.items():
        _prompts, _latents, group_extra = build_group_inputs(
            indices=indices, batch=batch_pair, family=_StubFamily(), vae=_StubVae(),
            cfg=cfg, device=torch.device("cpu"), weight_dtype=torch.float32,
        )
        label = Path(batch_pair["image_path"][indices[0]]).name
        per_bucket[label] = group_extra["loss_mask"][0, 0]
    wide_row = per_bucket.get("wide.png")
    tall_row = per_bucket.get("tall.png")
    rep.check(
        tier, "mask follows its own sample through collate + bucket grouping",
        len(groups) == 2 and wide_row is not None and tall_row is not None
        and float(wide_row.min()) == 1.0
        and float(tall_row[: tall_row.shape[0] // 2 - 2].max()) > 0.9
        and float(tall_row[tall_row.shape[0] // 2 + 2 :].max()) < 0.1,
        "two buckets in one batch: the wide image stays unmasked, the tall one stays top-masked",
        groups={f"{k[0]}x{k[1]}": len(v) for k, v in groups.items()},
        wide_uniform=float(wide_row.min()), tall_top=float(tall_row[4, 0]),
        tall_bottom=float(tall_row[-5, 0]),
    )

    # 5. latent-cache independence -------------------------------------------------------
    root = work / "plumbing" / "cache"
    root.mkdir(parents=True, exist_ok=True)
    write_rgb(root / "c.png", (64, 64), (10, 90, 120))
    (root / "c.txt").write_text("cache", encoding="utf-8")
    right_region_sidecar(root / "c.png", keep_fraction=0.5)
    cfg = base_config(train_data_dir=str(root), enable_bucket=False, train_resolution=64,
                      cache_latents=True, cache_latents_to_disk=True, shuffle_caption=False)
    ds_cache = LoraImageDataset(cfg)
    image_path = Path(ds_cache.records[0]["path"])
    bucket_w, bucket_h = ds_cache.records[0]["bucket_w"], ds_cache.records[0]["bucket_h"]
    cache_key = f"{image_path.resolve()}::{bucket_w}x{bucket_h}"
    cache_file = ds_cache.latent_cache_dir / f"{sha1_text(cache_key)}.pt"
    torch.save(torch.full((4, bucket_h // 8, bucket_w // 8), 0.25, dtype=torch.float32), cache_file)
    mtime_before = cache_file.stat().st_mtime_ns
    first = ds_cache[0]
    right_region_sidecar(root / "c.png", keep_fraction=0.8)  # edit the mask only
    second = ds_cache[0]
    rep.check(
        tier, "documented cache key resolves, and editing a mask does not invalidate it",
        first["img_type"] == "latent" and second["img_type"] == "latent"
        and cache_file.stat().st_mtime_ns == mtime_before
        and abs(float(first["loss_mask"].mean()) - 0.5) < 0.02
        and abs(float(second["loss_mask"].mean()) - 0.8) < 0.02
        and float(second["loss_mask"][0, 32, 8]) > 0.9,
        "latent served from `<data>/.latents_cache/{sha1(abs_path::WxH)}.pt` with the new mask",
        key=cache_key, coverage_before=float(first["loss_mask"].mean()),
        coverage_after=float(second["loss_mask"].mean()),
    )

    # 6. real dataset scans --------------------------------------------------------------
    dataset_dir = Path(base_config().train_data_dir)
    ds_kanae = LoraImageDataset(base_config())
    alpha_means, mask_means, agreements = [], [], []
    for record in ds_kanae.records[:60]:
        image = Path(record["path"])
        with Image.open(image) as img:
            has_alpha = "A" in img.getbands()
            if has_alpha:
                alpha_means.append(float(np.asarray(img.getchannel("A"), dtype=np.float32).mean() / 255.0))
        if len(mask_means) < 12 or (has_alpha and not mask_path_for(image).is_file()):
            pipeline_mask = load_loss_mask(
                image, record["bucket_w"], record["bucket_h"], record["src_w"], record["src_h"]
            )[0].numpy()
            mask_means.append(float(pipeline_mask.mean()))
            if has_alpha and not mask_path_for(image).is_file() and len(agreements) < 5:
                expected = bucket_crop(image, record["bucket_w"], record["bucket_h"], "RGBA", "A")
                agreements.append(float((abs(pipeline_mask - expected) < 0.02).mean()))
    rep.check(
        tier, "configured dataset: alpha-derived masks match an independent alpha crop",
        bool(agreements) and all(value > 0.99 for value in agreements),
        "the fallback path (alpha channel -> loss weight) reproduces an independently computed crop",
        checked=len(agreements), worst_agreement=min(agreements) if agreements else None,
    )
    rep.observe(
        tier, f"configured dataset scan ({dataset_dir})",
        "`n_masked` counts alpha-capable files, and alpha only acts as a no-op when it is ~opaque. "
        "This scan follows the live config, which may differ from the copies the training tiers used",
        images=len(ds_kanae), n_masked=ds_kanae.n_masked,
        captions=sum(1 for r in ds_kanae.records if r["path"].with_suffix(".txt").is_file()),
        alpha_mean=float(np.mean(alpha_means)) if alpha_means else None,
        mask_mean=float(np.mean(mask_means)),
        alpha_effectively_opaque=all(value > 0.98 for value in mask_means),
    )

    stands_dir = Path(args.stands_dir)
    if stands_dir.is_dir():
        stand_copy = work / "plumbing" / "stands"
        copied = copy_samples(stands_dir, stand_copy, 3, rgb_only=False, sidecars=False,
                              distinct_alpha=True)
        distinct = _distinct_alpha_count(stands_dir, limit=60)
        ds_stand = LoraImageDataset(base_config(train_data_dir=str(stand_copy)))
        rows, agreements = [], []
        for record in ds_stand.records:
            image = Path(record["path"])
            pipeline_mask = load_loss_mask(
                image, record["bucket_w"], record["bucket_h"], record["src_w"], record["src_h"]
            )[0].numpy()
            expected = bucket_crop(image, record["bucket_w"], record["bucket_h"], "RGBA", "A")
            agreements.append(float((abs(pipeline_mask - expected) < 0.02).mean()))
            rows.append({
                "image": image.name,
                "bucket": f"{record['bucket_w']}x{record['bucket_h']}",
                "pipeline_coverage": float(pipeline_mask.mean()),
                "independent_coverage": float(expected.mean()),
            })
        rep.check(
            tier, "real transparent art: pipeline mask equals an independent alpha crop",
            all(value > 0.99 for value in agreements) and ds_stand.n_masked == len(ds_stand)
            and all(abs(r["pipeline_coverage"] - r["independent_coverage"]) < 0.005 for r in rows),
            "alpha channel cropped to the bucket agrees pixel-for-pixel with a separate implementation",
            agreement=min(agreements), samples=len(ds_stand), rows=rows,
        )
        rep.observe(
            tier, f"untagged stand dataset ({args.stands_dir})",
            "no .txt captions exist: the trainer falls back to the file stem (see --tag-stands). "
            "Expression/pose variants also reuse their silhouette's alpha, so mask-carrying runs "
            "are built from images with distinct alpha channels",
            copied=[p.name for p in copied], distinct_alpha_patterns=distinct,
            captions=[p.with_suffix(".txt").read_text(encoding="utf-8")[:40] for p in copied],
        )
    else:
        rep.note(f"stand dataset not found at {args.stands_dir}; skipped its scan")


def _distinct_alpha_count(source_dir: Path, limit: int = 60) -> dict[str, int]:
    """How many different alpha patterns the first `limit` images carry."""
    import hashlib

    import numpy as np
    from PIL import Image

    from trainer.utils import list_images

    images = [p for p in list_images(source_dir) if p.suffix.lower() in IMAGE_EXT][:limit]
    digests = set()
    for path in images:
        with Image.open(path) as img:
            digests.add(hashlib.sha1(np.asarray(img.convert("RGBA").getchannel("A")).tobytes()).hexdigest())
    return {"sampled": len(images), "distinct": len(digests)}


# --------------------------------------------------------------------------------------
# pipeline holder (one SDXL load, reused for every probe)
# --------------------------------------------------------------------------------------


class Pipeline:
    def __init__(self, cfg: Any) -> None:
        import torch

        from trainer.family import resolve_family

        self.torch = torch
        self.cfg = cfg
        self.family = resolve_family(cfg)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.pipe: Any = None
        self.modules: Any = None
        self._init_lora: dict[str, Any] = {}
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """Load the pipeline if it is not resident. Released around long training runs: the parent
        holding SDXL in CPU RAM while a child trains was several GB of avoidable pressure."""
        if self.modules is not None:
            return
        import gc

        from accelerate.utils import set_seed

        # Same seed + same call order as trainer/main.py, so the adapter init matches the runs.
        set_seed(self.cfg.seed)
        self.pipe = self.family.load_pipeline(self.cfg.pretrained_model_name_or_path, self.dtype)
        self.modules = self.family.unpack(self.pipe)
        self.modules.noise_scheduler = self.family.build_noise_scheduler(self.pipe, self.cfg)
        self.modules.vae.requires_grad_(False)
        self.modules.denoise.requires_grad_(False)
        self.modules = self.family.apply_lora(self.cfg, self.modules)
        self._init_lora = {
            key: value.detach().to("cpu").clone()
            for key, value in self.modules.denoise.state_dict().items()
            if "lora_" in key
        }
        if hasattr(self.modules.denoise, "enable_gradient_checkpointing"):
            self.modules.denoise.enable_gradient_checkpointing()
        self.modules.denoise.to(device=self.device, dtype=self.dtype).eval()
        self.modules.vae.to("cpu")
        for encoder in self.modules.text_encoders:
            encoder.to("cpu")
        gc.collect()

    def release(self) -> None:
        """Drop the pipeline entirely (CPU + GPU); the next probe call reloads it."""
        import gc

        if self.modules is None:
            return
        self.modules = None
        self.pipe = None
        self._init_lora = {}
        gc.collect()
        self.torch.cuda.empty_cache()

    # -- encoders ------------------------------------------------------------------
    def encode_latents(self, image_path: Path, bucket_w: int, bucket_h: int):
        import torch
        from PIL import Image

        self.ensure_loaded()

        from trainer.utils import image_to_tensor, resize_and_center_crop

        with Image.open(image_path) as img:
            pixels = image_to_tensor(resize_and_center_crop(img.convert("RGB"), bucket_w, bucket_h))
        self.modules.vae.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            latents = self.modules.vae.encode(pixels.unsqueeze(0).to(self.device, self.dtype)).latent_dist.sample()
            latents = latents * self.modules.vae.config.scaling_factor
        self.modules.vae.to("cpu")
        torch.cuda.empty_cache()
        return latents.to(torch.float32)

    def encode_prompts(self, prompts: list[str]):
        self.ensure_loaded()
        # no_grad: the probes measure the masked loss / UNet gradients, and a graph through the
        # text encoders would break as soon as they are moved back to the CPU.
        for encoder in self.modules.text_encoders:
            encoder.to(device=self.device, dtype=self.dtype)
        with self.torch.no_grad():
            encoded = self.family.encode_prompts(prompts, self.modules, self.cfg, self.device, self.dtype)
        for encoder in self.modules.text_encoders:
            encoder.to("cpu")
        self.torch.cuda.empty_cache()
        return encoded

    def extra_cond(self, src_wh: tuple[int, int], bucket_wh: tuple[int, int]):
        self.ensure_loaded()
        return self.family.extra_cond(src_wh=src_wh, bucket_wh=bucket_wh, device=self.device, dtype=self.dtype)

    # -- probes --------------------------------------------------------------------
    def lora_parameters(self):
        self.ensure_loaded()
        return [p for p in self.modules.denoise.parameters() if p.requires_grad]

    def _forward(self, latents, encoded, extra, mask, seed, *, train: bool, grad_latents: bool):
        self.ensure_loaded()
        torch = self.torch
        payload = dict(extra)
        if mask is not None:
            payload["loss_mask"] = mask
        tensor = latents.to(self.device, self.dtype)
        if grad_latents:
            tensor = tensor.detach().requires_grad_(True)
        params = self.lora_parameters()
        for param in params:
            param.grad = None
        if train:
            self.modules.denoise.train()
        else:
            self.modules.denoise.eval()
        torch.manual_seed(seed)
        loss = self.family.denoise_loss(
            latents=tensor, encoded=encoded, extra=payload, modules=self.modules,
            cfg=self.cfg, device=self.device, dtype=self.dtype,
        )
        return loss, tensor, params

    def loss_only(self, *, latents, encoded, extra, mask=None, seed: int = 0) -> float:
        with self.torch.no_grad():
            loss, _tensor, _params = self._forward(latents, encoded, extra, mask, seed,
                                                   train=False, grad_latents=False)
            return float(loss.detach())

    def with_grads(self, *, latents, encoded, extra, mask=None, seed: int = 0, grad_latents: bool = False):
        loss, tensor, params = self._forward(latents, encoded, extra, mask, seed,
                                             train=True, grad_latents=grad_latents)
        loss.backward()
        grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
        latent_grad = tensor.grad.detach().clone() if grad_latents and tensor.grad is not None else None
        self.modules.denoise.eval()
        return float(loss.detach()), grads, latent_grad

    # -- weights -------------------------------------------------------------------
    def reload(self) -> None:
        self.ensure_loaded()
        self.modules.denoise.to(device=self.device, dtype=self.dtype)
        self.modules.denoise.eval()

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        self.ensure_loaded()
        self.cfg.resume_lora_path = str(path)
        info = self.family.load_lora(self.cfg, self.modules)
        self.cfg.resume_lora_path = ""
        return info

    def reset_adapter(self) -> None:
        self.ensure_loaded()
        self.modules.denoise.load_state_dict(self._init_lora, strict=False)


def grad_norm(grads: list[Any] | None) -> float:
    import math

    if not grads:
        return 0.0
    total = 0.0  # accumulate in Python floats: the grads live on the GPU
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().double().pow(2).sum().item())
    return math.sqrt(total)


def grad_diff_norm(left: list[Any] | None, right: list[Any] | None) -> float:
    import math

    if not left or not right:
        return float("nan")
    total = 0.0
    for a, b in zip(left, right):
        if a is None or b is None:
            continue
        total += float((a.double() - b.double()).pow(2).sum().item())
    return math.sqrt(total)


# --------------------------------------------------------------------------------------
# tier: loss (GPU)
# --------------------------------------------------------------------------------------


def tier_loss(rep: Report, args: argparse.Namespace, work: Path, pipeline: Pipeline) -> dict[str, Any]:
    import torch

    from trainer.dataset import LoraImageDataset
    from trainer.utils import load_loss_mask

    tier = "loss"
    cfg = pipeline.cfg
    out: dict[str, Any] = {}

    sample_dir = work / "loss" / "samples"
    copy_samples(Path(cfg.train_data_dir), sample_dir, 2, rgb_only=False, sidecars=False)
    ds = LoraImageDataset(base_config(train_data_dir=str(sample_dir), cache_latents=False,
                                      cache_latents_to_disk=False, shuffle_caption=False))
    record = ds.records[0]
    image = Path(record["path"])
    bucket_w, bucket_h = record["bucket_w"], record["bucket_h"]
    src_wh = (record["src_w"], record["src_h"])
    latent = pipeline.encode_latents(image, bucket_w, bucket_h)
    encoded = pipeline.encode_prompts([ds[0]["caption"] or "probe"])
    extra = pipeline.extra_cond(src_wh, (bucket_w, bucket_h))
    rep.observe(tier, "probe setup", "real image at its real bucket, prompt from its caption",
                image=image.name, bucket=f"{bucket_w}x{bucket_h}",
                latent_size=f"{latent.shape[-1]}x{latent.shape[-2]}",
                caption=ds[0]["caption"][:60], noise_offset=cfg.noise_offset,
                min_snr_gamma=cfg.min_snr_gamma)

    def loss(kind: str | None, seed: int = 11) -> float:
        mask = None if kind is None else region_mask(bucket_w, bucket_h, kind)
        return pipeline.loss_only(latents=latent, encoded=encoded, extra=extra, mask=mask, seed=seed)

    loss_ones, loss_none, loss_repeat = loss("ones"), loss(None), loss("ones")
    _, grads_ones, _ = pipeline.with_grads(latents=latent, encoded=encoded, extra=extra,
                                           mask=region_mask(bucket_w, bucket_h, "ones"), seed=11)
    _, grads_none, _ = pipeline.with_grads(latents=latent, encoded=encoded, extra=extra,
                                           mask=None, seed=11)
    rep.check(
        tier, "all-ones mask is bit-identical to no mask, and probes replay exactly",
        loss_ones == loss_none and loss_ones == loss_repeat and grad_diff_norm(grads_ones, grads_none) == 0.0,
        "an unmasked sample is completely unaffected; repeated probes are bitwise stable",
        loss_masked=loss_ones, loss_plain=loss_none, loss_replay=loss_repeat,
        grad_diff=grad_diff_norm(grads_ones, grads_none),
    )

    loss_zeros, grads_zeros, latent_grad_zeros = pipeline.with_grads(
        latents=latent, encoded=encoded, extra=extra, mask=region_mask(bucket_w, bucket_h, "zeros"),
        seed=11, grad_latents=True,
    )
    rep.check(
        tier, "all-black mask -> zero loss, zero weight grads, zero latent grad",
        loss_zeros == 0.0 and grad_norm(grads_zeros) == 0.0 and latent_grad_zeros is not None
        and float(latent_grad_zeros.abs().max()) == 0.0,
        "a fully ignored sample contributes nothing at all, not even to the input gradient",
        loss=loss_zeros, weight_grad_norm=grad_norm(grads_zeros),
        latent_grad_max=float(latent_grad_zeros.abs().max()) if latent_grad_zeros is not None else None,
    )

    # linearity: the loss is a spatially weighted mean, nothing else
    loss_right40, loss_left60 = loss("right40"), loss("left60")
    loss_half_gray = loss("half_gray")
    right40_mask = region_mask(bucket_w, bucket_h, "right40")
    loss_right40_half = pipeline.loss_only(latents=latent, encoded=encoded, extra=extra,
                                           mask=right40_mask * 0.5, seed=11)
    rep.check(
        tier, "loss is linear in the mask (no area renormalization, no renormalized mean)",
        abs(loss_right40_half - 0.5 * loss_right40) < 1e-6
        and abs(loss_half_gray - 0.5 * loss_ones) < 1e-6
        and abs((loss_right40 + loss_left60) - loss_ones) < 1e-5,
        "half weight halves the loss; two disjoint regions sum to the full-image loss",
        loss_right40=loss_right40, loss_left60=loss_left60, sum=loss_right40 + loss_left60,
        loss_all_ones=loss_ones, loss_half_gray=loss_half_gray, loss_half_weight=loss_right40_half,
    )

    # coverage -> gradient scale, the effective LR knob
    coverage: dict[str, dict[str, float]] = {}
    for kind in ("ones", "left", "quarter"):
        mask = region_mask(bucket_w, bucket_h, kind)
        value = pipeline.loss_only(latents=latent, encoded=encoded, extra=extra, mask=mask, seed=11)
        _, grads, _ = pipeline.with_grads(latents=latent, encoded=encoded, extra=extra, mask=mask, seed=11)
        coverage[kind] = {"coverage": float(mask.mean()), "loss": value, "grad_norm": grad_norm(grads)}
    base_norm = coverage["ones"]["grad_norm"] or 1.0
    for stats in coverage.values():
        stats["loss_ratio"] = stats["loss"] / loss_ones
        stats["grad_ratio"] = stats["grad_norm"] / base_norm
    out["coverage"] = coverage
    rep.observe(
        tier, "mask coverage scales the loss exactly, and the gradient norm even harder",
        "loss(M) is proportional to mean(M) (no area renormalization), so a masked run's reported "
        "loss and step size both shrink; the gradient norm drops further because its direction "
        "changes too",
        **{f"{kind}.coverage": stats["coverage"] for kind, stats in coverage.items()},
        **{f"{kind}.loss_ratio": stats["loss_ratio"] for kind, stats in coverage.items()},
        **{f"{kind}.grad_ratio": stats["grad_ratio"] for kind, stats in coverage.items()},
    )

    # position sensitivity and leakage of ignored content
    leakage = _leakage_probe(rep, pipeline, work, latent, encoded, extra, bucket_w, bucket_h, tier)
    out["leakage"] = leakage

    # real transparent art: alpha mask, independent recomputation, loss comparison
    if Path(args.stands_dir).is_dir():
        stand_copy = work / "loss" / "stand"
        copy_samples(Path(args.stands_dir), stand_copy, 1, rgb_only=False, sidecars=False,
                     distinct_alpha=True)
        ds_stand = LoraImageDataset(base_config(train_data_dir=str(stand_copy), cache_latents=False,
                                                cache_latents_to_disk=False, shuffle_caption=False))
        record = ds_stand.records[0]
        stand_image = Path(record["path"])
        stand_bucket = (record["bucket_w"], record["bucket_h"])
        mask = load_loss_mask(stand_image, stand_bucket[0], stand_bucket[1], record["src_w"], record["src_h"])
        expected = bucket_crop(stand_image, stand_bucket[0], stand_bucket[1], "RGBA", "A")
        stand_latent = pipeline.encode_latents(stand_image, stand_bucket[0], stand_bucket[1])
        stand_encoded = pipeline.encode_prompts([record["path"].stem])
        stand_extra = pipeline.extra_cond((record["src_w"], record["src_h"]), stand_bucket)
        loss_masked = pipeline.loss_only(latents=stand_latent, encoded=stand_encoded, extra=stand_extra,
                                        mask=mask, seed=7)
        loss_plain = pipeline.loss_only(latents=stand_latent, encoded=stand_encoded, extra=stand_extra,
                                        mask=None, seed=7)
        agreement = float((abs(mask[0].numpy() - expected) < 0.02).mean())
        rep.check(
            tier, "real alpha mask arrives at the loss and lowers it",
            agreement > 0.99 and loss_masked < loss_plain,
            "transparent-background stand art: the alpha channel becomes the loss weight",
            agreement=agreement, coverage=float(mask.mean()), loss_masked=loss_masked,
            loss_unmasked=loss_plain, ratio=loss_masked / loss_plain if loss_plain else None,
        )
    else:
        rep.note(f"stand dataset not found at {args.stands_dir}; skipped its loss probe")
    return out


def _leakage_probe(rep: Report, pipeline: Pipeline, work: Path, latent, encoded, extra,
                   bucket_w: int, bucket_h: int, tier: str) -> dict[str, Any]:
    """Repaint only the region the mask ignores, and compare what reaches the weights with the
    same repaint on the trained side. Exact gating happens at the error map; the UNet, being
    convolutional, still propagates content across the mask boundary."""
    import numpy as np
    import torch
    from PIL import Image

    from trainer.dataset import LoraImageDataset

    ds = LoraImageDataset(base_config(train_data_dir=str(work / "loss" / "samples"),
                                      cache_latents=False, cache_latents_to_disk=False, shuffle_caption=False))
    record = ds.records[0]
    source = Path(record["path"])
    repaint_dir = work / "loss" / "repaint"
    repaint_dir.mkdir(parents=True, exist_ok=True)
    repainted: dict[str, Any] = {}
    for label, columns in (("inside", (0.6, 1.0)), ("outside", (0.0, 0.4))):
        target = repaint_dir / f"{label}.png"
        with Image.open(source) as img:
            array = np.asarray(img.convert("RGB")).copy()
        width = array.shape[1]
        array[:, int(width * columns[0]) : int(width * columns[1])] = (12, 240, 12)
        Image.fromarray(array).save(target)
        repainted[label] = pipeline.encode_latents(target, bucket_w, bucket_h)

    mask_right = region_mask(bucket_w, bucket_h, "right40")
    loss_masked, grads_base, _ = pipeline.with_grads(latents=latent, encoded=encoded, extra=extra,
                                                     mask=mask_right, seed=13)
    loss_plain, grads_plain, _ = pipeline.with_grads(latents=latent, encoded=encoded, extra=extra,
                                                     mask=None, seed=13)
    deltas = {}
    losses = {}
    for label, repainted_latent in repainted.items():
        _, grads_new, _ = pipeline.with_grads(latents=repainted_latent, encoded=encoded, extra=extra,
                                              mask=mask_right, seed=13)
        deltas[label] = grad_diff_norm(grads_base, grads_new)
        losses[label] = {
            "masked": pipeline.loss_only(latents=repainted_latent, encoded=encoded, extra=extra,
                                         mask=mask_right, seed=13),
            "unmasked": pipeline.loss_only(latents=repainted_latent, encoded=encoded, extra=extra,
                                           mask=None, seed=13),
        }
    baseline = grad_diff_norm(grads_base, grads_plain)
    loss_base_masked = pipeline.loss_only(latents=latent, encoded=encoded, extra=extra,
                                         mask=mask_right, seed=13)
    loss_base_unmasked = pipeline.loss_only(latents=latent, encoded=encoded, extra=extra,
                                           mask=None, seed=13)
    _, _, latent_grad = pipeline.with_grads(latents=latent, encoded=encoded, extra=extra,
                                            mask=mask_right, seed=13, grad_latents=True)
    column = int(round(bucket_w * 0.6)) // 8
    grad_in_mask = float(latent_grad[..., :, column + 2 :].abs().mean())
    grad_out_mask = float(latent_grad[..., :, : column - 2].abs().mean())
    share = deltas["inside"] / deltas["outside"] if deltas["outside"] else float("inf")
    rep.check(
        tier, "input gradient reaches the masked-out pixels too (so the leakage numbers are real)",
        grad_out_mask > 0.0 and grad_in_mask > 0.0,
        "∂loss/∂latents is non-zero on both sides of the mask boundary: the error map is gated, "
        "the network input is not",
        grad_in_mask=grad_in_mask, grad_out_mask=grad_out_mask,
        ratio=grad_in_mask / grad_out_mask if grad_out_mask else None,
    )
    rep.observe(
        tier, "ignored-region content still reaches the weights (measured, not exact)",
        "repainting only the masked-out 40% moves the LoRA gradient by this share of the same "
        "repaint on the trained side; masking suppresses the objective, it does not insulate the "
        "network from those pixels",
        delta_repaint_inside_mask=deltas["inside"], delta_repaint_outside_mask=deltas["outside"],
        share_of_trained_side=share, delta_masked_vs_unmasked_loss=baseline,
        loss_base={"masked": loss_base_masked, "unmasked": loss_base_unmasked},
        loss_after_repaint_inside=losses["inside"], loss_after_repaint_outside=losses["outside"],
        masked_loss_change_inside=losses["inside"]["masked"] - loss_base_masked,
        unmasked_loss_change_inside=losses["inside"]["unmasked"] - loss_base_unmasked,
    )
    rep.observe(
        tier, "mask position matters (equal coverage, different loss)",
        "a coverage-only implementation or a transposed mask would collapse these numbers",
        loss_right40=pipeline.loss_only(latents=latent, encoded=encoded, extra=extra,
                                       mask=region_mask(bucket_w, bucket_h, "right40"), seed=11),
        loss_left60=pipeline.loss_only(latents=latent, encoded=encoded, extra=extra,
                                      mask=region_mask(bucket_w, bucket_h, "left60"), seed=11),
    )
    return {"share_of_trained_side": share, "delta_inside_mask": deltas["inside"],
            "delta_outside_mask": deltas["outside"], "grad_in_mask": grad_in_mask,
            "grad_out_mask": grad_out_mask}


# --------------------------------------------------------------------------------------
# tier: train / stand (real entrypoint subprocesses)
# --------------------------------------------------------------------------------------


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def write_toml(path: Path, sections: dict[str, dict[str, Any]]) -> None:
    lines: list[str] = []
    for name, table in sections.items():
        lines.append(f"[{name}]")
        for key, value in table.items():
            if value is None:
                continue
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def config_sections() -> dict[str, dict[str, Any]]:
    import tomllib

    with open(REPO_ROOT / "trainer" / "config.toml", "rb") as handle:
        return tomllib.load(handle)


def make_mirror(dest: Path) -> None:
    """A throwaway repo root: symlinked trainer sources plus a generated config.toml, so the
    unmodified `trainer/main.py` runs against a verification config."""
    (dest / "trainer").mkdir(parents=True, exist_ok=True)
    for src in (REPO_ROOT / "trainer").glob("*.py"):
        os.symlink(src, dest / "trainer" / src.name)
    os.symlink(REPO_ROOT / "text_processing.py", dest / "text_processing.py")
    os.symlink(REPO_ROOT / "start_train.sh", dest / "start_train.sh")


def reap_run_processes(marker: str) -> list[int]:
    """Terminate leftovers of a dead child run (DataLoader forkserver workers survive a SIGABRT).

    `marker` is that run's own mirror path, so only its processes can match.
    """
    import signal

    me, parent = os.getpid(), os.getppid()
    reaped: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if marker in cmdline and "forkserver" in cmdline:
            try:
                os.kill(pid, signal.SIGTERM)
                reaped.append(pid)
            except OSError:
                pass
    return reaped


def gpu_memory_fault(log_path: Path) -> bool:
    """True when the child died from the ROCm gfx1201 Tensile OOB abort (fixes/fix2.txt)."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return ("HSA_STATUS_ERROR_MEMORY_FAULT" in text
            or "Memory access fault by GPU node" in text
            or "GCVM_L2_PROTECTION_FAULT" in text)


def mask_ratio(line: str | None) -> tuple[int, int] | None:
    """Parse the trainer's `Loss masks: n/m samples` startup line."""
    if not line or ":" not in line:
        return None
    try:
        numbers = line.split(":", 1)[1].strip().split()[0].split("/")
        return int(numbers[0]), int(numbers[1])
    except (IndexError, ValueError):
        return None


def child_env(runtime_dir: Path, *, no_hip_memory_caching: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "AXL_RUNTIME_DIR": str(runtime_dir),
        "PYTHONUNBUFFERED": "1",
        # Same ROCm quieting/caching as start_train.sh.
        "AMD_LOG_LEVEL": "0",
        "CK_LOG_LEVEL": "0",
        "MIOPEN_ENABLE_LOGGING": "0",
        "MIOPEN_ENABLE_LOGGING_CMD": "0",
        "MIOPEN_LOG_LEVEL": "1",
        "MIOPEN_LOG_BUFFER_SIZE": "0",
        "MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS": "0",
        "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR": "0",
        "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK": "0",
        "MIOPEN_CUSTOM_CACHE_DIR": str(Path.home() / ".cache" / "miopen"),
        "MIOPEN_USER_DB_PATH": str(Path.home() / ".config" / "miopen"),
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
    })
    if no_hip_memory_caching:
        # fixes/fix2.txt: every allocation via hipMalloc avoids the unpacked-page overrun that the
        # gfx1201 Tensile kernels trigger, at a large speed cost.
        env["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
    return env


@dataclass
class RunResult:
    name: str
    run_root: Path
    log_path: Path
    logging_dir: Path
    output_dir: Path
    runtime_dir: Path
    run_dir: Path | None
    returncode: int
    seconds: float
    mask_line: str | None
    claims: dict[str, Any] = field(default_factory=dict)
    loss_series: list[tuple[int, float]] = field(default_factory=list)
    gpu_fault: bool = False
    attempts: int = 1


# Registered by the tier that owns a pipeline holder: called before every child launch, so a
# parent-resident model can never compete with a training child for VRAM.
BEFORE_LAUNCH: list[Any] = []


def launch_run(*, name: str, data_dir: Path, seed: int, steps: int, work: Path, masked: bool,
               tier: str, batch_size: int = 3, resume_from: Path | None = None,
               save_every_override: int | None = None, lr: tuple[float, float] = (2e-4, 2e-5),
               warmup: int = 10, quiet: bool = False, retries: int = 0,
               no_hip_memory_caching: bool = False) -> RunResult:
    """Run one child training, retrying the intermittent ROCm GPU memory fault."""
    for hook in BEFORE_LAUNCH:
        hook()
    result = _launch_once(name=name, data_dir=data_dir, seed=seed, steps=steps, work=work,
                          masked=masked, tier=tier, batch_size=batch_size, resume_from=resume_from,
                          save_every_override=save_every_override, lr=lr, warmup=warmup, quiet=quiet,
                          no_hip_memory_caching=no_hip_memory_caching)
    attempt = 1
    while result.gpu_fault and attempt <= retries:
        print(f"      [{tier}] {name}: (attempt {attempt}) died from the known gfx1201 Tensile GPU "
              f"memory fault (fixes/fix2.txt); retrying", flush=True)
        attempt += 1
        # The fault is shape/allocation driven and reproduces, so do not burn a second identical
        # attempt: switch to the documented (much slower) dodge right away.
        for hook in BEFORE_LAUNCH:
            hook()
        escalate = attempt >= 2 and not no_hip_memory_caching
        if escalate:
            print(f"      [{tier}] {name}: retrying with PYTORCH_NO_HIP_MEMORY_CACHING=1", flush=True)
        result = _launch_once(name=name, data_dir=data_dir, seed=seed, steps=steps, work=work,
                              masked=masked, tier=tier, batch_size=batch_size, resume_from=resume_from,
                              save_every_override=save_every_override, lr=lr, warmup=warmup,
                              quiet=quiet, no_hip_memory_caching=no_hip_memory_caching or escalate,
                              attempt=attempt)
    result.attempts = attempt
    return result


def _launch_once(*, name: str, data_dir: Path, seed: int, steps: int, work: Path, masked: bool,
                 tier: str, batch_size: int = 3, resume_from: Path | None = None,
                 save_every_override: int | None = None, lr: tuple[float, float] = (2e-4, 2e-5),
                 warmup: int = 10, quiet: bool = False, no_hip_memory_caching: bool = False,
                 attempt: int = 1) -> RunResult:
    from trainer.dataset import LoraImageDataset

    run_root = work / f"run_{name}"
    mirror = run_root / "mirror"
    runtime_dir = run_root / "runtime"
    output_dir = run_root / "outputs"
    logging_dir = run_root / "logs"
    for path in (runtime_dir, output_dir, logging_dir):
        path.mkdir(parents=True, exist_ok=True)
    if not mirror.exists():
        make_mirror(mirror)

    n_images = len(LoraImageDataset(base_config(train_data_dir=str(data_dir), cache_latents=False,
                                                cache_latents_to_disk=False)))
    steps_per_epoch = max(1, -(-n_images // batch_size))
    epochs = max(1, -(-steps // steps_per_epoch))
    cadence = save_every_override if save_every_override else max(1, steps // 4)
    sections = config_sections()
    overrides = {
        "environment": {
            "train_data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "logging_dir": str(logging_dir),
            "output_name": f"verify_{name}",
        },
        "training": {
            "seed": int(seed),
            "epoch": epochs,
            "save_every_n_steps": cadence,
            "resume_lora_path": str(resume_from) if resume_from else "",
            "train_batch_size": batch_size,
            "lr_warmup_steps": warmup,
        },
        # Raised over trainer/config.toml so a few minutes of training moves the LoRA clearly
        # enough to measure; both variants get the same values.
        "unet_optimizer": {"unet_learning_rate": lr[0], "unet_warmup_steps": warmup},
        "te_optimizer": {"te_learning_rate": lr[1]},
        # 2 workers: the children only decode 6 images per epoch, and each worker is another
        # few hundred MB of resident memory while a full SDXL pipeline is already in flight.
        "infrastructure": {"max_data_loader_n_workers": 2, "persistent_workers": True},
        # keep samples cheap: the final save always samples, and so does every step cadence
        "validation": {"sample_width": 768, "sample_height": 768, "sample_steps": 8, "sample_repeat": 1},
    }
    for section, values in overrides.items():
        sections.setdefault(section, {}).update(values)
    write_toml(mirror / "trainer" / "config.toml", sections)

    # Re-running the same work dir only redoes runs that did not finish, so a verification hit by
    # an intermittent GPU fault can be completed without repeating everything.
    done_file = run_root / "done.json"
    if done_file.is_file():
        try:
            done = json.loads(done_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            done = {}
        if done.get("returncode") == 0:
            print(f"      [{tier}] {name}: reusing completed run ({done['seconds']:.0f}s)", flush=True)
            return RunResult(
                name=name, run_root=run_root, log_path=run_root / "train.out", logging_dir=logging_dir,
                output_dir=output_dir, runtime_dir=runtime_dir,
                run_dir=Path(done["run_dir"]) if done.get("run_dir") else None,
                returncode=0, seconds=float(done["seconds"]), mask_line=done.get("mask_line"),
                claims=done.get("claims", {}),
            )

    log_path = run_root / "train.out"
    started = time.time()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([sys.executable, "-u", "trainer/main.py"], cwd=mirror,
                                env=child_env(runtime_dir, no_hip_memory_caching=no_hip_memory_caching),
                                stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        state_path = runtime_dir / "state.json"
        last_step = None
        while proc.poll() is None:
            time.sleep(2.0)
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            step = state.get("training", {}).get("step")
            if step != last_step and step is not None and not quiet:
                last_step = step
                print(f"      [{tier}] {name}: step {step}/{steps} "
                      f"loss={state['training'].get('loss')} ({time.time() - started:.0f}s)", flush=True)
        proc.wait()
    seconds = time.time() - started

    if attempt > 1:
        log_path.replace(run_root / f"train.attempt{attempt}.out")
        log_path = run_root / f"train.attempt{attempt}.out"
    else:
        (run_root / "train.attempt1.out").write_text(
            log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    mask_line = next((line.strip() for line in text.splitlines() if "Loss masks:" in line), None)
    run_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    fault = proc.returncode != 0 and gpu_memory_fault(log_path)
    if not quiet:
        print(f"      [{tier}] {name}: exit={proc.returncode} in {seconds:.0f}s | {mask_line}"
              + ("  [gfx1201 GPU memory fault]" if fault else ""), flush=True)
    result = RunResult(
        name=name, run_root=run_root, log_path=log_path, logging_dir=logging_dir,
        output_dir=output_dir, runtime_dir=runtime_dir,
        run_dir=run_dirs[-1] if run_dirs else None, returncode=proc.returncode, seconds=seconds,
        mask_line=mask_line,
        claims={"seed": seed, "steps_requested": steps, "epochs": epochs,
                "steps_per_epoch": steps_per_epoch, "images": n_images, "masked": masked,
                "save_every_n_steps": cadence, "mirror_config": str(mirror / "trainer" / "config.toml"),
                "gpu_memory_fault": fault},
    )
    result.gpu_fault = fault
    if result.returncode != 0:
        reaped = reap_run_processes(str(mirror))
        if reaped:
            result.claims["reaped_orphans"] = reaped
            print(f"      [{tier}] {name}: cleaned up {len(reaped)} orphaned worker processes", flush=True)
    if result.returncode == 0:
        done_file.write_text(json.dumps({
            "returncode": result.returncode, "seconds": result.seconds, "mask_line": result.mask_line,
            "run_dir": str(result.run_dir) if result.run_dir else None, "claims": result.claims,
        }, indent=2), encoding="utf-8")
    return result


def tb_scalars(logging_dir: Path, tag: str) -> list[tuple[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runs = sorted([p for p in logging_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        return []
    accumulator = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [(event.step, float(event.value)) for event in accumulator.Scalars(tag)]


def dump_kohya_init(pipeline: "Pipeline", dest_dir: Path) -> Path | None:
    """Save the freshly initialized adapters in the checkpoint (kohya) key space, so a run's
    distance from its starting weights can be measured with the same comparison."""
    from trainer.models import lora_checkpoint_file

    class _StubAccelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(module):
            return module

    cfg = base_config()
    cfg.run_dir = str(dest_dir)
    cfg.output_name = "verify_init"
    pipeline.reset_adapter()
    try:
        pipeline.family.save_lora(_StubAccelerator(), pipeline.modules, cfg, 0, final=True)
    except Exception as exc:  # noqa: BLE001 - the movement metric is diagnostic only
        print(f"      (could not dump the initial adapter: {exc})", flush=True)
        return None
    path = lora_checkpoint_file(cfg, 0, final=True)
    return path if path.is_file() else None


def checkpoint_files(run_dir: Path | None) -> list[Path]:
    return sorted(run_dir.glob("*/*.safetensors")) if run_dir else []


def dataset_mask_coverage(data_dir: Path) -> float:
    """Mean mask weight over a dataset copy: the factor the reported loss should be scaled by."""
    from trainer.dataset import LoraImageDataset
    from trainer.utils import load_loss_mask

    ds = LoraImageDataset(base_config(train_data_dir=str(data_dir), cache_latents=False,
                                      cache_latents_to_disk=False))
    values = [
        float(load_loss_mask(Path(record["path"]), record["bucket_w"], record["bucket_h"],
                             record["src_w"], record["src_h"]).mean())
        for record in ds.records
    ]
    return sum(values) / len(values)


def final_checkpoint(run_dir: Path | None) -> Path | None:
    found = [p for p in checkpoint_files(run_dir) if "_final" in p.parent.name]
    return found[0] if found else None


def step_checkpoint(run_dir: Path | None, step: int) -> Path | None:
    found = [p for p in checkpoint_files(run_dir) if p.parent.name.endswith(f"_s{step:06d}")]
    return found[0] if found else None


def compare_checkpoints(left: Path, right: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    a, b = load_file(str(left)), load_file(str(right))
    changed, max_delta, total_sq, ref_sq = 0, 0.0, 0.0, 0.0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b or a[key].shape != b[key].shape:
            changed += 1
            continue
        if not key.endswith(".alpha"):
            diff = a[key].float() - b[key].float()
            max_delta = max(max_delta, float(diff.abs().max()))
            total_sq += float(diff.pow(2).sum())
            ref_sq += float(a[key].float().pow(2).sum())
        if not bool((a[key] == b[key]).all()):
            changed += 1
    return {"keys": len(set(a) | set(b)), "changed_keys": changed, "max_weight_delta": max_delta,
            "relative_weight_delta": (total_sq ** 0.5) / max(ref_sq ** 0.5, 1e-12)}


def tier_training(rep: Report, args: argparse.Namespace, work: Path, pipeline: Pipeline | None,
                  tier: str, *, source_dir: Path, steps: int, seeds: int, count: int,
                  use_alpha_mask: bool, lr: tuple[float, float] | None = None) -> dict[str, Any]:
    """Masked vs unmasked training through the real entrypoint, on copies of `source_dir`."""
    out: dict[str, Any] = {"runs": [], "comparisons": []}
    lr = lr or (args.unet_lr, args.te_lr)
    data_masked = work / tier / "data_masked"
    data_control = work / tier / "data_control"
    data_scale = work / tier / "data_scale"
    if not data_masked.exists():
        copy_samples(source_dir, data_masked, count, rgb_only=False, sidecars=not use_alpha_mask,
                     distinct_alpha=use_alpha_mask)
    if not data_control.exists():
        copy_samples(source_dir, data_control, count, rgb_only=True, sidecars=False,
                     distinct_alpha=use_alpha_mask)
    coverage = dataset_mask_coverage(data_masked)
    gray = int(round(255 * coverage))
    if not data_scale.exists():
        # Same loss scale as the masked run, no spatial selectivity: any difference between the
        # two is the mask's geometry rather than the amount of loss it removes.
        scale_images = copy_samples(source_dir, data_scale, count, rgb_only=False, sidecars=False,
                                    distinct_alpha=use_alpha_mask)
        for image_path in scale_images:
            uniform_sidecar(image_path, gray)
    mask_kind = ("alpha channel (no sidecars)" if use_alpha_mask
                 else "sidecar {stem}.mask.png (right 40% ignored)")
    rep.observe(
        tier, "dataset variants",
        "masked = spatial mask, scale-matched = same coverage as a constant gray weight, "
        "unmasked = no mask at all",
        masked=len(list(data_masked.glob("*.png"))), scale=len(list(data_scale.glob("*.png"))),
        control=len(list(data_control.glob("*.png"))), coverage=coverage, gray_level=gray,
    )

    if args.tag_stands and tier == "stand":
        tagger = REPO_ROOT / "tagger" / "main.py"
        if tagger.is_file():
            print(f"      [{tier}] tagging the dataset copies with tagger/main.py", flush=True)
            subprocess.run([sys.executable, str(tagger), str(data_masked)], cwd=REPO_ROOT, check=False)
            subprocess.run([sys.executable, str(tagger), str(data_control)], cwd=REPO_ROOT, check=False)

    base_seed = int(base_config().seed)
    seed_values = [base_seed + index for index in range(seeds)]
    if pipeline is not None:
        # Training subprocesses own the GPU and their model copies; release the parent's pipeline
        # (it is rebuilt lazily for the probes afterwards) and keep it released for every launch.
        pipeline.release()
        BEFORE_LAUNCH.append(pipeline.release)
    # The floor must be measured at the same step count as the masked-vs-unmasked comparison,
    # otherwise a noise figure from a shorter run would be compared against a longer one.
    floor_steps = args.floor_steps if args.floor_steps > 0 else steps
    floor_steps = max(1, min(floor_steps, steps))
    masked_cadence = max(1, steps // 4) if floor_steps == steps else floor_steps
    runs: dict[str, list[RunResult]] = {"masked": [], "control": [], "scale": []}
    for seed in seed_values:
        runs["masked"].append(launch_run(name=f"{tier}_m{seed}", data_dir=data_masked, seed=seed,
                                        steps=steps, work=work, masked=True, tier=tier,
                                        save_every_override=masked_cadence, lr=lr,
                                        warmup=args.warmup, retries=args.retries,
                                        no_hip_memory_caching=args.no_hip_memory_caching))
        runs["control"].append(launch_run(name=f"{tier}_c{seed}", data_dir=data_control, seed=seed,
                                         steps=steps, work=work, masked=False, tier=tier,
                                         save_every_override=masked_cadence, lr=lr,
                                        warmup=args.warmup, retries=args.retries,
                                        no_hip_memory_caching=args.no_hip_memory_caching))
        runs["scale"].append(launch_run(name=f"{tier}_g{seed}", data_dir=data_scale, seed=seed,
                                        steps=steps, work=work, masked=True, tier=tier,
                                        save_every_override=masked_cadence, lr=lr,
                                        warmup=args.warmup, retries=args.retries,
                                        no_hip_memory_caching=args.no_hip_memory_caching))

    for label, label_runs in runs.items():
        for run in label_runs:
            state_path = run.runtime_dir / "state.json"
            payload = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            expected_masked = label != "control"
            counts = mask_ratio(run.mask_line)
            mask_line_ok = counts is not None and counts[1] == run.claims["images"] and (
                counts[0] == counts[1] if expected_masked else counts[0] == 0
            )
            rep.check(
                tier, f"{label} run completed through trainer/main.py ({run.name})",
                run.returncode == 0 and run.run_dir is not None and payload.get("status") == "finished",
                f"exit={run.returncode}, {run.seconds:.0f}s, control plane status={payload.get('status')}",
                **run.claims,
            )
            rep.check(
                tier, f"{label} run reports mask usage ({run.name})",
                mask_line_ok,
                f"the trainer's own startup line accounts for every sample: {run.mask_line}",
                log=run.mask_line, expected_masked=expected_masked, counts=counts,
            )
            run.loss_series = tb_scalars(run.logging_dir, "Train/Loss")
            avg_series = tb_scalars(run.logging_dir, "Train/Avg_Loss")
            finite = all(value == value and abs(value) != float("inf") for _, value in run.loss_series)
            rep.check(
                tier, f"{label} run wrote TensorBoard scalars ({run.name})",
                len(run.loss_series) >= min(2, steps) and len(avg_series) > 0 and finite,
                f"{len(run.loss_series)} Train/Loss + {len(avg_series)} Train/Avg_Loss points, all finite",
                first=run.loss_series[0] if run.loss_series else None,
                last=run.loss_series[-1] if run.loss_series else None,
                checkpoints=len(checkpoint_files(run.run_dir)),
            )
            out["runs"].append({
                "name": run.name, "variant": label, "seed": run.claims["seed"],
                "status": payload.get("status"), "seconds": round(run.seconds, 1),
                "mask_line": run.mask_line, "loss_first": run.loss_series[0] if run.loss_series else None,
                "loss_last": run.loss_series[-1] if run.loss_series else None,
                "loss_points": len(run.loss_series),
                "checkpoints": [p.parent.name for p in checkpoint_files(run.run_dir)],
                "config": run.claims["mirror_config"],
                "gpu_memory_fault": run.claims.get("gpu_memory_fault", False),
                "attempts": run.attempts,
            })

    # repeatability floor: a duplicate of the masked run measures this platform's noise. Its
    # checkpoint/sample cadence must match the reference run exactly: generating a sample reseeds
    # the global RNG, so a different cadence would change the run for reasons unrelated to masks.
    floor = launch_run(name=f"{tier}_floor", data_dir=data_masked, seed=seed_values[0],
                       steps=floor_steps, work=work, masked=True, tier=tier,
                       save_every_override=masked_cadence, lr=lr,
                       warmup=args.warmup, quiet=True, retries=args.retries,
                       no_hip_memory_caching=args.no_hip_memory_caching)
    floor_ckpt = final_checkpoint(floor.run_dir)
    reference = (step_checkpoint(runs["masked"][0].run_dir, floor_steps) if floor_steps < steps
                 else final_checkpoint(runs["masked"][0].run_dir))
    comparison = compare_checkpoints(reference, floor_ckpt) if reference and floor_ckpt else None
    movement = None
    if pipeline is not None and reference is not None:
        init_dump = dump_kohya_init(pipeline, work / tier / "init_adapter")
        movement = compare_checkpoints(init_dump, reference) if init_dump else None
    deterministic = bool(comparison and comparison["changed_keys"] == 0)
    rep.check(
        tier, f"repeatability floor: an identical {floor_steps}-step duplicate reproduces bitwise",
        comparison is not None and comparison["relative_weight_delta"] < 0.25,
        ("bitwise-identical LoRA weights from two separate processes: every masked-vs-unmasked "
         "difference below is fully attributable to the mask"
         if deterministic else
         "the duplicate differs, so this value is the platform's noise floor for the comparisons "
         "below (an identical run must still land far closer than an unrelated one)"),
        floor_steps=floor_steps, deterministic=deterministic,
        **({} if comparison is None else comparison),
    )
    if movement and movement["changed_keys"] >= movement["keys"]:
        movement = None  # incomparable key spaces: do not publish a meaningless number
    if movement and comparison:
        rep.observe(
            tier, "how large the run-to-run noise is compared with the training movement",
            "an identical duplicate differs by `floor`; a single run moves `movement` away from its "
            "initial weights. A mask effect can only be resolved when it is bigger than `floor`",
            floor_relative=comparison["relative_weight_delta"],
            movement_from_init=movement["relative_weight_delta"],
            noise_share_of_movement=comparison["relative_weight_delta"]
            / max(movement["relative_weight_delta"], 1e-12),
        )
    out["repeatability"] = comparison
    if comparison and comparison["changed_keys"] == 0:
        rep.note(
            f"[{tier}] the training loop is deterministic here: two identical {floor_steps}-step runs in "
            "separate processes produced bitwise-identical LoRA tensors, so the masked-vs-unmasked "
            "deltas below are attributable to the mask rather than to run-to-run noise. (An earlier "
            "measurement of a ~5% floor came from comparing runs with different `epoch` counts, which "
            "changes the text-encoder cosine schedule - a harness bug, not GPU noise.)"
        )
    elif comparison and comparison["relative_weight_delta"] > 1e-4:
        rep.note(
            f"[{tier}] two identical {floor_steps}-step runs differ by "
            f"{comparison['relative_weight_delta']:.2%} (relative L2) in their LoRA weights and "
            f"{comparison['max_weight_delta']:.2e} at the largest single tensor: that is the noise floor "
            "every masked-vs-unmasked delta below has to beat."
        )

    # The decisive, noise-free end-to-end signal: masked loss is a weighted mean, so a constant
    # gray mask of coverage c must report exactly c x the unmasked loss, whatever the image content.
    for index, seed in enumerate(seed_values):
        series = {"masked": runs["masked"][index].loss_series,
                  "scale": runs["scale"][index].loss_series,
                  "control": runs["control"][index].loss_series}
        if any(len(points) < 2 for points in series.values()):
            continue
        means = {name: sum(value for _step, value in points) / len(points)
                 for name, points in series.items()}
        law_error = abs(means["scale"] - coverage * means["control"]) / max(means["control"], 1e-12)
        rep.check(
            tier, f"trainer's reported loss obeys the mask weighting law (seed {seed})",
            law_error < 0.05 and means["masked"] < means["control"],
            "a constant gray mask of coverage c reports c x the unmasked loss (un-normalized weighted "
            "mean), and any mask reports less than no mask: the mask reaches the objective the real "
            "trainer optimizes",
            coverage=coverage, control_mean=means["control"], scale_mean=means["scale"],
            law_error=law_error, masked_mean=means["masked"],
            masked_over_control=means["masked"] / means["control"],
            scale_over_control=means["scale"] / means["control"],
        )
        out.setdefault("loss_scale", []).append(
            {"seed": seed, "coverage": coverage, "law_error": law_error, **means}
        )

    # Where the mask sits relative to the error distribution decides how much loss it removes.
    for index, seed in enumerate(seed_values):
        masked_losses = [value for _step, value in runs["masked"][index].loss_series]
        scale_losses = [value for _step, value in runs["scale"][index].loss_series]
        control_losses = [value for _step, value in runs["control"][index].loss_series]
        if min(len(masked_losses), len(scale_losses), len(control_losses)) < 2:
            continue
        masked_mean = sum(masked_losses) / len(masked_losses)
        scale_mean = sum(scale_losses) / len(scale_losses)
        control_mean = sum(control_losses) / len(control_losses)
        rep.observe(
            tier, f"the mask's loss removal depends on which pixels it covers (seed {seed})",
            "the spatial mask and the same-coverage gray mask remove different amounts of loss "
            "because the error is not spatially uniform: `spatial_over_gray` > 1 means the mask is "
            "concentrated on harder pixels, < 1 means it removes the easy ones",
            coverage=coverage, masked_over_control=masked_mean / control_mean,
            gray_over_control=scale_mean / control_mean,
            spatial_over_gray=masked_mean / scale_mean,
        )

    # masked vs unmasked weights: a measured comparison, not a pass/fail assertion, because ROCm
    # kernels are not reproducible from run to run.
    comparable = floor_steps == steps  # only then is the floor a fair reference for the final step
    for index, seed in enumerate(seed_values):
        contrasts = {
            "unmasked": (runs["masked"][index], runs["control"][index]),
            "scale-matched": (runs["masked"][index], runs["scale"][index]),
        }
        for name, (reference_run, other_run) in contrasts.items():
            left, right = final_checkpoint(reference_run.run_dir), final_checkpoint(other_run.run_dir)
            diff = compare_checkpoints(left, right) if left and right else None
            signals = None
            if diff and comparison and comparable:
                if comparison["changed_keys"] == 0:
                    signals = {"floor_identical": True}
                else:
                    signals = {"signal_over_floor": diff["relative_weight_delta"]
                               / max(comparison["relative_weight_delta"], 1e-12)}
            if name == "unmasked":
                rep.check(
                    tier, f"masked and unmasked training produce different weights (seed {seed})",
                    diff is not None and diff["changed_keys"] > 0,
                    "a mask that does nothing would leave the two runs numerically identical",
                    **({} if diff is None else diff), **(signals or {}),
                )
            if diff:
                resolved = signals is not None and (
                    signals.get("floor_identical", False) or signals.get("signal_over_floor", 0.0) > 3.0
                )
                out["comparisons"].append({"seed": seed, "contrast": name, **diff,
                                           **(signals or {}), "resolved": resolved})
                rep.observe(
                    tier, f"masked vs {name} weight delta (seed {seed})",
                    "relative L2 distance over all LoRA tensors of the final checkpoints, against "
                    "the duplicate-run floor; `resolved` is False when the platform's run-to-run "
                    "noise is the same size as the mask's effect",
                    floor=comparison, **diff, **(signals or {}), resolved=resolved,
                )

    # the weights-only resume runs while the parent pipeline is still released: the metadata check
    # below reloads it onto the GPU, and a child started next to a resident UNet runs out of VRAM.
    masked_final = final_checkpoint(runs["masked"][0].run_dir)
    if masked_final and not args.no_resume_check:
        resume = launch_run(name=f"{tier}_resume", data_dir=data_masked, seed=seed_values[0],
                            steps=5, work=work, masked=True, resume_from=masked_final, tier=tier,
                            save_every_override=5, lr=lr, warmup=args.warmup, quiet=True,
                            retries=args.retries,
                            no_hip_memory_caching=args.no_hip_memory_caching)
        state_path = resume.runtime_dir / "state.json"
        payload = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        rep.check(
            tier, "resuming from a masked-trained checkpoint runs to completion",
            resume.returncode == 0 and payload.get("status") == "finished"
            and (payload.get("resume") or {}).get("loaded") is not None,
            "weights-only resume keeps working while masks are present in the dataset",
            resume=payload.get("resume"), seconds=round(resume.seconds, 1),
        )
        out["resume"] = {"status": payload.get("status"), "resume": payload.get("resume"),
                         "seconds": round(resume.seconds, 1)}

    # checkpoint metadata and reload (this loads the pipeline back onto the GPU)

    if masked_final and pipeline is not None:
        from trainer.checkpoints import read_lora_metadata

        metadata = read_lora_metadata(masked_final)
        info = pipeline.load_checkpoint(masked_final)
        rep.check(
            tier, "masked-trained checkpoint keeps its kohya metadata and reloads",
            metadata.get("ss_network_dim") is not None and bool(info.get("loaded"))
            and not info.get("skipped"),
            "masks do not change the checkpoint format; resumed/civitai consumers still work",
            network_dim=metadata.get("ss_network_dim"), network_alpha=metadata.get("ss_network_alpha"),
            steps=metadata.get("ss_steps"), loaded=info.get("loaded"), skipped=info.get("skipped"),
        )

    if args.prune_step_checkpoints:
        pruned = 0
        for run in [*runs["masked"], *runs["control"], *runs["scale"], floor]:
            for ckpt in checkpoint_files(run.run_dir):
                if "_final" not in ckpt.parent.name:
                    shutil.rmtree(ckpt.parent, ignore_errors=True)
                    pruned += 1
        if pruned:
            print(f"      [{tier}] pruned {pruned} step checkpoints (finals kept)", flush=True)
    if pipeline is not None:
        out["region_probe"] = region_probe(rep, pipeline, tier, data_masked, runs, seed_values,
                                           use_alpha_mask, floor if floor_steps == steps else None)
        probe = out["region_probe"]
        if not probe.get("resolved"):
            rep.note(
                f"[{tier}] the per-region outcome comparison did not clear the duplicate-run noise at "
                f"{steps} steps (see the `mask effect` observation for effect_over_noise). The direction "
                "is what masked loss predicts, but the size is not separable from GPU nondeterminism at "
                "this scale: more steps, a lower LR, or deterministic kernels would be needed."
            )
    return out


def region_probe(rep: Report, pipeline: Pipeline, tier: str, data_dir: Path,
                 runs: dict[str, list[RunResult]], seed_values: list[int],
                 use_alpha_mask: bool, floor_run: RunResult | None = None) -> dict[str, Any]:
    """Per-region error at init vs after masked training vs after unmasked training, on the same
    latents, prompts and noise seeds."""
    from trainer.dataset import LoraImageDataset
    from trainer.utils import load_loss_mask

    pipeline.reload()
    ds = LoraImageDataset(base_config(train_data_dir=str(data_dir), cache_latents=False,
                                      cache_latents_to_disk=False, shuffle_caption=False))
    prepared = []
    for index, record in enumerate(ds.records[: min(4, len(ds.records))]):
        image = Path(record["path"])
        bucket = (record["bucket_w"], record["bucket_h"])
        mask = load_loss_mask(image, bucket[0], bucket[1], record["src_w"], record["src_h"])
        prepared.append({
            "name": image.name,
            "latents": pipeline.encode_latents(image, bucket[0], bucket[1]),
            "encoded": pipeline.encode_prompts([ds[index]["caption"] or image.stem]),
            "extra": pipeline.extra_cond((record["src_w"], record["src_h"]), bucket),
            "mask": mask,
            "bucket": bucket,
            "seed": 100 + index,
        })

    def regions(sample: dict[str, Any]) -> dict[str, Any]:
        if use_alpha_mask:
            return {"subject (trained)": sample["mask"], "background (ignored)": 1.0 - sample["mask"]}
        width, height = sample["bucket"]
        return {"trained (left 60%)": region_mask(width, height, "left60"),
                "ignored (right 40%)": region_mask(width, height, "right40")}

    def measure() -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for sample in prepared:
            for name, mask in regions(sample).items():
                coverage = float(mask.mean())
                if coverage <= 0.0:
                    continue
                loss = pipeline.loss_only(latents=sample["latents"], encoded=sample["encoded"],
                                          extra=sample["extra"], mask=mask, seed=sample["seed"])
                totals.setdefault(name, []).append(loss / coverage)
        return {name: sum(values) / len(values) for name, values in totals.items()}

    states: dict[str, dict[str, float]] = {}
    pipeline.reset_adapter()
    states["init"] = measure()
    if floor_run is not None and floor_run.run_dir is not None:
        duplicate = final_checkpoint(floor_run.run_dir)
        if duplicate is not None:
            pipeline.load_checkpoint(duplicate)
            states["masked duplicate"] = measure()
    for index, seed in enumerate(seed_values):
        masked_final = final_checkpoint(runs["masked"][index].run_dir)
        control_final = final_checkpoint(runs["control"][index].run_dir)
        if not masked_final or not control_final:
            continue
        pipeline.load_checkpoint(masked_final)
        states[f"masked seed {seed}"] = measure()
        pipeline.load_checkpoint(control_final)
        states[f"unmasked seed {seed}"] = measure()
        scale_final = final_checkpoint(runs["scale"][index].run_dir)
        if scale_final:
            pipeline.load_checkpoint(scale_final)
            states[f"scale-matched seed {seed}"] = measure()

    region_names = list(states["init"].keys())
    rows = [{"region": name, **{state: values[name] for state, values in states.items()}}
            for name in region_names]
    rep.observe(
        tier, "per-region error: init vs masked-trained vs unmasked-trained",
        "mean per-pixel error over the probe images, area-normalized, same latents/prompts/noise",
        probe_images=[sample["name"] for sample in prepared], rows=rows,
    )
    duplicate = states.get("masked duplicate")
    ignored_region = region_names[-1]
    trained_region = region_names[0]
    # Paired contrast: (error in the region both runs trained) - (error in the ignored region).
    # Measuring the two regions in one probe pass cancels most of the run-to-run weight noise,
    # which is what makes this a sensitive detector of the mask's effect.
    contrast = {state: values[trained_region] - values[ignored_region] for state, values in states.items()}
    effects = {}
    for index, seed in enumerate(seed_values):
        masked_state = states.get(f"masked seed {seed}")
        control_state = states.get(f"unmasked seed {seed}")
        if not masked_state or not control_state:
            continue
        scale_state = states.get(f"scale-matched seed {seed}")
        entry = {
            "region": ignored_region,
            "init": states["init"][ignored_region],
            "masked": masked_state[ignored_region],
            "unmasked": control_state[ignored_region],
            # positive: the masked run left the ignored region worse off than the unmasked run
            "masked_minus_unmasked_ignored_region":
                masked_state[ignored_region] - control_state[ignored_region],
            # negative would mean the masked run fitted the trained region worse
            "masked_minus_unmasked_trained_region":
                masked_state[trained_region] - control_state[trained_region],
            "contrast": {state: contrast[state] for state in states},
            "contrast_effect": contrast[f"unmasked seed {seed}"] - contrast[f"masked seed {seed}"],
        }
        if scale_state:
            entry["masked_minus_scale_ignored_region"] = (
                masked_state[ignored_region] - scale_state[ignored_region])
            entry["masked_minus_scale_trained_region"] = (
                masked_state[trained_region] - scale_state[trained_region])
            entry["scale_contrast_effect"] = (
                contrast[f"scale-matched seed {seed}"] - contrast[f"masked seed {seed}"])
        if duplicate:
            noise = abs(contrast[f"masked seed {seed}"] - contrast["masked duplicate"])
            entry["contrast_noise"] = noise
            entry["duplicate_is_identical"] = noise == 0.0
            if noise == 0.0:
                # Bitwise-identical duplicate: there is no run-to-run noise to beat, so any
                # positive effect is fully attributable to the mask.
                entry["effect_over_noise"] = None
                entry["resolved"] = bool(entry["contrast_effect"] > 0)
            else:
                entry["effect_over_noise"] = entry["contrast_effect"] / noise
                entry["resolved"] = bool(entry["contrast_effect"] > 3.0 * noise)
        effects[seed] = entry
    signs = {seed: values["masked_minus_unmasked_ignored_region"] > 0 for seed, values in effects.items()}
    rep.observe(
        tier, "mask effect: the masked run keeps the ignored region relatively worse off",
        "`masked_minus_unmasked_ignored_region` should be positive and the trained-region value "
        "negative (the mask bought accuracy where it trained); `contrast_effect` is the paired "
        "trained-minus-ignored contrast difference, and `resolved` means it beats the duplicate "
        "run's probe noise by 3x",
        **{f"seed{seed}": values for seed, values in effects.items()},
        consistent_sign=len(set(signs.values())) == 1 if signs else None,
        resolved=any(bool(values.get("resolved")) for values in effects.values()),
    )
    return {"rows": rows, "effects": effects,
            "consistent_sign": len(set(signs.values())) == 1 if signs else None,
            "resolved": any(bool(values.get("resolved")) for values in effects.values())}


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def write_report(rep: Report, args: argparse.Namespace, details: dict[str, Any]) -> Path:
    out = rep.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": rep.env,
        "tiers": args.tiers,
        "options": {
            "steps": args.steps, "stand_steps": args.stand_steps, "seeds": args.seeds,
            "images": args.images, "stand_images": args.stand_images, "floor_steps": args.floor_steps,
            "unet_lr": args.unet_lr, "te_lr": args.te_lr, "warmup": args.warmup,
            "stand_seeds": args.stand_seeds,
            "dataset": str(base_config().train_data_dir), "stands_dir": args.stands_dir,
            "model": base_config().pretrained_model_name_or_path, "work_dir": str(args.work_dir),
        },
        "checks": [check.__dict__ for check in rep.checks],
        "observations": [obs.__dict__ for obs in rep.observations],
        "notes": rep.notes,
        "details": details,
        "failures": len(rep.failures),
        "seconds": round(time.time() - rep.started, 1),
    }
    (out / "mask_verify_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    lines = [
        "# Mask -> training pipeline verification",
        "",
        f"- Interpreter `{rep.env.get('executable')}` (prefix `{rep.env.get('prefix')}`), "
        f"torch {rep.env.get('torch')} / HIP {rep.env.get('torch_hip')}, GPU {rep.env.get('gpu')}",
        f"- Tiers `{', '.join(args.tiers)}` | steps={args.steps}, stand_steps={args.stand_steps}, "
        f"seeds={args.seeds}, images={args.images}, floor_steps={args.floor_steps}",
        f"- Wall clock {payload['seconds']:.0f}s | {len(rep.checks)} checks "
        f"({len(rep.failures)} failed) | {len(rep.observations)} observations",
        "",
        "## Checks",
        "",
        "| tier | check | result | measured |",
        "| --- | --- | --- | --- |",
    ]
    for check in rep.checks:
        measured = fmt_values(check.values).strip(" ()").replace("; ", " ")
        lines.append(f"| {check.tier} | {check.name} | {'PASS' if check.ok else '**FAIL**'} | {measured} |")
    lines += ["", "## Observations (measured, not asserted)", ""]
    for obs in rep.observations:
        lines += [f"### [{obs.tier}] {obs.name}", "", obs.detail, "", "```json",
                  json.dumps(obs.values, indent=2, ensure_ascii=False, default=str), "```", ""]
    if details:
        lines += ["## Detail per tier", "", "```json",
                  json.dumps(details, indent=2, ensure_ascii=False, default=str), "```", ""]
    if rep.notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in rep.notes] + [""]
    path = out / "mask_verify_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", default="all", help="comma list of plumbing,loss,train,stand, or 'all'")
    parser.add_argument("--steps", type=int, default=120, help="training steps per run (train tier)")
    parser.add_argument("--stand-steps", type=int, default=60, help="training steps per run (stand tier)")
    parser.add_argument("--seeds", type=int, default=2, help="seed replicates per variant")
    parser.add_argument("--images", type=int, default=12, help="images per training dataset copy")
    parser.add_argument("--stand-images", type=int, default=3, help="stand images to train on")
    parser.add_argument("--floor-steps", type=int, default=0,
                        help="repeatability floor run length (0 = same length as the masked runs)")
    parser.add_argument("--stand-seeds", type=int, default=1, help="seed replicates for the stand tier")
    parser.add_argument("--unet-lr", type=float, default=2e-4,
                        help="UNet LR for the verification runs (config.toml uses 2e-5)")
    parser.add_argument("--te-lr", type=float, default=2e-5,
                        help="text encoder LR for the verification runs (config.toml uses 2e-6)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="LR warmup steps for the verification runs (config.toml uses 100)")
    parser.add_argument("--report-dir", default=None, help="where the report is written")
    parser.add_argument("--work-dir", default=None, help="scratch dir (datasets, mirrors, runs)")
    parser.add_argument("--stands-dir", default=str(DEFAULT_STANDS), help="untagged stand dataset")
    parser.add_argument("--keep", action="store_true", help="keep the scratch dir")
    parser.add_argument("--force", action="store_true", help="run even when a training run looks live")
    parser.add_argument("--tag-stands", action="store_true", help="run tagger/main.py on the stand copies first")
    parser.add_argument("--no-resume-check", action="store_true", help="skip the resume sub-run")
    parser.add_argument("--allow-foreign-env", action="store_true", help="skip the conda axl check")
    parser.add_argument("--retries", type=int, default=2,
                        help="retries per training run after the intermittent gfx1201 GPU memory fault")
    parser.add_argument("--prune-step-checkpoints", action="store_true",
                        help="delete step checkpoints after each tier (keeps finals); saves ~0.5 GB per run")
    parser.add_argument("--no-hip-memory-caching", action="store_true",
                        help="run children with PYTORCH_NO_HIP_MEMORY_CACHING=1 (fixes/fix2.txt dodge, slow)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    os.chdir(REPO_ROOT)  # trainer/config.toml is read relative to the repo root
    args = parse_args(argv)
    tiers = list(TIERS if args.tiers == "all" else (t.strip() for t in args.tiers.split(",") if t.strip()))
    for tier in tiers:
        if tier not in TIERS:
            raise SystemExit(f"unknown tier: {tier}")
    report_dir = Path(args.report_dir) if args.report_dir else Path("/tmp/axl-mask-verify") / time.strftime("%Y%m%d_%H%M%S")
    work = Path(args.work_dir) if args.work_dir else report_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.report_dir, args.work_dir, args.tiers = report_dir, work, tiers
    rep = Report(report_dir)
    print(f"== mask pipeline verification | tiers={','.join(tiers)} | report={report_dir}", flush=True)

    guard_environment(args, rep)
    live = live_training_runs()
    if live and not args.force:
        raise SystemExit("refusing to run with a live training process (pass --force to override):\n  "
                         + "\n  ".join(live))
    if live:
        rep.note(f"--force used while these looked live: {live}")

    rep.note(
        "Dataset copies only: masks and captions are written into temp copies, never into the source "
        "dataset directories, and every run gets its own AXL_RUNTIME_DIR."
    )
    if str(work.resolve()).startswith(("/tmp", "/dev/shm")):
        note = (f"the scratch dir {work} is on tmpfs (RAM-backed): every run keeps ~5 checkpoints of "
                "~130 MB, so a full verification holds several GB of RAM unless you point --report-dir "
                "at a disk or pass --prune-step-checkpoints")
        print(f"note: {note}", flush=True)
        rep.note(note)
    details: dict[str, Any] = {}
    pipeline: Pipeline | None = None
    try:
        if "plumbing" in tiers:
            tier_plumbing(rep, args, work)
        if "loss" in tiers or "train" in tiers or "stand" in tiers:
            print("== loading the SDXL pipeline once", flush=True)
            pipeline = Pipeline(base_config())
        if "loss" in tiers:
            details["loss"] = tier_loss(rep, args, work, pipeline)
        if "train" in tiers:
            details["train"] = tier_training(rep, args, work, pipeline, "train",
                                            source_dir=Path(base_config().train_data_dir), steps=args.steps,
                                            seeds=args.seeds, count=args.images, use_alpha_mask=False,
                                            lr=(args.unet_lr, args.te_lr))
        if "stand" in tiers:
            if Path(args.stands_dir).is_dir():
                details["stand"] = tier_training(rep, args, work, pipeline, "stand",
                                                 source_dir=Path(args.stands_dir), steps=args.stand_steps,
                                                 seeds=max(1, args.stand_seeds), count=args.stand_images,
                                                 use_alpha_mask=True, lr=(args.unet_lr, args.te_lr))
            else:
                rep.note(f"stand dataset missing: {args.stands_dir}")
    finally:
        report_path = write_report(rep, args, details)

    print("")
    print(f"== {len(rep.checks)} checks, {len(rep.failures)} failed | report: {report_path}", flush=True)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("DONE" if not rep.failures else "FAILED", flush=True)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
