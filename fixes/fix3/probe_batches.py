#!/usr/bin/env python3
"""CPU-only replay of the first batches of a run: bucket, caption chunk count, and the LoRA GEMM's M.

The step-1 abort in `fixes/fix3/README.md` happens on the very first backward, so the only dataset
properties that can influence it are the ones the *first* batch carries: its bucket (which sets the
UNet's spatial token count) and the number of CLIP chunks its captions need (which sets the encoder
sequence length, hence the row count `M` of the cross-attention LoRA GEMM). This script prints both
without touching the GPU, using the trainer's own `LoraImageDataset` and `BucketBatchSampler`.

    python probe_batches.py                          # the repo config, the real dataset
    python probe_batches.py --data-dir DIR --batch 2 --batches 4
    python probe_batches.py --buckets                # just the bucket histogram

The trainer resolves `config.toml` from the working directory, so this runs inside a throwaway
mirror (`/tmp/axl-fix3-probe`) that symlinks `trainer/` and holds the config under test. The dataset
itself is opened read-only: images are only measured, never copied or written, except that the
dataset constructor creates an empty `.latents_cache/` if the data directory has none.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "api.py").is_file() and (candidate / "trainer" / "main.py").is_file():
            return candidate
    raise SystemExit(f"could not find the repository root above {start}")


REPO_ROOT = find_repo_root(HERE)


def make_mirror(dest: Path, config: Path) -> None:
    """Symlinked trainer sources + the config under test, so `trainer/config.py` reads this file."""
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "trainer").mkdir(parents=True)
    for src in (REPO_ROOT / "trainer").glob("*.py"):
        os.symlink(src, dest / "trainer" / src.name)
    os.symlink(REPO_ROOT / "text_processing.py", dest / "text_processing.py")
    shutil.copy2(config, dest / "config.toml")


def chunk_count(text: str, tokenizer, max_token_length: int, chunk_size: int = 75) -> int:
    """`text_processing.tokenize_long_prompt`'s rule: 75-token chunks, capped by `max_token_length`."""
    token_ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
    token_ids = token_ids[:max_token_length]
    if not token_ids:
        return 1
    return min(max(1, math.ceil(len(token_ids) / chunk_size)),
               max(1, math.ceil(max_token_length / chunk_size)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.toml"))
    parser.add_argument("--data-dir", default=None, help="defaults to the config's train_data_dir")
    parser.add_argument("--batch", type=int, default=None, help="defaults to the config's setting")
    parser.add_argument("--batches", type=int, default=3, help="how many batches to print")
    parser.add_argument("--model", default=None, help="pipeline dir for the tokenizers")
    parser.add_argument("--json", default=None,
                        help="write the whole probe (bucket histogram, chunk counts, per-step "
                             "shapes) to this file — the step -> bucket map the scans are read against")
    parser.add_argument("--buckets", action="store_true", help="only the bucket histogram")
    parser.add_argument("--seed", type=int, default=None,
                        help="override training.seed; the seed sets the caption shuffle, hence the "
                             "CLIP chunk count, hence M")
    parser.add_argument("--names-per-bucket", type=int, default=0,
                        help="print that many image names per bucket, for building a dataset whose "
                             "every batch uses one bucket")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "axl-fix3-probe"))
    args = parser.parse_args()
    # Resolve before the chdir into the mirror, or a relative --json lands inside the mirror.
    if args.json:
        args.json = str(Path(args.json).resolve())

    mirror = Path(args.work) / "mirror"
    make_mirror(mirror, Path(args.config))
    os.chdir(mirror)
    sys.path.insert(0, str(mirror))

    from trainer.config import TrainConfig
    from trainer.dataset import BucketBatchSampler, LoraImageDataset

    cfg = TrainConfig()
    if args.data_dir:
        cfg.train_data_dir = args.data_dir
    if args.seed is not None:
        cfg.seed = args.seed
    batch_size = args.batch or cfg.train_batch_size

    import tomllib

    with open(args.config, "rb") as handle:
        toml = tomllib.load(handle)
    model_dir = args.model or toml["environment"]["pretrained_model_name_or_path"]

    dataset = LoraImageDataset(cfg)
    histogram = collections.Counter(
        (record["bucket_w"], record["bucket_h"]) for record in dataset.records
    )
    latent_histogram = collections.Counter(
        (record["bucket_w"] // 8, record["bucket_h"] // 8) for record in dataset.records
    )
    printed = {
        "config": str(args.config),
        "data_dir": str(cfg.train_data_dir),
        "images": len(dataset.records),
        "batch_size": batch_size,
        "seed": cfg.seed,
        "max_token_length": cfg.max_token_length,
        "bucket_reso_steps": cfg.bucket_reso_steps,
        "min_bucket_reso": cfg.min_bucket_reso,
        "max_bucket_reso": cfg.max_bucket_reso,
        "train_resolution": cfg.train_resolution,
        "masked_samples": dataset.n_masked,
        "buckets": {f"{w}x{h}": n for (w, h), n in sorted(histogram.items())},
        "latents": {f"{w}x{h}": n for (w, h), n in sorted(latent_histogram.items())},
        "n_buckets": len(histogram),
    }
    print(json.dumps(printed, indent=2), flush=True)
    if args.names_per_bucket:
        per_bucket: dict[str, list[str]] = {}
        for record in dataset.records:
            key = f"{record['bucket_w']}x{record['bucket_h']}"
            names = per_bucket.setdefault(key, [])
            if len(names) < args.names_per_bucket:
                names.append(Path(record["path"]).name)
        print(json.dumps({"names_per_bucket": per_bucket}, indent=2), flush=True)
    if args.buckets or args.names_per_bucket:
        return 0

    # Chunk counts need the real CLIP tokenizer: `model_max_length` is what fixes the 75-token chunk.
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    chunks_by_path: dict[str, int] = {}
    for record in dataset.records:
        path = record["path"]
        chunks_by_path[str(path)] = chunk_count(
            dataset._caption_for(path), tokenizer, cfg.max_token_length
        )
    chunk_histogram = collections.Counter(chunks_by_path.values())
    print(json.dumps({"caption_chunks": dict(sorted(chunk_histogram.items())),
                      "seq_lengths": {n * 77: c for n, c in sorted(chunk_histogram.items())}}),
          flush=True)

    sampler = BucketBatchSampler(dataset.buckets, batch_size, cfg.seed)
    sampler.set_epoch(args.epoch)
    print(f"\n--- epoch {args.epoch}: first {args.batches} batches (batch_size={batch_size}) ---")
    steps = []
    for index, indices in enumerate(sampler):
        if index >= args.batches:
            break
        records = [dataset.records[i] for i in indices]
        chunks = max(chunks_by_path[str(r["path"])] for r in records)
        rows = len(indices) * 77 * chunks
        bucket = (records[0]["bucket_w"], records[0]["bucket_h"])
        step = {
            "step": index,
            "bucket": f"{bucket[0]}x{bucket[1]}",
            "latent": f"{bucket[0] // 8}x{bucket[1] // 8}",
            "unet_tokens": (bucket[0] // 16) * (bucket[1] // 16),
            "images": [Path(r["path"]).name for r in records],
            "src_sizes": [f"{r['src_w']}x{r['src_h']}" for r in records],
            "chunks": [chunks_by_path[str(r["path"])] for r in records],
            "seq": 77 * chunks,
            "M_lora_gemm": rows,
            "M_partial_tile_64": rows % 64,
        }
        steps.append(step)
        print(json.dumps(step), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {**printed, "caption_chunks": dict(sorted(chunk_histogram.items())),
             "steps": steps}, indent=2), encoding="utf-8")
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
