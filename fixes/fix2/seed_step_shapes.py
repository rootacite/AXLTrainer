#!/usr/bin/env python3
"""Why does `training.seed` decide whether the run survives? Walk the shapes it controls.

The seed never touches a kernel's shape; what it decides is *which images are paired in which order*
and *how each caption is shuffled*, and both of those set the size of every tensor in the step:

  * `BucketBatchSampler` shuffles with `random.Random(cfg.seed + epoch)` -> the order of
    (bucket, batch) pairs, so the aspect ratio behind step N changes with the seed;
  * `_caption_for` shuffles each caption with `random.Random(cfg.seed + epoch + sha1(path))` ->
    the token stream, i.e. how many CLIP chunks the prompt needs (chunk size = 74 tokens of content),
    so the text-encoder sequence length — and therefore the LoRA GEMM's `M = batch * chunks * 77`.

Both are pure functions of the seed, so the whole per-step shape trace is computable on the CPU
with the trainer's own code. This script prints the trace for each seed and where two seeds diverge.

    python seed_step_shapes.py                                   # 1145141920 vs 1145141919
    python seed_step_shapes.py --seeds 1145141920,1145141921 --steps 120
    python seed_step_shapes.py --dataset <dir> --batch-size 2     # the fix2.txt combination
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
REPO_ROOT = next(parent for parent in (HERE, *HERE.parents)
                 if (parent / "trainer" / "config.toml").is_file())


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return str(value)


def build_config(work: Path, *, dataset: Path, batch_size: int | None, steps: int):
    """Materialise the packaged config in `work` and load it through the trainer's own class."""
    import tomllib

    with open(RESOURCES / "original-config.toml", "rb") as handle:
        sections = tomllib.load(handle)
    environment = sections["environment"]
    environment["train_data_dir"] = str(dataset)
    environment["output_dir"] = str(work / "outputs")
    environment["logging_dir"] = str(work / "logs")
    if not Path(environment["pretrained_model_name_or_path"]).is_dir():
        raise SystemExit(f"{environment['pretrained_model_name_or_path']} is missing; "
                         f"edit resources/original-config.toml or pass --model")

    training = sections["training"]
    if batch_size:
        training["train_batch_size"] = int(batch_size)
    # `training.epoch` is left as packaged: 60 epochs x 2 batches = the 120 steps the run takes.

    lines: list[str] = []
    for name, table in sections.items():
        lines.append(f"[{name}]")
        for key, value in table.items():
            if value is None:
                continue
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    # `trainer/config.py` reads the relative path `trainer/config.toml`, so the work dir has to
    # look like a repo root.
    (work / "trainer").mkdir(parents=True, exist_ok=True)
    (work / "trainer" / "config.toml").write_text("\n".join(lines), encoding="utf-8")

    sys.path.insert(0, str(REPO_ROOT / "trainer"))
    previous = os.getcwd()
    os.chdir(work)  # `_load_toml_config` resolves `trainer/config.toml` against the cwd
    try:
        from config import TrainConfig  # noqa: PLC0415  (import needs the cwd above)

        return TrainConfig()
    finally:
        os.chdir(previous)


def trace(cfg, steps: int) -> list[dict]:
    """Per-step (bucket, batch size, chunk counts, M) using the trainer's dataset/sampler."""
    sys.path.insert(0, str(REPO_ROOT))

    from transformers import CLIPTokenizer

    from dataset import BucketBatchSampler, LoraImageDataset  # noqa: PLC0415
    from text_processing import tokenize_long_prompt  # noqa: PLC0415
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
    dataset = LoraImageDataset(cfg)
    sampler = BucketBatchSampler(dataset.buckets, batch_size=cfg.train_batch_size, seed=cfg.seed)

    rows: list[dict] = []
    step = 0
    for epoch in range(cfg.epoch):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for batch in sampler:
            captions = [dataset._caption_for(dataset.images[index]) for index in batch]
            chunks = [tokenize_long_prompt(caption, tokenizer, cfg.max_token_length)[1]
                      for caption in captions]
            record = dataset.records[batch[0]]
            step += 1
            rows.append({
                "step": step, "epoch": epoch,
                "bucket": f"{record['bucket_w']}x{record['bucket_h']}",
                "batch": len(batch), "chunks": chunks,
                "max_chunks": max(chunks), "M": len(batch) * max(chunks) * 77,
            })
            if step >= steps:
                return rows
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1145141920,1145141919")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dataset", default=None,
                        help="training images; default: the packaged six (override paths with care)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=str(RESOURCES / "seed-shapes.json"))
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    work = Path(tempfile.mkdtemp(prefix="axl-seed-shapes-"))
    try:
        dataset = Path(args.dataset) if args.dataset else RESOURCES / "dataset"
        cfg = build_config(work, dataset=dataset, batch_size=args.batch_size, steps=args.steps)
        if args.model:
            cfg.pretrained_model_name_or_path = args.model
        print(f"== {dataset} | batch {cfg.train_batch_size} | max_token_length "
              f"{cfg.max_token_length} | {args.steps} steps | dtype bf16")

        traces: dict[str, list[dict]] = {}
        for seed in seeds:
            cfg.seed = int(seed)
            traces[str(seed)] = trace(cfg, args.steps)

        for seed in seeds:
            rows = traces[str(seed)]
            histogram = Counter(row["M"] for row in rows)
            switches = sum(1 for a, b in zip(rows, rows[1:]) if a["M"] != b["M"])
            print(f"\n-- seed {seed}: {len(rows)} steps, M histogram "
                  f"{dict(sorted(histogram.items()))}, {switches} shape switches")
            print("   step: " + " ".join(f"{row['step']}:{row['M']}" for row in rows))
            around = [row for row in rows if 95 <= row["step"] <= 105]
            for row in around:
                print(f"   step {row['step']:>3}  {row['bucket']:>9}  batch {row['batch']}  "
                      f"chunks {row['chunks']}  M {row['M']}")

        first, second = (traces[str(seed)] for seed in seeds[:2])
        for field in ("bucket", "M"):
            step = next((a["step"] for a, b in zip(first, second) if a[field] != b[field]), None)
            print(f"\n== first step where {field!r} differs between seeds {seeds[0]} and "
                  f"{seeds[1]}: {step}")
        for step in (100, 101, 102):
            left = next((row for row in first if row["step"] == step), None)
            right = next((row for row in second if row["step"] == step), None)
            print(f"   step {step:>3}: {seeds[0]} -> bucket {left['bucket']}, M {left['M']}"
                  f"   |   {seeds[1]} -> bucket {right['bucket']}, M {right['M']}")

        Path(args.out).write_text(json.dumps({"seeds": seeds, "traces": traces}, indent=2),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
        return 0
    finally:
        if os.environ.get("AXL_KEEP_SEED_WORK") != "1":
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
