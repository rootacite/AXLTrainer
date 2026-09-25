"""`[[environment.train_data]]`: one dataset folder per entry, each drawn `repeat` times per epoch.

Two layers are covered here:

1. `resolve_train_data_entries` (torch-free): blocks vs. the flat `train_data_dir` scalar, and the
   value errors a hand-edited config gets, each naming the offending entry.
2. `LoraImageDataset` + `BucketBatchSampler` + `build_dataloader`: the repeat reaches training as a
   repeated index in the bucket list, so the epoch length, the batch count and the LR-schedule
   step count all follow the sampler - the train loop itself is not involved.
"""

import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

# `python test/test_train_data_repeat.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import (
    TRAIN_DATA_REPEAT_RANGE,
    TrainConfig,
    TrainDataEntry,
    resolve_train_data_entries,
)
from trainer.dataset import BucketBatchSampler, LoraImageDataset, collate_fn
from trainer.loop import group_indices_by_bucket
from trainer.setup import build_dataloader


def _write_images(root: Path, count: int, stem: str = "img") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (512, 512), (i * 20, 40, 80)).save(root / f"{stem}_{i:03d}.png")
        (root / f"{stem}_{i:03d}.txt").write_text("tag_a, tag_b", encoding="utf-8")


def _cfg(folders: list[tuple[Path, int]], **overrides) -> TrainConfig:
    cfg = TrainConfig()
    cfg.train_data = [{"path": str(path), "repeat": repeat} for path, repeat in folders]
    cfg.train_data_dir = str(folders[0][0])
    cfg.enable_bucket = False
    cfg.train_resolution = 512
    cfg.cache_latents = True
    cfg.cache_latents_to_disk = True
    cfg.train_batch_size = 2
    cfg.max_data_loader_n_workers = 0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class ResolveTrainDataEntriesTest(unittest.TestCase):
    def test_no_blocks_falls_back_to_the_scalar(self):
        self.assertEqual(
            resolve_train_data_entries({"train_data_dir": "/data/one"}),
            [TrainDataEntry(path="/data/one", repeat=1)],
        )

    def test_blocks_carry_their_own_repeat(self):
        entries = resolve_train_data_entries(
            {
                "train_data_dir": "/first",
                "train_data": [
                    {"path": "/first/", "repeat": 3},
                    {"path": "/second"},
                ],
            }
        )
        self.assertEqual(
            entries,
            [TrainDataEntry(path="/first/", repeat=3), TrainDataEntry(path="/second", repeat=1)],
        )

    def test_the_scalar_is_ignored_while_blocks_exist(self):
        # The form keeps the scalar mirroring the first block, but a hand-edited file that left
        # the two out of step trains what the blocks say.
        entries = resolve_train_data_entries(
            {"train_data_dir": "/stale", "train_data": [{"path": "/real", "repeat": 2}]}
        )
        self.assertEqual(entries, [TrainDataEntry(path="/real", repeat=2)])

    def test_missing_or_blank_path_names_the_entry(self):
        for entry in ({"repeat": 2}, {"path": "   "}):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError) as ctx:
                    resolve_train_data_entries({"train_data": [{"path": "/ok"}, entry]})
                self.assertIn("train_data[2]: path must not be empty", str(ctx.exception))

    def test_repeat_outside_the_range_names_the_entry(self):
        low, high = TRAIN_DATA_REPEAT_RANGE
        for value in (0, high + 1):
            with self.subTest(repeat=value):
                with self.assertRaises(ValueError) as ctx:
                    resolve_train_data_entries({"train_data": [{"path": "/a", "repeat": value}]})
                self.assertIn(
                    f"train_data[1]: repeat must be between {low} and {high}", str(ctx.exception)
                )
        for value in ("abc", 1.5, True):
            with self.subTest(repeat=value):
                with self.assertRaises(ValueError) as ctx:
                    resolve_train_data_entries({"train_data": [{"path": "/a", "repeat": value}]})
                self.assertIn("train_data[1]: expected an integer", str(ctx.exception))

    def test_repeat_range_bounds_are_accepted(self):
        low, high = TRAIN_DATA_REPEAT_RANGE
        entries = resolve_train_data_entries(
            {"train_data": [{"path": "/a", "repeat": low}, {"path": "/b", "repeat": high}]}
        )
        self.assertEqual([entry.repeat for entry in entries], [low, high])

    def test_train_data_must_be_an_array_of_tables(self):
        for value in ("/one", {"path": "/one"}):
            with self.subTest(train_data=value):
                with self.assertRaises(ValueError) as ctx:
                    resolve_train_data_entries({"train_data_dir": "/x", "train_data": value})
                self.assertIn("must be an array of tables", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            resolve_train_data_entries({"train_data_dir": "/x", "train_data": ["/one"]})
        self.assertIn("train_data[1] must be a table", str(ctx.exception))

    def test_blocks_survive_the_flat_toml_load(self):
        """The blocks sit under `[environment]`, so the flattening loader hands them over as `train_data`."""
        from trainer.config import _load_toml_config

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text(
                '[environment]\ntrain_data_dir = "/fallback"\noutput_name = "x"\n\n'
                "[[environment.train_data]]\n"
                'path = "/a"\nrepeat = 3\n\n'
                "[[environment.train_data]]\n"
                'path = "/b"\n\n'
                '[model_spec]\nbase_model_version = "sdxl_base_v1-0"\n',
                encoding="utf-8",
            )
            flat = _load_toml_config(str(path))

        self.assertEqual(flat["train_data_dir"], "/fallback")
        entries = resolve_train_data_entries(flat)
        self.assertEqual([entry.path for entry in entries], ["/a", "/b"])
        self.assertEqual([entry.repeat for entry in entries], [3, 1])

    def test_no_folder_at_all_is_refused(self):
        for cfg in ({"train_data_dir": ""}, {"train_data_dir": "   "}, {}):
            with self.subTest(cfg=cfg):
                if cfg == {}:
                    # An empty mapping still sees the repo's own config.toml through `get_val`.
                    self.assertTrue(resolve_train_data_entries(cfg))
                    continue
                with self.assertRaises(ValueError) as ctx:
                    resolve_train_data_entries(cfg)
                self.assertIn("train_data_dir is blank", str(ctx.exception))


class DatasetRepeatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.first = root / "first"
        self.second = root / "second"
        _write_images(self.first, 3)
        _write_images(self.second, 2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_len_is_unique_images_and_total_samples_counts_repeats(self):
        ds = LoraImageDataset(_cfg([(self.first, 3), (self.second, 1)]))
        self.assertEqual(len(ds), 5)  # unique images, not per-epoch samples
        self.assertEqual(ds.entry_image_counts, [3, 2])
        self.assertEqual(ds.total_samples, 11)  # 3*3 + 2*1
        self.assertEqual(ds.entries[0].repeat, 3)
        self.assertEqual(ds.records[0]["repeat"], 3)
        self.assertEqual(ds.records[0]["root"], self.first)

    def test_buckets_hold_each_index_repeat_times(self):
        ds = LoraImageDataset(_cfg([(self.first, 3), (self.second, 1)]))
        counts = Counter(ds.buckets[(512, 512)])
        self.assertEqual(sorted(counts.values()), [1, 1, 3, 3, 3])
        self.assertEqual(sum(counts.values()), ds.total_samples)

    def test_missing_folder_names_its_entry(self):
        missing = Path(self.tmp.name) / "typo"
        with self.assertRaises(RuntimeError) as ctx:
            LoraImageDataset(_cfg([(self.first, 1), (missing, 1)]))
        self.assertIn("train_data[2] is not a directory", str(ctx.exception))
        self.assertIn(str(missing), str(ctx.exception))

    def test_no_usable_images_names_every_folder(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            LoraImageDataset(_cfg([(empty, 1)]))
        self.assertIn(str(empty), str(ctx.exception))

    def test_each_folder_keeps_its_own_latent_cache(self):
        ds = LoraImageDataset(_cfg([(self.first, 3), (self.second, 1)]))
        self.assertEqual(
            ds.latent_cache_dirs,
            [self.first / ".latents_cache", self.second / ".latents_cache"],
        )
        self.assertTrue(all(cache_dir.is_dir() for cache_dir in ds.latent_cache_dirs))
        self.assertEqual(ds.latent_cache_dir, ds.latent_cache_dirs[0])
        cache_paths = []
        for index, record in enumerate(ds.records):
            cache_dir = (
                ds.latent_cache_dirs[0] if record["root"] == self.first else ds.latent_cache_dirs[1]
            )
            cache_path = Path(ds[index]["cache_path"])
            self.assertEqual(cache_path.parent, cache_dir)
            cache_paths.append(cache_path)
        # Distinct images key distinct files, so one folder's cache never answers for another's.
        self.assertEqual(len(set(cache_paths)), len(ds.records))

    def test_one_epoch_draws_every_image_exactly_repeat_times(self):
        ds = LoraImageDataset(_cfg([(self.first, 3), (self.second, 1)]))
        sampler = BucketBatchSampler(ds.buckets, batch_size=2, seed=7)
        expected = Counter()
        for index, record in enumerate(ds.records):
            expected[index] = record["repeat"]
        for epoch in range(3):
            sampler.set_epoch(epoch)
            seen = Counter()
            for batch in sampler:
                seen.update(batch)
                # A batch is one bucket's worth of indices, repeats included.
                buckets = {(ds.records[i]["bucket_w"], ds.records[i]["bucket_h"]) for i in batch}
                self.assertEqual(buckets, {(512, 512)}, msg=f"mixed buckets: {batch}")
            self.assertEqual(seen, expected, msg=f"epoch {epoch}")

    def test_sampler_length_follows_the_repeats(self):
        repeated = LoraImageDataset(_cfg([(self.first, 3), (self.second, 1)]))
        plain = LoraImageDataset(_cfg([(self.first, 1), (self.second, 1)]))
        self.assertEqual(len(plain.buckets[(512, 512)]), 5)
        self.assertEqual(len(repeated.buckets[(512, 512)]), 11)
        self.assertEqual(
            math.ceil(len(repeated.buckets[(512, 512)]) / 2),
            len(BucketBatchSampler(repeated.buckets, batch_size=2, seed=0)),
        )

    def test_the_dataloader_the_loop_iterates_carries_the_repeats(self):
        """`build_dataloader` is what `main.py` derives `steps_per_epoch` from; no GPU needed."""
        repeated = _cfg([(self.first, 3), (self.second, 1)])
        ds, dataloader = build_dataloader(repeated)
        self.assertEqual(len(ds), 5)
        self.assertEqual(len(dataloader), 6)  # ceil(11 / 2)
        drawn = 0
        buckets_seen = set()
        for batch in dataloader:
            drawn += len(batch["caption"])
            buckets_seen.update(set(batch["bucket_w"]) | set(batch["bucket_h"]))
            self.assertEqual(len(group_indices_by_bucket(batch)), 1)
        self.assertEqual(drawn, ds.total_samples)
        self.assertEqual(buckets_seen, {512})

        steps_per_epoch = max(
            1, math.ceil(len(dataloader) / repeated.gradient_accumulation_steps)
        )
        self.assertEqual(steps_per_epoch, 6)
        # Same folder, no repeat: the epoch halves, which is what "steps follow the sampler" means.
        _, plain = build_dataloader(_cfg([(self.first, 1), (self.second, 1)]))
        self.assertEqual(len(plain), 3)

    def test_collate_accepts_one_index_twice(self):
        """A repeat can put the same image twice in one batch; the loop stacks it unchanged."""
        ds = LoraImageDataset(_cfg([(self.first, 2)]))
        item = ds[0]
        batch = collate_fn([item, item])
        self.assertEqual(len(batch["caption"]), 2)
        self.assertEqual(batch["image_path"], [item["image_path"]] * 2)
        self.assertEqual(batch["bucket_w"], [512, 512])
        self.assertEqual(len(batch["img_data"]), 2)
        self.assertTrue(all(torch.is_tensor(entry) for entry in batch["img_data"]))

    def test_legacy_single_folder_config_trains_one_pass(self):
        cfg = _cfg([(self.first, 1)])
        cfg.train_data = []  # a config written before the list existed
        ds = LoraImageDataset(cfg)
        self.assertEqual(ds.entries, [TrainDataEntry(path=str(self.first), repeat=1)])
        self.assertEqual(ds.total_samples, 3)
        self.assertEqual(len(ds.buckets[(512, 512)]), 3)


if __name__ == "__main__":
    unittest.main()
