"""The per-run prompt sets the Dashboard's Sampling Prompts editor saves.

`{logging_dir}/{run_id}/sample_sets.json` is an optional layer over the run's own config snapshot:
absent, every read path behaves exactly as it did before the file existed; present, it replaces the
`samples` entry of whatever the run's config resolves to. No test here needs torch or a GPU.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# `python test/test_sample_prompts.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import (
    SAMPLE_OVERRIDE_FILENAME,
    active_sample_sets,
    clear_sample_override,
    read_sample_override,
    resolve_sample_sets,
    run_config_mapping,
    run_log_dir,
    sample_override_path,
    write_sample_override,
)


def entry(prompt: str = "edited prompt", **overrides) -> dict:
    """One complete stored entry, as `write_sample_override` writes it."""
    base = {
        "name": "set",
        "prompt": prompt,
        "negative": "bad quality",
        "width": 1152,
        "height": 768,
        "steps": 35,
        "guidance_scale": 6.0,
        "guidance_rescale": 0.6,
        "seed": 0,
        "repeat": 2,
    }
    base.update(overrides)
    return base


def stored(log_dir: Path, entries: list | None = None) -> Path:
    """Write a raw JSON payload where the editor's own writes land (no atomic-write path)."""
    path = sample_override_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries if entries is not None else [entry()]), encoding="utf-8")
    return path


def run_cfg(root: Path, run_id: str, samples: list) -> SimpleNamespace:
    """A `TrainConfig`-shaped stand-in: `run_dir` is `{output_dir}/{run_id}`, as main.py sets it."""
    return SimpleNamespace(
        run_dir=str(root / "outputs" / run_id),
        logging_dir=str(root / "logs"),
        samples=list(samples),
    )


class PathTest(unittest.TestCase):
    def test_the_file_sits_in_the_run_log_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs" / "kanae_20261002_120000"
            log_dir.mkdir(parents=True)
            self.assertEqual(
                sample_override_path(log_dir),
                log_dir / SAMPLE_OVERRIDE_FILENAME,
            )

    def test_run_log_dir_is_the_logging_root_plus_the_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = run_cfg(root, "kanae_20261002_120000", [entry()])
            self.assertEqual(run_log_dir(cfg), root / "logs" / "kanae_20261002_120000")

    def test_run_log_dir_needs_both_halves(self):
        self.assertIsNone(run_log_dir(SimpleNamespace(run_dir="", logging_dir="/logs")))
        self.assertIsNone(run_log_dir(SimpleNamespace(run_dir="/out/run_1", logging_dir="")))


class ReadTest(unittest.TestCase):
    def test_no_file_is_no_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_sample_override(Path(tmp) / "logs" / "run_1"))

    def test_a_complete_file_reads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            stored(log_dir, [entry("first"), entry("second", seed=11, repeat=1)])
            raw = read_sample_override(log_dir)
            self.assertIsNotNone(raw)
            self.assertEqual([item["prompt"] for item in raw], ["first", "second"])
            self.assertEqual(raw[1]["seed"], 11)

    def test_broken_json_is_warned_about_and_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            log_dir.mkdir(parents=True)
            sample_override_path(log_dir).write_text("{not json", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertIsNone(read_sample_override(log_dir))
            self.assertIn("could not be read", stderr.getvalue())

    def test_a_table_instead_of_a_list_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            stored(log_dir, {"prompt": "not a list"})
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertIsNone(read_sample_override(log_dir))
            self.assertIn("not a non-empty list", stderr.getvalue())

    def test_an_empty_list_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            stored(log_dir, [])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertIsNone(read_sample_override(log_dir))

    def test_a_partial_entry_is_ignored_rather_than_filled_from_the_repo(self):
        """An omitted key would otherwise fall back to the repo `config.toml`'s scalar."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            incomplete = entry()
            del incomplete["width"]
            stored(log_dir, [incomplete])
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                self.assertIsNone(read_sample_override(log_dir))
            self.assertIn("is unusable", stderr.getvalue())

    def test_a_blank_prompt_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            stored(log_dir, [entry("  ")])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertIsNone(read_sample_override(log_dir))


class WriteTest(unittest.TestCase):
    def test_write_then_read_round_trips_every_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            sets = resolve_sample_sets({"samples": [entry("a"), entry("b", repeat=3)]})
            path = write_sample_override(log_dir, sets)
            self.assertEqual(path, sample_override_path(log_dir))
            raw = read_sample_override(log_dir)
            self.assertEqual([item["prompt"] for item in raw], ["a", "b"])
            for item in raw:
                self.assertEqual(sorted(item), sorted(entry()))
            parsed = resolve_sample_sets({"samples": raw})
            self.assertEqual([(item.name, item.repeat) for item in parsed], [("set", 2), ("set", 3)])

    def test_a_write_replaces_the_previous_file_and_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            write_sample_override(log_dir, resolve_sample_sets({"samples": [entry("old")]}))
            write_sample_override(log_dir, resolve_sample_sets({"samples": [entry("new")]}))
            self.assertEqual([item["prompt"] for item in read_sample_override(log_dir)], ["new"])
            leftovers = [p.name for p in log_dir.iterdir() if p.name != SAMPLE_OVERRIDE_FILENAME]
            self.assertEqual(leftovers, [])

    def test_clear_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run_1"
            write_sample_override(log_dir, resolve_sample_sets({"samples": [entry()]}))
            clear_sample_override(log_dir)
            self.assertIsNone(read_sample_override(log_dir))
            clear_sample_override(log_dir)  # a second call is not an error


class MappingOverlayTest(unittest.TestCase):
    """`_load_toml_config` flattens tables only, so a snapshot's scalars live under `[validation]`."""

    def test_the_snapshot_is_the_source_when_nothing_was_edited(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs" / "kanae_20261002_120000"
            log_dir.mkdir(parents=True)
            (log_dir / "config.toml").write_text(
                '[validation]\nsample_prompts = "from the snapshot"\nsample_width = 1024\n',
                encoding="utf-8",
            )
            mapping, source = run_config_mapping(log_dir)
            self.assertEqual(source, str((log_dir / "config.toml").resolve()))
            self.assertEqual(mapping["sample_prompts"], "from the snapshot")

    def test_the_override_replaces_samples_and_is_reported_as_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs" / "kanae_20261002_120000"
            log_dir.mkdir(parents=True)
            (log_dir / "config.toml").write_text(
                '[validation]\nsample_prompts = "from the snapshot"\nsample_width = 1024\n',
                encoding="utf-8",
            )
            write_sample_override(log_dir, resolve_sample_sets({"samples": [entry("edited")]}))
            mapping, source = run_config_mapping(log_dir)
            self.assertEqual(source, str(sample_override_path(log_dir).resolve()))
            self.assertEqual([item["prompt"] for item in mapping["samples"]], ["edited"])
            # Everything else still comes from the snapshot.
            self.assertEqual(mapping["sample_prompts"], "from the snapshot")
            sets = resolve_sample_sets(mapping)
            self.assertEqual(len(sets), 1)
            self.assertEqual(sets[0].prompt, "edited")
            self.assertEqual(sets[0].width, 1152)

    def test_the_override_also_layers_over_the_repo_fallback(self):
        """A run with no snapshot (a run from before them) resolves the repo file, plus the overlay."""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs" / "runs_without_a_snapshot"
            log_dir.mkdir(parents=True)
            write_sample_override(log_dir, resolve_sample_sets({"samples": [entry("edited")]}))
            mapping, source = run_config_mapping(log_dir)
            self.assertEqual(source, str(sample_override_path(log_dir).resolve()))
            self.assertEqual(resolve_sample_sets(mapping)[0].prompt, "edited")

    def test_a_corrupt_override_falls_back_to_the_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs" / "kanae_20261002_120000"
            log_dir.mkdir(parents=True)
            (log_dir / "config.toml").write_text(
                '[validation]\nsample_prompts = "from the snapshot"\n', encoding="utf-8"
            )
            sample_override_path(log_dir).write_text("[]", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                mapping, source = run_config_mapping(log_dir)
            self.assertEqual(source, str((log_dir / "config.toml").resolve()))
            sets = resolve_sample_sets(mapping)
            self.assertEqual(sets[0].prompt, "from the snapshot")


class ActiveSetsTest(unittest.TestCase):
    """What the trainer's own sample points use (`trainer/sampling.py`)."""

    def test_the_run_saved_prompts_win_over_the_config_it_started_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs" / "kanae_20261002_120000"
            log_dir.mkdir(parents=True)
            write_sample_override(log_dir, resolve_sample_sets({"samples": [entry("edited")]}))
            cfg = run_cfg(root, "kanae_20261002_120000", [entry("from the config")])
            sets = active_sample_sets(cfg)
            self.assertEqual(len(sets), 1)
            self.assertEqual(sets[0].prompt, "edited")

    def test_without_a_file_it_is_the_configs_own_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs" / "kanae_20261002_120000").mkdir(parents=True)
            cfg = run_cfg(root, "kanae_20261002_120000", [entry("from the config")])
            self.assertEqual(active_sample_sets(cfg)[0].prompt, "from the config")

    def test_a_corrupt_file_falls_back_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs" / "kanae_20261002_120000"
            log_dir.mkdir(parents=True)
            sample_override_path(log_dir).write_text("not json", encoding="utf-8")
            cfg = run_cfg(root, "kanae_20261002_120000", [entry("from the config")])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(active_sample_sets(cfg)[0].prompt, "from the config")

    def test_a_cfg_with_no_run_directory_cannot_have_been_edited(self):
        cfg = SimpleNamespace(
            run_dir="",
            logging_dir="/logs",
            samples=[entry("from the config")],
        )
        self.assertEqual(active_sample_sets(cfg)[0].prompt, "from the config")


if __name__ == "__main__":
    unittest.main()
