import tempfile
import unittest

import sys
from pathlib import Path

# `python test/test_validation.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import (
    TrainConfig,
    _load_toml_config,
    resolve_sample_sets,
    tracker_hparams,
)


def flat(**overrides) -> dict:
    """A flattened config mapping shaped like `_load_toml_config()` output."""
    base = {
        "sample_prompts": "base prompt",
        "sample_negative": "base negative",
        "sample_width": 1024,
        "sample_height": 768,
        "sample_steps": 30,
        "sample_seed": 7,
        "sample_repeat": 2,
        "guidance_scale": 5.5,
    }
    base.update(overrides)
    return base


class NoSetsTest(unittest.TestCase):
    """A config without `[[validation.samples]]` keeps the single-prompt behaviour."""

    def test_one_set_from_the_flat_scalars(self):
        sets = resolve_sample_sets(flat())
        self.assertEqual(len(sets), 1)
        only = sets[0]
        self.assertEqual(only.prompt, "base prompt")
        self.assertEqual(only.negative, "base negative")
        self.assertEqual((only.width, only.height), (1024, 768))
        self.assertEqual(only.steps, 30)
        self.assertEqual(only.seed, 7)
        self.assertEqual(only.repeat, 2)
        self.assertEqual(only.guidance_scale, 5.5)

    def test_name_defaults_to_the_first_prompt_tag(self):
        self.assertEqual(resolve_sample_sets(flat())[0].name, "base prompt")
        self.assertEqual(resolve_sample_sets(flat(sample_prompts="tag, rest, more"))[0].name, "tag")

    def test_a_dataclass_and_a_mapping_resolve_the_same(self):
        from_mapping = resolve_sample_sets(_load_toml_config())
        from_config = resolve_sample_sets(TrainConfig())
        self.assertEqual(from_mapping, from_config)

    def test_blank_prompt_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_sample_sets(flat(sample_prompts="   "))
        self.assertIn("sample_prompts", str(ctx.exception))


class SetEntryTest(unittest.TestCase):
    def test_each_entry_overrides_its_own_keys(self):
        sets = resolve_sample_sets(
            flat(
                samples=[
                    {"name": "close up", "prompt": "p1", "steps": 8, "repeat": 1},
                    {"prompt": "p2", "width": 512, "guidance_scale": 3},
                ]
            )
        )
        self.assertEqual([s.name for s in sets], ["close up", "p2"])
        self.assertEqual((sets[0].steps, sets[0].repeat), (8, 1))
        self.assertEqual((sets[1].width, sets[1].height), (512, 768))
        self.assertEqual(sets[1].guidance_scale, 3.0)
        self.assertEqual(sets[1].negative, "base negative")

    def test_name_falls_back_to_the_prompt_tag_then_the_index(self):
        sets = resolve_sample_sets(flat(samples=[{"prompt": "a"}, {"prompt": ",,"}]))
        self.assertEqual(sets[0].name, "a")
        self.assertEqual(sets[1].name, "Set 2")

    def test_a_set_omitting_a_key_uses_the_flat_scalar(self):
        sets = resolve_sample_sets(flat(samples=[{"prompt": "only a prompt"}]))
        self.assertEqual(sets[0].seed, 7)
        self.assertEqual(sets[0].repeat, 2)

    def test_same_seed_in_two_sets_keeps_the_seed_sequences_aligned(self):
        sets = resolve_sample_sets(flat(samples=[{"prompt": "a"}, {"prompt": "b"}], sample_seed=100, sample_repeat=2))
        seeds = [[s.seed + repeat for repeat in range(s.repeat)] for s in sets]
        self.assertEqual(seeds[0], seeds[1])
        self.assertEqual(seeds[0], [100, 101])

    def test_a_single_set_keeps_a_single_prompt_behaviour(self):
        sets = resolve_sample_sets(flat(samples=[{"prompt": "solo"}]))
        self.assertEqual(len(sets), 1)


class ValidationErrorTest(unittest.TestCase):
    def test_out_of_range_names_the_entry(self):
        cases = [
            ({"prompt": "p", "steps": 0}, "steps"),
            ({"prompt": "p", "repeat": 0}, "repeat"),
            ({"prompt": "p", "width": 32}, "width"),
            ({"prompt": "p", "height": 8192}, "height"),
            ({"prompt": "p", "guidance_scale": 60}, "guidance_scale"),
            ({"prompt": "p", "seed": -1}, "seed"),
            ({"prompt": "p", "seed": 2**32}, "seed"),
        ]
        for entry, expected in cases:
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError) as ctx:
                    resolve_sample_sets(flat(samples=[{"prompt": "ok"}, entry]))
                message = str(ctx.exception)
                self.assertIn("validation.samples[2]", message)
                self.assertIn(expected, message)

    def test_an_empty_prompt_names_the_entry(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_sample_sets(flat(samples=[{"prompt": " "}]))
        self.assertIn("validation.samples[1]: prompt must not be empty", str(ctx.exception))

    def test_a_non_table_entry_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_sample_sets(flat(samples=["nope"]))
        self.assertIn("must be a table", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            resolve_sample_sets(flat(samples="nope"))
        self.assertIn("array of tables", str(ctx.exception))

    def test_a_non_numeric_value_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_sample_sets(flat(samples=[{"prompt": "p", "steps": "many"}]))
        self.assertIn("expected an integer", str(ctx.exception))


class TrackerHparamsTest(unittest.TestCase):
    """A prompt set in the config must not abort `accelerator.init_trackers`.

    TensorBoard's `add_hparams` accepts int/float/str/bool/torch.Tensor and skips `None`. Once
    `[[validation.samples]]` put a list of tables into `vars(cfg)`, the tracker init raised — after
    the pipeline was loaded and the latent cache had been built, i.e. minutes into a run.
    """

    def setUp(self):
        self.cfg = TrainConfig(samples=[{"name": "one", "prompt": "p1", "repeat": 2}])

    def test_non_scalars_become_json_and_scalars_pass_through(self):
        params = tracker_hparams(self.cfg)
        self.assertIsInstance(params["samples"], str)
        self.assertIn('"prompt": "p1"', params["samples"])
        self.assertEqual(params["train_batch_size"], self.cfg.train_batch_size)
        self.assertEqual(params["guidance_scale"], self.cfg.guidance_scale)
        # the kohya metadata placeholders are None, which the tracker skips anyway
        self.assertNotIn("ss_session_id", params)

    def test_tensorboard_accepts_the_sanitized_record(self):
        from torch.utils.tensorboard import SummaryWriter

        with tempfile.TemporaryDirectory() as tmp:
            writer = SummaryWriter(log_dir=tmp)
            try:
                writer.add_hparams(tracker_hparams(self.cfg), metric_dict={})
                with self.assertRaises(ValueError):
                    writer.add_hparams(vars(self.cfg), metric_dict={})
            finally:
                writer.close()

    def test_a_flattened_mapping_is_accepted_too(self):
        params = tracker_hparams(flat(samples=[{"prompt": "p"}]))
        self.assertIn('"prompt": "p"', params["samples"])
        self.assertEqual(params["sample_steps"], 30)


def _load_verifier():
    """The mask verifier, imported by path (`test/` is deliberately not a package)."""
    import importlib.util

    path = Path(__file__).resolve().parent / "verify_mask_pipeline.py"
    spec = importlib.util.spec_from_file_location("verify_mask_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves a field's annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


class VerifierMirrorConfigTest(unittest.TestCase):
    """The mask verifier re-serializes the live config for its mirror repo.

    Its writer emits `key = value` lines, so an array of tables used to come out as one
    `samples = "{'name': …}"` string and the child aborted at startup with "validation.samples must
    be an array of tables". Left out of the mirror, the child builds its single set from the flat
    `sample_*` scalars — which is what those runs want anyway.
    """

    def test_the_mirror_config_drops_the_array_and_still_resolves_a_set(self):
        import tomllib

        module = _load_verifier()
        sections = {
            "environment": {"train_data_dir": "/tmp/data"},
            "validation": {
                "sample_prompts": "flat prompt",
                "sample_negative": "flat negative",
                "sample_repeat": 2,
                "samples": [{"name": "one", "prompt": "p1"}],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            module.write_toml(path, sections)
            with open(path, "rb") as handle:
                written = tomllib.load(handle)
            self.assertNotIn("samples", written["validation"])
            self.assertEqual(written["validation"]["sample_prompts"], "flat prompt")

            sets = resolve_sample_sets(_load_toml_config(str(path)))
        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0].prompt, "flat prompt")
        self.assertEqual(sets[0].repeat, 2)


if __name__ == "__main__":
    unittest.main()
