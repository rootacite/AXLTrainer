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


if __name__ == "__main__":
    unittest.main()
