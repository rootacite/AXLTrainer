"""The detached generator's own contract: its job spec, its cancel flag and its plan helpers.

Importing this module pulls torch and diffusers in; nothing here touches the GPU.
"""

import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path

# `python test/test_generate_sample.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import genjob
from trainer import generate_sample as generator


class CancelFlagTest(unittest.TestCase):
    """`cancel_generation` sends SIGTERM; the runner turns that into a check-pointed stop."""

    def setUp(self):
        self.previous = {}
        for name in ("SIGTERM", "SIGINT"):
            signum = getattr(signal, name)
            self.previous[signum] = signal.getsignal(signum)
        self._was_asked = generator._cancel["asked"]
        generator._cancel["asked"] = False
        generator._install_signal_handler()

    def tearDown(self):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)
        generator._cancel["asked"] = self._was_asked

    def test_a_signal_sets_the_flag_without_killing_the_process(self):
        self.assertFalse(generator._cancel_asked())
        os.kill(os.getpid(), signal.SIGTERM)
        self.assertTrue(generator._cancel_asked())
        # The loop's check turns the flag into the exception the modes catch.
        self.assertTrue(issubclass(generator._Cancelled, Exception))

    def test_interrupt_is_handled_too(self):
        os.kill(os.getpid(), signal.SIGINT)
        self.assertTrue(generator._cancel_asked())

    def test_the_modes_catch_a_cancel_as_a_state(self):
        """`main` marks the job cancelled (not an error) and exits 0."""
        source = Path(generator.__file__).read_text(encoding="utf-8")
        self.assertIn("except _Cancelled:", source)
        self.assertIn("state=genjob.STATE_CANCELLED", source)
        for mode in ("MODE_SINGLE", "MODE_SETS", "MODE_BATCH"):
            self.assertIn(mode, source)


class ShapeKeyTest(unittest.TestCase):
    """What forces a batch to rebuild the pipeline instead of only reloading the weights."""

    def _cfg(self, **fields):
        from types import SimpleNamespace

        return SimpleNamespace(
            **{
                "pretrained_model_name_or_path": "/models/sdxl",
                "network_type": "standard",
                "network_dim": 32,
                "network_alpha": 16,
                "conv_dim": 0,
                "conv_alpha": 0,
                **fields,
            }
        )

    def test_the_weights_are_not_part_of_it(self):
        self.assertEqual(generator._shape_key(self._cfg()), generator._shape_key(self._cfg()))

    def test_a_different_base_kind_or_rank_rebuilds(self):
        base = generator._shape_key(self._cfg())
        for fields in (
            {"pretrained_model_name_or_path": "/models/other"},
            {"network_type": "locon"},
            {"network_dim": 64},
            {"network_alpha": 8},
            {"conv_dim": 16},
        ):
            with self.subTest(fields=fields):
                self.assertNotEqual(base, generator._shape_key(self._cfg(**fields)))

    def test_a_locons_conv_shape_matters(self):
        self.assertNotEqual(
            generator._shape_key(self._cfg(network_type="locon", conv_dim=16)),
            generator._shape_key(self._cfg(network_type="locon", conv_dim=32)),
        )


class BatchSpecTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.generated = genjob.generated_dir(Path(self.tmp.name) / "rein_samples")
        self.generated.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _spec(self, checkpoints):
        job = genjob.new_batch_job(
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoints=checkpoints,
            from_step=0,
            to_step=1000,
            images_per_checkpoint=1,
        )
        genjob.write_job(self.generated, job)
        return job

    def test_an_empty_plan_is_refused(self):
        job = self._spec([])
        with self.assertRaises(RuntimeError):
            generator.run_sample_batch(job, self.generated)

    def test_a_missing_checkpoint_fails_that_entry_and_leaves_the_batch_recorded(self):
        """One unusable checkpoint must not stop the range — the batch keeps going and reports it."""
        job = self._spec([{"path": str(self.generated / "gone.safetensors"), "step": 100}])
        generator.run_sample_batch(job, self.generated)
        stored = genjob.read_job(genjob.job_path(self.generated, str(job["id"])))
        self.assertEqual(stored["state"], genjob.STATE_ERROR)  # the only entry failed
        self.assertEqual(len(stored["failed"]), 1)
        self.assertIn("gone.safetensors", stored["failed"][0]["checkpoint"])
        self.assertEqual(stored["images_done"], 0)


if __name__ == "__main__":
    unittest.main()
