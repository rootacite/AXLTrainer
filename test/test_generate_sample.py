"""The detached generator's own contract: its job spec, its cancel flag and its plan helpers.

Importing this module pulls torch and diffusers in; nothing here touches the GPU.
"""

import contextlib
import io
import os
import signal
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

# `python test/test_generate_sample.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import genjob
from trainer import generate_sample as generator
from trainer.config import SampleSet


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
        for mode in ("MODE_SINGLE", "MODE_SETS", "MODE_BATCH", "MODE_EVALUATE"):
            self.assertIn(mode, source)


class PlanSlotsTest(unittest.TestCase):
    """Which `(set, repeat)` an evaluation renders: only the slots the plan lists, per set."""

    def setUp(self):
        from trainer.config import SampleSet

        self.sets = [
            SampleSet(name="a", prompt="pa", negative="", width=64, height=64, steps=2,
                      guidance_scale=1.0, guidance_rescale=0.0, seed=1, repeat=2),
            SampleSet(name="b", prompt="pb", negative="", width=64, height=64, steps=2,
                      guidance_scale=1.0, guidance_rescale=0.0, seed=2, repeat=1),
        ]

    def test_no_plan_is_the_whole_pass(self):
        self.assertEqual(generator._plan_slots(self.sets), {0: [0, 1], 1: [0]})

    def test_a_plan_renders_exactly_its_slots(self):
        self.assertEqual(generator._plan_slots(self.sets, [(1, 0), (0, 2)]), {1: [0], 0: [2]})

    def test_a_set_with_nothing_to_draw_is_absent(self):
        # An evaluation that only has to top set 1 up must not encode set 0's prompt.
        self.assertEqual(generator._plan_slots(self.sets, [(1, 1)]), {1: [1]})

    def test_an_empty_plan_renders_nothing(self):
        self.assertEqual(generator._plan_slots(self.sets, []), {})


class RecordedPlanTest(unittest.TestCase):
    """The spec is the work order: its sets and its config file are read back, not re-invented."""

    def test_the_recorded_sets_come_back_as_sample_sets(self):
        spec = {
            "sample_sets": [
                {"name": "a", "prompt": "pa", "negative": "n", "width": 64, "height": 64,
                 "steps": 3, "guidance_scale": 4.0, "guidance_rescale": 0.5, "seed": 7, "repeat": 2}
            ]
        }
        sets = generator._record_sets(spec)
        self.assertEqual(len(sets), 1)
        self.assertEqual((sets[0].prompt, sets[0].seed, sets[0].repeat, sets[0].steps), ("pa", 7, 2, 3))

    def test_a_record_without_usable_sets_reads_as_empty(self):
        self.assertEqual(generator._record_sets({}), [])
        self.assertEqual(generator._record_sets({"sample_sets": "nonsense"}), [])
        # A table this config does not know (or not a table at all) is skipped, not raised.
        self.assertEqual(generator._record_sets({"sample_sets": [{"prompt": "x", "extra": 1}, "junk"]}), [])

    def test_the_recorded_config_is_the_one_the_run_saved(self):
        # The runner resolves the same way api.py did, from the run's own log directory.
        with tempfile.TemporaryDirectory() as raw:
            log_dir = Path(raw) / "rein_20260911_120000"
            log_dir.mkdir()
            (log_dir / "config.toml").write_text(
                "[validation]\nsample_steps = 12\nsample_prompts = \"snapshot\"\n", encoding="utf-8"
            )
            cfg = generator._record_config({"config_log_dir": str(log_dir)})
        self.assertEqual(cfg.sample_steps, 12)

    def test_the_runs_recorded_hparams_stand_in_for_a_missing_snapshot(self):
        from torch.utils.tensorboard import SummaryWriter

        from trainer.config import tracker_hparams

        with tempfile.TemporaryDirectory() as raw:
            log_dir = Path(raw) / "rein_20260911_120000"
            log_dir.mkdir()
            writer = SummaryWriter(log_dir=str(log_dir))
            try:
                writer.add_hparams(tracker_hparams({"output_name": "rein", "sample_steps": 21}), {})
            finally:
                writer.close()
            cfg = generator._record_config({"config_log_dir": str(log_dir)})
        self.assertEqual(cfg.sample_steps, 21)

    def test_no_run_log_directory_falls_back_to_todays(self):
        from trainer.config import TrainConfig

        self.assertEqual(generator._record_config({}).sample_steps, TrainConfig().sample_steps)


class EvaluationWithoutRenderTest(unittest.TestCase):
    """A checkpoint that already has enough images: tag, score, never load the diffusion model.

    The empty image list is what keeps this off the GPU — the pass resolves no tagger either — while
    the whole record path (phases, counters, scores) still runs for real.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.generated = genjob.generated_dir(Path(self.tmp.name) / "rein_samples")
        self.generated.mkdir(parents=True)

    def test_the_pass_scores_an_empty_set_and_ends_done(self):
        from safetensors.torch import save_file

        checkpoint = Path(self.tmp.name) / "rein_s000100" / "rein.safetensors"
        checkpoint.parent.mkdir(parents=True)
        save_file({}, str(checkpoint), metadata={"ss_steps": "100"})

        job = genjob.new_evaluation_job(
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint=str(checkpoint),
            step=100,
            depth=4,
            threshold=0.35,
            categories=["general"],
            config_source="/tmp/gone-config.toml",
            config_log_dir="/tmp/gone-log-dir",
            plan={"k": 0, "needed": False, "render_total": 0, "sets": []},
            images=[],
            sample_sets=[],
        )
        genjob.write_job(self.generated, job)
        previous = generator._cancel["asked"]
        generator._cancel["asked"] = False
        self.addCleanup(lambda: generator._cancel.__setitem__("asked", previous))

        generator.run_evaluation(job, self.generated)

        stored = genjob.read_job(genjob.job_path(self.generated, str(job["id"])))
        self.assertEqual(stored["state"], genjob.STATE_DONE)
        self.assertEqual(stored["phase"], genjob.PHASE_DONE)
        self.assertEqual(stored["scores"]["images_scored"], 0)
        self.assertEqual(stored["scores"]["f1"], 0.0)
        self.assertEqual((stored["images_done"], stored["total_images"]), (0, 0))
        self.assertEqual(list(self.generated.glob("*.png")), [])

    def test_the_recorded_tag_selection_narrows_the_scoring(self):
        from safetensors.torch import save_file

        checkpoint = Path(self.tmp.name) / "rein_s000100" / "rein.safetensors"
        checkpoint.parent.mkdir(parents=True)
        save_file({}, str(checkpoint), metadata={"ss_steps": "100"})

        images = [
            {
                "path": str(self.generated / "a.png"),
                "set_index": 0,
                "repeat_idx": 0,
                "prompt": "1girl, anal",
            }
        ]
        (self.generated / "a.png").write_bytes(b"png")
        job = genjob.new_evaluation_job(
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint=str(checkpoint),
            step=100,
            depth=1,
            threshold=0.35,
            categories=["general"],
            config_source="/tmp/gone-config.toml",
            config_log_dir="/tmp/gone-log-dir",
            plan={"k": 0, "needed": False, "render_total": 0, "sets": []},
            images=images,
            sample_sets=[{"prompt": "1girl, anal", "repeat": 1}],
            tags=["anal"],
        )
        genjob.write_job(self.generated, job)
        previous = generator._cancel["asked"]
        generator._cancel["asked"] = False
        self.addCleanup(lambda: generator._cancel.__setitem__("asked", previous))

        def fake_tag_paths(paths, *, threshold, categories, on_entry=None):
            entries = []
            for path in paths:
                entry = {"path": str(path), "name": Path(path).name, "tags": ["anal", "outdoor"], "error": None}
                entries.append(entry)
                if on_entry is not None:
                    on_entry(entry)
            return entries

        with mock.patch("tagger2.main.tag_paths", side_effect=fake_tag_paths):
            generator.run_evaluation(job, self.generated)

        stored = genjob.read_job(genjob.job_path(self.generated, str(job["id"])))
        self.assertEqual(stored["state"], genjob.STATE_DONE)
        self.assertEqual(stored["scores"]["tags"], ["anal"])
        # `outdoor` was not asked about, so it is not a false positive; `1girl` is not a miss.
        self.assertEqual((stored["scores"]["tp"], stored["scores"]["fp"], stored["scores"]["fn"]), (1, 0, 0))
        self.assertEqual(stored["scores"]["recall"], 1.0)


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

    def test_the_recorded_sets_are_what_a_range_renders(self):
        """The range's prompts come off the record, the way a single-checkpoint pass takes them.

        `run_sample_batch` used to resolve them from `config_log_dir` itself. For a run whose
        snapshot carries no `[[validation.samples]]` (a config with only the flat `sample_*`
        scalars) that fell through to *today's* repo `config.toml`, so a range could render prompts,
        sizes and step counts the run never used.
        """
        recorded = SampleSet(
            name="planned", prompt="the planned prompt", negative="n", width=8, height=8, steps=1,
            guidance_scale=4.0, guidance_rescale=0.5, seed=7, repeat=1,
        )
        job = genjob.new_batch_job(
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoints=[{"path": "rein_s000100", "step": 100}],
            from_step=100,
            to_step=100,
            images_per_checkpoint=1,
            sample_sets=[asdict(recorded)],
        )
        genjob.write_job(self.generated, job)
        cfg = types.SimpleNamespace(
            pretrained_model_name_or_path="base.safetensors",
            network_type="standard",
            network_dim=8,
            network_alpha=4,
            conv_dim=0,
            conv_alpha=0,
            mixed_precision="bf16",
        )
        seen: dict = {}

        def fake_render(**kwargs):
            seen["prompts"] = [entry.prompt for entry in kwargs["sets"]]
            return []

        def refuse(cfg):
            raise AssertionError("the recorded sets must be used, not re-resolved from a config")

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(generator, "resolve_resume_path", lambda raw: Path(str(raw)).with_suffix(".safetensors")))
            stack.enter_context(mock.patch.object(generator, "read_lora_metadata", lambda path: {}))
            stack.enter_context(mock.patch.object(generator, "_record_config", lambda spec: cfg))
            stack.enter_context(
                mock.patch.object(generator, "_build_config", lambda metadata, checkpoint, base_cfg=None: cfg)
            )
            stack.enter_context(mock.patch.object(generator, "resolve_sample_sets", refuse))
            stack.enter_context(mock.patch.object(generator, "_load_family", lambda cfg, dtype: (None, None)))
            stack.enter_context(mock.patch.object(generator, "_render_sets", fake_render))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            generator.run_sample_batch(job, self.generated)

        self.assertEqual(seen["prompts"], ["the planned prompt"])

    def test_a_missing_checkpoint_fails_that_entry_and_leaves_the_batch_recorded(self):
        """One unusable checkpoint must not stop the range — the batch keeps going and reports it."""
        job = self._spec([{"path": str(self.generated / "gone.safetensors"), "step": 100}])
        generator.run_sample_batch(job, self.generated)
        stored = genjob.read_job(genjob.job_path(self.generated, str(job["id"])))
        self.assertEqual(stored["state"], genjob.STATE_ERROR)  # the only entry failed
        self.assertEqual(len(stored["failed"]), 1)
        self.assertIn("gone.safetensors", stored["failed"][0]["checkpoint"])
        self.assertEqual(stored["images_done"], 0)

    def test_a_range_that_renders_finishes_every_checkpoint(self):
        """The render's own list and the counter the batch reports are two different things.

        They once shared the name `rendered`, so `rendered += 1` added an image slot to an image
        list: every checkpoint of a range that actually rendered was recorded as failed while its
        images were written anyway. The model side is stubbed here — the point is the loop's own
        bookkeeping, and no GPU is involved.
        """
        job = self._spec(
            [
                {"path": "rein_s000100", "step": 100},
                {"path": "rein_s000200", "step": 200},
            ]
        )
        cfg = types.SimpleNamespace(
            pretrained_model_name_or_path="base.safetensors",
            network_type="standard",
            network_dim=8,
            network_alpha=4,
            conv_dim=0,
            conv_alpha=0,
            mixed_precision="bf16",
        )
        sets = [
            SampleSet(
                name="probe",
                prompt="a prompt",
                negative="",
                width=8,
                height=8,
                steps=1,
                guidance_scale=1.0,
                guidance_rescale=0.0,
                seed=1,
                repeat=2,
            )
        ]
        calls = []

        def fake_render(**kwargs):
            calls.append(kwargs["job_id"])
            return [(0, index, f"{kwargs['job_id']}_p0_{index}.png") for index in range(2)]

        stdout = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(generator, "resolve_resume_path", lambda raw: Path(str(raw)).with_suffix(".safetensors")))
            stack.enter_context(mock.patch.object(generator, "read_lora_metadata", lambda path: {}))
            stack.enter_context(mock.patch.object(generator, "_record_config", lambda spec: cfg))
            stack.enter_context(
                mock.patch.object(generator, "_build_config", lambda metadata, checkpoint, base_cfg=None: cfg)
            )
            stack.enter_context(mock.patch.object(generator, "resolve_sample_sets", lambda cfg: sets))
            stack.enter_context(mock.patch.object(generator, "_render_sets", fake_render))
            stack.enter_context(mock.patch.object(generator, "_load_family", lambda cfg, dtype: (None, None)))
            stack.enter_context(
                mock.patch.object(
                    generator,
                    "resolve_family",
                    lambda cfg: types.SimpleNamespace(load_lora=lambda cfg, modules: {}),
                )
            )
            stack.enter_context(contextlib.redirect_stdout(stdout))
            generator.run_sample_batch(job, self.generated)

        stored = genjob.read_job(genjob.job_path(self.generated, str(job["id"])))
        self.assertEqual(stored["state"], genjob.STATE_DONE)
        self.assertEqual(stored["failed"], [])
        self.assertEqual(stored["checkpoint_index"], 2)
        self.assertEqual(stored["images_done"], 4)
        self.assertEqual(len(stored["job_ids"]), 2)
        for job_id in stored["job_ids"]:
            entry = genjob.read_job(genjob.job_path(self.generated, job_id))
            self.assertEqual(entry["state"], genjob.STATE_DONE)
            self.assertEqual(len(entry["files"]), 2)
        self.assertEqual(len(calls), 2, "the pass renders once per checkpoint")
        self.assertIn("2/2 checkpoint(s), 4 image(s)", stdout.getvalue())
        for job_id in stored["job_ids"]:
            entry = genjob.read_job(genjob.job_path(self.generated, job_id))
            # The child spec stands on its own: it says which prompts it rendered.
            self.assertEqual([item["prompt"] for item in entry["sample_sets"]], ["a prompt"])


class SetsPassPromptsTest(unittest.TestCase):
    """A `sets` pass renders the prompts the spec planned, not whatever the repo file says today."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.generated = genjob.generated_dir(Path(self.tmp.name) / "rein_samples")
        self.generated.mkdir(parents=True)
        self.cfg = types.SimpleNamespace(
            pretrained_model_name_or_path="base.safetensors",
            network_type="standard",
            network_dim=8,
            network_alpha=4,
            conv_dim=0,
            conv_alpha=0,
            mixed_precision="bf16",
        )

    def _spec(self, **extra):
        request = {"step": 100, "prompt": "", "negative_prompt": "", "cfg": 6.0, "steps": 1, "seed": 1}
        return genjob.new_job(
            request,
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint="/out/rein_s000100/rein.safetensors",
            mode=genjob.MODE_SETS,
            total_images=1,
            extra=extra,
        )

    def _render_with(self, spec, resolver):
        """Run one `sets` pass with the family stubbed; return the prompts `_render_sets` got."""
        seen: dict = {}

        class FakeModule:
            def to(self, *args, **kwargs):
                return self

        class FakeModules:
            denoise = FakeModule()
            text_encoders = [FakeModule(), FakeModule()]

        def fake_render(**kwargs):
            seen["prompts"] = [entry.prompt for entry in kwargs["sets"]]
            return []

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(generator, "resolve_resume_path", lambda raw: Path(str(raw)))
            )
            stack.enter_context(mock.patch.object(generator, "read_lora_metadata", lambda path: {}))
            stack.enter_context(
                mock.patch.object(generator, "_build_config", lambda metadata, checkpoint, base_cfg=None: self.cfg)
            )
            stack.enter_context(mock.patch.object(generator, "_record_config", lambda spec: self.cfg))
            stack.enter_context(mock.patch.object(generator, "resolve_sample_sets", resolver))
            stack.enter_context(mock.patch.object(generator, "_load_family", lambda cfg, dtype: (None, FakeModules())))
            stack.enter_context(mock.patch.object(generator, "_render_sets", fake_render))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            generator.run_sample_sets(spec, self.generated)
        return seen

    def test_the_recorded_sets_are_what_renders(self):
        def refuse(cfg):
            raise AssertionError("the recorded sets must be used, not re-resolved from a config")

        recorded = SampleSet(
            name="planned",
            prompt="the planned prompt",
            negative="n",
            width=64,
            height=64,
            steps=1,
            guidance_scale=4.0,
            guidance_rescale=0.5,
            seed=7,
            repeat=1,
        )
        from dataclasses import asdict

        spec = self._spec(
            sample_sets=[asdict(recorded)],
            config_log_dir="/logs/rein_20260101_000000",
        )
        self.assertEqual(self._render_with(spec, refuse)["prompts"], ["the planned prompt"])

    def test_a_spec_without_recorded_sets_falls_back_to_the_run_config(self):
        fallback = SampleSet(
            name="from-config",
            prompt="the run's config",
            negative="",
            width=64,
            height=64,
            steps=1,
            guidance_scale=1.0,
            guidance_rescale=0.0,
            seed=1,
            repeat=1,
        )
        spec = self._spec(config_log_dir="/logs/rein_20260101_000000")
        self.assertEqual(self._render_with(spec, lambda cfg: [fallback])["prompts"], ["the run's config"])


if __name__ == "__main__":
    unittest.main()
