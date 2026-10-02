import json
import tempfile
import unittest
from pathlib import Path

import sys

# `python test/test_genjob.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import genjob


class JobIdTest(unittest.TestCase):
    def test_directory_checkpoint_is_named_after_its_directory(self):
        stem = genjob.job_stem("/out/lllj_20260915_134334/lllj_s003050/lllj.safetensors")
        self.assertEqual(stem, "lllj_s003050")

    def test_flat_checkpoint_is_named_after_the_file(self):
        self.assertEqual(genjob.job_stem("/out/lllj_20260915_134334/lllj.safetensors"), "lllj")
        self.assertEqual(genjob.job_stem("lllj_final.safetensors"), "lllj_final")

    def test_an_unrelated_directory_does_not_rename_the_job(self):
        self.assertEqual(genjob.job_stem("/models/loras/mylora.safetensors"), "mylora")

    def test_job_id_carries_a_timestamp_and_survives_odd_names(self):
        job_id = genjob.new_job_id("lllj_s003050")
        self.assertRegex(job_id, r"^lllj_s003050_gen_\d{8}_\d{6}$")
        self.assertNotIn("/", genjob.new_job_id("a/b c"))
        self.assertEqual(genjob.new_job_id("a/b c").split("_gen_")[0], "a_b_c")

    def test_paths_share_the_job_id(self):
        generated = Path("/tmp/run/lllj_samples/generated")
        self.assertEqual(
            genjob.job_path(generated, "job1"), generated / "job1.json"
        )
        self.assertEqual(genjob.image_path(generated, "job1"), generated / "job1.png")
        self.assertEqual(genjob.log_path(generated, "job1"), generated / "job1.log")

    def test_a_sets_job_is_marked_in_its_id(self):
        job_id = genjob.new_job_id("lllj_s003050", mode=genjob.MODE_SETS)
        self.assertRegex(job_id, r"^lllj_s003050_sets_gen_\d{8}_\d{6}$")

    def test_set_image_names_follow_the_run_sample_convention(self):
        generated = Path("/tmp/run/lllj_samples/generated")
        self.assertEqual(
            genjob.set_image_path(generated, "job1", 2, 3),
            generated / "job1_p2_3.png",
        )


class SetsJobTest(unittest.TestCase):
    def test_defaults_mark_a_job_as_single_image(self):
        job = genjob.new_job(
            {"prompt": "p", "steps": 12},
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint="/out/rein_s000100/rein.safetensors",
        )
        self.assertEqual(job["mode"], genjob.MODE_SINGLE)
        self.assertEqual(job["total_images"], 1)
        self.assertEqual(job["images_done"], 0)
        self.assertEqual(job["files"], [])

    def test_a_sets_job_carries_its_image_count_and_files(self):
        job = genjob.new_job(
            {},
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint="/out/rein_s000100/rein.safetensors",
            mode=genjob.MODE_SETS,
            total_images=5,
            extra={"sample_sets": [{"name": "a"}]},
        )
        self.assertEqual(job["mode"], genjob.MODE_SETS)
        self.assertEqual(job["total_images"], 5)
        self.assertEqual(job["sample_sets"], [{"name": "a"}])
        self.assertIn("_sets_gen_", job["id"])

    def test_no_counter_is_ever_null(self):
        """An explicit null where the client declares an Int fails its decode; a `sets` job has no
        per-image `steps`, so the record must still say `total_steps = 0`."""
        single = genjob.new_job(
            {"prompt": "p", "steps": 12},
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint="/out/rein_s000100/rein.safetensors",
        )
        self.assertEqual(single["total_steps"], 12)

        sets = genjob.new_job(
            {},
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint="/out/rein_s000100/rein.safetensors",
            mode=genjob.MODE_SETS,
            total_images=6,
        )
        for key in ("current_step", "total_steps", "images_done", "total_images"):
            self.assertIsInstance(sets[key], int)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            genjob.new_job(
                {},
                run_id="r",
                output_name="rein",
                checkpoint="/out/c.safetensors",
                mode="everything",
            )

    def test_a_job_file_without_a_mode_reads_as_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            generated = genjob.generated_dir(Path(tmp) / "rein_samples")
            generated.mkdir(parents=True)
            (generated / "old_gen_1.json").write_text(
                json.dumps({"id": "old_gen_1", "state": "done"}), encoding="utf-8"
            )
            self.assertEqual(genjob.list_jobs(generated)[0]["mode"], genjob.MODE_SINGLE)


    def test_a_jobs_id_and_set_names_carry_their_mode(self):
        self.assertRegex(
            genjob.new_job_id("lllj_s003050", mode=genjob.MODE_EVALUATE),
            r"^lllj_s003050_evaluate_gen_\d{8}_\d{6}$",
        )
        self.assertRegex(
            genjob.new_job_id("lllj_s003050", mode=genjob.MODE_BATCH),
            r"^lllj_s003050_batch_gen_\d{8}_\d{6}$",
        )


class BatchJobTest(unittest.TestCase):
    """The plan a step-range pass is started from."""

    def _job(self, **fields):
        return genjob.new_batch_job(
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoints=[{"path": "/out/rein_s00100/rein.safetensors", "step": 100, "dir": "rein_s00100"}],
            from_step=100,
            to_step=200,
            images_per_checkpoint=3,
            **fields,
        )

    def test_it_names_the_run_config_its_prompts_come_from(self):
        job = self._job(config_log_dir="/logs/rein_20260101_000000")
        self.assertEqual(job["config_log_dir"], "/logs/rein_20260101_000000")

    def test_a_batch_without_one_reads_as_empty(self):
        # An older record (or a hand-written spec) must still load: the runner then falls back to
        # today's config.toml rather than failing.
        self.assertEqual(self._job()["config_log_dir"], "")

    def test_it_records_the_prompt_sets_the_range_renders_with(self):
        # api.py resolves the run's prompts when it plans the batch, and the record has to carry
        # them: the runner must not derive them again from a config file that may have moved on.
        sets = [{"name": "planned", "prompt": "p", "width": 64, "height": 64, "steps": 1,
                 "guidance_scale": 4.0, "guidance_rescale": 0.5, "seed": 7, "negative": "n",
                 "repeat": 2}]
        self.assertEqual(self._job(sample_sets=sets)["sample_sets"], sets)

    def test_a_batch_without_recorded_sets_reads_as_empty(self):
        self.assertEqual(self._job()["sample_sets"], [])


class EvaluationJobTest(unittest.TestCase):
    """The record api.py writes before an evaluation: a plan plus the set of images to score."""

    def _job(self, *, plan, images=None, **fields):
        return genjob.new_evaluation_job(
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint="/out/rein_s000100/rein.safetensors",
            step=100,
            depth=fields.pop("depth", 7),
            threshold=fields.pop("threshold", 0.35),
            categories=fields.pop("categories", ["general"]),
            config_source="/logs/rein_20260101_000000/config.toml",
            config_log_dir="/logs/rein_20260101_000000",
            plan=plan,
            images=images if images is not None else [],
            sample_sets=fields.pop("sample_sets", [{"prompt": "a set", "repeat": 1}]),
            **fields,
        )

    def test_a_top_up_starts_in_the_rendering_phase(self):
        plan = {"k": 1, "needed": True, "render_total": 4, "sets": []}
        job = self._job(plan=plan)
        self.assertEqual(job["mode"], genjob.MODE_EVALUATE)
        self.assertEqual(job["phase"], genjob.PHASE_RENDERING)
        # The counters belong to the phase: images to render first, then the scored set.
        self.assertEqual((job["images_done"], job["total_images"]), (0, 4))
        self.assertEqual(job["depth"], 7)
        self.assertEqual(job["threshold"], 0.35)
        self.assertEqual(job["categories"], ["general"])
        self.assertEqual(job["config_source"], "/logs/rein_20260101_000000/config.toml")
        self.assertEqual(job["config_log_dir"], "/logs/rein_20260101_000000")
        self.assertEqual(job["plan"], plan)
        self.assertIsNone(job["scores"])
        self.assertEqual(job["step"], 100)

    def test_a_deep_enough_checkpoint_starts_in_the_tagging_phase(self):
        plan = {"k": 0, "needed": False, "render_total": 0, "sets": []}
        images = [{"path": "/x/a.png", "set_index": 0, "repeat_idx": 0} for _ in range(9)]
        job = self._job(plan=plan, images=images, depth=7)
        self.assertEqual(job["phase"], genjob.PHASE_TAGGING)
        self.assertEqual((job["images_done"], job["total_images"]), (0, 9))
        self.assertEqual(len(job["images"]), 9)

    def test_the_scored_tag_selection_is_recorded(self):
        plan = {"k": 0, "needed": False, "render_total": 0, "sets": []}
        job = self._job(plan=plan, tags=["anal", "pussy"])
        self.assertEqual(job["tags"], ["anal", "pussy"])
        # No selection at all is still a list: the reply's shape must not change with the request.
        self.assertEqual(self._job(plan=plan)["tags"], [])

    def test_no_counter_is_ever_null(self):
        job = self._job(plan={"k": 1, "needed": True, "render_total": 2, "sets": []})
        for key in ("current_step", "total_steps", "images_done", "total_images"):
            self.assertIsInstance(job[key], int, key)
        payload = json.loads(json.dumps(job))
        self.assertIsNone(payload["scores"])
        self.assertIsInstance(payload["images"], list)

    def test_listing_keeps_an_evaluation_a_mode_of_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            generated = genjob.generated_dir(Path(tmp) / "rein_samples")
            generated.mkdir(parents=True)
            job = self._job(plan={"k": 0, "needed": False, "render_total": 0, "sets": []})
            genjob.write_job(generated, job)
            stored = genjob.list_jobs(generated)[0]
            self.assertEqual(stored["mode"], genjob.MODE_EVALUATE)
            self.assertEqual(stored["phase"], genjob.PHASE_TAGGING)
            self.assertIn(stored["phase"], genjob.PHASES)


class RequestValidationTest(unittest.TestCase):
    defaults = {
        "prompt": "config prompt",
        "negative_prompt": "config negative",
        "cfg": 5.0,
        "steps": 35,
        "seed": 0,
        "width": 1152,
        "height": 768,
    }

    def test_defaults_fill_every_field(self):
        request = genjob.normalize_request({}, self.defaults)
        self.assertEqual(request["prompt"], "config prompt")
        self.assertEqual(request["negative_prompt"], "config negative")
        self.assertEqual(request["cfg"], 5.0)
        self.assertEqual(request["steps"], 35)
        self.assertEqual(request["seed"], 0)
        self.assertEqual((request["width"], request["height"]), (1152, 768))
        self.assertIsNone(request["step"])

    def test_explicit_values_win(self):
        request = genjob.normalize_request(
            {"prompt": " mine ", "cfg": "7.5", "steps": 12, "seed": "99", "step": "3050"},
            self.defaults,
        )
        self.assertEqual(request["prompt"], "mine")
        self.assertEqual(request["cfg"], 7.5)
        self.assertEqual(request["steps"], 12)
        self.assertEqual(request["seed"], 99)
        self.assertEqual(request["step"], 3050)

    def test_a_cleared_prompt_is_rejected_not_replaced(self):
        with self.assertRaises(ValueError) as ctx:
            genjob.normalize_request({"prompt": ""}, self.defaults)
        self.assertIn("prompt", str(ctx.exception))

    def test_an_explicit_null_falls_back_to_the_config(self):
        self.assertEqual(genjob.normalize_request({"cfg": None}, self.defaults)["cfg"], 5.0)

    def test_rejects_bad_values(self):
        cases = [
            ({"prompt": "   "}, "prompt"),
            ({"cfg": 0.5}, "cfg"),
            ({"cfg": 31}, "cfg"),
            ({"cfg": "high"}, "cfg"),
            ({"steps": 0}, "steps"),
            ({"steps": 151}, "steps"),
            ({"seed": -1}, "seed"),
            ({"seed": 2**32}, "seed"),
            ({"step": "abc"}, "step"),
            ({"width": 16}, "width"),
            ({"height": 99999}, "height"),
        ]
        for params, expected in cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError) as ctx:
                    genjob.normalize_request(params, self.defaults)
                self.assertIn(expected, str(ctx.exception))


class JobStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.generated = genjob.generated_dir(Path(self.tmp.name) / "rein_samples")

    def tearDown(self):
        self.tmp.cleanup()

    def _job(self, job_id: str, **fields):
        job = genjob.new_job(
            {
                "prompt": "p",
                "negative_prompt": "",
                "cfg": 5.0,
                "steps": 12,
                "seed": 7,
                "width": 1024,
                "height": 1024,
                "step": 100,
            },
            run_id="rein_20260101_000000",
            output_name="rein",
            checkpoint="/out/rein_s000100/rein.safetensors",
        )
        job.update(fields)
        job["id"] = job_id
        return job

    def test_round_trip_and_progress_updates(self):
        job = genjob.write_job(self.generated, self._job("a_gen_1"))
        self.assertEqual(job["state"], genjob.STATE_RUNNING)
        self.assertEqual(job["total_steps"], 12)

        genjob.update_job(self.generated, "a_gen_1", current_step=5)
        stored = json.loads(genjob.job_path(self.generated, "a_gen_1").read_text())
        self.assertEqual(stored["current_step"], 5)
        self.assertEqual(stored["prompt"], "p")
        self.assertIn("updated_at", stored)

    def test_listing_is_newest_first_and_tolerates_junk(self):
        genjob.write_job(self.generated, self._job("older", started_at=100.0))
        genjob.write_job(self.generated, self._job("newer", started_at=200.0))
        (self.generated / "broken.json").write_text("{not json")
        (self.generated / "notes.txt").write_text("ignore me")

        self.assertEqual([job["id"] for job in genjob.list_jobs(self.generated)], ["newer", "older"])
        self.assertEqual(genjob.running_job(self.generated)["id"], "newer")

    def test_a_finished_job_is_not_running(self):
        genjob.write_job(self.generated, self._job("done", state=genjob.STATE_DONE))
        genjob.write_job(self.generated, self._job("failed", state=genjob.STATE_ERROR, error="boom"))
        self.assertIsNone(genjob.running_job(self.generated))
        self.assertEqual(
            [job["state"] for job in genjob.list_jobs(self.generated)],
            ["error", "done"],
        )

    def test_missing_directory_lists_nothing(self):
        self.assertEqual(genjob.list_jobs(self.generated), [])
        self.assertIsNone(genjob.running_job(self.generated))

    def test_generated_dir_sits_inside_the_sample_dir(self):
        self.assertEqual(genjob.generated_dir("/out/run/rein_samples").name, "generated")
        self.assertEqual(genjob.generated_dir("/out/run/rein_samples").parent.name, "rein_samples")


if __name__ == "__main__":
    unittest.main()
