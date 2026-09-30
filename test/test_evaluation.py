"""The config snapshot a run keeps, and the evaluation built on top of it.

Both halves need no GPU and no torch: the snapshot is one file copy plus the read-back that decides
whether a run's own prompts are used, and the evaluation half is inventory, an expansion plan and
the two scoreboards over already-tagged images.
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

# `python test/test_evaluation.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import (
    RUN_CONFIG_FILENAME,
    TrainConfig,
    _load_toml_config,
    resolve_sample_sets,
    run_config_mapping,
    save_run_config,
    tracker_hparams,
)
from trainer.evaluation import (
    MAX_DEPTH,
    MAX_RENDER,
    ImageRef,
    collect_images,
    expansion_plan,
    images_from_payload,
    normalize_depth,
    normalize_tag,
    normalize_threshold,
    prompt_tags,
    render_slots,
    sample_name_parts,
    score_images,
)

SOURCE = """\
[environment]
output_name = "rein"
logging_dir = "/tmp/logs"

[validation]
sample_steps = 12
sample_repeat = 1
guidance_rescale = 0.6
sample_prompts = "flat prompt"

[[validation.samples]]
name = "old run"
prompt = "snapshot prompt"
steps = 12
repeat = 1
"""


class RunConfigSnapshotTest(unittest.TestCase):
    """`{logging_dir}/{run_id}/config.toml`: the file a run trained with, kept beside its logs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.logs = self.root / "logs"
        self.source = self.root / "repo-config.toml"
        self.source.write_text(SOURCE, encoding="utf-8")

    def test_save_copies_the_file_verbatim_into_the_run_log_dir(self):
        target = save_run_config(self.logs, "rein_20260911_120000", source=self.source)
        self.assertEqual(target, self.logs / "rein_20260911_120000" / RUN_CONFIG_FILENAME)
        self.assertEqual(target.read_bytes(), self.source.read_bytes())

    def test_save_creates_the_run_directory_when_the_run_wrote_no_logs_yet(self):
        target = save_run_config(self.logs, "rein_20260911_120000", source=self.source)
        self.assertTrue(target.parent.is_dir())

    def test_missing_source_warns_and_writes_nothing(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            target = save_run_config(self.logs, "rein_20260911_120000", source=self.root / "gone.toml")
        self.assertIsNone(target)
        self.assertIn("saves no config snapshot", stderr.getvalue())
        self.assertFalse(self.logs.exists())

    def test_the_run_snapshot_wins_over_the_repo_file(self):
        log_dir = self.logs / "rein_20260911_120000"
        save_run_config(self.logs, "rein_20260911_120000", source=self.source)

        mapping, source = run_config_mapping(log_dir)
        self.assertEqual(source, str((log_dir / RUN_CONFIG_FILENAME).resolve()))
        self.assertEqual(mapping["sample_steps"], 12)
        self.assertEqual([s["prompt"] for s in mapping["samples"]], ["snapshot prompt"])
        self.assertEqual([s.prompt for s in resolve_sample_sets(mapping)], ["snapshot prompt"])

    def test_a_run_without_a_snapshot_falls_back_to_the_repo_config(self):
        mapping, source = run_config_mapping(self.logs / "rein_20200101_000000")
        self.assertEqual(source, str(Path("config.toml").resolve()))
        self.assertEqual(mapping, _load_toml_config("config.toml"))
        # The fallback is the resolver's own path, so both spell the same prompts.
        self.assertEqual(
            [s.prompt for s in resolve_sample_sets(mapping)],
            [s.prompt for s in resolve_sample_sets(_load_toml_config("config.toml"))],
        )


class RunHparamsFallbackTest(unittest.TestCase):
    """A run from before the `config.toml` copies still has its own config in its TensorBoard logs.

    `accelerator.init_trackers` records the whole config as it starts (`tracker_hparams` into an
    `add_hparams` subdirectory), which is what the evaluation reads back here: same prompts, same
    sampling values, and usable types.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logs = Path(self.tmp.name) / "logs"
        self.run_id = "rein_20260911_120000"
        self.log_dir = self.logs / self.run_id
        self.log_dir.mkdir(parents=True)

    def record_hparams(self, *, samples=None, steps=17, output_name="rein"):
        """Exactly what the trainer does at startup: `add_hparams(tracker_hparams(cfg), {})`."""
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(self.log_dir))
        try:
            writer.add_hparams(
                tracker_hparams(
                    {
                        "output_name": output_name,
                        "sample_steps": steps,
                        "sample_repeat": 1,
                        "samples": samples
                        if samples is not None
                        else [{"name": "one", "prompt": "recorded prompt", "steps": 17, "repeat": 2}],
                    }
                ),
                {},
            )
        finally:
            writer.close()

    def test_the_runs_own_hparams_supply_the_prompts(self):
        self.record_hparams()
        mapping, source = run_config_mapping(self.log_dir)

        self.assertTrue(source.startswith(str(self.log_dir)), source)
        self.assertIn("events.out.tfevents.", source)
        self.assertEqual(mapping["output_name"], "rein")
        # Numbers come back as numbers: `sample_steps` as an int, and `repeat` usable as a range.
        self.assertEqual(mapping["sample_steps"], 17)
        self.assertIsInstance(mapping["sample_steps"], int)
        sets = resolve_sample_sets(mapping)
        self.assertEqual([s.prompt for s in sets], ["recorded prompt"])
        self.assertEqual([s.repeat for s in sets], [2])
        self.assertEqual(list(range(sets[0].repeat)), [0, 1])

    def test_the_saved_snapshot_still_wins_over_the_hparams(self):
        self.record_hparams()
        save_run_config(
            self.logs,
            self.run_id,
            source=self._write_source('[validation]\nsample_prompts = "snapshot prompt"\n'),
        )
        mapping, source = run_config_mapping(self.log_dir)
        self.assertEqual(source, str((self.log_dir / RUN_CONFIG_FILENAME).resolve()))
        self.assertEqual([s.prompt for s in resolve_sample_sets(mapping)], ["snapshot prompt"])

    def test_a_run_with_neither_still_falls_back_to_the_repo_config(self):
        mapping, source = run_config_mapping(self.log_dir)
        self.assertEqual(source, str(Path("config.toml").resolve()))
        self.assertEqual(mapping, _load_toml_config("config.toml"))

    def test_an_unreadable_event_file_is_not_an_error(self):
        (self.log_dir / "1790780085.6183703").mkdir()
        (self.log_dir / "1790780085.6183703" / "events.out.tfevents.1790780085.host.1.0").write_bytes(
            b"\x00\x01  torn"
        )
        mapping, source = run_config_mapping(self.log_dir)
        self.assertEqual(source, str(Path("config.toml").resolve()))
        self.assertTrue(mapping)

    def test_a_run_that_recorded_no_samples_keeps_its_scalars(self):
        # A run for which the tracker recorded the config but the list-valued key did not survive:
        # the `sample_*` scalars it did record are still its own, not today's.
        self.record_hparams(samples=[], steps=23)
        mapping, _source = run_config_mapping(self.log_dir)
        sets = resolve_sample_sets(mapping)
        self.assertEqual([s.steps for s in sets], [23])

    def _write_source(self, text: str) -> Path:
        source = Path(self.tmp.name) / "source.toml"
        source.write_text(text, encoding="utf-8")
        return source


class TrainConfigFromMappingTest(unittest.TestCase):
    """A config built from a mapping, for rendering a run's own samples with its own values."""

    def test_mapping_values_win_over_the_repo_file(self):
        mapping = {
            "epoch": 3,
            "mixed_precision": "fp16",
            "guidance_rescale": 0.4,
            "min_snr_gamma": 7.5,
            "samples": [{"prompt": "snapshot prompt", "repeat": 2, "steps": 9}],
        }
        cfg = TrainConfig.from_mapping(mapping)
        self.assertEqual(cfg.epoch, 3)
        self.assertEqual(cfg.mixed_precision, "fp16")
        self.assertEqual(cfg.guidance_rescale, 0.4)
        self.assertEqual([entry["prompt"] for entry in cfg.samples], ["snapshot prompt"])
        self.assertEqual([s.prompt for s in resolve_sample_sets(cfg)], ["snapshot prompt"])

    def test_keys_the_config_does_not_declare_are_ignored(self):
        cfg = TrainConfig.from_mapping({"bookkeeping": "nonsense", "epoch": 2, "_current_epoch": 99})
        self.assertEqual(cfg.epoch, 2)
        self.assertEqual(cfg._current_epoch, 0)

    def test_an_empty_mapping_is_the_same_config_as_the_repo_file(self):
        plain = TrainConfig()
        mapped = TrainConfig.from_mapping({})
        self.assertEqual(mapped.epoch, plain.epoch)
        self.assertEqual(mapped.pretrained_model_name_or_path, plain.pretrained_model_name_or_path)
        self.assertEqual(mapped.prediction_type, plain.prediction_type)

    def test_derived_values_are_still_derived(self):
        # `prediction_type` / `zero_terminal_snr` come from the base, never from the file, so a
        # mapping cannot smuggle a disagreeing pair in.
        cfg = TrainConfig.from_mapping({"prediction_type": "epsilon", "zero_terminal_snr": True})
        self.assertEqual(cfg.prediction_type, TrainConfig().prediction_type)
        self.assertEqual(cfg.zero_terminal_snr, TrainConfig().zero_terminal_snr)

    def test_an_unusable_output_name_still_fails_at_construction(self):
        with self.assertRaises(ValueError):
            TrainConfig.from_mapping({"output_name": "not a name"})


class SampleSet:
    """The shape `resolve_sample_sets` yields, without importing the dataclass."""

    def __init__(self, prompt: str, repeat: int):
        self.prompt = prompt
        self.repeat = repeat


def table(prompt: str, repeat: int) -> dict:
    return {"prompt": prompt, "repeat": repeat}


class PromptTagsTest(unittest.TestCase):
    def test_a_comma_list_loses_its_noise(self):
        self.assertEqual(
            prompt_tags("1girl, solo, , Long_Hair , 1girl"),
            ["1girl", "solo", "long hair"],
        )

    def test_weights_and_wrappers_come_off(self):
        self.assertEqual(normalize_tag("(anal:1.2)"), "anal")
        self.assertEqual(normalize_tag("(anal : 1.2)"), "anal")
        self.assertEqual(normalize_tag("(anal)"), "anal")
        self.assertEqual(normalize_tag("[long hair]"), "long hair")
        self.assertEqual(normalize_tag("{{wide shot}}"), "wide shot")
        self.assertEqual(normalize_tag("(zoom:0.8)"), "zoom")

    def test_a_parenthesised_danbooru_suffix_is_part_of_the_tag(self):
        # `tagger2.tag_text` turns `masking_tape_(medium)` into `masking tape (medium)`; the two
        # sides have to agree, so the wrapper strip must not eat that tail.
        self.assertEqual(normalize_tag("masking_tape_(medium)"), "masking tape (medium)")
        self.assertEqual(normalize_tag("masking tape (medium)"), "masking tape (medium)")

    def test_a_colon_that_is_not_a_weight_stays(self):
        self.assertEqual(normalize_tag("dark:blue"), "dark:blue")

    def test_nothing_becomes_nothing(self):
        self.assertEqual(prompt_tags(None), [])
        self.assertEqual(prompt_tags("  ,  , "), [])
        self.assertEqual(prompt_tags("( :1.2 )"), [])


class SampleNamePartsTest(unittest.TestCase):
    def test_the_current_naming_carries_step_set_and_repeat(self):
        self.assertEqual(sample_name_parts("rein_000150_p2_3.png"), (150, 2, 3))

    def test_the_older_two_number_naming_is_set_zero(self):
        self.assertEqual(sample_name_parts("rein_000150_3.png"), (150, 0, 3))

    def test_a_name_without_a_step_carries_none(self):
        # A foreign or hand-written name inside a samples directory is left alone, not guessed at.
        self.assertIsNone(sample_name_parts("rein_000150_p2.3.png"))
        self.assertIsNone(sample_name_parts("cover.png"))


class CollectImagesTest(unittest.TestCase):
    """The checkpoint's images: the run's own sample points, plus the recorded generations."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = self.root / "rein_samples"
        self.generated = self.samples / "generated"
        self.generated.mkdir(parents=True)
        self.checkpoint = str(self.root / "rein_s000150" / "rein.safetensors")
        self.other = str(self.root / "rein_s000200" / "rein.safetensors")
        self.sets = [SampleSet("first prompt", 2), SampleSet("second prompt", 1)]
        self._write_samples()
        self._write_job(
            "a_sets_gen_1.json",
            checkpoint=self.checkpoint,
            files=[self.generated / "a_p0_0.png", self.generated / "a_p0_1.png"],
            sample_sets=[table("recorded prompt", 2), table("second prompt", 1)],
        )
        self._write_job(
            "b_sets_gen_1.json",
            checkpoint=self.other,
            files=[self.generated / "b_p0_0.png"],
            sample_sets=[table("other prompt", 1)],
        )
        self._write_job(
            "c_sets_gen_1.json",
            checkpoint=self.checkpoint,
            files=[self.generated / "c_p0_0.png"],
            sample_sets=[table("recorded prompt", 1)],
            state="running",
        )
        self._write_job(
            "d_single_gen_1.json",
            checkpoint=self.checkpoint,
            image_path=self.generated / "d.png",
            prompt="ad hoc prompt",
            state="cancelled",
            mode="single",
        )

    def _write_samples(self):
        for name in (
            "rein_000150_p0_0.png",
            "rein_000150_p1_1.png",
            "rein_000150_0.png",
            "rein_000200_p0_0.png",
            "rein_nostep.png",
            "rein_000150_p0_0.mask.png",
        ):
            (self.samples / name).write_bytes(b"png")

    def _write_job(self, name, *, state="done", **fields):
        payload = {"id": name.removesuffix(".json"), "state": state, "mode": "sets", **fields}
        if "image_path" in payload:
            payload["image_path"] = str(payload["image_path"])
        if "files" in payload:
            payload["files"] = [str(item) for item in payload["files"]]
        (self.generated / name).write_text(json.dumps(payload), encoding="utf-8")

    def collect(self, step=150):
        return collect_images(
            samples_dir=self.samples,
            generated_dir=self.generated,
            checkpoint=self.checkpoint,
            step=step,
            sets=self.sets,
        )

    def test_only_this_steps_samples_are_read(self):
        names = [image.name for image in self.collect()]
        self.assertNotIn("rein_000200_p0_0.png", names)
        self.assertNotIn("rein_nostep.png", names)
        self.assertNotIn("rein_000150_p0_0.mask.png", names)

    def test_run_samples_carry_the_prompt_of_their_own_set(self):
        by_name = {image.name: image for image in self.collect()}
        self.assertEqual(by_name["rein_000150_p0_0.png"].prompt, "first prompt")
        self.assertEqual(by_name["rein_000150_p1_1.png"].prompt, "second prompt")
        # The older naming is set 0.
        self.assertEqual(by_name["rein_000150_0.png"].prompt, "first prompt")
        self.assertEqual(by_name["rein_000150_0.png"].source, "run")

    def test_a_recorded_pass_supplies_its_own_prompts(self):
        by_name = {image.name: image for image in self.collect()}
        self.assertEqual(by_name["a_p0_0.png"].prompt, "recorded prompt")
        self.assertEqual(by_name["a_p0_1.png"].prompt, "recorded prompt")
        self.assertEqual(by_name["a_p0_1.png"].source, "generated")
        self.assertEqual((by_name["a_p0_1.png"].set_index, by_name["a_p0_1.png"].repeat_idx), (0, 1))

    def test_another_checkpoints_job_and_a_running_one_are_left_out(self):
        names = [image.name for image in self.collect()]
        self.assertNotIn("b_p0_0.png", names)
        self.assertNotIn("c_p0_0.png", names)

    def test_a_single_generation_counts_with_its_own_prompt(self):
        by_name = {image.name: image for image in self.collect()}
        self.assertEqual(by_name["d.png"].prompt, "ad hoc prompt")
        self.assertEqual(by_name["d.png"].set_index, -1)
        # A cancelled pass keeps the images it wrote, exactly as its card shows them.
        self.assertEqual(by_name["d.png"].source, "generated")

    def test_images_are_listed_run_first_then_by_slot(self):
        images = self.collect()
        self.assertEqual(
            [image.name for image in images],
            [
                "rein_000150_0.png",
                "rein_000150_p0_0.png",
                "rein_000150_p1_1.png",
                "a_p0_0.png",
                "a_p0_1.png",
                "d.png",
            ],
        )

    def test_a_job_without_recorded_sets_falls_back_to_the_config(self):
        self._write_job(
            "e_sets_gen_1.json",
            checkpoint=self.checkpoint,
            files=[self.generated / "e_p1_0.png"],
        )
        by_name = {image.name: image for image in self.collect()}
        self.assertEqual(by_name["e_p1_0.png"].prompt, "second prompt")

    def test_an_unknown_step_keeps_the_generations_and_drops_the_samples(self):
        images = self.collect(step=None)
        self.assertEqual({image.source for image in images}, {"generated"})
        self.assertIn("a_p0_0.png", [image.name for image in images])

    def test_a_sample_of_a_set_the_config_no_longer_has_has_no_prompt(self):
        images = collect_images(
            samples_dir=self.samples,
            generated_dir=self.generated,
            checkpoint=self.checkpoint,
            step=150,
            sets=[SampleSet("only prompt", 1)],
        )
        by_name = {image.name: image for image in images}
        self.assertEqual(by_name["rein_000150_p1_1.png"].prompt, "")
        self.assertEqual(by_name["rein_000150_p0_0.png"].prompt, "only prompt")

    def test_a_missing_directory_is_not_an_error(self):
        images = collect_images(
            samples_dir=self.root / "nope",
            generated_dir=self.root / "nope",
            checkpoint=self.checkpoint,
            step=150,
            sets=self.sets,
        )
        self.assertEqual(images, [])


class ExpansionPlanTest(unittest.TestCase):
    """Depth is a floor, and the top-up is whole passes of the config: enough of them to reach it."""

    def setUp(self):
        self.sets = [SampleSet("a", 3), SampleSet("b", 2), SampleSet("c", 2)]  # 7 per pass

    def image(self, set_index, repeat_idx, source="run"):
        return ImageRef(
            path=f"/x/img_{set_index}_{repeat_idx}.png",
            set_index=set_index,
            repeat_idx=repeat_idx,
            source=source,
            prompt="a",
        )

    def test_an_already_deep_enough_checkpoint_renders_nothing(self):
        images = [self.image(index % 3, index) for index in range(20)]
        plan = expansion_plan(self.sets, 20, images)
        self.assertFalse(plan["needed"])
        self.assertEqual(plan["passes"], 0)
        self.assertEqual(plan["render_total"], 0)
        self.assertEqual(render_slots(plan), [])
        self.assertEqual(plan["existing_images"], 20)
        # One image short is a shortfall again, and that costs one whole pass.
        plan = expansion_plan(self.sets, 21, images)
        self.assertTrue(plan["needed"])
        self.assertEqual(plan["passes"], 1)
        self.assertEqual(plan["render_total"], 7)

    def test_the_top_up_is_as_many_whole_passes_as_the_shortfall_needs(self):
        # 21 images already there (three `sets` passes over the same 7 slots), Depth 50: the
        # shortfall is 29, one pass is 7, so five passes (35 images) reach it — 56 in all.
        images = [self.image(index % 3, index) for index in range(20)] + [self.image(0, 20)]
        plan = expansion_plan(self.sets, 50, images)
        self.assertEqual(plan["passes"], 5)
        self.assertEqual(plan["render_total"], 35)
        self.assertEqual(plan["existing_images"], 21)
        self.assertEqual(plan["per_pass"], 7)
        self.assertEqual(21 + plan["render_total"], 56)

    def test_every_set_renders_its_own_repeat_per_pass(self):
        plan = expansion_plan(self.sets, 15, [])
        self.assertEqual(plan["passes"], 3)  # ceil(15 / 7)
        self.assertEqual([len(entry["render"]) for entry in plan["sets"]], [9, 6, 6])
        self.assertEqual(plan["sets"][0]["render"][:3], [0, 1, 2])
        self.assertEqual(plan["sets"][2]["render"][-1], 5)

    def test_a_new_pass_continues_each_sets_own_numbering(self):
        # Two images at slots 0 and 1 of set 0, one slot 4 of set 2: the new pass must not write
        # over any of them, so it starts after the highest index each set already uses.
        images = [self.image(0, 0), self.image(0, 1), self.image(2, 4)]
        plan = expansion_plan(self.sets, 10, images)
        self.assertEqual(plan["passes"], 1)  # ceil(7 / 7)
        self.assertEqual(plan["sets"][0]["existing"], [0, 1])
        self.assertEqual(plan["sets"][0]["render"], [2, 3, 4])
        self.assertEqual(plan["sets"][2]["render"], [5, 6])
        self.assertEqual(plan["sets"][1]["render"], [0, 1])

    def test_a_depth_below_one_pass_still_renders_one_pass(self):
        plan = expansion_plan(self.sets, 1, [])
        self.assertEqual(plan["passes"], 1)
        self.assertEqual(plan["render_total"], 7)

    def test_an_oversized_job_is_refused_with_the_numbers(self):
        sets = [SampleSet("a", 1), SampleSet("b", 1), SampleSet("c", 1)]
        plan = expansion_plan(sets, 510, [])  # 170 passes x 3 = 510
        self.assertEqual(plan["render_total"], 510)
        with self.assertRaises(ValueError) as caught:
            expansion_plan(sets, MAX_DEPTH, [])
        self.assertIn(str(MAX_RENDER), str(caught.exception))

    def test_a_config_without_images_is_refused(self):
        with self.assertRaises(ValueError):
            expansion_plan([], 10, [])

    def test_the_set_share_comes_from_the_config(self):
        plan = expansion_plan([SampleSet("a", 1)], 12, [])
        self.assertEqual(plan["passes"], 12)
        self.assertEqual(plan["render_total"], 12)


class NormalizeDepthTest(unittest.TestCase):
    def test_a_number_in_range_comes_back_as_an_int(self):
        self.assertEqual(normalize_depth(12), 12)
        self.assertEqual(normalize_depth("12"), 12)
        self.assertEqual(normalize_depth(1), 1)
        self.assertEqual(normalize_depth(MAX_DEPTH), MAX_DEPTH)

    def test_anything_else_is_refused(self):
        for value in (0, -1, MAX_DEPTH + 1, 12.5, "twelve", None, True, ""):
            with self.assertRaises(ValueError, msg=str(value)):
                normalize_depth(value)


class NormalizeThresholdTest(unittest.TestCase):
    """The tagger floor, checked before an evaluation pays for a GPU."""

    def test_a_number_in_range_comes_back_as_a_float(self):
        self.assertEqual(normalize_threshold(0.35), 0.35)
        self.assertEqual(normalize_threshold("0.2"), 0.2)
        self.assertEqual(normalize_threshold(0), 0.0)
        self.assertEqual(normalize_threshold(1), 1.0)

    def test_anything_else_is_refused(self):
        for value in (-0.1, 1.5, "high", None, True, [0.5]):
            with self.assertRaises(ValueError, msg=str(value)):
                normalize_threshold(value)


class ScoreImagesTest(unittest.TestCase):
    def image(self, prompt, tags, source="run", error=None):
        return ImageRef(
            path=f"/x/{prompt}-{len(tags)}.png",
            set_index=0,
            repeat_idx=0,
            source=source,
            prompt=prompt,
            tags=list(tags),
            error=error,
        )

    def test_micro_counts_every_image_and_union_pools_them(self):
        images = [
            self.image("1girl, blue eyes", ["1girl", "solo", "blue eyes"]),
            self.image("1girl, blue eyes", ["1girl"]),
        ]
        scores = score_images(images)
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (3, 1, 1))
        self.assertEqual(scores["precision"], 0.75)
        self.assertEqual(scores["recall"], 0.75)
        self.assertEqual(scores["f1"], 0.75)
        # Pooled, `blue eyes` was found on one of the two images.
        self.assertEqual((scores["union_tp"], scores["union_fp"], scores["union_fn"]), (2, 1, 0))
        self.assertEqual(scores["union_recall"], 1.0)
        self.assertEqual(scores["union_f1"], 0.8)
        self.assertEqual(scores["images_scored"], 2)
        self.assertEqual(scores["top_false_positives"], [{"tag": "solo", "count": 1}])
        self.assertEqual(scores["top_false_negatives"], [{"tag": "blue eyes", "count": 1}])

    def test_one_prompt_is_one_group(self):
        images = [
            self.image("1girl", ["1girl"]),
            self.image("1girl", ["1girl", "outdoors"]),
            self.image("a castle", ["castle"]),
        ]
        scores = score_images(images)
        self.assertEqual([row["prompt"] for row in scores["groups"]], ["1girl", "a castle"])
        first, second = scores["groups"]
        self.assertEqual((first["images"], first["tp"], first["fp"], first["fn"]), (2, 2, 1, 0))
        self.assertEqual(first["f1"], 0.8)
        self.assertEqual(first["union_tp"], 1)
        self.assertEqual(first["union_fn"], 0)
        self.assertEqual((second["images"], second["f1"]), (1, 0.0))
        self.assertEqual(second["union_fn"], 1)

    def test_a_failed_image_is_counted_and_not_scored(self):
        images = [
            self.image("1girl", ["1girl"]),
            self.image("1girl", ["1girl"], error="RuntimeError: boom"),
        ]
        scores = score_images(images)
        self.assertEqual(scores["images_scored"], 1)
        self.assertEqual(scores["images_failed"], 1)
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (1, 0, 0))

    def test_an_image_without_a_prompt_is_counted_and_not_scored(self):
        images = [self.image("", ["1girl"]), self.image("1girl", ["1girl"])]
        scores = score_images(images)
        self.assertEqual(scores["images_skipped"], 1)
        self.assertEqual(scores["fp"], 0)

    def test_nothing_found_scores_zero_without_dividing_by_zero(self):
        scores = score_images([self.image("1girl, solo", [])])
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (0, 0, 2))
        self.assertEqual(scores["precision"], 0.0)
        self.assertEqual(scores["recall"], 0.0)
        self.assertEqual(scores["f1"], 0.0)

    def test_the_tagger_side_is_normalized_the_same_way(self):
        images = [self.image("(anal:1.2), Long_Hair", ["anal", "long hair"])]
        scores = score_images(images)
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (2, 0, 0))
        self.assertEqual(scores["f1"], 1.0)

    def test_an_unrelated_tag_is_a_false_positive_on_every_image(self):
        images = [self.image("1girl", ["1girl", "solo"]) for _ in range(3)]
        scores = score_images(images)
        self.assertEqual(scores["top_false_positives"], [{"tag": "solo", "count": 3}])
        # The union board counts it once, however many images showed it.
        self.assertEqual(scores["union_fp"], 1)

    def test_the_offender_list_is_capped_but_ordered_by_count(self):
        extra = [f"tag {index}" for index in range(25)]
        scores = score_images([self.image("wanted", ["wanted"] + extra)])
        self.assertEqual(len(scores["top_false_positives"]), 20)
        self.assertEqual(scores["top_false_positives"][0]["tag"], "tag 0")
        self.assertEqual(scores["fp"], 25)

    def test_no_images_at_all_is_an_empty_report(self):
        scores = score_images([])
        self.assertEqual(scores["images_scored"], 0)
        self.assertEqual(scores["f1"], 0.0)
        self.assertEqual(scores["groups"], [])


class ImageRefRecordTest(unittest.TestCase):
    """The record on disk is the boundary between api.py and the runner."""

    def test_round_trip(self):
        image = ImageRef(
            path="/out/rein_000150_p1_2.png",
            set_index=1,
            repeat_idx=2,
            source="generated",
            prompt="a prompt",
            tags=["1girl", "solo"],
        )
        payload = image.to_dict()
        self.assertEqual(payload["name"], "rein_000150_p1_2.png")
        self.assertEqual(payload["set_index"], 1)
        self.assertEqual(payload["tags"], ["1girl", "solo"])
        self.assertIsNone(payload["error"])
        self.assertEqual(ImageRef.from_dict(payload), image)

    def test_a_partial_record_reads_leniently(self):
        image = ImageRef.from_dict({"path": "/out/a.png"})
        self.assertEqual(image.name, "a.png")
        self.assertEqual(image.set_index, -1)
        self.assertEqual(image.repeat_idx, -1)
        self.assertEqual(image.source, "run")
        self.assertEqual(image.tags, [])
        self.assertIsNone(image.error)

    def test_a_record_of_images_that_are_not_tables_is_skipped(self):
        self.assertEqual(images_from_payload([{"path": "/a.png"}, "nonsense", None]), [ImageRef(path="/a.png")])
        self.assertEqual(images_from_payload(None), [])


class ScoreRecordShapeTest(unittest.TestCase):
    """The numbers the Dashboard decodes: ints and floats, never null."""

    def test_every_counter_is_a_number(self):
        scores = score_images(
            [
                ImageRef(path="/x/a.png", prompt="1girl", tags=["1girl", "solo"]),
                ImageRef(path="/x/b.png", prompt="", tags=[]),
            ]
        )
        for key in (
            "tp",
            "fp",
            "fn",
            "union_tp",
            "union_fp",
            "union_fn",
            "images_scored",
            "images_failed",
            "images_skipped",
        ):
            self.assertIsInstance(scores[key], int, key)
        for key in ("precision", "recall", "f1", "union_precision", "union_recall", "union_f1"):
            self.assertIsInstance(scores[key], float, key)

    def test_the_payload_is_json_safe(self):
        scores = score_images([ImageRef(path="/x/a.png", prompt="1girl", tags=["1girl"])])
        self.assertEqual(json.loads(json.dumps(scores)), scores)


if __name__ == "__main__":
    unittest.main()
