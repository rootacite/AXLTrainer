import json
import os
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import sys

# `python test/test_runs.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.runs import (
    chart_view_path,
    create_run_dirs,
    default_chart_view,
    find_latest_run,
    find_samples_dir,
    list_runs,
    make_run_id,
    read_chart_view,
    read_steps_per_epoch,
    run_id_re,
    run_output_name,
    safe_name,
    steps_per_epoch_path,
    validate_output_name,
    write_chart_view,
    write_steps_per_epoch,
)


class RunIdTest(unittest.TestCase):
    def test_make_run_id_format(self):
        run_id = make_run_id("rein", now=datetime(2026, 9, 11, 12, 30, 45))
        self.assertEqual(run_id, "rein_20260911_123045")
        self.assertTrue(run_id_re("rein").match(run_id))

    def test_make_run_id_accepts_timestamp(self):
        stamp = datetime(2026, 1, 2, 3, 4, 5).timestamp()
        self.assertEqual(make_run_id("rein", now=stamp), "rein_20260102_030405")

    def test_make_run_id_sanitizes_name(self):
        self.assertEqual(safe_name("re in/x"), "re_in_x")
        self.assertEqual(
            make_run_id("re in/x", now=datetime(2026, 9, 11, 12, 0, 0)),
            "re_in_x_20260911_120000",
        )

    def test_run_id_re_ignores_checkpoint_and_other_dirs(self):
        pattern = run_id_re("rein")
        self.assertTrue(pattern.match("rein_20260911_120000"))
        self.assertTrue(pattern.match("rein_20260911_120000_2"))
        self.assertIsNone(pattern.match("rein_s000100"))
        self.assertIsNone(pattern.match("rein_final"))
        self.assertIsNone(pattern.match("reinfo_20260911_120000"))
        self.assertIsNone(pattern.match("rein_2026_09_11"))


class CreateRunDirsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        self.logs = Path(self.tmp.name) / "logs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_both_directories(self):
        run_id = create_run_dirs(
            self.out, self.logs, "rein", now=datetime(2026, 9, 11, 12, 0, 0)
        )
        self.assertEqual(run_id, "rein_20260911_120000")
        self.assertTrue((self.out / run_id).is_dir())
        self.assertTrue((self.logs / run_id).is_dir())

    def test_collision_gets_suffix(self):
        first = create_run_dirs(
            self.out, self.logs, "rein", now=datetime(2026, 9, 11, 12, 0, 0)
        )
        second = create_run_dirs(
            self.out, self.logs, "rein", now=datetime(2026, 9, 11, 12, 0, 0)
        )
        self.assertEqual(first, "rein_20260911_120000")
        self.assertEqual(second, "rein_20260911_120000_2")
        self.assertTrue((self.logs / second).is_dir())

    def test_collision_in_log_dir_only(self):
        (self.logs / "rein_20260911_120000").mkdir(parents=True)
        run_id = create_run_dirs(
            self.out, self.logs, "rein", now=datetime(2026, 9, 11, 12, 0, 0)
        )
        self.assertEqual(run_id, "rein_20260911_120000_2")


class FindLatestRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logs = Path(self.tmp.name) / "logs"
        self.logs.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_logging_dir(self):
        self.assertIsNone(find_latest_run(Path(self.tmp.name) / "nope", "rein"))

    def test_picks_newest_run(self):
        older = self.logs / "rein_20260101_000000"
        newer = self.logs / "rein_20260911_120000"
        older.mkdir()
        newer.mkdir()
        self.assertEqual(find_latest_run(self.logs, "rein"), "rein_20260911_120000")

    def test_ignores_legacy_dirs_and_files(self):
        (self.logs / "rein").mkdir()
        (self.logs / "rein_s000100").mkdir()
        (self.logs / "rein_20260911_120000").write_text("not a dir")
        self.assertIsNone(find_latest_run(self.logs, "rein"))


class ListRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        self.logs = Path(self.tmp.name) / "logs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_reports_output_and_log_presence(self):
        import os as _os

        both = self.out / "rein_20260911_120000"
        both.mkdir(parents=True)
        (both / "rein_final").mkdir()
        (both / "rein_final" / "rein.safetensors").write_bytes(b"0123456789")
        (self.logs / "rein_20260911_120000").mkdir(parents=True)
        log_only = self.logs / "rein_20260910_120000"
        log_only.mkdir(parents=True)
        # list_runs reports most recent activity first.
        _os.utime(log_only, (1_700_000_000, 1_700_000_000))
        _os.utime(both, (1_700_000_100, 1_700_000_100))

        runs = list_runs(self.out, self.logs, "rein")
        self.assertEqual([run["run_id"] for run in runs], ["rein_20260911_120000", "rein_20260910_120000"])
        first, second = runs
        self.assertTrue(first["has_output"])
        self.assertTrue(first["has_log"])
        self.assertEqual(first["size_bytes"], 10)
        self.assertFalse(second["has_output"])
        self.assertTrue(second["has_log"])
        self.assertTrue(re.match(r"rein_\d{8}_\d{6}$", second["run_id"]))

    def test_scoped_listing_reports_the_name(self):
        (self.out / "rein_20260911_120000").mkdir(parents=True)
        runs = list_runs(self.out, self.logs, "rein")
        self.assertEqual(runs[0]["output_name"], "rein")


class RunOutputNameTest(unittest.TestCase):
    def test_strips_the_stamp(self):
        self.assertEqual(run_output_name("rein_20260911_120000"), "rein")

    def test_strips_a_collision_suffix(self):
        self.assertEqual(run_output_name("rein_20260911_120000_2"), "rein")

    def test_a_name_with_a_stamp_inside_keeps_it(self):
        self.assertEqual(run_output_name("rein_20260911_120000_20260912_130000"), "rein_20260911_120000")

    def test_underscores_survive(self):
        self.assertEqual(run_output_name("towa_2_20260911_120000"), "towa_2")

    def test_non_run_names_are_empty(self):
        self.assertEqual(run_output_name("rein_s000100"), "")
        self.assertEqual(run_output_name("rein"), "")
        self.assertEqual(run_output_name(""), "")


class ValidateOutputNameTest(unittest.TestCase):
    def test_plain_names_pass(self):
        for name in ("rein", "towa_2", "babara-v2", "a.b", "月子"):
            self.assertIsNone(validate_output_name(name), name)

    def test_spaces_are_refused(self):
        for name in ("re in", " rein", "rein ", "rein\tv2"):
            self.assertIn("must be letters, digits", validate_output_name(name) or "")

    def test_slashes_and_other_characters_are_refused(self):
        for name in ("re/in", "re\\in", "rein:v2", "rein·v2", 're"in'):
            self.assertIsNotNone(validate_output_name(name), name)

    def test_empty_is_refused(self):
        self.assertEqual(validate_output_name(""), "output_name is empty")
        self.assertEqual(validate_output_name("   "), "output_name is empty")
        self.assertEqual(validate_output_name(None), "output_name is empty")


class FindSamplesDirTest(unittest.TestCase):
    """Artifact dirs use the raw name while a run id carries the sanitized one."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "re_in_20260911_120000"
        self.run.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefers_the_named_directory(self):
        (self.run / "re in_samples").mkdir()
        self.assertEqual(find_samples_dir(self.run, "re in"), self.run / "re in_samples")

    def test_falls_back_to_the_only_sample_directory(self):
        (self.run / "re in_samples").mkdir()
        self.assertEqual(find_samples_dir(self.run, "re_in"), self.run / "re in_samples")

    def test_missing_directory_is_still_named(self):
        self.assertEqual(find_samples_dir(self.run, "rein"), self.run / "rein_samples")
        self.assertFalse(find_samples_dir(Path(self.tmp.name) / "nope", "rein").is_dir())


class ListAllRunsTest(unittest.TestCase):
    """`list_runs(..., output_name=None)` backs the dashboard's run history list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        self.logs = Path(self.tmp.name) / "logs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_lists_every_output_name_with_its_own(self):
        (self.out / "konomi_20260910_120000").mkdir(parents=True)
        (self.out / "towa_20260911_120000").mkdir(parents=True)
        runs = list_runs(self.out, self.logs, None)
        self.assertEqual(
            {run["run_id"]: run["output_name"] for run in runs},
            {"konomi_20260910_120000": "konomi", "towa_20260911_120000": "towa"},
        )

    def test_ignores_flat_legacy_and_plain_dirs(self):
        for name in ("rein", "rein_s000100", "rein_samples", "random"):
            (self.out / name).mkdir(parents=True)
        self.assertEqual(list_runs(self.out, self.logs, None), [])

    def test_summarises_samples_steps_and_checkpoints(self):
        run = self.out / "rein_20260911_120000"
        samples = run / "rein_samples"
        samples.mkdir(parents=True)
        for name in ("rein_000100_0.png", "rein_000100_1.png", "rein_000200_p0_0.png"):
            (samples / name).write_bytes(b"x")
        # A generated sample and its job record stay out of the count.
        (samples / "generated").mkdir()
        (samples / "generated" / "p0001_01.png").write_bytes(b"x")
        (samples / "notes.txt").write_text("x")
        for step in (100, 200):
            weight_dir = run / f"rein_s{step:06d}"
            weight_dir.mkdir()
            (weight_dir / "rein.safetensors").write_bytes(b"w")
        (run / "rein_final").mkdir()
        (run / "rein_final" / "rein.safetensors").write_bytes(b"w")

        run_info = list_runs(self.out, self.logs, None)[0]
        self.assertEqual(run_info["samples"], 3)
        self.assertEqual(run_info["last_step"], 200)
        self.assertEqual(run_info["checkpoints"], 3)
        self.assertTrue(run_info["has_output"])
        self.assertFalse(run_info["has_log"])

    def test_a_weight_dir_lifts_the_step_when_samples_are_gone(self):
        run = self.out / "rein_20260911_120000"
        (run / "rein_samples").mkdir(parents=True)
        (run / "rein_e002_s000450").mkdir()
        (run / "rein_e002_s000450" / "rein.safetensors").write_bytes(b"w")
        run_info = list_runs(self.out, self.logs, None)[0]
        self.assertEqual(run_info["last_step"], 450)
        self.assertEqual(run_info["samples"], 0)
        self.assertEqual(run_info["checkpoints"], 1)

    def test_log_only_run_is_listed(self):
        (self.logs / "rein_20260911_120000").mkdir(parents=True)
        run_info = list_runs(self.out, self.logs, None)[0]
        self.assertFalse(run_info["has_output"])
        self.assertTrue(run_info["has_log"])
        self.assertEqual(run_info["output_name"], "rein")
        self.assertEqual(run_info["samples"], 0)
        self.assertIsNone(run_info["last_step"])

    def test_counts_samples_under_the_raw_name(self):
        # `re in` becomes `re_in` in the run id, while the sample dir keeps the raw name.
        run = self.out / "re_in_20260911_120000"
        (run / "re in_samples").mkdir(parents=True)
        (run / "re in_samples" / "re in_000100_0.png").write_bytes(b"x")
        run_info = list_runs(self.out, self.logs, None)[0]
        self.assertEqual(run_info["samples"], 1)
        self.assertEqual(run_info["last_step"], 100)


class StepsPerEpochFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "logs" / "rein_20260911_120000"

    def tearDown(self):
        self.tmp.cleanup()

    def test_path_is_inside_the_run_log_directory(self):
        self.assertEqual(steps_per_epoch_path(self.log), self.log / "steps_per_epoch.json")

    def test_missing_file_reads_as_none(self):
        self.assertIsNone(read_steps_per_epoch(self.log))

    def test_write_creates_the_directory_and_leaves_no_temp(self):
        path = write_steps_per_epoch(self.log, 12)
        self.assertEqual(path, self.log / "steps_per_epoch.json")
        self.assertEqual(read_steps_per_epoch(self.log), 12)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"steps_per_epoch": 12}')
        self.assertEqual(list(self.log.glob(".steps_per_epoch.json.*.tmp")), [])

    def test_corrupt_or_non_positive_reads_as_none(self):
        self.log.mkdir(parents=True)
        path = self.log / "steps_per_epoch.json"
        path.write_text("{", encoding="utf-8")
        self.assertIsNone(read_steps_per_epoch(self.log))
        path.write_text('{"steps_per_epoch": 0}', encoding="utf-8")
        self.assertIsNone(read_steps_per_epoch(self.log))
        path.write_text('{"steps_per_epoch": -4}', encoding="utf-8")
        self.assertIsNone(read_steps_per_epoch(self.log))
        path.write_text('{"steps_per_epoch": "nope"}', encoding="utf-8")
        self.assertIsNone(read_steps_per_epoch(self.log))
        path.write_text("[]", encoding="utf-8")
        self.assertIsNone(read_steps_per_epoch(self.log))

    def test_a_non_positive_write_is_refused(self):
        with self.assertRaises(ValueError):
            write_steps_per_epoch(self.log, 0)


class StepsPerEpochDashboardTest(unittest.TestCase):
    """`dashboard` reports the file, else this run's state.json when it divides evenly."""

    RUN_ID = "rein_20260911_120000"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        import api
        from trainer import control

        self.api = api
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.reset_to_idle()
        self.control = control
        self.logs = Path(self.tmp.name) / "logs"
        self.cfg = {
            "output_dir": str(Path(self.tmp.name) / "out"),
            "logging_dir": str(self.logs),
            "output_name": "rein",
        }
        self._orig = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        self.api._train_config_dict = self._orig
        self.control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_the_file_wins(self):
        write_steps_per_epoch(self.logs / self.RUN_ID, 15)
        self.control.write_state(
            {
                "status": "finished",
                "pid": None,
                "output_name": "rein",
                "run_id": self.RUN_ID,
                "training": {"epochs": 2, "total_steps": 40},
            },
            force=True,
        )
        result = self.api.dispatch("dashboard", {"run_id": self.RUN_ID})
        self.assertEqual(result["steps_per_epoch"], 15)

    def test_state_json_fills_in_only_for_the_run_it_names(self):
        self.control.write_state(
            {
                "status": "training",
                "pid": None,
                "output_name": "rein",
                "run_id": self.RUN_ID,
                "training": {"epochs": 4, "total_steps": 40},
            },
            force=True,
        )
        self.assertEqual(
            self.api.dispatch("dashboard", {"run_id": self.RUN_ID})["steps_per_epoch"],
            10,
        )
        self.assertIsNone(
            self.api.dispatch("dashboard", {"run_id": "rein_20260101_000000"})["steps_per_epoch"]
        )

    def test_an_uneven_total_and_a_missing_record_draw_nothing(self):
        self.control.write_state(
            {
                "status": "finished",
                "pid": None,
                "output_name": "rein",
                "run_id": self.RUN_ID,
                "training": {"epochs": 3, "total_steps": 40},
            },
            force=True,
        )
        self.assertIsNone(self.api.dispatch("dashboard", {"run_id": self.RUN_ID})["steps_per_epoch"])
        self.control.reset_to_idle()
        self.assertIsNone(self.api.dispatch("dashboard", {})["steps_per_epoch"])


class ChartViewFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "logs" / "rein_20260911_120000"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_is_the_defaults(self):
        self.assertEqual(
            read_chart_view(self.log),
            {"smooth_extra_dp": 1.2, "outlier_clip": 0.15, "step_span": 800},
        )
        self.assertEqual(default_chart_view()["outlier_clip"], 0.15)
        self.assertEqual(default_chart_view()["step_span"], 800)
        self.assertFalse(chart_view_path(self.log).exists())

    def test_write_round_trips_and_leaves_no_temp(self):
        path = write_chart_view(self.log, 2.5, 0.0)
        self.assertEqual(path, self.log / "chart_view.json")
        self.assertEqual(
            read_chart_view(self.log),
            {"smooth_extra_dp": 2.5, "outlier_clip": 0.0, "step_span": 800},
        )
        self.assertEqual(list(self.log.glob(".chart_view.json.*.tmp")), [])

    def test_a_bad_field_falls_back_on_its_own(self):
        self.log.mkdir(parents=True)
        (self.log / "chart_view.json").write_text(
            '{"smooth_extra_dp": 80, "outlier_clip": 0.2}',
            encoding="utf-8",
        )
        self.assertEqual(
            read_chart_view(self.log),
            {"smooth_extra_dp": 1.2, "outlier_clip": 0.2, "step_span": 800},
        )

    def test_a_corrupt_file_is_the_defaults(self):
        self.log.mkdir(parents=True)
        (self.log / "chart_view.json").write_text("{", encoding="utf-8")
        self.assertEqual(read_chart_view(self.log), default_chart_view())

    def test_a_value_outside_its_range_is_refused(self):
        with self.assertRaises(ValueError):
            write_chart_view(self.log, 1.2, 0.5)
        with self.assertRaises(ValueError):
            write_chart_view(self.log, -0.1, 0.15)
        with self.assertRaises(ValueError):
            write_chart_view(self.log, 1.2, 0.15, 50)


class ChartViewIpcTest(unittest.TestCase):
    RUN_ID = "rein_20260911_120000"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        import api
        from trainer import control

        self.api = api
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.reset_to_idle()
        self.control = control
        self.logs = Path(self.tmp.name) / "logs"
        self.cfg = {
            "output_dir": str(Path(self.tmp.name) / "out"),
            "logging_dir": str(self.logs),
            "output_name": "rein",
        }
        self._orig = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)
        control.write_state(
            {"status": "finished", "pid": None, "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )

    def tearDown(self):
        self.api._train_config_dict = self._orig
        self.control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_a_run_with_no_file_answers_the_defaults(self):
        result = self.api.dispatch("chart_view", {"run_id": self.RUN_ID})
        self.assertEqual(result["smooth_extra_dp"], 1.2)
        self.assertEqual(result["outlier_clip"], 0.15)
        self.assertEqual(result["step_span"], 800)
        self.assertEqual(result["file"], str(self.logs / self.RUN_ID / "chart_view.json"))
        self.assertFalse((self.logs / self.RUN_ID / "chart_view.json").exists())

    def test_setting_one_slider_keeps_the_other(self):
        self.api.dispatch("chart_view_set", {"run_id": self.RUN_ID, "smooth_extra_dp": 3})
        result = self.api.dispatch(
            "chart_view_set",
            {"run_id": self.RUN_ID, "outlier_clip": 0.05},
        )
        self.assertEqual(result["smooth_extra_dp"], 3)
        self.assertEqual(result["outlier_clip"], 0.05)
        self.assertEqual(result["step_span"], 800)
        stored = json.loads((self.logs / self.RUN_ID / "chart_view.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, {"smooth_extra_dp": 3.0, "outlier_clip": 0.05, "step_span": 800})

    def test_another_run_is_left_alone(self):
        other = "rein_20260101_000000"
        self.api.dispatch("chart_view_set", {"run_id": self.RUN_ID, "outlier_clip": 0.1})
        other_view = self.api.dispatch("chart_view", {"run_id": other})
        self.assertEqual(other_view["outlier_clip"], 0.15)
        self.assertFalse((self.logs / other / "chart_view.json").exists())

    def test_no_run_refuses_the_write_and_reads_the_defaults(self):
        self.control.reset_to_idle()
        self.assertIsNone(self.api.dispatch("chart_view", {})["run_id"])
        self.assertEqual(self.api.dispatch("chart_view", {})["outlier_clip"], 0.15)
        with self.assertRaises(ValueError):
            self.api.dispatch("chart_view_set", {"outlier_clip": 0.1})

    def test_an_out_of_range_value_is_refused(self):
        with self.assertRaises(ValueError):
            self.api.dispatch("chart_view_set", {"run_id": self.RUN_ID, "smooth_extra_dp": 9})
        self.assertFalse((self.logs / self.RUN_ID / "chart_view.json").exists())
        self.api.dispatch("chart_view_set", {"run_id": self.RUN_ID, "step_span": 1200})
        with self.assertRaises(ValueError):
            self.api.dispatch("chart_view_set", {"run_id": self.RUN_ID, "step_span": 9000})
        self.assertEqual(self.api.dispatch("chart_view", {"run_id": self.RUN_ID})["step_span"], 1200)


if __name__ == "__main__":
    unittest.main()
