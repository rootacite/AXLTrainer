import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import torch


# `python test/test_api_ipc.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# `api.subprocess` *is* the subprocess module, so the spawn tests' `mock.patch.object(api.subprocess,
# "Popen")` replaces it globally; the tests that need a real child process keep this reference.
REAL_POPEN = subprocess.Popen

import api
from trainer import config as trainer_config
from trainer import genjob
from trainer.config import sample_override_path
from trainer.loss_log import LossRecorder, synthesize_avg_loss


class ScanSamplesTest(unittest.TestCase):
    def test_groups_by_step_and_sorts_repeats(self):
        with tempfile.TemporaryDirectory() as raw:
            sample_dir = Path(raw)
            (sample_dir / "run_200_1.png").write_bytes(b"x")
            (sample_dir / "run_200_0.png").write_bytes(b"x")
            (sample_dir / "run_100_0.png").write_bytes(b"x")
            (sample_dir / "orphan.png").write_bytes(b"x")

            grouped = api.scan_samples(sample_dir)
            self.assertEqual(list(grouped.keys()), ["200", "100", "-1"])
            self.assertEqual([item["repeat_idx"] for item in grouped["200"]], [0, 1])
            self.assertTrue(Path(grouped["200"][0]["path"]).is_absolute())
            self.assertEqual(grouped["-1"][0]["filename"], "orphan.png")

    def test_missing_dir_is_empty(self):
        self.assertEqual(api.scan_samples(Path("/tmp/axl-missing-samples-dir")), {})


class ScanSampleSetsTest(unittest.TestCase):
    """`_p{set}_{repeat}` naming, with the two-number form still mapping to set 0."""

    def _names(self, *names):
        with tempfile.TemporaryDirectory() as raw:
            sample_dir = Path(raw)
            for name in names:
                (sample_dir / name).write_bytes(b"x")
            return api.scan_samples(sample_dir)

    def test_set_index_is_parsed(self):
        grouped = self._names("run_100_p0_0.png", "run_100_p1_0.png", "run_100_p1_1.png")
        self.assertEqual(
            [(item["set_index"], item["repeat_idx"]) for item in grouped["100"]],
            [(0, 0), (1, 0), (1, 1)],
        )

    def test_legacy_names_are_set_zero(self):
        grouped = self._names("run_100_0.png", "run_100_1.png")
        self.assertEqual([item["set_index"] for item in grouped["100"]], [0, 0])
        self.assertEqual([item["repeat_idx"] for item in grouped["100"]], [0, 1])

    def test_both_layouts_group_under_the_same_step(self):
        grouped = self._names("run_100_p0_0.png", "run_100_0.png")
        self.assertEqual(len(grouped["100"]), 2)
        self.assertEqual(list(grouped.keys()), ["100"])


class DashboardSampleSetsTest(unittest.TestCase):
    """The dashboard's sample defaults: a run that resolves uses that run's own prompts, so these
    cases keep no run recorded and read the config mapping instead."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        from trainer import control

        control._state = {}
        control._last_write_mono = 0.0
        control.reset_to_idle()

        self._orig_config = api._train_config_dict
        api._train_config_dict = lambda: {
            "sample_prompts": "flat prompt",
            "sample_negative": "flat negative",
            "sample_width": 1152,
            "sample_height": 768,
            "sample_steps": 35,
            "sample_seed": 0,
            "sample_repeat": 3,
            "guidance_scale": 5.0,
            "samples": [
                {"name": "one", "prompt": "p1", "steps": 8, "repeat": 1},
                {"prompt": "p2", "width": 512},
            ],
        }

    def tearDown(self):
        api._train_config_dict = self._orig_config
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_dashboard_reports_every_set(self):
        result = api.dispatch("dashboard", {"name": "__missing_run__"})
        sets = result["sample_sets"]
        self.assertEqual([entry["name"] for entry in sets], ["one", "p2"])
        self.assertEqual(sets[1]["width"], 512)
        self.assertEqual(sets[1]["height"], 768)
        json.dumps(result)

    def test_flat_config_keys_mirror_the_first_set(self):
        result = api.dispatch("dashboard", {"name": "__missing_run__"})
        config = result["config"]
        self.assertEqual(config["sample_prompts"], "p1")
        self.assertEqual(config["sample_steps"], 8)
        self.assertEqual(config["sample_repeat"], 1)

    def test_without_sets_the_flat_keys_are_untouched(self):
        api._train_config_dict = lambda: {
            "sample_prompts": "flat prompt",
            "sample_negative": "flat negative",
            "sample_steps": 35,
            "guidance_scale": 5.0,
        }
        # A key this mapping omits falls back to the config file's scalar, so pin that scalar
        # instead of asserting on whatever the author's local config.toml happens to hold.
        with mock.patch.dict(trainer_config._CONFIG, {"sample_repeat": 3}, clear=False):
            result = api.dispatch("dashboard", {"name": "__missing_run__"})
        self.assertEqual(result["config"]["sample_prompts"], "flat prompt")
        self.assertEqual(len(result["sample_sets"]), 1)
        self.assertEqual(result["sample_sets"][0]["repeat"], 3)

    def test_a_broken_entry_degrades_to_no_sets(self):
        api._train_config_dict = lambda: {"samples": [{"prompt": "p", "steps": 0}]}
        result = api.dispatch("dashboard", {"name": "__missing_run__"})
        self.assertEqual(result["sample_sets"], [])
        json.dumps(result)


class DispatchTest(unittest.TestCase):
    def test_ping(self):
        self.assertEqual(api.dispatch("ping"), {"status": "ok"})

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            api.dispatch("generate")

    def test_dashboard_empty_logs_does_not_crash(self):
        result = api.dispatch("dashboard", {"name": "__missing_run__"})
        self.assertIn("config", result)
        self.assertIn("latest_stats", result)
        self.assertIn("metrics", result)
        self.assertIsInstance(result["metrics"], dict)
        json.dumps(result)


class AvgLossTest(unittest.TestCase):
    def test_epoch0_is_cumulative_mean(self):
        rec = LossRecorder()
        rec.add(epoch=0, step=0, loss=1.0)
        rec.add(epoch=0, step=1, loss=3.0)
        self.assertAlmostEqual(rec.moving_average, 2.0)

    def test_later_epoch_overwrites_slot(self):
        rec = LossRecorder()
        rec.add(epoch=0, step=0, loss=1.0)
        rec.add(epoch=0, step=1, loss=3.0)
        rec.add(epoch=1, step=0, loss=5.0)
        self.assertAlmostEqual(rec.moving_average, 4.0)

    def test_synthesize_with_known_window(self):
        points = [{"step": i, "value": float(i), "wall_time": 0.0} for i in range(1, 6)]
        out = synthesize_avg_loss(points, steps_per_epoch=2)
        self.assertEqual([p["value"] for p in out], [1.0, 1.5, 2.5, 3.5, 4.5])
        self.assertEqual([p["step"] for p in out], [1, 2, 3, 4, 5])

    def test_dashboard_synthesizes_when_tag_missing(self):
        fake = {
            "Train/Loss": [
                {"step": 1, "value": 2.0, "wall_time": 1.0},
                {"step": 2, "value": 4.0, "wall_time": 2.0},
            ]
        }
        orig = api._get_tensorboard_metrics
        api._get_tensorboard_metrics = lambda *a, **k: dict(fake)
        try:
            result = api.handle_dashboard(
                {"name": "__avg_loss_synth__", "run_id": "__avg_loss_synth___20260101_000000"}
            )
            series = result["metrics"]["Train/Avg_Loss"]
            self.assertEqual(series[0]["value"], 2.0)
            self.assertEqual(series[1]["value"], 3.0)
            self.assertEqual(result["latest_stats"]["Train/Avg_Loss"], 3.0)
        finally:
            api._get_tensorboard_metrics = orig

    def test_dashboard_keeps_logged_avg(self):
        fake = {
            "Train/Loss": [{"step": 1, "value": 2.0, "wall_time": 1.0}],
            "Train/Avg_Loss": [{"step": 1, "value": 1.5, "wall_time": 1.0}],
        }
        orig = api._get_tensorboard_metrics
        api._get_tensorboard_metrics = lambda *a, **k: dict(fake)
        try:
            result = api.handle_dashboard(
                {"name": "__avg_loss_keep__", "run_id": "__avg_loss_keep___20260101_000000"}
            )
            self.assertEqual(result["metrics"]["Train/Avg_Loss"][0]["value"], 1.5)
        finally:
            api._get_tensorboard_metrics = orig

    def test_dashboard_cadence_is_the_run_snapshot_not_the_repo(self):
        # The repo file is whatever this checkout has. A run with its own snapshot must not
        # answer with that, and a run with no snapshot must not answer with it either.
        bare = api.handle_dashboard(
            {"name": "__no_snapshot__", "run_id": "__no_snapshot___20260101_000000"}
        )
        self.assertIsNone(bare["save_every_n_steps"])

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log_dir = Path(tmp.name) / "logs" / "rein_20260101_000000"
        log_dir.mkdir(parents=True)
        (log_dir / "config.toml").write_text(
            "[training]\nsave_every_n_steps = 40\n",
            encoding="utf-8",
        )
        orig = api._train_config_dict
        api._train_config_dict = lambda: {
            "logging_dir": str(Path(tmp.name) / "logs"),
            "output_dir": str(Path(tmp.name) / "out"),
            "output_name": "rein",
            "save_every_n_steps": 999,
        }
        try:
            result = api.handle_dashboard({"run_id": "rein_20260101_000000"})
        finally:
            api._train_config_dict = orig
        self.assertEqual(result["save_every_n_steps"], 40)
        self.assertEqual(result["config"]["save_every_n_steps"], 999)


class RunScopedIpcTest(unittest.TestCase):
    """dashboard / list_samples / list_checkpoints / train_reset are run-scoped."""

    RUN_ID = "rein_20260911_120000"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        from trainer import control

        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.reset_to_idle()

        self.out = Path(self.tmp.name) / "out"
        self.logs = Path(self.tmp.name) / "logs"
        self.cfg = {
            "output_dir": str(self.out),
            "logging_dir": str(self.logs),
            "output_name": "rein",
        }
        self._orig_config = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        from trainer import control

        api._train_config_dict = self._orig_config
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _make_run(self, run_id: str = RUN_ID) -> Path:
        run_dir = self.out / run_id
        (run_dir / "rein_samples").mkdir(parents=True)
        (self.logs / run_id).mkdir(parents=True)
        return run_dir

    def _record_run(self) -> None:
        """`state.json` as a finished run leaves it — what makes a run "the current one"."""
        from trainer import control

        control.write_state(
            {"status": "finished", "pid": None, "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )

    def test_dashboard_without_run_is_empty(self):
        result = api.dispatch("dashboard", {})
        self.assertIsNone(result["run_id"])
        self.assertEqual(result["metrics"], {})
        self.assertEqual(result["latest_stats"], {})

    def test_dashboard_prefers_state_run_id(self):
        from trainer import control

        control.write_state(
            {"status": "training", "pid": os.getpid(), "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )
        result = api.dispatch("dashboard", {})
        self.assertEqual(result["run_id"], self.RUN_ID)

    def test_dashboard_without_a_recorded_run_is_empty(self):
        self._make_run("rein_20260101_000000")
        self._make_run("rein_20260911_120000")
        (self.logs / "rein").mkdir(parents=True)  # legacy flat dir is ignored
        # Following means the run state.json is on; a run the trainer never recorded is
        # reached by name instead of being served as "current".
        self.assertIsNone(api.dispatch("dashboard", {})["run_id"])

    def test_dashboard_resolves_an_explicitly_named_run(self):
        self._make_run("rein_20260101_000000")
        self._make_run("rein_20260911_120000")
        result = api.dispatch("dashboard", {"name": "rein"})
        self.assertEqual(result["run_id"], "rein_20260911_120000")

    def test_dashboard_explicit_run_id_wins(self):
        self._make_run("rein_20260101_000000")
        result = api.dispatch("dashboard", {"run_id": self.RUN_ID})
        self.assertEqual(result["run_id"], self.RUN_ID)

    def test_list_samples_reads_run_dir(self):
        run_dir = self._make_run()
        (run_dir / "rein_samples" / "rein_000100_0.png").write_bytes(b"x")
        (run_dir / "rein_samples" / "rein_000200_0.png").write_bytes(b"x")
        result = api.dispatch("list_samples", {"run_id": self.RUN_ID})
        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(list(result["samples"].keys()), ["200", "100"])

    def test_list_samples_without_run_is_empty(self):
        result = api.dispatch("list_samples", {})
        self.assertIsNone(result["run_id"])
        self.assertEqual(result["samples"], {})

    def test_list_samples_without_a_recorded_run_is_empty(self):
        self._make_run()
        self.assertIsNone(api.dispatch("list_samples", {})["run_id"])

    def test_list_checkpoints_reports_metadata(self):
        from safetensors.torch import save_file

        run_dir = self._make_run()
        weight_dir = run_dir / "rein_s000100"
        weight_dir.mkdir(parents=True)
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(weight_dir / "rein.safetensors"),
            metadata={"ss_steps": "100", "ss_network_dim": "4", "ss_network_alpha": "2"},
        )
        final_dir = run_dir / "rein_final"
        final_dir.mkdir(parents=True)
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(final_dir / "rein.safetensors"),
            metadata={"ss_steps": "300", "ss_network_dim": "4", "ss_network_alpha": "2"},
        )
        legacy = self.out / "rein_s000999"
        legacy.mkdir(parents=True)
        save_file({"lora_unet_x.lora_down.weight": torch.zeros(4, 2)}, str(legacy / "rein.safetensors"))

        result = api.dispatch("list_checkpoints", {})
        checkpoints = result["checkpoints"]
        self.assertEqual([item["step"] for item in checkpoints], [300, 100])
        self.assertEqual(checkpoints[0]["final"], True)
        self.assertEqual(checkpoints[0]["run_id"], self.RUN_ID)
        self.assertEqual(checkpoints[0]["network_dim"], 4)
        self.assertEqual(checkpoints[0]["output_name"], "rein")
        self.assertTrue(Path(checkpoints[0]["path"]).is_file())

    def test_list_checkpoints_empty_output_dir(self):
        self.assertEqual(api.dispatch("list_checkpoints", {})["checkpoints"], [])

    def test_reset_keeps_samples_and_logs(self):
        run_dir = self._make_run()
        (run_dir / "rein_samples" / "a.png").write_bytes(b"x")
        (self.logs / self.RUN_ID / "events.out.tfevents.1").write_bytes(b"e")
        weights = run_dir / "rein_s000100"
        weights.mkdir(parents=True)
        (weights / "rein.safetensors").write_bytes(b"w")

        self._record_run()
        result = api.dispatch("train_reset", {})
        self.assertEqual(result["status"], "idle")
        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertTrue((run_dir / "rein_samples" / "a.png").exists())
        self.assertTrue((self.logs / self.RUN_ID / "events.out.tfevents.1").exists())
        self.assertTrue(weights.exists())
        self.assertEqual(result["cleanup"]["removed"], [])
        self.assertFalse(result["cleanup"]["delete_samples"])
        self.assertFalse(result["cleanup"]["delete_logs"])
        # The run it just cleared is still part of the history, logs and all.
        runs = api.dispatch("list_runs", {})["runs"]
        self.assertEqual([run["run_id"] for run in runs], [self.RUN_ID])
        self.assertTrue(runs[0]["has_log"])

    def test_reset_never_deletes_weights(self):
        run_dir = self._make_run()
        weights = run_dir / "rein_final"
        weights.mkdir(parents=True)
        (weights / "rein.safetensors").write_bytes(b"w")

        self._record_run()
        # Reset clears the state and nothing else: the flag that used to wipe a run's weights is
        # gone, and an older client still sending it changes nothing.
        result = api.dispatch("train_reset", {"delete_weights": True})
        self.assertTrue(weights.exists())
        self.assertEqual(result["cleanup"]["removed"], [])
        self.assertFalse(result["cleanup"]["delete_weights"])
        self.assertEqual(result["cleanup"]["run_id"], self.RUN_ID)
        self.assertTrue((run_dir / "rein_samples").is_dir())

    def test_reset_without_run_leaves_legacy_alone(self):
        legacy_samples = self.out / "rein_samples"
        legacy_samples.mkdir(parents=True)
        (legacy_samples / "a.png").write_bytes(b"x")
        (self.logs / "rein").mkdir(parents=True)

        result = api.dispatch("train_reset", {})
        self.assertIsNone(result["run_id"])
        self.assertTrue(legacy_samples.exists())
        self.assertTrue((self.logs / "rein").exists())


class RequestedSettingsIpcTest(unittest.TestCase):
    """`requested`: the change a live run has been asked for but has not adopted yet.

    The trainer reads `settings.json` once per optimizer step, so a switch flipped during a sample
    pass stays a request for a while; the reply has to say so instead of repeating the effective
    values and leaving the dashboard to wait for the trainer.
    """

    EFFECTIVE = {"save_every_n_steps": 50, "sampling_enabled": True, "next_save_step": 3350}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        from trainer import control

        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.reset_to_idle()
        self.control = control

        self.cfg = {
            "output_dir": str(Path(self.tmp.name) / "out"),
            "logging_dir": str(Path(self.tmp.name) / "logs"),
            "output_name": "rein",
        }
        self._orig_config = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        api._train_config_dict = self._orig_config
        self.control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _live_run(self, status: str = "training", pid: int | None = None) -> None:
        self.control.write_state(
            {
                "pid": os.getpid() if pid is None else pid,
                "status": status,
                "output_name": "rein",
                "run_id": "rein_20260911_120000",
                "settings": dict(self.EFFECTIVE),
            },
            force=True,
        )
        # `train_start` seeds the request file from `config.toml`, so a live run always holds both
        # keys: a later request merges onto that file rather than writing a fresh one.
        self.control.request_settings(
            save_every_n_steps=self.EFFECTIVE["save_every_n_steps"],
            sampling_enabled=self.EFFECTIVE["sampling_enabled"],
        )

    def test_a_request_that_differs_is_reported_by_both_replies(self):
        self._live_run()
        result = api.handle_train_settings({"sampling_enabled": False})
        self.assertEqual(result["requested"], {"save_every_n_steps": 50, "sampling_enabled": False})
        # The effective values are untouched: the run is still sampling until the trainer adopts.
        self.assertEqual(result["settings"], self.EFFECTIVE)
        self.assertEqual(api.handle_train_status({})["requested"], result["requested"])

    def test_a_request_equal_to_the_effective_values_is_not_reported(self):
        self._live_run()
        api.handle_train_settings({"save_every_n_steps": 50, "sampling_enabled": True})
        self.assertIsNone(api.handle_train_status({})["requested"])

    def test_a_paused_run_reports_what_it_will_adopt_on_resume(self):
        self._live_run(status="paused")
        api.handle_train_settings({"save_every_n_steps": 100})
        self.assertEqual(
            api.handle_train_status({})["requested"],
            {"save_every_n_steps": 100, "sampling_enabled": True},
        )

    def test_a_request_with_no_file_keeps_the_running_cadence(self):
        """A hand-started run has no `settings.json`: flipping the switch must not zero the cadence.

        `bash start_train.sh` (or a wiped runtime dir) leaves no request file, and the merge used to
        fill it from the defaults — `save_every_n_steps = 0` means "write no checkpoints", so asking
        for the switch off would also have stopped the checkpoints it never mentioned.
        """
        self._live_run()
        self.control.settings_path().unlink()
        result = api.handle_train_settings({"sampling_enabled": False})
        self.assertEqual(
            result["requested"],
            {"save_every_n_steps": self.EFFECTIVE["save_every_n_steps"], "sampling_enabled": False},
        )
        self.assertEqual(self.control.read_settings(), result["requested"])

    def test_no_live_process_means_nothing_to_report(self):
        self._live_run()
        api.handle_train_settings({"sampling_enabled": False})
        self.assertIsNotNone(api.handle_train_status({})["requested"])

        # The request file survives the run; the report must not: nobody is left to adopt it.
        self.control.write_state({"pid": None, "status": "idle"}, force=True)
        self.assertIsNone(api.handle_train_status({})["requested"])

    def test_reset_clears_the_request(self):
        self._live_run(status="finished")
        api.handle_train_settings({"sampling_enabled": False})
        result = api.handle_train_reset({})
        self.assertIsNone(result["requested"])
        self.assertIsNone(self.control.read_settings())

    def test_no_request_file_reports_nothing(self):
        """The request channel *is* `settings.json`: no file, nothing asked for."""
        self._live_run()
        self.control.settings_path().unlink()
        self.assertIsNone(api.handle_train_status({})["requested"])


class TensorboardCacheTest(unittest.TestCase):
    """`dashboard` asks for the whole history every poll; the reader must not re-read all of it.

    A fresh `EventAccumulator(...).Reload()` per call costs 343 ms at 24 k points and grows with the
    run, so the reader is kept per run directory and reloaded in place.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name) / "logs" / "rein_20260911_120000"
        self.log_dir.mkdir(parents=True)
        api.reset_tensorboard_cache()

        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def tearDown(self):
        self.writer.close()
        api.reset_tensorboard_cache()
        self.tmp.cleanup()

    def _steps(self, count: int, start: int = 0) -> None:
        for step in range(start, start + count):
            self.writer.add_scalar("Train/Loss", 1.0 / (step + 1), step)
        self.writer.flush()

    def test_a_second_read_reuses_the_reader_and_sees_new_steps(self):
        self._steps(3)
        first = api._get_tensorboard_metrics(str(self.log_dir))
        self.assertEqual([0, 1, 2], [point["step"] for point in first["Train/Loss"]])
        self.assertEqual({"entries": 1, "builds": 1}, api.tensorboard_cache_stats())

        self._steps(2, start=3)
        second = api._get_tensorboard_metrics(str(self.log_dir))
        self.assertEqual([0, 1, 2, 3, 4], [point["step"] for point in second["Train/Loss"]])
        self.assertEqual(1, api.tensorboard_cache_stats()["builds"], "the reader was rebuilt")

    def test_a_second_read_of_the_same_history_is_still_the_same_history(self):
        self._steps(4)
        first = api._get_tensorboard_metrics(str(self.log_dir))
        second = api._get_tensorboard_metrics(str(self.log_dir))
        self.assertEqual(first, second)
        self.assertEqual(1, api.tensorboard_cache_stats()["builds"])

    def test_the_step_range_is_applied_to_the_cached_reader(self):
        self._steps(6)
        sliced = api._get_tensorboard_metrics(str(self.log_dir), start_step=2, end_step=4)
        self.assertEqual([2, 3, 4], [point["step"] for point in sliced["Train/Loss"]])
        again = api._get_tensorboard_metrics(str(self.log_dir), start_step=5)
        self.assertEqual([5], [point["step"] for point in again["Train/Loss"]])
        self.assertEqual(1, api.tensorboard_cache_stats()["builds"])

    def test_a_new_event_file_in_the_same_directory_rebuilds_the_reader(self):
        self._steps(3)
        api._get_tensorboard_metrics(str(self.log_dir))
        self.assertEqual(1, api.tensorboard_cache_stats()["builds"])

        # A second writer in the same directory is a new file with a newer mtime: the reader has to
        # follow it rather than keep serving what the first one wrote.
        from torch.utils.tensorboard import SummaryWriter

        self.writer.close()
        other = SummaryWriter(log_dir=str(self.log_dir))
        try:
            other.add_scalar("Train/Loss", 9.0, 0)
            other.flush()
            metrics = api._get_tensorboard_metrics(str(self.log_dir))
        finally:
            other.close()
        # A directory-level reader: the second writer's events join the series, as TensorBoard
        # shows them, but the reader itself had to follow the newer file.
        self.assertEqual(
            [1.0, 0.5, 0.3333333432674408, 9.0],
            [point["value"] for point in metrics["Train/Loss"]],
        )
        self.assertEqual(2, api.tensorboard_cache_stats()["builds"])

    def test_the_cache_is_per_run_directory(self):
        self._steps(2)
        other_dir = Path(self.tmp.name) / "logs" / "konomi_20260911_130000"
        other_dir.mkdir(parents=True)
        from torch.utils.tensorboard import SummaryWriter

        other = SummaryWriter(log_dir=str(other_dir))
        try:
            other.add_scalar("Train/Loss", 5.0, 0)
            other.flush()
            mine = api._get_tensorboard_metrics(str(self.log_dir))
            theirs = api._get_tensorboard_metrics(str(other_dir))
        finally:
            other.close()
        self.assertEqual(2, len(mine["Train/Loss"]))
        self.assertEqual(1, len(theirs["Train/Loss"]))
        self.assertEqual({"entries": 2, "builds": 2}, api.tensorboard_cache_stats())

    def test_a_missing_directory_is_empty_and_caches_nothing(self):
        self.assertEqual({}, api._get_tensorboard_metrics(str(Path(self.tmp.name) / "nope")))
        self.assertEqual({"entries": 0, "builds": 0}, api.tensorboard_cache_stats())


class ListRunsIpcTest(unittest.TestCase):
    """list_runs: the dashboard's run history (every output name, live vs stopped)."""

    RUN_ID = "rein_20260911_120000"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        from trainer import control

        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.reset_to_idle()

        self.out = Path(self.tmp.name) / "out"
        self.logs = Path(self.tmp.name) / "logs"
        self.cfg = {
            "output_dir": str(self.out),
            "logging_dir": str(self.logs),
            "output_name": "rein",
        }
        self._orig_config = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        from trainer import control

        api._train_config_dict = self._orig_config
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _make_run(self, run_id: str, name: str = "rein", samples: tuple[str, ...] = ()) -> Path:
        run_dir = self.out / run_id
        samples_dir = run_dir / f"{name}_samples"
        samples_dir.mkdir(parents=True)
        for filename in samples:
            (samples_dir / filename).write_bytes(b"x")
        return run_dir

    def test_lists_runs_newest_first_with_brief_data(self):
        older = self._make_run("rein_20260910_120000")
        newer = self._make_run(
            self.RUN_ID, samples=("rein_000200_0.png", "rein_000300_p0_1.png")
        )
        weights = newer / "rein_s000300"
        weights.mkdir()
        (weights / "rein.safetensors").write_bytes(b"w")
        (self.logs / self.RUN_ID).mkdir(parents=True)
        os.utime(older, (1_700_000_000, 1_700_000_000))
        os.utime(newer, (1_700_000_100, 1_700_000_100))

        runs = api.dispatch("list_runs", {})["runs"]
        self.assertEqual([run["run_id"] for run in runs], [self.RUN_ID, "rein_20260910_120000"])
        first = runs[0]
        self.assertEqual(first["output_name"], "rein")
        self.assertEqual(first["samples"], 2)
        self.assertEqual(first["last_step"], 300)
        self.assertEqual(first["checkpoints"], 1)
        self.assertTrue(first["has_log"])
        self.assertFalse(first["current"])
        self.assertFalse(first["live"])
        self.assertEqual(first["size_bytes"], 3)  # two samples + one weight file

    def test_marks_the_state_run_current_and_live(self):
        from trainer import control

        self._make_run(self.RUN_ID)
        control.write_state(
            {"status": "training", "pid": os.getpid(), "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )
        runs = api.dispatch("list_runs", {})["runs"]
        self.assertEqual([run["run_id"] for run in runs], [self.RUN_ID])
        self.assertTrue(runs[0]["current"])
        self.assertTrue(runs[0]["live"])

    def test_a_finished_run_is_current_but_not_live(self):
        from trainer import control

        self._make_run(self.RUN_ID)
        control.write_state(
            {"status": "finished", "pid": None, "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )
        runs = api.dispatch("list_runs", {})["runs"]
        self.assertTrue(runs[0]["current"])
        self.assertFalse(runs[0]["live"])

    def test_the_state_run_is_listed_without_its_directories(self):
        from trainer import control

        control.write_state(
            {"status": "starting", "pid": os.getpid(), "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )
        runs = api.dispatch("list_runs", {})["runs"]
        self.assertEqual([run["run_id"] for run in runs], [self.RUN_ID])
        self.assertTrue(runs[0]["current"])
        self.assertFalse(runs[0]["has_output"])

    def test_lists_another_output_name_and_resolves_its_samples(self):
        self._make_run("konomi_20260912_090000", name="konomi", samples=("konomi_000100_0.png",))

        runs = api.dispatch("list_runs", {})["runs"]
        self.assertEqual([run["run_id"] for run in runs], ["konomi_20260912_090000"])
        self.assertEqual(runs[0]["output_name"], "konomi")

        # Only the run id is needed: the name it was built from opens its sample dir.
        samples = api.dispatch("list_samples", {"run_id": "konomi_20260912_090000"})
        self.assertEqual(list(samples["samples"]), ["100"])
        self.assertEqual(samples["samples"]["100"][0]["filename"], "konomi_000100_0.png")

    def test_a_run_without_logs_is_reached_by_id(self):
        """A log-less run stays readable: the history entry carries the run id it needs."""
        self._make_run("konomi_20260912_090000", name="konomi", samples=("konomi_000100_0.png",))

        result = api.dispatch("list_samples", {"run_id": "konomi_20260912_090000"})
        self.assertEqual(result["run_id"], "konomi_20260912_090000")
        self.assertEqual(list(result["samples"]), ["100"])

    def test_the_state_run_wins_whatever_the_config_names(self):
        from trainer import control

        self._make_run("konomi_20260912_090000", name="konomi")
        self._make_run("rein_20260101_000000")
        control.write_state(
            {
                "status": "finished",
                "pid": None,
                "output_name": "konomi",
                "run_id": "konomi_20260912_090000",
            },
            force=True,
        )
        self.assertEqual(api.dispatch("dashboard", {})["run_id"], "konomi_20260912_090000")

    def test_samples_dir_of_a_sanitized_name_is_found(self):
        # A name with a space reaches the run id as `re_in`, but its sample dir keeps the raw name.
        run_dir = self.out / "re_in_20260911_120000"
        (run_dir / "re in_samples").mkdir(parents=True)
        (run_dir / "re in_samples" / "re in_000200_0.png").write_bytes(b"x")

        result = api.dispatch("list_samples", {"run_id": "re_in_20260911_120000"})
        self.assertEqual(list(result["samples"]), ["200"])


class GeneratedFixture:
    """The generation/evaluation tests' fixture: a temp run with a real checkpoint file, a mocked
    spawner (no test may start a real generator) and the helpers that read the spec back."""

    RUN_ID = "rein_20260911_120000"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        from trainer import control

        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.reset_to_idle()

        self.out = Path(self.tmp.name) / "out"
        self.logs = Path(self.tmp.name) / "logs"
        self.cfg = {
            "output_dir": str(self.out),
            "logging_dir": str(self.logs),
            "output_name": "rein",
            "sample_prompts": "config prompt",
            "sample_negative": "config negative",
            "guidance_scale": 5.0,
            "sample_steps": 35,
            "sample_seed": 0,
            "sample_width": 1152,
            "sample_height": 768,
        }
        self._orig_config = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

        # No test may spawn a real generator (it would load SDXL on the GPU); the spawn tests assert
        # against this mock instead.
        self._popen_patcher = mock.patch.object(api.subprocess, "Popen")
        self.popen = self._popen_patcher.start()
        self.popen.return_value.pid = 4242

        self.run_dir = self.out / self.RUN_ID
        self.samples = self.run_dir / "rein_samples"
        self.samples.mkdir(parents=True)
        (self.logs / self.RUN_ID).mkdir(parents=True)
        # What `main.py` leaves beside a run's logs, and what its prompts resolve from: the run's
        # own copy of the config, not today's `config.toml`.
        self._write_run_config()
        self.checkpoint_dir = self.run_dir / "rein_s003050"
        self.checkpoint_dir.mkdir()
        self.checkpoint = self.checkpoint_dir / "rein.safetensors"
        self.checkpoint.write_bytes(b"weights")
        # The run the page follows: its id comes from state.json, never from the newest
        # directory on disk.
        control.write_state(
            {"status": "finished", "pid": None, "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )

    def tearDown(self):
        from trainer import control

        self._popen_patcher.stop()
        api._train_config_dict = self._orig_config
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    @property
    def generated(self) -> Path:
        return self.samples / "generated"

    def _write_run_config(self) -> Path:
        """Write the run's own `config.toml` copy from `self.cfg` (`[validation]` + its sets)."""
        lines = ["[validation]"]
        for key in (
            "sample_prompts",
            "sample_negative",
            "sample_width",
            "sample_height",
            "sample_steps",
            "sample_seed",
            "sample_repeat",
            "guidance_scale",
            "guidance_rescale",
        ):
            if key in self.cfg:
                lines.append(f"{key} = {json.dumps(self.cfg[key])}")
        for entry in self.cfg.get("samples") or []:
            lines.append("")
            lines.append("[[validation.samples]]")
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in entry.items())
        target = self.logs / self.RUN_ID / "config.toml"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def _set_run_samples(self, entries: list[dict]) -> None:
        """Pin this run's prompt sets (and the snapshot they resolve from)."""
        self.cfg["samples"] = list(entries)
        self._write_run_config()

    def _write_job(self, job_id: str, **fields) -> dict:
        from trainer import genjob

        job = genjob.new_job(
            {
                "prompt": "p",
                "negative_prompt": "",
                "cfg": 5.0,
                "steps": 12,
                "seed": 7,
                "width": 1024,
                "height": 1024,
                "step": 3050,
            },
            run_id=self.RUN_ID,
            output_name="rein",
            checkpoint=str(self.checkpoint),
        )
        job["id"] = job_id
        job.update(fields)
        return genjob.write_job(self.generated, job)

    def _spawn(self, params: dict | None = None):
        """Run the handler against the mocked generator process."""
        return api.handle_generate_sample(
            {
                "checkpoint": str(self.checkpoint),
                "prompt": "my prompt",
                "cfg": 7.0,
                "steps": 12,
                "seed": 42,
                **(params or {}),
            }
        )

    def _spec_written_by_last_spawn(self) -> dict:
        return json.loads(Path(self.popen.call_args.args[0][4]).read_text())


class GeneratedSampleIpcTest(GeneratedFixture, unittest.TestCase):
    """generate_sample / list_generated_samples: job records, GPU guard, validation."""

    def test_dispatch_registered(self):
        self.assertIn("generate_sample", api._HANDLERS)
        self.assertIn("list_generated_samples", api._HANDLERS)

    def test_listing_without_generated_dir_is_empty(self):
        result = api.dispatch("list_generated_samples", {})
        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(result["jobs"], [])

    def test_listing_is_newest_first_and_json_safe(self):
        self._write_job("old_gen_1", started_at=100.0, state="done", image_path="/x/old.png")
        self._write_job("new_gen_1", started_at=200.0, state="done", image_path="/x/new.png")
        result = api.dispatch("list_generated_samples", {})
        self.assertEqual([job["id"] for job in result["jobs"]], ["new_gen_1", "old_gen_1"])
        self.assertEqual(result["jobs"][0]["cfg"], 5.0)
        json.dumps(result)

    def test_a_dead_generator_is_reported_as_error(self):
        self._write_job("stuck_gen_1", pid=999_999_999)
        jobs = api.dispatch("list_generated_samples", {})["jobs"]
        self.assertEqual(jobs[0]["state"], "error")
        self.assertIn("exited before finishing", jobs[0]["error"])

    def test_a_live_generator_stays_running(self):
        self._write_job("live_gen_1", pid=os.getpid())
        self.assertEqual(api.dispatch("list_generated_samples", {})["jobs"][0]["state"], "running")

    def test_spawns_a_detached_generator_and_records_the_job(self):
        result = self._spawn()
        argv = self.popen.call_args.args[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1], "-u")
        self.assertTrue(argv[2].endswith("trainer/generate_sample.py"))
        self.assertEqual(argv[3], "--spec")
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])

        spec = Path(argv[4])
        self.assertTrue(spec.is_file())
        self.assertEqual(spec.parent, self.generated)
        stored = json.loads(spec.read_text())
        self.assertEqual(stored["state"], "running")
        self.assertEqual(stored["pid"], 4242)
        self.assertEqual(stored["prompt"], "my prompt")
        self.assertEqual(stored["cfg"], 7.0)
        self.assertEqual(stored["steps"], 12)
        self.assertEqual(stored["seed"], 42)
        self.assertEqual(stored["width"], 1152)
        self.assertEqual(stored["height"], 768)
        self.assertEqual(stored["checkpoint"], str(self.checkpoint))
        self.assertTrue(stored["id"].startswith("rein_s003050_gen_"))
        self.assertEqual(result["job"]["id"], stored["id"])
        self.assertTrue(result["log_path"].endswith(".log"))
        self.assertEqual(Path(result["log_path"]).parent, self.generated)

    def test_form_values_default_to_the_config(self):
        self._spawn({"prompt": None, "cfg": None, "steps": None, "seed": None})
        stored = self._spec_written_by_last_spawn()
        self.assertEqual(stored["prompt"], "config prompt")
        self.assertEqual(stored["negative_prompt"], "config negative")
        self.assertEqual(stored["cfg"], 5.0)
        self.assertEqual(stored["steps"], 35)
        self.assertEqual(stored["seed"], 0)
        self.assertEqual((stored["width"], stored["height"]), (1152, 768))

    def test_form_values_default_to_the_first_sample_set(self):
        for key in ("sample_prompts", "sample_negative", "guidance_scale", "sample_steps",
                    "sample_width", "sample_height"):
            self.cfg.pop(key)
        self._set_run_samples([
            {
                "prompt": "set one",
                "negative": "set one negative",
                "steps": 9,
                "guidance_scale": 4.0,
                "width": 640,
                "height": 960,
                "seed": 11,
            },
            {"prompt": "set two", "steps": 40},
        ])
        self._spawn({"prompt": None, "cfg": None, "steps": None, "seed": None})
        stored = self._spec_written_by_last_spawn()
        self.assertEqual(stored["prompt"], "set one")
        self.assertEqual(stored["negative_prompt"], "set one negative")
        self.assertEqual(stored["cfg"], 4.0)
        self.assertEqual(stored["steps"], 9)
        self.assertEqual(stored["seed"], 11)
        self.assertEqual((stored["width"], stored["height"]), (640, 960))

    def test_refuses_while_the_gpu_is_busy(self):
        from trainer import control

        for status in ("starting", "encoding", "training", "sampling", "pausing", "resuming", "stopping"):
            with self.subTest(status=status):
                control.write_state({"status": status, "pid": os.getpid()}, force=True)
                with self.assertRaises(ValueError) as ctx:
                    api.handle_generate_sample({"checkpoint": str(self.checkpoint), "prompt": "p"})
                self.assertIn("GPU", str(ctx.exception))

    def test_generating_is_allowed_while_paused(self):
        """Pause has offloaded every module, so a one-off image can use the card next to it."""
        from trainer import control

        control.write_state({"status": "paused", "pid": os.getpid()}, force=True)
        result = self._spawn()
        self.popen.assert_called_once()
        self.assertEqual(result["job"]["state"], "running")
        self.assertEqual(result["job"]["checkpoint"], str(self.checkpoint))

    def test_refuses_a_second_job_while_one_runs(self):
        self._write_job("live_gen_1", pid=os.getpid())
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_sample({"checkpoint": str(self.checkpoint), "prompt": "p"})
        self.assertIn("already running", str(ctx.exception))

    def test_requires_a_checkpoint_file(self):
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_sample({"checkpoint": str(self.run_dir / "nope.safetensors"), "prompt": "p"})
        self.assertIn("not a checkpoint file", str(ctx.exception))

    def test_rejects_a_bad_form(self):
        cases = [
            ({"prompt": ""}, "prompt"),
            ({"prompt": "p", "cfg": 99}, "cfg"),
            ({"prompt": "p", "steps": 0}, "steps"),
            ({"prompt": "p", "seed": -5}, "seed"),
        ]
        for params, expected in cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError) as ctx:
                    api.handle_generate_sample({"checkpoint": str(self.checkpoint), **params})
                self.assertIn(expected, str(ctx.exception))

    def test_requires_a_resolved_run(self):
        from trainer import control

        with tempfile.TemporaryDirectory() as empty:
            self.cfg["logging_dir"] = empty
            self.cfg["output_dir"] = empty
            control.reset_to_idle()  # no run recorded, and no run directory to find one in
            with self.assertRaises(ValueError) as ctx:
                api.handle_generate_sample({"checkpoint": str(self.checkpoint), "prompt": "p"})
            self.assertIn("no run", str(ctx.exception))

    def test_checkpoint_samples_spawn_a_sets_job(self):
        self._set_run_samples([
            {"prompt": "set one", "steps": 9, "repeat": 2},
            {"prompt": "set two", "steps": 40, "repeat": 3},
        ])
        result = api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})

        stored = self._spec_written_by_last_spawn()
        self.assertEqual(stored["mode"], "sets")
        self.assertEqual(stored["state"], "running")
        self.assertEqual(stored["total_images"], 5)
        self.assertEqual(stored["images_done"], 0)
        self.assertEqual(stored["files"], [])
        self.assertEqual(stored["checkpoint"], str(self.checkpoint))
        # The sets are recorded so the panel can show what the pass renders.
        self.assertEqual([entry["steps"] for entry in stored["sample_sets"]], [9, 40])
        # ...and they are this run's own, with the directory they were resolved in.
        self.assertEqual(stored["config_log_dir"], str(self.logs / self.RUN_ID))
        self.assertIn("_sets_gen_", stored["id"])
        self.assertEqual(result["job"]["id"], stored["id"])
        self.popen.assert_called_once()

    def test_a_sets_job_carries_its_step_and_no_null_counters(self):
        """Both halves of one bug: a null counter fails the client's decode, and a null step
        leaves the rendered images on no checkpoint card at all."""
        self._set_run_samples([{"prompt": "set one", "steps": 9, "repeat": 2}])
        api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})
        stored = self._spec_written_by_last_spawn()
        self.assertEqual(stored["step"], 3050)
        for key in ("current_step", "total_steps", "images_done", "total_images"):
            self.assertIsInstance(stored[key], int, f"{key} must be an int, not {stored[key]!r}")
        self.assertEqual(stored["total_steps"], 0)

    def test_a_single_job_fills_the_step_from_the_checkpoint(self):
        # The panel sends the row it is showing; a client that sends nothing (or null) still gets a
        # job the Checkpoints section can place.
        self._spawn({"step": None})
        self.assertEqual(self._spec_written_by_last_spawn()["step"], 3050)

    def test_the_checkpoint_step_falls_back_to_its_metadata(self):
        # A directory name that carries no step (a copied or renamed checkpoint) reads `ss_steps`.
        from safetensors.torch import save_file

        import torch

        target = self.run_dir / "rein_something" / "rein.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(target),
            metadata={"ss_steps": "4242"},
        )
        self._set_run_samples([{"prompt": "set one", "steps": 9, "repeat": 1}])
        api.handle_generate_checkpoint_samples({"checkpoint": str(target)})
        self.assertEqual(self._spec_written_by_last_spawn()["step"], 4242)

    def test_checkpoint_samples_refuse_a_busy_gpu_and_a_second_job(self):
        from trainer import control

        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})
        self.assertIn("GPU", str(ctx.exception))

        control.write_state({"status": "finished", "pid": None}, force=True)
        self._write_job("live_gen_1", pid=os.getpid())
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})
        self.assertIn("already running", str(ctx.exception))

    def test_checkpoint_samples_require_a_checkpoint_and_sets(self):
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_checkpoint_samples({"checkpoint": str(self.run_dir / "nope.safetensors")})
        self.assertIn("not a checkpoint file", str(ctx.exception))

        self._set_run_samples([{"prompt": ""}])
        with self.assertRaises(ValueError):
            api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})

    def _checkpoint_dir(self, step: int):
        """A real checkpoint directory (`{name}_s{step:06d}`) with a safetensors file in it."""
        from safetensors.torch import save_file

        import torch

        target = self.run_dir / f"rein_s{step:06d}" / "rein.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(target),
            metadata={"ss_steps": str(step)},
        )
        return target

    def test_a_batch_covers_the_checkpoints_inside_the_range(self):
        for step in (100, 200):
            self._checkpoint_dir(step)
        self._set_run_samples([
            {"prompt": "a", "steps": 9, "repeat": 2},
            {"prompt": "b", "steps": 9, "repeat": 1},
        ])

        result = api.handle_generate_checkpoint_samples_batch({"from_step": 150, "to_step": 250})
        stored = self._spec_written_by_last_spawn()

        self.assertEqual(stored["mode"], "batch")
        self.assertEqual(stored["state"], "running")
        self.assertEqual((stored["from_step"], stored["to_step"]), (150, 250))
        self.assertEqual([entry["step"] for entry in stored["checkpoints"]], [200])
        self.assertEqual(stored["total_checkpoints"], 1)
        # One checkpoint × 3 images, and nothing rendered yet.
        self.assertEqual(stored["total_images"], 3)
        self.assertEqual(stored["images_done"], 0)
        self.assertEqual(stored["checkpoint_index"], 0)
        # The prompts this run samples with travel on the record, exactly as api.py resolved them.
        self.assertEqual([entry["prompt"] for entry in stored["sample_sets"]], ["a", "b"])
        self.assertEqual([entry["repeat"] for entry in stored["sample_sets"]], [2, 1])
        self.assertEqual(stored["job_ids"], [])
        self.assertEqual(stored["failed"], [])
        self.assertEqual(stored["config_log_dir"], str(self.logs / self.RUN_ID))
        self.assertIn("_batch_gen_", stored["id"])
        self.assertEqual(result["job"]["id"], stored["id"])
        self.popen.assert_called_once()

    def test_a_batch_hints_where_its_prompts_came_from(self):
        """The runner resolves the range's prompts once, from the run the batch names."""
        self._checkpoint_dir(100)
        self._set_run_samples([{"prompt": "a", "steps": 9, "repeat": 1}])
        api.handle_generate_checkpoint_samples_batch({"from_step": 0, "to_step": 200})
        stored = self._spec_written_by_last_spawn()
        self.assertEqual(stored["config_log_dir"], str(self.logs / self.RUN_ID))

    def test_a_sets_job_records_the_runs_own_prompts_not_todays_file(self):
        """The prompts are the ones that run samples with, which is what the panel shows too."""
        self._set_run_samples([{"prompt": "this run's prompt", "steps": 9, "repeat": 1}])
        api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})
        stored = self._spec_written_by_last_spawn()
        self.assertEqual([entry["prompt"] for entry in stored["sample_sets"]], ["this run's prompt"])
        self.assertEqual(stored["total_images"], 1)

    def test_a_run_without_prompts_is_refused_before_spawning(self):
        self._set_run_samples([{"prompt": "", "steps": 9}])
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})
        self.assertIn("no sample prompts", str(ctx.exception))
        self.popen.assert_not_called()

    def test_a_batch_runs_oldest_step_first(self):
        for step in (300, 100, 200):
            self._checkpoint_dir(step)
        self._set_run_samples([{"prompt": "a", "steps": 9, "repeat": 1}])
        api.handle_generate_checkpoint_samples_batch({"from_step": 0, "to_step": 500})
        stored = self._spec_written_by_last_spawn()
        # Oldest step first, and the fixture's own step-3050 checkpoint is outside the range.
        self.assertEqual([entry["step"] for entry in stored["checkpoints"]], [100, 200, 300])

    def test_a_batch_leaves_out_another_runs_checkpoints(self):
        self._checkpoint_dir(100)
        other = self.out / "elsewhere_20260910_120000" / "rein_s00200"
        other.mkdir(parents=True)
        (other / "rein.safetensors").write_bytes(b"not a checkpoint")
        self._set_run_samples([{"prompt": "a", "steps": 9, "repeat": 1}])

        api.handle_generate_checkpoint_samples_batch({"from_step": 0, "to_step": 150})
        stored = self._spec_written_by_last_spawn()
        self.assertEqual([entry["step"] for entry in stored["checkpoints"]], [100])

    def test_a_batch_validates_its_range_before_spawning(self):
        self._checkpoint_dir(100)
        self._set_run_samples([{"prompt": "a", "steps": 9, "repeat": 1}])
        cases = [
            ({"from_step": "x", "to_step": 10}, "from_step must be an integer"),
            ({"from_step": -1, "to_step": 10}, "from_step must be >= 0"),
            ({"from_step": 20, "to_step": 10}, "must not be greater"),
            ({}, "from_step must be an integer"),
        ]
        for params, expected in cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError) as ctx:
                    api.handle_generate_checkpoint_samples_batch(params)
                self.assertIn(expected, str(ctx.exception))

        # A range with nothing in it, and a config with no sets to render.
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_checkpoint_samples_batch({"from_step": 500, "to_step": 600})
        self.assertIn("no checkpoints between step 500 and 600", str(ctx.exception))
        self.popen.assert_not_called()

    def test_a_batch_shares_the_gpu_gate(self):
        from trainer import control

        self._checkpoint_dir(100)
        self._set_run_samples([{"prompt": "a", "steps": 9, "repeat": 1}])
        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_checkpoint_samples_batch({"from_step": 0, "to_step": 200})
        self.assertIn("GPU", str(ctx.exception))

        control.write_state({"status": "finished", "pid": None}, force=True)
        self._write_job("live_gen_1", pid=os.getpid())
        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_checkpoint_samples_batch({"from_step": 0, "to_step": 200})
        self.assertIn("already running", str(ctx.exception))
        self.popen.assert_not_called()

    def test_a_batch_record_names_no_images_of_its_own(self):
        """The images belong to the per-checkpoint jobs, so a batch never renders as an image."""
        from trainer import genjob as genjob_module

        job = genjob_module.new_batch_job(
            run_id=self.RUN_ID,
            output_name="rein",
            checkpoints=[{"path": "/out/rein_s000100/rein.safetensors", "step": 100}],
            from_step=100,
            to_step=100,
            images_per_checkpoint=1,
        )
        self.assertEqual(job["mode"], genjob_module.MODE_BATCH)
        self.assertNotIn("files", job)
        self.assertNotIn("image_path", job)

    def _sleeping_generator(self, job_id: str = "live_gen_1"):
        """A stand-in for a generator: a real process in its own session, recorded as a job."""
        child = REAL_POPEN([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        self._write_job(job_id, pid=child.pid)
        return child

    @staticmethod
    def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                Path(f"/proc/{pid}/stat").read_text()
            except OSError:
                return True
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()[0]
            except (OSError, IndexError):
                return True
            if state == "Z":  # killed and unreaped: it is no longer running
                return True
            time.sleep(0.02)
        return False

    def test_cancel_asks_the_running_generation_to_stop(self):
        child = self._sleeping_generator()
        try:
            result = api.handle_cancel_generation({})
            self.assertTrue(result["cancelled"])
            stored = json.loads(
                genjob.job_path(self.generated, "live_gen_1").read_text(encoding="utf-8")
            )
            # Still `running` while the process winds down, so the card is not handed over yet.
            self.assertEqual(stored["state"], "running")
            self.assertTrue(stored["cancel_requested"])
            self.assertTrue(self._wait_gone(child.pid))
        finally:
            child.wait()

    def test_a_cancelled_job_closes_as_cancelled_once_the_process_is_gone(self):
        from trainer import control

        child = self._sleeping_generator()
        try:
            api.handle_cancel_generation({})
            self._wait_gone(child.pid)
            jobs = api.dispatch("list_generated_samples", {})["jobs"]
            self.assertEqual(jobs[0]["state"], "cancelled")
            self.assertIsNone(jobs[0]["error"])
        finally:
            child.wait()

        # A job that died on its own is still an error.
        self._write_job("dead_gen_1", pid=999_999_999)
        jobs = api.dispatch("list_generated_samples", {})["jobs"]
        states = {job["id"]: job["state"] for job in jobs}
        self.assertEqual(states["dead_gen_1"], "error")

    def test_a_cancelling_job_still_holds_the_card(self):
        """A generator handles SIGTERM and finishes its step: until it is really gone, nothing else
        may start on that card. (A process that dies at once frees it — see the close test.)"""
        # The child says when it is ready: a SIGTERM that arrives before its handler is installed
        # would kill it outright, and the card would legitimately be free.
        ready = Path(self.tmp.name) / "stubborn.ready"
        script = (
            "import signal, time, pathlib; "
            "signal.signal(signal.SIGTERM, lambda *a: None); "
            f"pathlib.Path({str(ready)!r}).write_text('ready'); "
            "time.sleep(60)"
        )
        child = REAL_POPEN([sys.executable, "-c", script], start_new_session=True)
        deadline = time.time() + 10
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(ready.exists(), "the stand-in generator never got ready")
        self._write_job("stubborn_gen_1", pid=child.pid)
        try:
            api.handle_cancel_generation({})
            with self.assertRaises(ValueError) as ctx:
                api.handle_generate_checkpoint_samples({"checkpoint": str(self.checkpoint)})
            self.assertIn("already running", str(ctx.exception))
            self.popen.assert_not_called()
        finally:
            child.kill()
            child.wait()

    def test_cancel_refuses_when_nothing_runs(self):
        with self.assertRaises(ValueError) as ctx:
            api.handle_cancel_generation({})
        self.assertIn("no generation is running", str(ctx.exception))

        self._write_job("done_gen_1", state="done")
        with self.assertRaises(ValueError) as ctx:
            api.handle_cancel_generation({"id": "done_gen_1"})
        self.assertIn("done_gen_1 is not running", str(ctx.exception))

    def test_cancel_by_id_needs_no_run_of_its_own(self):
        """The reported bug: a pass started from a past run's card, with `state.json` carrying no run.

        The card sends the running job's id; the helper must find that job where it lives instead of
        resolving a run from the request (which found none and answered `no run to cancel a
        generation for`).
        """
        from trainer import control

        child = self._sleeping_generator("past_run_gen_1")
        try:
            self.cfg["logging_dir"] = str(Path(self.tmp.name) / "empty-logs")
            self.cfg["output_dir"] = str(self.out)  # the job is still under the configured root
            control.reset_to_idle()  # no run recorded at all

            result = api.handle_cancel_generation({"id": "past_run_gen_1"})

            self.assertTrue(result["cancelled"])
            stored = json.loads(
                genjob.job_path(self.generated, "past_run_gen_1").read_text(encoding="utf-8")
            )
            self.assertTrue(stored["cancel_requested"])
            self.assertTrue(self._wait_gone(child.pid))
        finally:
            child.wait()

    def test_cancel_with_no_id_still_finds_the_only_running_job(self):
        """What a client that sends no id at all relies on: the GPU is single-tenant."""
        from trainer import control

        child = self._sleeping_generator("only_gen_1")
        try:
            self.cfg["output_dir"] = str(self.out)
            control.reset_to_idle()  # no run recorded, so nothing names the job either

            result = api.handle_cancel_generation({})

            self.assertTrue(result["cancelled"])
            self.assertEqual(result["job"]["id"], "only_gen_1")
            self.assertTrue(self._wait_gone(child.pid))
        finally:
            child.wait()

    def test_cancel_by_id_refuses_an_unknown_or_finished_job(self):
        """With a run named, an id that is not a running job of it (or anywhere) is refused."""
        with self.assertRaises(ValueError) as ctx:
            api.handle_cancel_generation({"id": "never_ran_gen_1", "run_id": self.RUN_ID})
        self.assertIn("never_ran_gen_1 is not running", str(ctx.exception))

        self._write_job("done_gen_1", state="done")
        with self.assertRaises(ValueError) as ctx:
            api.handle_cancel_generation({"id": "done_gen_1", "run_id": self.RUN_ID})
        self.assertIn("is not running", str(ctx.exception))

        # Nothing named and no run recorded: the old message still says what is missing.
        from trainer import control

        control.reset_to_idle()
        with self.assertRaises(ValueError) as ctx:
            api.handle_cancel_generation({})
        self.assertIn("no run to cancel a generation for", str(ctx.exception))

    def test_cancel_can_name_one_job(self):
        first = self._sleeping_generator("first_gen_1")
        second = self._sleeping_generator("second_gen_1")
        try:
            api.handle_cancel_generation({"id": "second_gen_1"})
            recorded = {
                job_id: json.loads(
                    genjob.job_path(self.generated, job_id).read_text(encoding="utf-8")
                )["cancel_requested"]
                for job_id in ("first_gen_1", "second_gen_1")
            }
            self.assertEqual(recorded, {"first_gen_1": False, "second_gen_1": True})
            self.assertTrue(self._wait_gone(second.pid))
            self.assertIsNone(first.poll())
        finally:
            first.terminate()
            first.wait()
            second.wait()

    def test_a_batch_is_cancellable_like_any_other_job(self):
        from trainer import genjob as genjob_module

        child = REAL_POPEN([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        batch = genjob_module.new_batch_job(
            run_id=self.RUN_ID,
            output_name="rein",
            checkpoints=[{"path": "/out/rein_s000100/rein.safetensors", "step": 100}],
            from_step=100,
            to_step=100,
            images_per_checkpoint=1,
        )
        batch["pid"] = child.pid
        genjob_module.write_job(self.generated, batch)
        try:
            api.handle_cancel_generation({"id": str(batch["id"])})
            stored = json.loads(
                genjob_module.job_path(self.generated, str(batch["id"])).read_text(encoding="utf-8")
            )
            self.assertTrue(stored["cancel_requested"])
            self.assertEqual(stored["mode"], "batch")
            self.assertTrue(self._wait_gone(child.pid))
        finally:
            child.wait()

    def test_resume_is_refused_while_a_generation_runs(self):
        from trainer import control

        control.write_state({"status": "paused", "pid": os.getpid()}, force=True)
        self._write_job("live_gen_1", pid=os.getpid())
        with self.assertRaises(ValueError) as ctx:
            api.handle_train_resume({})
        self.assertIn("generation is using the GPU", str(ctx.exception))

        # A job whose process is gone is closed as an error instead of blocking the resume.
        self._write_job("live_gen_1", pid=999_999_999)
        control.write_state({"status": "paused", "pid": os.getpid()}, force=True)
        api.handle_train_resume({})
        self.assertTrue(control.peek_command())

    def test_train_settings_writes_the_request(self):
        from trainer import control

        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        result = api.handle_train_settings({"save_every_n_steps": 50, "sampling_enabled": False})
        self.assertEqual(result["status"], "training")
        self.assertEqual(control.read_settings(), {"save_every_n_steps": 50, "sampling_enabled": False})

        # One field at a time keeps the other where it was.
        api.handle_train_settings({"sampling_enabled": True})
        self.assertEqual(control.read_settings(), {"save_every_n_steps": 50, "sampling_enabled": True})

    def test_train_settings_validates_and_needs_a_live_run(self):
        from trainer import control

        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        for params, expected in (
            ({"save_every_n_steps": -1}, ">= 0"),
            ({"save_every_n_steps": "many"}, "integer"),
            ({"sampling_enabled": "yes"}, "boolean"),
            ({}, "nothing to change"),
        ):
            with self.subTest(params=params):
                with self.assertRaises(ValueError) as ctx:
                    api.handle_train_settings(params)
                self.assertIn(expected, str(ctx.exception))

        control.write_state({"status": "idle", "pid": None}, force=True)
        with self.assertRaises(ValueError):
            api.handle_train_settings({"save_every_n_steps": 10})

    def test_a_generation_from_another_run_blocks_a_new_one(self):
        """The GPU is single-tenant whatever run the job belongs to."""
        from trainer import genjob

        other = self.out / "elsewhere_20260910_120000" / "elsewhere_samples" / "generated"
        other.mkdir(parents=True)
        job = genjob.new_job(
            {"prompt": "p", "cfg": 5.0, "steps": 4, "seed": 1, "width": 512, "height": 512},
            run_id="elsewhere_20260910_120000",
            output_name="elsewhere",
            checkpoint=str(self.checkpoint),
            pid=os.getpid(),
        )
        genjob.write_job(other, job)

        with self.assertRaises(ValueError) as ctx:
            api.handle_generate_sample({"checkpoint": str(self.checkpoint), "prompt": "p"})
        self.assertIn("already running", str(ctx.exception))


class EvaluateCheckpointIpcTest(GeneratedFixture, unittest.TestCase):
    """`evaluate_checkpoint`: the plan it writes, the run it belongs to, and its refusals."""

    def _write_snapshot(self, *sets, run_id=None):
        """The `config.toml` a run saves beside its logs, as `trainer/main.py` writes it."""
        text = "".join(
            "[[validation.samples]]\n"
            f'name = "set {index}"\n'
            f'prompt = "{prompt}"\n'
            f"steps = {steps}\n"
            f"repeat = {repeat}\n\n"
            for index, (prompt, steps, repeat) in enumerate(sets)
        )
        target = self.logs / (run_id or self.RUN_ID) / "config.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def _evaluate(self, **params):
        return api.handle_evaluate_checkpoint(
            {"checkpoint": str(self.checkpoint), "depth": 4, **params}
        )

    def _stored(self) -> dict:
        return self._spec_written_by_last_spawn()

    def _selection(self):
        """This run's saved tag selection, or None when it has none."""
        from trainer import evaluation

        return evaluation.read_evaluation_tags(self.logs / self.RUN_ID)

    def test_dispatch_registered(self):
        self.assertIn("evaluate_checkpoint", api._HANDLERS)
        self.assertIn("evaluation_prompts", api._HANDLERS)

    def test_the_requested_tags_are_cleaned_and_recorded(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        self._evaluate(tags=[" (Anal:1.2) ", "1girl", "anal"])
        self.assertEqual(self._stored()["tags"], ["1girl", "anal"])

        self._evaluate()
        self.assertEqual(self._stored()["tags"], [])

        self._evaluate(tags="anal, pussy")
        self.assertEqual(self._stored()["tags"], ["anal", "pussy"])

    def test_starting_an_evaluation_saves_the_selection_for_this_run(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        self._evaluate(tags=[" (Anal:1.2) ", "1girl"])
        self.assertEqual(self._selection(), ["1girl", "anal"])

        # Clearing the picker is a choice like any other, and it is remembered as one.
        self._evaluate(tags=[])
        self.assertEqual(self._selection(), [])

    def test_a_refused_evaluation_saves_no_selection(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        with self.assertRaises(ValueError):
            self._evaluate(depth=0, tags=["anal"])
        self.assertIsNone(self._selection())

    def test_a_bad_tags_payload_is_refused(self):
        self._write_snapshot(("1girl", 9, 1))
        with self.assertRaises(ValueError):
            self._evaluate(tags={"anal": True})

    def test_the_prompt_picker_reads_the_runs_own_config(self):
        self._write_snapshot(("1girl, solo, anal", 9, 2), ("1girl, anal", 9, 1))
        result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})

        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(result["output_name"], "rein")
        self.assertEqual(
            result["config_source"], str((self.logs / self.RUN_ID / "config.toml").resolve())
        )
        self.assertEqual([entry["prompt"] for entry in result["sample_sets"]],
                         ["1girl, solo, anal", "1girl, anal"])
        self.assertEqual(
            result["tags"],
            [
                {"tag": "1girl", "count": 2, "frequency": 100.0},
                {"tag": "anal", "count": 2, "frequency": 100.0},
                {"tag": "solo", "count": 1, "frequency": 50.0},
            ],
        )

    def test_the_prompt_picker_opens_on_the_runs_saved_selection(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        self._evaluate(tags="anal")

        result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})

        self.assertEqual(result["selected_tags"], ["anal"])

    def test_the_picker_asks_for_everything_when_the_run_saved_nothing(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})
        self.assertEqual(result["selected_tags"], [])

    def test_the_picker_falls_back_to_the_newest_finished_evaluation(self):
        """A run from before `evaluation_tags.json`: its own last evaluation is its record."""
        self._write_snapshot(("1girl, anal", 9, 1))
        self._write_job(
            "old_evaluate_gen_1", mode="evaluate", state="done", tags=["1girl"], started_at=100.0
        )
        # `_write_job` leaves the state alone, so the newer record is still running.
        self._write_job("live_evaluate_gen_2", mode="evaluate", tags=["pussy"], started_at=200.0)

        result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})

        self.assertEqual(result["selected_tags"], ["1girl"])

    def test_a_finished_evaluation_is_read_from_its_scores_when_it_has_no_tags_field(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        self._write_job(
            "evaluate_gen_1",
            mode="evaluate",
            state="done",
            scores={"tags": [" (Anal:1.2) ", "1girl"]},
        )

        result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})

        self.assertEqual(result["selected_tags"], ["1girl", "anal"])

    def test_other_kinds_of_job_are_no_selection(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        self._write_job("sets_gen_1", mode="sets", state="done", tags=["anal"])
        self._write_job("single_gen_2", state="done", tags=["1girl"])

        result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})

        self.assertEqual(result["selected_tags"], [])

    def test_a_broken_saved_selection_warns_and_falls_back_to_the_evaluations(self):
        self._write_snapshot(("1girl, anal", 9, 1))
        log_dir = self.logs / self.RUN_ID
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "evaluation_tags.json").write_text("{not a list", encoding="utf-8")
        self._write_job("evaluate_gen_1", mode="evaluate", state="done", tags=["anal"])

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})

        self.assertEqual(result["selected_tags"], ["anal"])
        self.assertIn("[Warn]", stderr.getvalue())

    def test_the_selection_belongs_to_the_checkpoints_own_run(self):
        other = "kanae_20260101_000000"
        other_checkpoint = self.out / other / "kanae_s000100" / "kanae.safetensors"
        other_checkpoint.parent.mkdir(parents=True)
        other_checkpoint.write_bytes(b"weights")
        self._write_snapshot(("1girl, anal", 9, 1), run_id=other)
        self._write_snapshot(("1girl, anal", 9, 1))

        api.handle_evaluate_checkpoint(
            {"checkpoint": str(other_checkpoint), "depth": 1, "tags": "1girl"}
        )

        self.assertEqual(
            api.handle_evaluation_prompts({"checkpoint": str(other_checkpoint)})["selected_tags"],
            ["1girl"],
        )
        self.assertEqual(
            api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})["selected_tags"],
            [],
        )

    def test_the_prompt_picker_never_fails_over_an_unusable_config(self):
        # No snapshot and no hparams: the repo's `config.toml` is read instead, and whatever it
        # holds is offered rather than an error.
        result = api.handle_evaluation_prompts({"checkpoint": str(self.checkpoint)})
        self.assertTrue(result["sample_sets"])
        self.assertTrue(result["tags"])
        self.assertEqual(result["reason"], "")

    def test_the_prompt_picker_refuses_a_missing_checkpoint_and_writes_nothing(self):
        with self.assertRaises(ValueError):
            api.handle_evaluation_prompts({"checkpoint": ""})
        with self.assertRaises(ValueError):
            api.handle_evaluation_prompts({"checkpoint": str(self.out / "nope.safetensors")})

    def test_a_short_checkpoint_gets_a_plan_to_render(self):
        self._write_snapshot(("first prompt", 9, 1), ("second prompt", 9, 1))
        result = self._evaluate(depth=3)

        stored = self._stored()
        self.assertEqual(stored["mode"], "evaluate")
        self.assertEqual(stored["state"], "running")
        self.assertEqual(stored["phase"], "rendering")
        self.assertEqual(stored["depth"], 3)
        self.assertEqual(stored["threshold"], 0.35)
        self.assertEqual(stored["categories"], ["general"])
        self.assertEqual(stored["step"], 3050)
        self.assertEqual(stored["run_id"], self.RUN_ID)
        self.assertEqual(stored["checkpoint"], str(self.checkpoint))
        self.assertEqual(
            stored["config_source"], str((self.logs / self.RUN_ID / "config.toml").resolve())
        )
        self.assertEqual(stored["images"], [])
        # Two images a pass, a depth of three and nothing on disk yet: two whole passes.
        self.assertEqual(stored["plan"]["passes"], 2)
        self.assertEqual(stored["plan"]["per_pass"], 2)
        self.assertEqual(stored["plan"]["existing_images"], 0)
        self.assertTrue(stored["plan"]["needed"])
        self.assertEqual(stored["plan"]["render_total"], 4)
        self.assertEqual((stored["images_done"], stored["total_images"]), (0, 4))
        self.assertEqual(
            [entry["prompt"] for entry in stored["sample_sets"]], ["first prompt", "second prompt"]
        )
        # The runner rebuilds `SampleSet(**entry)` from these, so the record must carry exactly the
        # dataclass's fields.
        from dataclasses import fields

        from trainer.config import SampleSet

        self.assertEqual(
            set(stored["sample_sets"][0]), {item.name for item in fields(SampleSet)}
        )
        self.assertIsNone(stored["scores"])
        for key in ("current_step", "total_steps", "images_done", "total_images"):
            self.assertIsInstance(stored[key], int, key)
        self.assertEqual(result["job"]["id"], stored["id"])
        json.dumps(result)
        self.popen.assert_called_once()
        self.assertTrue(self.generated.is_dir())

    def test_enough_images_mean_no_render_at_all(self):
        self._write_snapshot(("first prompt", 9, 1), ("second prompt", 9, 1))
        for set_index in range(2):
            (self.samples / f"rein_003050_p{set_index}_0.png").write_bytes(b"png")

        self._evaluate(depth=2)

        stored = self._stored()
        self.assertEqual(stored["phase"], "tagging")
        self.assertFalse(stored["plan"]["needed"])
        self.assertEqual(stored["plan"]["render_total"], 0)
        # The counters are the phase's own: nothing to render, so they count the scored set.
        self.assertEqual((stored["images_done"], stored["total_images"]), (0, 2))
        self.assertEqual([image["set_index"] for image in stored["images"]], [0, 1])
        self.assertEqual(
            [image["prompt"] for image in stored["images"]], ["first prompt", "second prompt"]
        )
        self.assertEqual([image["source"] for image in stored["images"]], ["run", "run"])

    def test_images_of_an_earlier_pass_count_and_keep_their_own_prompt(self):
        self._write_snapshot(("first prompt", 9, 2))
        job_id = "rein_s003050_sets_gen_20260911_120000"
        self.generated.mkdir(parents=True, exist_ok=True)
        (self.generated / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "id": job_id,
                    "state": "done",
                    "mode": "sets",
                    "checkpoint": str(self.checkpoint),
                    "step": 3050,
                    "files": [
                        str(self.generated / "a_p0_0.png"),
                        str(self.generated / "a_p0_1.png"),
                    ],
                    "sample_sets": [{"prompt": "recorded prompt", "repeat": 2}],
                }
            ),
            encoding="utf-8",
        )

        self._evaluate(depth=2)

        stored = self._stored()
        self.assertFalse(stored["plan"]["needed"])
        self.assertEqual(len(stored["images"]), 2)
        self.assertEqual([image["source"] for image in stored["images"]], ["generated", "generated"])
        # Its own pass recorded what it drew with, which is not necessarily today's snapshot.
        self.assertEqual([image["prompt"] for image in stored["images"]], ["recorded prompt"] * 2)

    def test_a_bad_depth_or_threshold_is_refused_before_spawning(self):
        self._write_snapshot(("p", 9, 1))
        for params in ({"depth": 0}, {"depth": 513}, {"depth": "many"}, {"depth": 4, "threshold": 1.5}):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    api.handle_evaluate_checkpoint({"checkpoint": str(self.checkpoint), **params})
        self.popen.assert_not_called()

    def test_a_depth_past_the_render_cap_is_refused(self):
        self._write_snapshot(("a", 9, 1), ("b", 9, 1), ("c", 9, 1))
        with self.assertRaises(ValueError) as ctx:
            self._evaluate(depth=512)
        self.assertIn(str(512), str(ctx.exception))
        self.popen.assert_not_called()

    def test_the_categories_default_and_follow_the_request(self):
        self._write_snapshot(("p", 9, 1))
        self._evaluate(depth=1)
        self.assertEqual(self._stored()["categories"], ["general"])

        self._evaluate(depth=1, categories="general,rating", threshold=0.2)
        self.assertEqual(self._stored()["categories"], ["general", "rating"])
        self.assertEqual(self._stored()["threshold"], 0.2)

    def test_a_live_trainer_and_another_generation_are_refused(self):
        from trainer import control

        self._write_snapshot(("p", 9, 1))
        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        with self.assertRaises(ValueError) as ctx:
            self._evaluate(depth=1)
        self.assertIn("GPU", str(ctx.exception))

        control.write_state({"status": "finished", "pid": None}, force=True)
        self._write_job("live_gen_1", pid=os.getpid())
        with self.assertRaises(ValueError) as ctx:
            self._evaluate(depth=1)
        self.assertIn("already running", str(ctx.exception))
        self.popen.assert_not_called()

    def test_a_checkpoint_that_is_not_a_file_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            api.handle_evaluate_checkpoint(
                {"checkpoint": str(self.run_dir / "nope.safetensors"), "depth": 1}
            )
        self.assertIn("not a checkpoint file", str(ctx.exception))
        self.popen.assert_not_called()

    def test_a_broken_snapshot_is_reported_before_spawning(self):
        self._write_snapshot(("", 9, 1))
        with self.assertRaises(ValueError):
            self._evaluate(depth=1)
        self.popen.assert_not_called()

    def test_a_run_without_a_snapshot_falls_back_to_the_current_config(self):
        self.cfg.pop("samples", None)
        with mock.patch.object(
            api,
            "run_config_mapping",
            return_value=(
                {"samples": [{"prompt": "current config prompt", "steps": 9, "repeat": 1}]},
                "/repo/config.toml",
            ),
        ):
            self._evaluate(depth=1)
        stored = self._stored()
        self.assertEqual(stored["config_source"], "/repo/config.toml")
        self.assertEqual([entry["prompt"] for entry in stored["sample_sets"]], ["current config prompt"])

    def test_a_run_without_a_snapshot_uses_the_prompts_it_recorded_itself(self):
        """A run from before the `config.toml` copies: its own hparams supply the prompts."""
        from torch.utils.tensorboard import SummaryWriter

        from trainer.config import tracker_hparams

        (self.logs / self.RUN_ID / "config.toml").unlink()  # a run from before those copies
        writer = SummaryWriter(log_dir=str(self.logs / self.RUN_ID))
        try:
            writer.add_hparams(
                tracker_hparams(
                    {
                        "output_name": "rein",
                        "samples": [{"name": "recorded", "prompt": "its own prompt", "steps": 17, "repeat": 1}],
                    }
                ),
                {},
            )
        finally:
            writer.close()

        self._evaluate(depth=1)

        stored = self._stored()
        self.assertIn("events.out.tfevents.", stored["config_source"])
        self.assertIn(self.RUN_ID, stored["config_source"])
        self.assertEqual([entry["prompt"] for entry in stored["sample_sets"]], ["its own prompt"])
        self.assertEqual(stored["plan"]["per_pass"], 1)

    def test_the_checkpoint_of_another_run_is_evaluated_with_that_runs_config(self):
        other = "kanae_20260101_000000"
        other_checkpoint = self.out / other / "kanae_s000100" / "kanae.safetensors"
        other_checkpoint.parent.mkdir(parents=True)
        other_checkpoint.write_bytes(b"weights")
        self._write_snapshot(("other run prompt", 9, 1), run_id=other)
        self._write_snapshot(("the named run prompt", 9, 1))

        api.handle_evaluate_checkpoint({"checkpoint": str(other_checkpoint), "depth": 1})

        stored = self._stored()
        self.assertEqual(stored["run_id"], other)
        self.assertEqual(stored["output_name"], "kanae")
        self.assertEqual(
            stored["config_source"], str((self.logs / other / "config.toml").resolve())
        )
        self.assertEqual([entry["prompt"] for entry in stored["sample_sets"]], ["other run prompt"])
        # The job lives beside the samples it is about, which is where that card reads it from.
        self.assertTrue(
            (self.out / other / "kanae_samples" / "generated" / f"{stored['id']}.json").is_file()
        )

    def test_a_checkpoint_outside_the_output_root_uses_the_named_run(self):
        stray = Path(self.tmp.name) / "elsewhere" / "rein.safetensors"
        stray.parent.mkdir(parents=True)
        stray.write_bytes(b"weights")
        self._write_snapshot(("the named run prompt", 9, 1))

        api.handle_evaluate_checkpoint({"checkpoint": str(stray), "depth": 1})

        stored = self._stored()
        self.assertEqual(stored["run_id"], self.RUN_ID)
        self.assertEqual(stored["config_source"], str((self.logs / self.RUN_ID / "config.toml").resolve()))
        self.assertTrue((self.generated / f"{stored['id']}.json").is_file())

    def test_the_step_falls_back_to_the_metadata(self):
        from safetensors.torch import save_file

        target = self.run_dir / "rein_renamed" / "rein.safetensors"
        target.parent.mkdir(parents=True)
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(target),
            metadata={"ss_steps": "4242"},
        )
        self._write_snapshot(("p", 9, 1))
        (self.samples / "rein_004242_p0_0.png").write_bytes(b"png")

        api.handle_evaluate_checkpoint({"checkpoint": str(target), "depth": 1})

        stored = self._stored()
        self.assertEqual(stored["step"], 4242)
        self.assertFalse(stored["plan"]["needed"])
        self.assertEqual([image["name"] for image in stored["images"]], ["rein_004242_p0_0.png"])


class DatasetTagIpcTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        from trainer import control

        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.reset_to_idle()

    def tearDown(self):
        from trainer import control

        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_dispatch_registered(self):
        self.assertIn("dataset_tag", api._HANDLERS)

    def test_missing_directory(self):
        with self.assertRaises(ValueError):
            api.handle_dataset_tag({"directory": "/tmp/axl-missing-tag-dir", "threshold": 0.35})

    def test_no_directory_falls_back_to_the_first_train_data_entry(self):
        """A hand-written config carrying only `[[environment.train_data]]` blocks still tags."""
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(
                api,
                "_load_toml_config",
                return_value={"train_data": [{"path": raw, "repeat": 2}, {"path": "/tmp/second"}]},
            ):
                with mock.patch.object(
                    api, "run_tagger_process", return_value={"processed": 0}
                ) as tagged:
                    api.handle_dataset_tag({"threshold": 0.35})
            self.assertEqual(str(tagged.call_args.args[0]), raw)

    def test_bad_threshold(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                api.handle_dataset_tag({"directory": raw, "threshold": 1.5})

    def test_blocked_while_training(self):
        from trainer import control

        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35})

    def test_success_uses_tagger_result(self):
        fake = {
            "directory": "/tmp/alice",
            "threshold": 0.35,
            "provider": "MIGraphXExecutionProvider",
            "total": 2,
            "processed": 2,
            "failed": 0,
            "seconds": 0.1,
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(api, "run_tagger_process", return_value=fake) as tagged:
                result = api.dispatch("dataset_tag", {"directory": raw, "threshold": 0.4})
        self.assertEqual(result["processed"], 2)
        tagged.assert_called_once()
        args, kwargs = tagged.call_args
        self.assertEqual(args[1], 0.4)
        # No batch size asked for: the script's own default applies, so no `--batch-size` is passed.
        self.assertIsNone(kwargs["batch_size"])

    def test_categories_are_cleaned_and_joined(self):
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(api, "run_tagger_process", return_value={}) as tagged:
                api.handle_dataset_tag(
                    {"directory": raw, "threshold": 0.35, "categories": [" rating ", "general", "rating"]}
                )
            self.assertEqual(tagged.call_args.kwargs["categories"], "rating,general")

            with mock.patch.object(api, "run_tagger_process", return_value={}) as tagged:
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35, "categories": "meta, style"})
            self.assertEqual(tagged.call_args.kwargs["categories"], "meta,style")

            with mock.patch.object(api, "run_tagger_process", return_value={}) as tagged:
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35})
            self.assertIsNone(tagged.call_args.kwargs["categories"])

    def test_bad_categories_payload(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35, "categories": {"general": True}})

    def test_batch_size_is_validated_and_passed_through(self):
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(api, "run_tagger_process", return_value={}) as tagged:
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35, "batch_size": 4})
            self.assertEqual(tagged.call_args.kwargs["batch_size"], 4)

            with self.assertRaises(ValueError):
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35, "batch_size": 0})
            with self.assertRaises(ValueError):
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35, "batch_size": "many"})

    def test_only_tags_is_cleaned_and_passed_through(self):
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(api, "run_tagger_process", return_value={}) as tagged:
                api.handle_dataset_tag(
                    {"directory": raw, "threshold": 0.6, "only_tags": [" anal ", "pussy", "anal"]}
                )
            self.assertEqual(tagged.call_args.kwargs["only_tags"], ["anal", "pussy"])
            self.assertEqual(tagged.call_args.args[1], 0.6)

            with mock.patch.object(api, "run_tagger_process", return_value={}) as tagged:
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35, "only_tags": "anal, pussy"})
            self.assertEqual(tagged.call_args.kwargs["only_tags"], ["anal", "pussy"])

            with mock.patch.object(api, "run_tagger_process", return_value={}) as tagged:
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35})
            self.assertIsNone(tagged.call_args.kwargs["only_tags"])

    def test_a_bad_only_tags_payload_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                api.handle_dataset_tag({"directory": raw, "threshold": 0.35, "only_tags": {"anal": True}})

    def test_a_partial_pass_does_not_send_categories(self):
        """The script looks a requested tag up in every category, so `--categories` stays out."""
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(api.subprocess, "run") as run:
                run.return_value = mock.Mock(returncode=0, stdout='{"processed": 0}', stderr="")
                api.run_tagger_process(raw, 0.35, categories="general", only_tags=["anal"])
            argv = run.call_args.args[0]
            self.assertIn("--only-tags", argv)
            self.assertNotIn("--categories", argv)

            with mock.patch.object(api.subprocess, "run") as run:
                run.return_value = mock.Mock(returncode=0, stdout='{"processed": 0}', stderr="")
                api.run_tagger_process(raw, 0.35, categories="general")
            argv = run.call_args.args[0]
            self.assertIn("--categories", argv)
            self.assertNotIn("--only-tags", argv)


class DatasetCountsIpcTest(unittest.TestCase):
    """`dataset_counts`: the image counts the Training section's step estimate is built from."""

    def test_registered(self):
        self.assertIn("dataset_counts", api._HANDLERS)

    def test_explicit_dirs_are_counted(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "1.png").write_bytes(b"x")
            (root / "2.jpg").write_bytes(b"x")
            result = api.dispatch(
                "dataset_counts", {"dirs": [{"path": raw, "repeat": 3}]}
            )
        self.assertEqual(result["entries"][0]["images"], 2)
        self.assertEqual(result["images"], 2)
        self.assertEqual(result["samples"], 6)

    def test_no_dirs_falls_back_to_the_configured_train_data(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "1.png").write_bytes(b"x")
            with mock.patch.object(
                api,
                "_load_toml_config",
                return_value={"train_data": [{"path": raw, "repeat": 2}]},
            ):
                result = api.handle_dataset_counts({})
        self.assertEqual(result["entries"][0]["path"], raw)
        self.assertEqual(result["samples"], 2)

    def test_a_missing_folder_answers_with_its_reason(self):
        with tempfile.TemporaryDirectory() as raw:
            result = api.handle_dataset_counts({"dirs": [{"path": str(Path(raw) / "gone")}]})
        self.assertEqual(result["entries"][0]["error"], "not a directory")
        self.assertEqual(result["images"], 0)

    def test_a_malformed_dirs_payload_is_refused(self):
        with self.assertRaises(ValueError):
            api.handle_dataset_counts({"dirs": "all of them"})
        with self.assertRaises(ValueError):
            api.handle_dataset_counts({"dirs": [["/tmp"]]})


class TaggerInfoIpcTest(unittest.TestCase):
    def test_registered(self):
        self.assertIn("tagger_info", api._HANDLERS)

    def test_payload_is_passed_through(self):
        payload = {"available": True, "engine": "pixai-tagger-v1.0", "categories": [{"key": "general"}]}
        with mock.patch.object(api, "run_tagger_info", return_value=payload):
            self.assertEqual(api.handle_tagger_info({}), payload)

    def test_a_failure_answers_with_a_reason_instead_of_raising(self):
        with mock.patch.object(api, "run_tagger_info", side_effect=RuntimeError("no model")):
            result = api.handle_tagger_info({})
        self.assertFalse(result["available"])
        self.assertIn("no model", result["reason"])
        self.assertEqual(result["categories"], [])

    def test_a_missing_script_answers_with_a_reason(self):
        with mock.patch.object(api, "_repo_root", return_value=Path("/tmp/axl-no-repo")):
            result = api.dispatch("tagger_info", {})
        self.assertFalse(result["available"])
        self.assertTrue(result["reason"])


class HardwareStatusTest(unittest.TestCase):
    def setUp(self):
        from trainer import hardware as hw

        hw.reset_cpu_tracker()
        self.hw = hw

    def tearDown(self):
        self.hw.reset_cpu_tracker()

    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_parse_nvtop_metric_strings(self):
        self.assertEqual(self.hw.parse_metric_number("92%"), 92.0)
        self.assertEqual(self.hw.parse_metric_number("303W"), 303.0)
        self.assertEqual(self.hw.parse_metric_number("72C"), 72.0)
        self.assertEqual(self.hw.parse_metric_number("2165MHz"), 2165.0)
        self.assertEqual(self.hw.parse_bytes("17095983104"), 17095983104)
        self.assertIsNone(self.hw.parse_metric_number("N/A"))
        self.assertIsNone(self.hw.parse_metric_number(None))

    def test_collect_parses_snapshot_and_drops_processes(self):
        snapshot = [
            {
                "device_name": "AMD Radeon RX 9070 XT",
                "gpu_clock": "2165MHz",
                "mem_clock": "2500MHz",
                "temp": "72C",
                "fan_speed": "30%",
                "power_draw": "303W",
                "gpu_util": "92%",
                "mem_util": "76%",
                "mem_total": "17095983104",
                "mem_used": "13000000000",
                "mem_free": "4095983104",
                "processes": [{"pid": "1", "cmdline": "x" * 5000}],
            }
        ]
        result = self.hw.collect_hardware_status(
            nvtop_runner=lambda: snapshot,
            drm_root="/tmp/axl-missing-drm",
            proc_stat="/tmp/axl-missing-stat",
            proc_cpuinfo="/tmp/axl-missing-cpuinfo",
            proc_meminfo="/tmp/axl-missing-meminfo",
            thermal_root="/tmp/axl-missing-thermal",
            now=1710000000.12,
            amdfq_choice="none",
        )
        self.assertTrue(result["available"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["ts"], 1710000000.12)
        gpu = result["gpus"][0]
        self.assertNotIn("processes", gpu)
        self.assertEqual(gpu["name"], "AMD Radeon RX 9070 XT")
        self.assertEqual(gpu["gpu_util_pct"], 92.0)
        self.assertEqual(gpu["power_w"], 303.0)
        self.assertEqual(gpu["temp_edge_c"], 72.0)
        self.assertIsNone(gpu["temp_junction_c"])
        self.assertEqual(gpu["mem_used_bytes"], 13000000000)
        self.assertIsNone(result["cpu"]["mem_total_bytes"])
        json.dumps(result)

    def test_missing_nvtop_is_unavailable_not_an_ipc_error(self):
        def boom():
            raise FileNotFoundError("nvtop not found on PATH")

        forced = self.hw.collect_hardware_status(
            nvtop_runner=boom,
            drm_root="/tmp/axl-missing-drm",
            proc_stat="/tmp/axl-missing-stat",
            proc_cpuinfo="/tmp/axl-missing-cpuinfo",
            proc_meminfo="/tmp/axl-missing-meminfo",
            thermal_root="/tmp/axl-missing-thermal",
            amdfq_choice="none",
        )
        self.assertFalse(forced["available"])
        self.assertIn("nvtop", forced["error"])
        self.assertEqual(forced["gpus"], [])
        self.assertIn("cpu", forced)
        self.assertIn("hardware_status", api._HANDLERS)

    def test_hwmon_fills_edge_and_junction(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            hwmon = root / "card1" / "device" / "hwmon" / "hwmon2"
            self._write(hwmon / "temp1_label", "edge\n")
            self._write(hwmon / "temp1_input", "72000\n")
            self._write(hwmon / "temp2_label", "junction\n")
            self._write(hwmon / "temp2_input", "85000\n")
            self._write(hwmon / "temp3_label", "mem\n")
            self._write(hwmon / "temp3_input", "80000\n")
            snapshot = [
                {
                    "device_name": "AMD Radeon RX 9070 XT",
                    "temp": "70C",
                    "gpu_util": "10%",
                    "power_draw": "50W",
                    "mem_total": "100",
                    "mem_used": "40",
                    "mem_free": "60",
                }
            ]
            result = self.hw.collect_hardware_status(
                nvtop_runner=lambda: snapshot,
                drm_root=root,
                proc_stat="/tmp/axl-missing-stat",
                proc_cpuinfo="/tmp/axl-missing-cpuinfo",
                proc_meminfo="/tmp/axl-missing-meminfo",
                thermal_root="/tmp/axl-missing-thermal",
                amdfq_choice="none",
            )
            gpu = result["gpus"][0]
            self.assertEqual(gpu["temp_edge_c"], 72.0)
            self.assertEqual(gpu["temp_c"], 72.0)
            self.assertEqual(gpu["temp_junction_c"], 85.0)
            self.assertEqual(gpu["temp_mem_c"], 80.0)

    def test_cpu_util_is_proc_stat_delta_and_prefers_pkg_temp(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stat_path = root / "stat"
            cpuinfo = root / "cpuinfo"
            thermal = root / "thermal"
            self._write(cpuinfo, "processor\t: 0\nmodel name\t: Test CPU\n")
            self._write(thermal / "thermal_zone0" / "type", "acpitz\n")
            self._write(thermal / "thermal_zone0" / "temp", "27800\n")
            self._write(thermal / "thermal_zone1" / "type", "iwlwifi_1\n")
            self._write(thermal / "thermal_zone1" / "temp", "57000\n")
            self._write(thermal / "thermal_zone2" / "type", "x86_pkg_temp\n")
            self._write(thermal / "thermal_zone2" / "temp", "41000\n")

            def snapshot():
                return [{"device_name": "GPU", "gpu_util": "1%", "temp": "40C"}]

            self._write(stat_path, "cpu  100 0 50 850 0 0 0 0 0 0\n")
            first = self.hw.collect_hardware_status(
                nvtop_runner=snapshot,
                drm_root=root / "missing-drm",
                proc_stat=stat_path,
                proc_cpuinfo=cpuinfo,
                proc_meminfo="/tmp/axl-missing-meminfo",
                thermal_root=thermal,
                amdfq_choice="none",
            )
            self.assertIsNone(first["cpu"]["util_pct"])
            self.assertEqual(first["cpu"]["name"], "Test CPU")
            self.assertEqual(first["cpu"]["temp_c"], 41.0)

            # 50 more busy, 50 more idle → 50% util
            self._write(stat_path, "cpu  150 0 50 900 0 0 0 0 0 0\n")
            second = self.hw.collect_hardware_status(
                nvtop_runner=snapshot,
                drm_root=root / "missing-drm",
                proc_stat=stat_path,
                proc_cpuinfo=cpuinfo,
                proc_meminfo="/tmp/axl-missing-meminfo",
                thermal_root=thermal,
                amdfq_choice="none",
            )
            self.assertAlmostEqual(second["cpu"]["util_pct"], 50.0)

    def test_cpu_meminfo_used_from_available(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            meminfo = root / "meminfo"
            self._write(meminfo, "MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n")

            result = self.hw.collect_hardware_status(
                nvtop_runner=lambda: [{"device_name": "GPU", "gpu_util": "1%", "temp": "40C"}],
                drm_root=root / "missing-drm",
                proc_stat="/tmp/axl-missing-stat",
                proc_cpuinfo="/tmp/axl-missing-cpuinfo",
                proc_meminfo=meminfo,
                thermal_root=root / "missing-thermal",
                amdfq_choice="none",
            )
            self.assertEqual(result["cpu"]["mem_total_bytes"], 16384000 * 1024)
            self.assertEqual(result["cpu"]["mem_used_bytes"], 8192000 * 1024)

    def test_vmm_va_absent_when_patch_is_not_vmm(self):
        result = self.hw.collect_hardware_status(
            nvtop_runner=lambda: [{"device_name": "GPU", "gpu_util": "1%"}],
            drm_root="/tmp/axl-missing-drm",
            proc_stat="/tmp/axl-missing-stat",
            proc_cpuinfo="/tmp/axl-missing-cpuinfo",
            proc_meminfo="/tmp/axl-missing-meminfo",
            thermal_root="/tmp/axl-missing-thermal",
            amdfq_choice="tail",
        )
        self.assertIsNone(result["vmm_va"])

    def test_vmm_va_reads_status_file_for_live_pid(self):
        with tempfile.TemporaryDirectory() as raw:
            stem = Path(raw) / "amdfq_vmm_va"
            pid = os.getpid()
            (Path(raw) / f"amdfq_vmm_va.{pid}.json").write_text(
                json.dumps(
                    {"pid": pid, "used_bytes": 8388608, "spans": 4, "never_reuse": True, "ts": 1.0}
                ),
                encoding="utf-8",
            )
            journal = "amdgpu 0000:03:00.0: vm size is 262144 GB, 4 levels\n"
            result = self.hw.collect_hardware_status(
                nvtop_runner=lambda: [{"device_name": "GPU", "gpu_util": "1%"}],
                drm_root="/tmp/axl-missing-drm",
                proc_stat="/tmp/axl-missing-stat",
                proc_cpuinfo="/tmp/axl-missing-cpuinfo",
                proc_meminfo="/tmp/axl-missing-meminfo",
                thermal_root="/tmp/axl-missing-thermal",
                amdfq_choice="vmm",
                journal_text=journal,
                dmesg_text="",
                vm_size_param="-1",
                trainer_pid=pid,
                va_status_stem=stem,
            )
            va = result["vmm_va"]
            self.assertEqual(va["patch"], "vmm")
            self.assertEqual(va["used_bytes"], 8388608)
            self.assertEqual(va["spans"], 4)
            self.assertEqual(va["pid"], pid)
            self.assertEqual(va["total_source"], "journal")
            self.assertEqual(va["total_bytes"], 262144 * 1024 * 1024 * 1024)
            self.assertTrue(va["never_reuse"])

    def test_vmm_va_status_file_mode_wins_over_config(self):
        """The hook's own mode is what the panel describes, not what the next run is configured for."""
        with tempfile.TemporaryDirectory() as raw:
            stem = Path(raw) / "amdfq_vmm_va"
            pid = os.getpid()
            (Path(raw) / f"amdfq_vmm_va.{pid}.json").write_text(
                json.dumps({"pid": pid, "used_bytes": 0, "spans": 0, "never_reuse": False}),
                encoding="utf-8",
            )
            va = self.hw.collect_vmm_va(
                amdfq_choice="vmm",
                vmm_total=(1 << 40, "default"),
                trainer_pid=pid,
                va_status_stem=stem,
            )
            self.assertFalse(va["never_reuse"])

    def test_vmm_va_without_status_file_reports_the_configured_mode(self):
        with tempfile.TemporaryDirectory() as raw:
            stem = Path(raw) / "amdfq_vmm_va"
            va = self.hw.collect_vmm_va(
                amdfq_choice="vmm",
                vmm_total=(1 << 40, "default"),
                trainer_pid=os.getpid(),
                va_status_stem=stem,
            )
            self.assertEqual(va["used_bytes"], 0)
            self.assertEqual(va["spans"], 0)
            self.assertIsInstance(va["never_reuse"], bool)

    def test_parse_vm_size_text_takes_last_match(self):
        text = (
            "amdgpu 0000:03:00.0: vm size is 128 GB\n"
            "amdgpu 0000:03:00.0: vm size is 262144 GB, 4 levels\n"
        )
        self.assertEqual(self.hw.parse_vm_size_text(text), 262144 * 1024 * 1024 * 1024)
        self.assertIsNone(self.hw.parse_vm_size_param("-1"))
        self.assertEqual(self.hw.parse_vm_size_param("256"), 256 * 1024 * 1024 * 1024)

    def test_dispatch_hardware_status_never_raises(self):
        result = api.dispatch("hardware_status", {})
        self.assertIn("available", result)
        self.assertIn("gpus", result)
        self.assertIn("cpu", result)
        json.dumps(result)


class SamplePromptsIpcTest(GeneratedFixture, unittest.TestCase):
    """`sample_prompts` / `sample_prompts_set`: one run's prompts, and editing them.

    The fixture's run has its own `config.toml` copy (`_write_run_config`), which is what its
    prompts resolve from; `sample_sets.json` is the layer the editor writes on top of it.
    """

    def _read(self, **params):
        return api.handle_sample_prompts({"name": "rein", "run_id": self.RUN_ID, **params})

    def _set(self, sets, **params):
        return api.handle_sample_prompts_set(
            {"name": "rein", "run_id": self.RUN_ID, "sets": sets, **params}
        )

    def test_dispatch_registered(self):
        self.assertIn("sample_prompts", api._HANDLERS)
        self.assertIn("sample_prompts_set", api._HANDLERS)

    def test_the_runs_own_config_is_the_source_when_nothing_was_edited(self):
        result = self._read()
        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(result["output_name"], "rein")
        self.assertFalse(result["edited"])
        self.assertIsNone(result["file"])
        self.assertEqual(
            result["config_source"], str((self.logs / self.RUN_ID / "config.toml").resolve())
        )
        self.assertEqual(result["sets"][0]["prompt"], "config prompt")
        self.assertEqual(result["reason"], "")
        json.dumps(result)

    def test_an_edit_replaces_the_prompt_sets_and_is_reported_as_the_source(self):
        # This run's own config says 640 / x2; the repo's file says something else. An omitted key
        # has to come from the run, or a partial edit would mix two runs' settings.
        self.cfg["sample_width"] = 640
        self.cfg["sample_repeat"] = 2
        before = self._write_run_config().read_text(encoding="utf-8")

        result = self._set([{"prompt": "a new prompt", "steps": 12}])

        self.assertTrue(result["edited"])
        self.assertEqual(result["file"], str(sample_override_path(self.logs / self.RUN_ID)))
        self.assertEqual(result["config_source"], result["file"])
        self.assertEqual([set_["prompt"] for set_ in result["sets"]], ["a new prompt"])
        self.assertEqual(result["sets"][0]["width"], 640)
        self.assertEqual(result["sets"][0]["repeat"], 2)
        # Nothing was written to the snapshot itself: it stays what the run trained with.
        self.assertEqual(
            (self.logs / self.RUN_ID / "config.toml").read_text(encoding="utf-8"), before
        )

    def test_the_edit_is_what_a_reader_of_the_run_resolves(self):
        from trainer.config import resolve_sample_sets, run_config_mapping

        self._set([{"prompt": "edited prompt", "steps": 12, "repeat": 1}])
        mapping, source = run_config_mapping(self.logs / self.RUN_ID)
        self.assertEqual(source, str(sample_override_path(self.logs / self.RUN_ID)))
        self.assertEqual([set_.prompt for set_ in resolve_sample_sets(mapping)], ["edited prompt"])

    def test_null_sets_drop_the_edit(self):
        self._set([{"prompt": "edited prompt", "steps": 12}])
        result = self._set(None)
        self.assertFalse(result["edited"])
        self.assertIsNone(result["file"])
        self.assertEqual(result["sets"][0]["prompt"], "config prompt")
        self.assertFalse(sample_override_path(self.logs / self.RUN_ID).exists())

    def test_a_live_run_is_flagged_but_a_stopped_one_is_not(self):
        from trainer import control

        self.assertFalse(self._read()["live"])
        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        self.assertTrue(self._read()["live"])
        # Another run being live does not make this one live.
        control.write_state({"status": "training", "pid": os.getpid(), "run_id": "other_20260101_000000"}, force=True)
        self.assertFalse(self._read()["live"])

    def test_an_unusable_config_answers_with_a_reason(self):
        self._set_run_samples([{"prompt": "", "steps": 9}])
        result = self._read()
        self.assertEqual(result["sets"], [])
        self.assertIn("prompt", result["reason"])

    def test_no_run_answers_empty(self):
        from trainer import control

        control.reset_to_idle()
        result = api.handle_sample_prompts({"name": "nothing_here"})
        self.assertIsNone(result["run_id"])
        self.assertEqual(result["sets"], [])
        self.assertIn("no run", result["reason"])

    def test_a_bad_edit_is_refused(self):
        cases = [
            ([{"prompt": ""}], "prompt"),
            ([{"prompt": "p", "steps": 0}], "steps"),
            ([{"prompt": "p", "width": 8}], "width"),
            ("not a list", "non-empty array"),
            ([], "non-empty array"),
        ]
        for sets, expected in cases:
            with self.subTest(sets=sets):
                with self.assertRaises(ValueError) as ctx:
                    self._set(sets)
                self.assertIn(expected, str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            api.handle_sample_prompts_set({"run_id": self.RUN_ID})
        self.assertIn("sets is required", str(ctx.exception))
        # Nothing was written by any of them.
        self.assertFalse(sample_override_path(self.logs / self.RUN_ID).exists())

    def test_editing_a_past_run_creates_its_log_directory(self):
        run_id = "gone_20260101_000000"
        result = api.handle_sample_prompts_set(
            {"run_id": run_id, "name": "gone", "sets": [{"prompt": "for a past run", "steps": 9}]}
        )
        self.assertEqual(result["run_id"], run_id)
        self.assertTrue((self.logs / run_id / "sample_sets.json").is_file())

    def test_no_run_to_edit_is_refused(self):
        from trainer import control

        control.reset_to_idle()
        with self.assertRaises(ValueError) as ctx:
            api.handle_sample_prompts_set({"name": "nothing_here", "sets": [{"prompt": "p"}]})
        self.assertIn("no run", str(ctx.exception))


class ClearCheckpointSamplesIpcTest(GeneratedFixture, unittest.TestCase):
    """`clear_checkpoint_samples`: one card's images and the records that produced them."""

    def setUp(self):
        super().setUp()
        self.generated.mkdir(parents=True, exist_ok=True)

    def _own_sample(self, step: int, set_index: int = 0, repeat: int = 0) -> Path:
        path = self.samples / f"rein_{step:06d}_p{set_index}_{repeat}.png"
        path.write_bytes(b"png")
        return path

    def _job_with_images(self, job_id: str, step: int, checkpoint: str, count: int = 2) -> tuple[dict, list[Path]]:
        files = []
        for index in range(count):
            path = self.generated / f"{job_id}_p0_{index}.png"
            path.write_bytes(b"png")
            files.append(path)
        job = self._write_job(job_id, step=step, checkpoint=checkpoint, state="done", pid=None)
        job = api.genjob.update_job(self.generated, job_id, files=[str(path) for path in files])
        return job, files

    def test_the_steps_own_samples_and_their_passes_go(self):
        own = self._own_sample(3050, 0, 0)
        other_step = self._own_sample(3000, 0, 0)
        job, files = self._job_with_images("rein_s003050_sets_gen_1", 3050, str(self.checkpoint))

        result = api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint), "run_id": self.RUN_ID})

        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(result["step"], 3050)
        self.assertEqual(result["images"], 3)
        self.assertEqual(result["jobs"], ["rein_s003050_sets_gen_1"])
        self.assertFalse(own.exists())
        for path in files:
            self.assertFalse(path.exists())
        self.assertFalse(api.genjob.job_path(self.generated, job["id"]).exists())
        self.assertIn(str(own), result["files"])
        json.dumps(result)

        # Another step's samples are untouched.
        self.assertTrue(other_step.exists())

    def test_another_checkpoints_pass_stays(self):
        other = self._checkpoint_file("rein_s003000")
        kept_job, kept_files = self._job_with_images("rein_s003000_sets_gen_1", 3000, str(other))
        mine = self._job_with_images("rein_s003050_sets_gen_1", 3050, str(self.checkpoint))

        result = api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint), "run_id": self.RUN_ID})

        self.assertEqual(result["jobs"], [mine[0]["id"]])
        for path in kept_files:
            self.assertTrue(path.exists())
        self.assertTrue(api.genjob.job_path(self.generated, kept_job["id"]).exists())

    def test_a_record_that_names_no_checkpoint_is_matched_by_its_step(self):
        # A pass recorded without a checkpoint path (an older record) still belongs to the card of
        # the step it was written at — the rule the section itself groups by.
        job = self._write_job("rein_s003050_gen_old", step=3050, checkpoint="", state="done", pid=None)
        api.genjob.update_job(self.generated, "rein_s003050_gen_old", files=[])

        result = api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint), "run_id": self.RUN_ID})

        self.assertEqual(result["jobs"], [job["id"]])
        self.assertFalse(api.genjob.job_path(self.generated, job["id"]).exists())

    def test_the_log_of_a_cleared_pass_goes_too(self):
        job, _files = self._job_with_images("rein_s003050_sets_gen_1", 3050, str(self.checkpoint), count=1)
        log = api.genjob.log_path(self.generated, job["id"])
        log.write_text("noise", encoding="utf-8")

        api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint), "run_id": self.RUN_ID})

        self.assertFalse(log.exists())

    def test_a_checkpoint_without_images_is_a_no_op(self):
        result = api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint), "run_id": self.RUN_ID})
        self.assertEqual((result["images"], result["jobs"], result["files"]), (0, [], []))

    def test_a_live_trainer_on_the_gpu_is_refused(self):
        from trainer import control

        for status in ("starting", "encoding", "training", "sampling", "pausing", "resuming", "stopping"):
            with self.subTest(status=status):
                control.write_state({"status": status, "pid": os.getpid()}, force=True)
                with self.assertRaises(ValueError) as ctx:
                    api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint)})
                self.assertIn("GPU", str(ctx.exception))

    def test_a_paused_run_may_clear(self):
        from trainer import control

        own = self._own_sample(3050)
        control.write_state({"status": "paused", "pid": os.getpid()}, force=True)
        result = api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint), "run_id": self.RUN_ID})
        self.assertEqual(result["images"], 1)
        self.assertFalse(own.exists())

    def test_a_running_generation_is_refused(self):
        self._write_job("live_gen_1", pid=os.getpid(), state="running")
        with self.assertRaises(ValueError) as ctx:
            api.handle_clear_checkpoint_samples({"checkpoint": str(self.checkpoint), "run_id": self.RUN_ID})
        self.assertIn("still using this card", str(ctx.exception))

    def test_a_file_that_is_not_a_checkpoint_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            api.handle_clear_checkpoint_samples({"checkpoint": str(self.run_dir / "nope.safetensors")})
        self.assertIn("not a checkpoint file", str(ctx.exception))

    def _checkpoint_file(self, directory: str) -> Path:
        target = self.run_dir / directory / "rein.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")
        return target


if __name__ == "__main__":
    unittest.main()
