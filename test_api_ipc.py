import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import api
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

    def test_dashboard_falls_back_to_latest_run_dir(self):
        self._make_run("rein_20260101_000000")
        self._make_run("rein_20260911_120000")
        (self.logs / "rein").mkdir(parents=True)  # legacy flat dir is ignored
        result = api.dispatch("dashboard", {})
        self.assertEqual(result["run_id"], "rein_20260911_120000")

    def test_dashboard_explicit_run_id_wins(self):
        self._make_run("rein_20260101_000000")
        result = api.dispatch("dashboard", {"run_id": self.RUN_ID})
        self.assertEqual(result["run_id"], self.RUN_ID)

    def test_list_samples_reads_run_dir(self):
        run_dir = self._make_run()
        (run_dir / "rein_samples" / "rein_000100_0.png").write_bytes(b"x")
        (run_dir / "rein_samples" / "rein_000200_0.png").write_bytes(b"x")
        result = api.dispatch("list_samples", {})
        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(list(result["samples"].keys()), ["200", "100"])

    def test_list_samples_without_run_is_empty(self):
        result = api.dispatch("list_samples", {})
        self.assertIsNone(result["run_id"])
        self.assertEqual(result["samples"], {})

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

    def test_reset_cleans_run_dir(self):
        run_dir = self._make_run()
        (run_dir / "rein_samples" / "a.png").write_bytes(b"x")
        (self.logs / self.RUN_ID / "events.out.tfevents.1").write_bytes(b"e")
        weights = run_dir / "rein_s000100"
        weights.mkdir(parents=True)
        (weights / "rein.safetensors").write_bytes(b"w")

        result = api.dispatch("train_reset", {})
        self.assertEqual(result["status"], "idle")
        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertFalse((run_dir / "rein_samples").exists())
        self.assertFalse((self.logs / self.RUN_ID).exists())
        self.assertTrue(weights.exists())
        self.assertEqual(result["cleanup"]["weight_dirs"], [str(weights)])

    def test_reset_can_delete_weights_and_run_dir(self):
        run_dir = self._make_run()
        weights = run_dir / "rein_final"
        weights.mkdir(parents=True)
        (weights / "rein.safetensors").write_bytes(b"w")

        result = api.dispatch("train_reset", {"delete_weights": True})
        self.assertFalse(weights.exists())
        self.assertFalse(run_dir.exists())
        self.assertEqual(result["cleanup"]["run_id"], self.RUN_ID)

    def test_reset_without_run_leaves_legacy_alone(self):
        legacy_samples = self.out / "rein_samples"
        legacy_samples.mkdir(parents=True)
        (legacy_samples / "a.png").write_bytes(b"x")
        (self.logs / "rein").mkdir(parents=True)

        result = api.dispatch("train_reset", {})
        self.assertIsNone(result["run_id"])
        self.assertTrue(legacy_samples.exists())
        self.assertTrue((self.logs / "rein").exists())


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


if __name__ == "__main__":
    unittest.main()
