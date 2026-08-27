import json
import tempfile
import unittest
from pathlib import Path

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
            result = api.handle_dashboard({"name": "__avg_loss_synth__"})
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
            result = api.handle_dashboard({"name": "__avg_loss_keep__"})
            self.assertEqual(result["metrics"]["Train/Avg_Loss"][0]["value"], 1.5)
        finally:
            api._get_tensorboard_metrics = orig


if __name__ == "__main__":
    unittest.main()
