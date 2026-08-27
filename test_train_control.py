import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import api
from trainer import control
from trainer.device_swap import optimizer_tensors_to
from trainer.models import lora_checkpoint_file, safe_output_name


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_runtime_dir_override(self):
        self.assertEqual(control.runtime_dir(), Path(self.tmp.name))

    def test_atomic_write_roundtrip(self):
        control.write_state({"status": "training", "pid": 7}, force=True)
        loaded = control.read_state()
        self.assertEqual(loaded["status"], "training")
        self.assertEqual(loaded["pid"], 7)
        self.assertTrue((Path(self.tmp.name) / "state.json").is_file())
        self.assertFalse((Path(self.tmp.name) / "state.json.tmp").exists())

    def test_stale_pid_reconcile(self):
        control.write_state({"status": "training", "pid": 99999999}, force=True)
        result = control.reconcile()
        self.assertEqual(result["status"], "error")
        self.assertIn("no longer running", result["error"])

    def test_command_seq(self):
        first = control.request("pause")
        self.assertEqual(first["op"], "pause")
        self.assertEqual(control.peek_command(), "pause")
        self.assertEqual(control.poll_command(), "pause")
        self.assertIsNone(control.peek_command())
        second = control.request("resume")
        self.assertGreater(second["seq"], first["seq"])
        self.assertEqual(control.poll_command(), "resume")

    def test_concurrent_state_writes(self):
        import threading

        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                for step in range(30):
                    control.write_state(
                        {"status": "training", "pid": index, "detail": str(step)},
                        force=True,
                    )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        loaded = control.read_state()
        self.assertEqual(loaded["status"], "training")
        self.assertTrue(control.state_path().is_file())
        leftovers = list(Path(self.tmp.name).glob("state.json.*.tmp"))
        self.assertEqual(leftovers, [])

    def test_lock_same_process(self):
        self.assertTrue(control.try_acquire_lock())
        self.assertTrue(control.try_acquire_lock())
        control.release_lock()

    def test_dispatch_status_idle(self):
        result = api.dispatch("train_status")
        self.assertEqual(result["status"], "idle")
        self.assertFalse(result["alive"])
        self.assertTrue(str(result["log_path"]).endswith("train.log"))

    def test_pause_without_process_fails(self):
        with self.assertRaises(ValueError):
            api.dispatch("train_pause")

    def test_train_start_refuses_live_pid(self):
        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        with self.assertRaises(ValueError):
            api.dispatch("train_start")

    def test_train_start_spawns_detached(self):
        spawned = {}

        class FakeProc:
            pid = 4242

        def fake_popen(*args, **kwargs):
            spawned["kwargs"] = kwargs
            spawned["args"] = args
            return FakeProc()

        with mock.patch("api.subprocess.Popen", side_effect=fake_popen):
            result = api.dispatch("train_start")
        self.assertEqual(result["pid"], 4242)
        self.assertEqual(result["status"], "starting")
        self.assertTrue(spawned["kwargs"]["start_new_session"])
        self.assertIs(spawned["kwargs"]["stdin"], subprocess.DEVNULL)

    def test_start_refuses_finished_while_pid_alive(self):
        control.write_state({"status": "finished", "pid": os.getpid()}, force=True)
        with self.assertRaises(ValueError):
            api.dispatch("train_start")

    def test_reset_clears_finished_and_logs(self):
        out = Path(self.tmp.name) / "out"
        logs = Path(self.tmp.name) / "logs"
        samples = out / "rein_samples"
        tb = logs / "rein"
        weights = out / "rein_s000100"
        samples.mkdir(parents=True)
        (samples / "a.png").write_bytes(b"x")
        tb.mkdir(parents=True)
        (tb / "events.out.tfevents.1").write_bytes(b"e")
        weights.mkdir(parents=True)
        (weights / "rein.safetensors").write_bytes(b"w")
        control.write_state({"status": "finished", "pid": None, "output_name": "rein"}, force=True)
        orig = api._train_config_dict
        api._train_config_dict = lambda: {
            "output_dir": str(out),
            "logging_dir": str(logs),
            "output_name": "rein",
        }
        try:
            result = api.dispatch("train_reset", {"delete_weights": False})
        finally:
            api._train_config_dict = orig
        self.assertEqual(result["status"], "idle")
        self.assertIsNone(result.get("pid"))
        self.assertFalse(samples.exists())
        self.assertFalse(tb.exists())
        self.assertTrue(weights.exists())
        self.assertIn(str(weights), result["cleanup"]["weight_dirs"])

    def test_reset_can_delete_weights(self):
        out = Path(self.tmp.name) / "out"
        logs = Path(self.tmp.name) / "logs"
        weights = out / "rein_final"
        weights.mkdir(parents=True)
        logs.mkdir(parents=True)
        orig = api._train_config_dict
        api._train_config_dict = lambda: {
            "output_dir": str(out),
            "logging_dir": str(logs),
            "output_name": "rein",
        }
        try:
            api.dispatch("train_reset", {"delete_weights": True})
        finally:
            api._train_config_dict = orig
        self.assertFalse(weights.exists())

    def test_reset_refuses_live_pid(self):
        control.write_state({"status": "training", "pid": os.getpid()}, force=True)
        with self.assertRaises(ValueError):
            api.dispatch("train_reset")

    def test_reset_allows_finished_even_if_pid_still_listed(self):
        control.write_state({"status": "finished", "pid": os.getpid()}, force=True)
        orig = api._train_config_dict
        api._train_config_dict = lambda: {
            "output_dir": str(Path(self.tmp.name) / "out"),
            "logging_dir": str(Path(self.tmp.name) / "logs"),
            "output_name": "rein",
        }
        try:
            result = api.dispatch("train_reset", {})
        finally:
            api._train_config_dict = orig
        self.assertEqual(result["status"], "idle")


class FilenameTest(unittest.TestCase):
    def test_safe_name(self):
        self.assertEqual(safe_output_name("rein"), "rein")
        self.assertEqual(safe_output_name("re in/x"), "re_in_x")
        self.assertEqual(safe_output_name("   "), "lora")

    def test_checkpoint_file_uses_output_name(self):
        class Cfg:
            output_dir = "/tmp/out"
            output_name = "rein"

        path = lora_checkpoint_file(Cfg(), 100)
        self.assertEqual(path.name, "rein.safetensors")
        self.assertEqual(path.parent.name, "rein_s000100")
        final = lora_checkpoint_file(Cfg(), 12, final=True)
        self.assertEqual(final.name, "rein.safetensors")
        self.assertEqual(final.parent.name, "rein_final")


class OptimizerSwapTest(unittest.TestCase):
    def test_walks_state_and_param_group_tensors(self):
        module = torch.nn.Linear(4, 4)
        try:
            from schedulefree import AdamWScheduleFree

            opt = AdamWScheduleFree(module.parameters(), lr=1e-3, warmup_steps=0, foreach=False)
            opt.train()
            loss = module(torch.ones(2, 4)).sum()
            loss.backward()
            opt.step()
            extra_keys = ("z", "exp_avg_sq")
        except Exception:
            opt = torch.optim.AdamW(module.parameters(), lr=1e-3)
            loss = module(torch.ones(2, 4)).sum()
            loss.backward()
            opt.step()
            extra_keys = ("exp_avg", "exp_avg_sq")

        opt.param_groups[0]["scheduled_lr_tensor"] = torch.tensor(1.0)
        optimizer_tensors_to(opt, torch.device("cpu"))
        for bucket in opt.state.values():
            for key in extra_keys:
                if key in bucket and torch.is_tensor(bucket[key]):
                    self.assertEqual(bucket[key].device.type, "cpu")
        self.assertEqual(opt.param_groups[0]["scheduled_lr_tensor"].device.type, "cpu")
        self.assertTrue(all(p.grad is None for g in opt.param_groups for p in g["params"]))


if __name__ == "__main__":
    unittest.main()
