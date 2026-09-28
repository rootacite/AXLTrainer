import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import sys

# `python test/test_train_control.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

    def test_start_refuses_a_bad_sample_set_before_spawning(self):
        from types import SimpleNamespace

        bad = SimpleNamespace(
            base_model_version="sdxl_base_v1-0",
            modelspec_architecture="stable-diffusion-xl-v1-base/lora",
            modelspec_implementation="https://github.com/Stability-AI/generative-models",
            modelspec_sai_model_spec="1.0.0",
            resume_lora_path="",
            sample_prompts="p",
            sample_negative="",
            sample_width=1024,
            sample_height=1024,
            sample_steps=30,
            sample_seed=0,
            sample_repeat=1,
            guidance_scale=5.0,
            samples=[{"prompt": "p", "steps": 999}],
        )
        with mock.patch.object(api, "TrainConfig", lambda: bad):
            with mock.patch("api.subprocess.Popen") as popen:
                with self.assertRaises(ValueError) as ctx:
                    api.dispatch("train_start")
        self.assertIn("validation.samples[1]", str(ctx.exception))
        popen.assert_not_called()

    def test_begin_run_records_run_id(self):
        control.begin_run(4242, "rein", run_id="rein_20260911_120000")
        loaded = control.read_state()
        self.assertEqual(loaded["run_id"], "rein_20260911_120000")
        self.assertEqual(loaded["status"], "starting")
        self.assertIsNone(loaded["resume"])

    def test_set_resume_roundtrip(self):
        control.begin_run(4242, "rein", run_id="rein_20260911_120000")
        control.set_resume({"path": "/tmp/rein.safetensors", "step": 300, "loaded": 42})
        loaded = control.read_state()
        self.assertEqual(loaded["resume"]["step"], 300)
        self.assertEqual(loaded["resume"]["loaded"], 42)
        control.set_resume({})
        self.assertIsNone(control.read_state()["resume"])

    def test_sampling_progress_reports_the_prompt_set(self):
        control.set_sampling(
            active=True,
            repeat=3,
            repeats=9,
            denoise_step=5,
            denoise_steps=20,
            global_step=3000,
            prompt_set=2,
            prompt_sets=3,
        )
        sampling = control.status_payload()["sampling"]
        self.assertEqual(sampling["prompt_set"], 2)
        self.assertEqual(sampling["prompt_sets"], 3)
        self.assertEqual(sampling["repeat"], 3)

    def test_sampling_without_sets_defaults_to_zero(self):
        control.set_sampling(active=True, repeat=0, repeats=1, denoise_step=1, denoise_steps=10)
        sampling = control.status_payload()["sampling"]
        self.assertEqual(sampling["prompt_set"], 0)
        self.assertEqual(sampling["prompt_sets"], 0)

    def test_reset_clears_finished_and_keeps_artifacts(self):
        out = Path(self.tmp.name) / "out"
        logs = Path(self.tmp.name) / "logs"
        run_id = "rein_20260911_120000"
        run_dir = out / run_id
        samples = run_dir / "rein_samples"
        tb = logs / run_id
        weights = run_dir / "rein_s000100"
        samples.mkdir(parents=True)
        (samples / "a.png").write_bytes(b"x")
        tb.mkdir(parents=True)
        (tb / "events.out.tfevents.1").write_bytes(b"e")
        weights.mkdir(parents=True)
        (weights / "rein.safetensors").write_bytes(b"w")
        control.write_state(
            {
                "status": "finished",
                "pid": None,
                "output_name": "rein",
                "run_id": run_id,
            },
            force=True,
        )
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
        self.assertEqual(result["run_id"], run_id)
        # Samples and logs stay: the run is still browsable in the dashboard's history.
        self.assertTrue(samples.exists())
        self.assertTrue(tb.exists())
        self.assertTrue(weights.exists())
        self.assertIn(str(weights), result["cleanup"]["weight_dirs"])
        self.assertEqual(result["cleanup"]["removed"], [])

    def test_reset_can_delete_weights(self):
        out = Path(self.tmp.name) / "out"
        logs = Path(self.tmp.name) / "logs"
        run_id = "rein_20260911_120000"
        weights = out / run_id / "rein_final"
        weights.mkdir(parents=True)
        (weights / "rein.safetensors").write_bytes(b"w")
        control.write_state(
            {"status": "finished", "pid": None, "output_name": "rein", "run_id": run_id},
            force=True,
        )
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
        # Nothing was kept, so the now-empty run directory is still pruned.
        self.assertFalse((out / run_id).exists())

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

    def test_checkpoint_file_follows_run_dir(self):
        class Cfg:
            output_dir = "/tmp/out"
            output_name = "rein"
            run_dir = "/tmp/out/rein_20260911_120000"

        path = lora_checkpoint_file(Cfg(), 100)
        self.assertEqual(path.parent.parent.name, "rein_20260911_120000")
        final = lora_checkpoint_file(Cfg(), 12, final=True)
        self.assertEqual(final.parent.parent.name, "rein_20260911_120000")


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
