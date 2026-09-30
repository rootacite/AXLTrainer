import os
import subprocess
import sys
import tempfile
import time
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

    def test_train_start_publishes_the_settings_the_run_starts_with(self):
        """The card must read config.toml's cadence from the moment Start is pressed."""
        from types import SimpleNamespace

        config = SimpleNamespace(
            base_model_version="sdxl_base_v1-0",
            modelspec_architecture="stable-diffusion-xl-v1-base/lora",
            modelspec_implementation="https://github.com/Stability-AI/generative-models",
            modelspec_sai_model_spec="1.0.0",
            resume_lora_path="",
            save_every_n_steps=250,
            sampling_enabled=False,
            sample_prompts="p",
            sample_negative="",
            sample_width=1024,
            sample_height=1024,
            sample_steps=30,
            sample_seed=0,
            sample_repeat=1,
            guidance_scale=5.0,
            samples=[],
        )

        class FakeProc:
            pid = 4242

        with mock.patch.object(api, "TrainConfig", lambda: config):
            with mock.patch.object(api, "_train_config_dict", lambda: {"output_name": "rein"}):
                with mock.patch("api.subprocess.Popen", return_value=FakeProc()):
                    result = api.dispatch("train_start")

        self.assertEqual(
            result["settings"],
            {"save_every_n_steps": 250, "sampling_enabled": False, "next_save_step": 250},
        )
        self.assertEqual(
            control.read_settings(),
            {"save_every_n_steps": 250, "sampling_enabled": False},
        )

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

    def test_reset_never_deletes_weights(self):
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
            # The old `delete_weights` flag is gone; sending it anyway must not remove anything.
            result = api.dispatch("train_reset", {"delete_weights": True})
        finally:
            api._train_config_dict = orig
        self.assertTrue(weights.exists())
        self.assertEqual(result["cleanup"]["removed"], [])
        self.assertIn(str(weights), result["cleanup"]["weight_dirs"])

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


class ProcessIdentityTest(unittest.TestCase):
    """`is_pid_alive`: a finished run must not look live to the api.py that spawned it."""

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

    def _state_of(self, pid: int, want: str, timeout: float = 10.0) -> str:
        deadline = time.time() + timeout
        state = ""
        while time.time() < deadline:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()[0]
            except (OSError, IndexError):
                state = ""
            if state == want:
                return state
            time.sleep(0.01)
        return state

    def test_a_running_child_is_alive(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertEqual(self._state_of(child.pid, "S"), "S")
            self.assertTrue(control.is_pid_alive(child.pid))
        finally:
            child.terminate()
            child.wait()

    def test_a_child_nobody_waited_on_is_not_alive(self):
        """The reported bug: api.py never `wait()`s the trainer it spawns, so after a normal finish
        the trainer is a zombie — `kill(pid, 0)` still succeeds on it, and the dashboard kept
        `alive` true (and the panel's Generate button disabled) until Ranko was restarted."""
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        try:
            self.assertEqual(self._state_of(child.pid, "Z"), "Z")
            self.assertFalse(control.is_pid_alive(child.pid))
        finally:
            child.wait()  # reap it, the way api.py does not

    def test_a_reaped_or_missing_pid_is_not_alive(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        self.assertFalse(control.is_pid_alive(child.pid))
        self.assertFalse(control.is_pid_alive(None))
        self.assertFalse(control.is_pid_alive(-1))
        self.assertFalse(control.is_pid_alive("not a pid"))

    def test_reconcile_closes_a_run_whose_trainer_was_never_reaped(self):
        """A live status plus a zombie pid is a dead run, not a live one."""
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        try:
            self.assertEqual(self._state_of(child.pid, "Z"), "Z")
            control.write_state({"status": "training", "pid": child.pid}, force=True)
            payload = control.status_payload()
            self.assertEqual(payload["status"], "error")
            self.assertFalse(payload["alive"])
            self.assertIn("no longer running", payload["error"])
        finally:
            child.wait()

    def test_reconcile_keeps_a_live_trainer_live(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertEqual(self._state_of(child.pid, "S"), "S")
            control.write_state({"status": "training", "pid": child.pid}, force=True)
            payload = control.status_payload()
            self.assertEqual(payload["status"], "training")
            self.assertTrue(payload["alive"])
        finally:
            child.terminate()
            child.wait()


class LiveSettingsTest(unittest.TestCase):
    """The runtime cadence / sampling switch: settings.json, the state block, and the schedule."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()
        control.clear_settings()

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _config(self, **fields):
        from types import SimpleNamespace

        return SimpleNamespace(**{"save_every_n_steps": 100, "sampling_enabled": True, **fields})

    def test_from_config_anchors_the_first_save_at_n(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=25))
        self.assertEqual(settings.save_every_n_steps, 25)
        self.assertEqual(settings.next_save_step, 25)
        self.assertFalse(settings.due(24))
        self.assertTrue(settings.due(25))

    def test_untouched_cadence_keeps_the_modulo_sequence(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=100))
        saved = []
        for step in range(1, 501):
            if settings.due(step):
                saved.append(step)
                settings.mark_saved(step)
        self.assertEqual(saved, [100, 200, 300, 400, 500])

    def test_a_missed_boundary_still_saves_at_the_next_step(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=100))
        # Step 100 went by without a save (paused, or the loop was between epochs): the next
        # step that reaches the loop still writes the checkpoint.
        self.assertTrue(settings.due(137))
        settings.mark_saved(137)
        self.assertEqual(settings.next_save_step, 237)

    def test_zero_disables_the_schedule(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=0))
        self.assertFalse(settings.due(0))
        self.assertFalse(settings.due(1000))

    def test_a_changed_cadence_restarts_from_the_change(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=100))
        settings.mark_saved(100)
        adopted = settings.adopt({"save_every_n_steps": 50, "sampling_enabled": True}, global_step=120)
        self.assertIsNot(adopted, settings)
        self.assertEqual(adopted.next_save_step, 170)
        self.assertFalse(adopted.due(169))
        self.assertTrue(adopted.due(170))

    def test_adopting_nothing_keeps_the_same_object(self):
        settings = control.LiveSettings.from_config(self._config())
        self.assertIs(settings.adopt(None, 10), settings)
        self.assertIs(
            settings.adopt({"save_every_n_steps": 100, "sampling_enabled": True}, 10),
            settings,
        )

    def test_the_switch_can_change_on_its_own(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=100))
        adopted = settings.adopt({"save_every_n_steps": 100, "sampling_enabled": False}, 30)
        self.assertFalse(adopted.sampling_enabled)
        # Only the switch moved, so the schedule does not restart.
        self.assertEqual(adopted.next_save_step, 100)

    def test_a_corrupt_request_is_ignored(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=100))
        adopted = settings.adopt({"save_every_n_steps": "soon", "sampling_enabled": True}, 30)
        self.assertEqual(adopted.save_every_n_steps, 100)

    def test_read_settings_needs_a_usable_file(self):
        self.assertIsNone(control.read_settings())
        control.settings_path().write_text("{not json", encoding="utf-8")
        self.assertIsNone(control.read_settings())
        control.settings_path().write_text('{"sampling_enabled": false}', encoding="utf-8")
        self.assertIsNone(control.read_settings())

    def test_request_settings_merges_and_resets(self):
        self.assertEqual(
            control.request_settings(save_every_n_steps=40),
            {"save_every_n_steps": 40, "sampling_enabled": True},
        )
        self.assertEqual(
            control.request_settings(sampling_enabled=False),
            {"save_every_n_steps": 40, "sampling_enabled": False},
        )
        self.assertEqual(
            control.request_settings(save_every_n_steps=-5)["save_every_n_steps"],
            0,
        )

    def test_request_settings_falls_back_to_the_running_values(self):
        # No `settings.json` at all (a run started by hand with `bash start_train.sh`): a request
        # that names one field must not write the default 0 for the other.
        baseline = {"save_every_n_steps": 50, "sampling_enabled": True, "next_save_step": 3350}
        self.assertEqual(
            control.request_settings(sampling_enabled=False, baseline=baseline),
            {"save_every_n_steps": 50, "sampling_enabled": False},
        )
        # The file wins over the baseline for a field it does carry: it is the newer request.
        control.settings_path().write_text(
            '{"save_every_n_steps": 100, "sampling_enabled": true}', encoding="utf-8"
        )
        self.assertEqual(
            control.request_settings(sampling_enabled=False, baseline=baseline),
            {"save_every_n_steps": 100, "sampling_enabled": False},
        )
        # The request itself wins over both.
        self.assertEqual(
            control.request_settings(save_every_n_steps=7, baseline=baseline),
            {"save_every_n_steps": 7, "sampling_enabled": False},
        )
        # A baseline without a usable value leaves the old default in place.
        control.settings_path().unlink()
        self.assertEqual(
            control.request_settings(sampling_enabled=True, baseline={"save_every_n_steps": "lots"}),
            {"save_every_n_steps": 0, "sampling_enabled": True},
        )

    def test_publish_settings_lands_in_the_state_block(self):
        settings = control.LiveSettings.from_config(self._config(save_every_n_steps=7))
        control.publish_settings(settings)
        self.assertEqual(
            control.read_state()["settings"],
            {"save_every_n_steps": 7, "sampling_enabled": True, "next_save_step": 7},
        )
        # The other blocks of the state file survive a settings publish.
        control.write_state({"status": "training", "training": {"step": 12}}, force=True)
        control.publish_settings(settings)
        loaded = control.read_state()
        self.assertEqual(loaded["training"]["step"], 12)
        self.assertEqual(loaded["settings"]["save_every_n_steps"], 7)

    def test_reset_clears_the_request(self):
        control.request_settings(save_every_n_steps=33)
        control.reset_to_idle()
        self.assertIsNone(control.read_settings())
        self.assertFalse(control.settings_path().exists())
        self.assertEqual(control.read_state()["settings"]["save_every_n_steps"], 0)


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
