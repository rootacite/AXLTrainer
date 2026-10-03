"""The Checkpoints section's pinned checkpoints: the per-run JSON file and the two IPC methods."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# `python test/test_checkpoint_pins.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api
from trainer.checkpoints import (
    PIN_FILENAME,
    pin_entry,
    pins_path,
    read_pins,
    unpin_entry,
    write_pins,
)


class PinStoreTest(unittest.TestCase):
    """The file itself: what it accepts, what it writes, and the two list edits."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = Path(self.tmp.name) / "logs" / "rein_20260911_120000"

    def test_pins_path_is_the_runs_own_log_directory(self):
        self.assertEqual(pins_path(self.log_dir), self.log_dir / PIN_FILENAME)

    def test_missing_file_reads_as_no_pins(self):
        self.assertEqual(read_pins(self.log_dir), [])

    def test_write_creates_the_directory_and_round_trips(self):
        entries = pin_entry([], "/out/rein_s000100/rein.safetensors", dir_name="rein_s000100", step=100)
        path = write_pins(self.log_dir, "rein_20260911_120000", entries)

        self.assertEqual(path, self.log_dir / PIN_FILENAME)
        self.assertTrue(path.is_file())
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["run_id"], "rein_20260911_120000")
        self.assertEqual(read_pins(self.log_dir), payload["pins"])
        self.assertEqual(read_pins(self.log_dir)[0]["step"], 100)
        self.assertEqual(read_pins(self.log_dir)[0]["dir"], "rein_s000100")

    def test_write_replaces_the_previous_list(self):
        write_pins(self.log_dir, "run", pin_entry([], "/a.safetensors"))
        write_pins(self.log_dir, "run", pin_entry([], "/b.safetensors"))
        self.assertEqual([entry["path"] for entry in read_pins(self.log_dir)], ["/b.safetensors"])

    def test_pinning_appends_at_the_end(self):
        first = pin_entry([], "/a.safetensors")
        both = pin_entry(first, "/b.safetensors")
        self.assertEqual([entry["path"] for entry in both], ["/a.safetensors", "/b.safetensors"])

    def test_pinning_twice_changes_nothing(self):
        once = pin_entry([], "/a.safetensors", dir_name="rein_s000100", step=100)
        again = pin_entry(once, "/a.safetensors", dir_name="other", step=200)
        self.assertEqual(again, once)
        self.assertEqual(again[0]["dir"], "rein_s000100")

    def test_pinning_an_empty_path_is_refused(self):
        with self.assertRaises(ValueError):
            pin_entry([], "   ")

    def test_unpinning_drops_only_that_path(self):
        pins = pin_entry(pin_entry([], "/a.safetensors"), "/b.safetensors")
        self.assertEqual([entry["path"] for entry in unpin_entry(pins, "/a.safetensors")], ["/b.safetensors"])

    def test_unpinning_something_not_pinned_changes_nothing(self):
        pins = pin_entry([], "/a.safetensors")
        self.assertEqual(unpin_entry(pins, "/gone.safetensors"), pins)

    def test_the_pinned_at_stamp_is_a_float(self):
        entry = pin_entry([], "/a.safetensors")[0]
        self.assertIsInstance(entry["pinned_at"], float)
        self.assertEqual(pin_entry([], "/a.safetensors", pinned_at=1730000000.0)[0]["pinned_at"], 1730000000.0)

    def test_a_corrupt_file_reads_as_no_pins(self):
        self.log_dir.mkdir(parents=True)
        (self.log_dir / PIN_FILENAME).write_text("{ not json", encoding="utf-8")
        self.assertEqual(read_pins(self.log_dir), [])

    def test_a_hand_written_file_is_read_leniently(self):
        self.log_dir.mkdir(parents=True)
        (self.log_dir / PIN_FILENAME).write_text(
            json.dumps([" /a.safetensors ", "/a.safetensors", "/b.safetensors", 7, {"nope": 1}]),
            encoding="utf-8",
        )
        # A bare list of paths works, a duplicate and a path-less entry are dropped, a non-entry
        # is skipped rather than raising: the file is the user's state, not a run artifact.
        self.assertEqual([entry["path"] for entry in read_pins(self.log_dir)], ["/a.safetensors", "/b.safetensors"])
        self.assertEqual(read_pins(self.log_dir)[0]["step"], None)

    def test_a_file_holding_something_else_reads_as_no_pins(self):
        self.log_dir.mkdir(parents=True)
        (self.log_dir / PIN_FILENAME).write_text(json.dumps({"pins": "soon"}), encoding="utf-8")
        self.assertEqual(read_pins(self.log_dir), [])


class CheckpointPinIpcTest(unittest.TestCase):
    """`checkpoint_pins` / `checkpoint_pin_set` through the dispatcher."""

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
        (run_dir / "rein_s000100").mkdir(parents=True)
        (self.logs / run_id).mkdir(parents=True)
        return run_dir

    def _checkpoint(self, run_id: str = RUN_ID, step: int = 100) -> Path:
        run_dir = self.out / run_id
        target = run_dir / f"rein_s{step:06d}" / "rein.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"lora")
        return target

    def _record_run(self, run_id: str = RUN_ID) -> None:
        from trainer import control

        control.write_state(
            {"status": "finished", "pid": None, "output_name": "rein", "run_id": run_id},
            force=True,
        )

    def _pin_file(self, run_id: str = RUN_ID) -> Path:
        return self.logs / run_id / PIN_FILENAME

    def test_without_a_run_there_is_nothing_to_read(self):
        self.assertEqual(
            api.dispatch("checkpoint_pins", {}),
            {"run_id": None, "file": None, "pins": []},
        )

    def test_without_a_run_pinning_is_refused(self):
        with self.assertRaises(ValueError):
            api.dispatch("checkpoint_pin_set", {"path": "/a.safetensors", "pinned": True})

    def test_pinning_writes_the_runs_log_file(self):
        self._make_run()
        self._record_run()
        target = self._checkpoint()

        result = api.dispatch(
            "checkpoint_pin_set",
            {"path": str(target), "pinned": True, "dir": "rein_s000100", "step": 100},
        )

        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(result["file"], str(self._pin_file()))
        self.assertEqual([entry["path"] for entry in result["pins"]], [str(target)])
        # The reply is what the file holds, so a client that polls reads the same list back.
        self.assertEqual(api.dispatch("checkpoint_pins", {})["pins"], result["pins"])
        self.assertEqual(
            [entry["path"] for entry in api.dispatch("checkpoint_pins", {"run_id": self.RUN_ID})["pins"]],
            [str(target)],
        )

    def test_pinning_keeps_the_order_it_was_done_in(self):
        self._make_run()
        self._record_run()
        first = self._checkpoint(step=100)
        second = self._checkpoint(step=200)

        api.dispatch("checkpoint_pin_set", {"path": str(second), "pinned": True, "step": 200})
        result = api.dispatch("checkpoint_pin_set", {"path": str(first), "pinned": True, "step": 100})

        self.assertEqual([entry["path"] for entry in result["pins"]], [str(second), str(first)])

    def test_pinning_twice_keeps_one_entry(self):
        self._make_run()
        self._record_run()
        target = self._checkpoint()

        api.dispatch("checkpoint_pin_set", {"path": str(target), "pinned": True})
        result = api.dispatch("checkpoint_pin_set", {"path": str(target), "pinned": True})

        self.assertEqual(len(result["pins"]), 1)

    def test_unpinning_leaves_the_others(self):
        self._make_run()
        self._record_run()
        kept = self._checkpoint(step=100)
        dropped = self._checkpoint(step=200)
        api.dispatch("checkpoint_pin_set", {"path": str(kept), "pinned": True})
        api.dispatch("checkpoint_pin_set", {"path": str(dropped), "pinned": True})

        result = api.dispatch("checkpoint_pin_set", {"path": str(dropped), "pinned": False})

        self.assertEqual([entry["path"] for entry in result["pins"]], [str(kept)])
        self.assertEqual(len(api.dispatch("checkpoint_pins", {})["pins"]), 1)

    def test_unpinning_a_checkpoint_whose_file_is_gone_still_works(self):
        # Reset deletes the weights and keeps the pin file, so a stale entry has to stay removable.
        self._make_run()
        self._record_run()
        target = self._checkpoint()
        api.dispatch("checkpoint_pin_set", {"path": str(target), "pinned": True})
        target.unlink()

        result = api.dispatch("checkpoint_pin_set", {"path": str(target), "pinned": False})

        self.assertEqual(result["pins"], [])

    def test_pinning_a_path_that_is_not_a_file_is_refused(self):
        self._make_run()
        self._record_run()

        with self.assertRaises(ValueError):
            api.dispatch("checkpoint_pin_set", {"path": str(self.out / "nope.safetensors"), "pinned": True})
        self.assertEqual(api.dispatch("checkpoint_pins", {})["pins"], [])

    def test_a_missing_pinned_flag_is_refused(self):
        self._make_run()
        self._record_run()
        target = self._checkpoint()

        with self.assertRaises(ValueError):
            api.dispatch("checkpoint_pin_set", {"path": str(target)})

    def test_an_empty_path_is_refused(self):
        self._make_run()
        self._record_run()

        with self.assertRaises(ValueError):
            api.dispatch("checkpoint_pin_set", {"path": "  ", "pinned": True})

    def test_pins_belong_to_one_run(self):
        other = "rein_20260912_130000"
        self._make_run()
        self._make_run(other)
        self._record_run()
        first = self._checkpoint(step=100)
        second = self._checkpoint(other, step=300)

        api.dispatch("checkpoint_pin_set", {"path": str(first), "pinned": True})
        api.dispatch("checkpoint_pin_set", {"path": str(second), "pinned": True, "run_id": other})

        self.assertEqual(
            [entry["path"] for entry in api.dispatch("checkpoint_pins", {"run_id": other})["pins"]],
            [str(second)],
        )
        self.assertEqual(
            [entry["path"] for entry in api.dispatch("checkpoint_pins", {"run_id": self.RUN_ID})["pins"]],
            [str(first)],
        )

    def test_pinning_a_past_run_creates_its_log_directory(self):
        # A run whose TensorBoard directory was deleted is still shown in the history list; the pin
        # file is where its pins belong, so the directory comes back rather than the write failing.
        other = "rein_20260101_000000"
        self._make_run(other)
        (self.logs / other).rmdir()
        target = self._checkpoint(other, step=50)

        result = api.dispatch(
            "checkpoint_pin_set",
            {"path": str(target), "pinned": True, "name": "rein", "run_id": other},
        )

        self.assertTrue(self._pin_file(other).is_file())
        self.assertEqual(result["run_id"], other)
        self.assertEqual(result["file"], str(self._pin_file(other)))

    def test_reading_pins_never_writes_the_file(self):
        self._make_run()
        self._record_run()

        api.dispatch("checkpoint_pins", {})

        self.assertFalse(self._pin_file().exists())

    def _sample(self, run_id: str = RUN_ID, name: str = "rein_000100_0.png") -> Path:
        samples = self.out / run_id / "rein_samples"
        samples.mkdir(parents=True, exist_ok=True)
        image = samples / name
        image.write_bytes(b"png")
        return image

    def test_clear_unpinned_keeps_pinned_weights_and_samples(self):
        self._make_run()
        self._record_run()
        pinned = self._checkpoint(step=100)
        loose = self._checkpoint(step=200)
        image = self._sample()
        api.dispatch("checkpoint_pin_set", {"path": str(pinned), "pinned": True, "step": 100})

        result = api.dispatch("clear_unpinned_checkpoints", {"run_id": self.RUN_ID, "name": "rein"})

        self.assertEqual(result["run_id"], self.RUN_ID)
        self.assertEqual(result["removed"], [str(loose.parent.resolve())])
        self.assertEqual(result["kept"], [str(pinned.resolve())])
        self.assertEqual(result["errors"], [])
        self.assertTrue(pinned.is_file())
        self.assertFalse(loose.parent.exists())
        self.assertTrue(image.is_file())
        self.assertTrue(self._pin_file().is_file())

    def test_clear_unpinned_leaves_another_run_alone(self):
        other = "rein_20260912_130000"
        self._make_run()
        self._make_run(other)
        self._record_run()
        loose = self._checkpoint(step=100)
        elsewhere = self._checkpoint(other, step=50)

        result = api.dispatch("clear_unpinned_checkpoints", {"run_id": self.RUN_ID})

        self.assertEqual(result["removed"], [str(loose.parent.resolve())])
        self.assertTrue(elsewhere.is_file())

    def test_clear_unpinned_refuses_while_training_and_while_a_generation_runs(self):
        from trainer import control

        self._make_run()
        self._record_run()
        target = self._checkpoint(step=100)
        control.write_state(
            {"status": "training", "pid": os.getpid(), "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )
        with self.assertRaises(ValueError) as raised:
            api.dispatch("clear_unpinned_checkpoints", {"run_id": self.RUN_ID})
        self.assertIn("GPU", str(raised.exception))
        self.assertTrue(target.is_file())

        control.write_state(
            {"status": "paused", "pid": os.getpid(), "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )
        generated = self.out / self.RUN_ID / "rein_samples" / "generated"
        generated.mkdir(parents=True)
        (generated / "job1.json").write_text(
            json.dumps({"id": "job1", "state": "running", "pid": os.getpid()}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as raised:
            api.dispatch("clear_unpinned_checkpoints", {"run_id": self.RUN_ID})
        self.assertIn("still using this card", str(raised.exception))
        self.assertTrue(target.is_file())

    def test_a_paused_run_with_no_generation_may_clear(self):
        from trainer import control

        self._make_run()
        target = self._checkpoint(step=100)
        control.write_state(
            {"status": "paused", "pid": os.getpid(), "output_name": "rein", "run_id": self.RUN_ID},
            force=True,
        )

        result = api.dispatch("clear_unpinned_checkpoints", {"run_id": self.RUN_ID})

        self.assertEqual(result["removed"], [str(target.parent.resolve())])
        self.assertFalse(target.parent.exists())

    def test_clear_unpinned_without_a_run_is_refused(self):
        with self.assertRaises(ValueError):
            api.dispatch("clear_unpinned_checkpoints", {})

    def test_a_pinned_file_keeps_the_directory_it_shares(self):
        self._make_run()
        self._record_run()
        directory = self.out / self.RUN_ID / "rein_s000100"
        directory.mkdir(parents=True, exist_ok=True)
        pinned = directory / "rein.safetensors"
        extra = directory / "extra.safetensors"
        pinned.write_bytes(b"lora")
        extra.write_bytes(b"lora")
        api.dispatch("checkpoint_pin_set", {"path": str(pinned.resolve()), "pinned": True})

        result = api.dispatch("clear_unpinned_checkpoints", {"run_id": self.RUN_ID})

        self.assertEqual(result["removed"], [])
        self.assertIn(str(pinned.resolve()), result["kept"])
        self.assertTrue(extra.is_file())


if __name__ == "__main__":
    unittest.main()
