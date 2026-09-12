import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from trainer.runs import (
    create_run_dirs,
    find_latest_run,
    list_runs,
    make_run_id,
    run_id_re,
    safe_name,
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


if __name__ == "__main__":
    unittest.main()
