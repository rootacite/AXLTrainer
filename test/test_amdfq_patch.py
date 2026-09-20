import tempfile
import unittest
from pathlib import Path

from trainer.amdfq_patch import (
    launch_env_line,
    normalize_amdfq,
    read_amdfq,
    resolve_preload,
)


class AmdfqPatchTest(unittest.TestCase):
    def test_normalize_accepts_known_values(self):
        self.assertEqual(normalize_amdfq("none"), "none")
        self.assertEqual(normalize_amdfq("TAIL"), "tail")
        self.assertEqual(normalize_amdfq(" VMM "), "vmm")
        self.assertEqual(normalize_amdfq(None), "none")
        with self.assertRaises(ValueError):
            normalize_amdfq("pool")

    def test_read_amdfq_from_environment_table(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text(
                '[environment]\noutput_name = "x"\namdfq = "vmm"\n',
                encoding="utf-8",
            )
            self.assertEqual(read_amdfq(path), "vmm")

    def test_read_amdfq_missing_file_is_none(self):
        self.assertEqual(read_amdfq(Path("/tmp/axl-no-such-config.toml")), "none")

    def test_resolve_preload_none_has_no_so(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "none"\n',
                encoding="utf-8",
            )
            info = resolve_preload(root)
            self.assertEqual(info["choice"], "none")
            self.assertIsNone(info["so"])
            self.assertIsNone(info["va_status"])
            self.assertEqual(launch_env_line(root), "none||")

    def test_resolve_preload_missing_so_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "tail"\n',
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_preload(root)
            self.assertIn("libamdfq_tail_rs.so", str(ctx.exception))
            self.assertIn("cargo build --release", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
