import tempfile
import unittest
from pathlib import Path

from trainer.amdfq_patch import (
    GIB,
    launch_env_line,
    normalize_amdfq,
    normalize_va_never_reuse,
    normalize_vram_reserve_gib,
    read_amdfq,
    read_va_never_reuse,
    read_vram_reserve_gib,
    resolve_preload,
    vram_reserve_bytes,
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
            self.assertEqual(info["vram_reserve"], "0")
            self.assertEqual(info["va_never_reuse"], "0")
            self.assertEqual(launch_env_line(root), "none|||0|0")

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

    def test_vram_reserve_defaults_to_off(self):
        self.assertEqual(normalize_vram_reserve_gib(None), 0.0)
        self.assertEqual(normalize_vram_reserve_gib(""), 0.0)
        self.assertEqual(vram_reserve_bytes(1.0), GIB)
        self.assertEqual(read_vram_reserve_gib(Path("/tmp/axl-no-such-config.toml")), 0.0)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text('[environment]\namdfq = "none"\n', encoding="utf-8")
            self.assertEqual(read_vram_reserve_gib(path), 0.0)

    def test_vram_reserve_zero_disables(self):
        self.assertEqual(normalize_vram_reserve_gib(0), 0.0)
        self.assertEqual(vram_reserve_bytes(0.0), 0)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "none"\namdfq_vram_reserve_gib = 0\n',
                encoding="utf-8",
            )
            info = resolve_preload(root)
            self.assertEqual(info["vram_reserve"], "0")
            self.assertEqual(launch_env_line(root), "none|||0|0")

    def test_vram_reserve_fractional_gib(self):
        self.assertEqual(vram_reserve_bytes(1.5), int(1.5 * GIB))
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "none"\namdfq_vram_reserve_gib = 1.5\n',
                encoding="utf-8",
            )
            self.assertEqual(read_vram_reserve_gib(root / "config.toml"), 1.5)
            self.assertEqual(launch_env_line(root), f"none|||{int(1.5 * GIB)}|0")

    def test_va_never_reuse_defaults_to_off(self):
        self.assertFalse(normalize_va_never_reuse(None))
        self.assertFalse(normalize_va_never_reuse(""))
        self.assertFalse(read_va_never_reuse(Path("/tmp/axl-no-such-config.toml")))
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text('[environment]\namdfq = "none"\n', encoding="utf-8")
            self.assertFalse(read_va_never_reuse(path))

    def test_va_never_reuse_true_reaches_the_launch_line(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "none"\namdfq_va_never_reuse = true\n',
                encoding="utf-8",
            )
            self.assertTrue(read_va_never_reuse(root / "config.toml"))
            info = resolve_preload(root)
            self.assertEqual(info["va_never_reuse"], "1")
            self.assertEqual(launch_env_line(root), "none|||0|1")

    def test_va_never_reuse_rejects_junk(self):
        self.assertTrue(normalize_va_never_reuse(True))
        self.assertFalse(normalize_va_never_reuse("false"))
        with self.assertRaises(ValueError):
            normalize_va_never_reuse("maybe")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text('[environment]\namdfq_va_never_reuse = "maybe"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_va_never_reuse(path)

    def test_vram_reserve_rejects_negative(self):
        with self.assertRaises(ValueError):
            normalize_vram_reserve_gib(-1)
        with self.assertRaises(ValueError):
            normalize_vram_reserve_gib("nope")


if __name__ == "__main__":
    unittest.main()
