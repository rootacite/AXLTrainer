import tempfile
import unittest
from pathlib import Path

from trainer.amdfq_patch import (
    GIB,
    MIB,
    launch_env_line,
    normalize_amdfq,
    normalize_pool_mib,
    normalize_va_never_reuse,
    normalize_vram_reserve_gib,
    pool_bytes,
    read_amdfq,
    read_pool_mib,
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
            self.assertEqual(info["pool"], str(64 * MIB))
            self.assertEqual(launch_env_line(root), f"none|||0|0|{64 * MIB}")

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
            self.assertEqual(launch_env_line(root), f"none|||0|0|{64 * MIB}")

    def test_vram_reserve_fractional_gib(self):
        self.assertEqual(vram_reserve_bytes(1.5), int(1.5 * GIB))
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "none"\namdfq_vram_reserve_gib = 1.5\n',
                encoding="utf-8",
            )
            self.assertEqual(read_vram_reserve_gib(root / "config.toml"), 1.5)
            self.assertEqual(launch_env_line(root), f"none|||{int(1.5 * GIB)}|0|{64 * MIB}")

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
            self.assertEqual(launch_env_line(root), f"none|||0|1|{64 * MIB}")

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

    def test_pool_default_is_the_shipped_sixty_four_mib(self):
        self.assertEqual(normalize_pool_mib(None), 64)
        self.assertEqual(normalize_pool_mib(""), 64)
        self.assertEqual(read_pool_mib(Path("/tmp/axl-no-such-config.toml")), 64)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text('[environment]\namdfq = "none"\n', encoding="utf-8")
            self.assertEqual(read_pool_mib(path), 64)

    def test_pool_zero_is_still_off(self):
        # The hook's own default is off as well: only a config row (or an exported variable) asks
        # for a pool.
        self.assertEqual(normalize_pool_mib(0), 0)
        self.assertEqual(normalize_pool_mib("0"), 0)
        self.assertEqual(pool_bytes(0), 0)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "none"\namdfq_pool_mib = 0\n',
                encoding="utf-8",
            )
            self.assertEqual(read_pool_mib(root / "config.toml"), 0)
            self.assertEqual(launch_env_line(root), "none|||0|0|0")

    def test_pool_size_reaches_the_launch_line(self):
        self.assertEqual(normalize_pool_mib(64), 64)
        self.assertEqual(pool_bytes(64), 64 * MIB)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "config.toml").write_text(
                '[environment]\namdfq = "none"\namdfq_pool_mib = 64\n',
                encoding="utf-8",
            )
            self.assertEqual(read_pool_mib(root / "config.toml"), 64)
            info = resolve_preload(root)
            # Bytes on the wire, like AMDFQ_VRAM_RESERVE: the hook does not know about MiB.
            self.assertEqual(info["pool"], str(64 * MIB))
            self.assertEqual(launch_env_line(root), f"none|||0|0|{64 * MIB}")

    def test_pool_size_out_of_range_is_refused(self):
        # The hook clamps too, but a number this far out is a typo and the start should say so.
        for raw in (8, 1024, -16):
            with self.assertRaises(ValueError):
                normalize_pool_mib(raw)
        for raw in ("sixty", "64.5", True, [64]):
            with self.assertRaises(ValueError):
                normalize_pool_mib(raw)
        self.assertEqual(normalize_pool_mib(16), 16)
        self.assertEqual(normalize_pool_mib(512), 512)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.toml"
            path.write_text('[environment]\namdfq_pool_mib = 4096\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_pool_mib(path)

    def test_vram_reserve_rejects_negative(self):
        with self.assertRaises(ValueError):
            normalize_vram_reserve_gib(-1)
        with self.assertRaises(ValueError):
            normalize_vram_reserve_gib("nope")


if __name__ == "__main__":
    unittest.main()
