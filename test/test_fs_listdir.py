import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api
from trainer import fsrpc


class FsListdirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dir_a").mkdir()
        (self.root / "dir_b").mkdir()
        (self.root / "file.txt").write_text("x", encoding="utf-8")
        self.data = self.root / "data"
        self.data.mkdir()
        self.out = self.root / "out"
        self.out.mkdir()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.cfg = {
            "train_data_dir": str(self.data),
            "output_dir": str(self.out),
            "logging_dir": str(self.logs),
        }
        self._orig = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        api._train_config_dict = self._orig
        self.tmp.cleanup()

    def test_lists_dirs_and_files(self):
        result = api.dispatch("fs_listdir", {"path": str(self.root)})
        names = {row["name"]: row for row in result["entries"]}
        self.assertEqual(result["path"], str(self.root.resolve()))
        self.assertTrue(names["dir_a"]["is_dir"])
        self.assertFalse(names["file.txt"]["is_dir"])
        self.assertEqual(names["file.txt"]["size"], 1)
        self.assertNotIn("base64", names["file.txt"])

    def test_file_path_refused(self):
        with self.assertRaises(ValueError):
            api.dispatch("fs_listdir", {"path": str(self.root / "file.txt")})

    def test_dotdot_canonicalizes(self):
        nested = self.root / "dir_a" / "nested"
        nested.mkdir()
        result = api.dispatch("fs_listdir", {"path": str(nested / ".." / ".." / "dir_b")})
        self.assertEqual(Path(result["path"]), (self.root / "dir_b").resolve())

    def test_missing_path(self):
        with self.assertRaises(ValueError):
            api.dispatch("fs_listdir", {})

    def test_roots_include_home_repo_and_config(self):
        result = api.dispatch("fs_roots", {})
        names = {row["name"]: row["path"] for row in result["roots"]}
        self.assertIn("Home", names)
        self.assertEqual(Path(names["Home"]), Path.home().resolve())
        self.assertIn("Repo", names)
        self.assertTrue(Path(names["Repo"]).is_dir())
        self.assertEqual(Path(names["Train data"]), self.data.resolve())
        self.assertEqual(Path(names["Output"]), self.out.resolve())
        self.assertEqual(Path(names["Logs"]), self.logs.resolve())


class AllowlistTest(unittest.TestCase):
    def test_empty_list_admits_loopback_only(self):
        nets = api.parse_allow_networks([])
        self.assertTrue(api.client_ip_allowed("127.0.0.1", nets))
        self.assertTrue(api.client_ip_allowed("::1", nets))
        self.assertFalse(api.client_ip_allowed("192.168.1.20", nets))

    def test_cidr_match(self):
        nets = api.parse_allow_networks(["192.168.1.0/24"])
        self.assertTrue(api.client_ip_allowed("192.168.1.20", nets))
        self.assertFalse(api.client_ip_allowed("10.0.0.2", nets))
        self.assertTrue(api.client_ip_allowed("127.0.0.1", nets))

    def test_single_ip(self):
        nets = api.parse_allow_networks(["10.0.0.5"])
        self.assertTrue(api.client_ip_allowed("10.0.0.5", nets))
        self.assertFalse(api.client_ip_allowed("10.0.0.6", nets))

    def test_mapped_ipv4(self):
        nets = api.parse_allow_networks(["192.168.1.20"])
        self.assertTrue(api.client_ip_allowed("::ffff:192.168.1.20", nets))

    def test_invalid_entry_exits(self):
        with self.assertRaises(SystemExit):
            api.parse_allow_networks(["not-an-ip"])

    def test_unset_allow_defaults_to_the_lan(self):
        nets = api.parse_allow_networks(api.resolve_allow_entries([], ""))
        self.assertTrue(api.client_ip_allowed("192.168.1.20", nets))
        self.assertTrue(api.client_ip_allowed("::ffff:192.168.0.1", nets))
        self.assertTrue(api.client_ip_allowed("127.0.0.1", nets))
        self.assertFalse(api.client_ip_allowed("10.0.0.1", nets))

    def test_explicit_allow_replaces_the_default(self):
        nets = api.parse_allow_networks(api.resolve_allow_entries(["10.1.2.3"], ""))
        self.assertTrue(api.client_ip_allowed("10.1.2.3", nets))
        self.assertFalse(api.client_ip_allowed("192.168.1.20", nets))

    def test_env_allow_replaces_the_default(self):
        nets = api.parse_allow_networks(api.resolve_allow_entries([], "10.0.0.0/8, 192.168.1.9"))
        self.assertTrue(api.client_ip_allowed("10.2.0.1", nets))
        self.assertTrue(api.client_ip_allowed("192.168.1.9", nets))
        self.assertFalse(api.client_ip_allowed("192.168.1.8", nets))

    def test_blank_env_still_uses_the_default(self):
        self.assertEqual(api.resolve_allow_entries([], " , "), ["192.168.0.0/16"])


class QuietWsCloseTest(unittest.TestCase):
    def test_filter_is_attached_to_server_logger(self):
        import logging

        api._quiet_websockets_log()
        names = {type(f).__name__ for f in logging.getLogger("websockets.server").filters}
        self.assertIn("_QuietWsClose", names)

    def test_drops_browser_disconnect_tracebacks(self):
        import logging

        from websockets.exceptions import InvalidMessage

        filt = api._QuietWsClose()
        record = logging.LogRecord(
            "websockets.server",
            logging.ERROR,
            __file__,
            1,
            "connection handler failed",
            (),
            None,
        )
        self.assertFalse(filt.filter(record))
        handshake = logging.LogRecord(
            "websockets.server",
            logging.ERROR,
            __file__,
            1,
            "opening handshake failed",
            (),
            (InvalidMessage, InvalidMessage("did not receive a valid HTTP request"), None),
        )
        self.assertFalse(filt.filter(handshake))
        other = logging.LogRecord(
            "websockets.server",
            logging.ERROR,
            __file__,
            1,
            "something else exploded",
            (),
            None,
        )
        self.assertTrue(filt.filter(other))
