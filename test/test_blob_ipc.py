import base64
import json
import os
import sys
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import api
from trainer import blobcodec, fsrpc


def _jpeg(path: Path, size=(2000, 1000), color=(10, 20, 30)) -> Path:
    Image.new("RGB", size, color).save(path, "JPEG", quality=95)
    return path


def _png_gray_b64(size=(8, 8), fill=255) -> str:
    buf = BytesIO()
    Image.new("L", size, fill).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class BlobIpcTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.out = self.root / "out"
        self.cache = self.root / "blob-cache"
        self.data.mkdir()
        self.out.mkdir()
        self.cache.mkdir()
        os.environ["AXL_BLOB_CACHE_DIR"] = str(self.cache)
        os.environ["AXL_BLOB_CACHE_BYTES"] = "1048576"
        os.environ["AXL_BLOB_WORKERS"] = "0"
        blobcodec.reset_for_tests()
        self.cfg = {
            "train_data_dir": str(self.data),
            "output_dir": str(self.out),
            "output_name": "rein",
        }
        self._orig = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        api._train_config_dict = self._orig
        blobcodec.reset_for_tests()
        self.tmp.cleanup()
        for key in ("AXL_BLOB_CACHE_DIR", "AXL_BLOB_CACHE_BYTES", "AXL_BLOB_WORKERS"):
            os.environ.pop(key, None)

    def test_max_edge_required(self):
        path = _jpeg(self.data / "a.jpg")
        with self.assertRaises(ValueError):
            api.dispatch("blob_batch", {"paths": [str(path)]})

    def test_resizes_and_jpegs(self):
        path = _jpeg(self.data / "wide.jpg", (2000, 1000))
        result = api.dispatch(
            "blob_batch",
            {"paths": [str(path)], "max_edge": 256, "quality": 80, "format": "jpeg"},
        )
        item = result["items"][0]
        self.assertNotIn("error", item)
        self.assertEqual(item["cache"], "miss")
        self.assertEqual(item["mime"], "image/jpeg")
        self.assertEqual(max(item["width"], item["height"]), 256)
        raw = base64.b64decode(item["base64"])
        with Image.open(BytesIO(raw)) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(max(img.size), 256)

    def test_rgba_flattens_leftover_rgb_onto_black(self):
        """Transparent pixels often still hold RGB (a previous pose). Dropping
        alpha would show that leftover; JPEG must composite onto black first."""
        img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
        pix = img.load()
        for y in range(32):
            for x in range(32):
                pix[x, y] = (255, 0, 0, 0)
            for x in range(32, 64):
                pix[x, y] = (0, 255, 0, 255)
        path = self.data / "ghost.png"
        img.save(path, "PNG")
        item = api.dispatch(
            "blob_batch",
            {"paths": [str(path)], "max_edge": 64, "quality": 95, "format": "jpeg"},
        )["items"][0]
        self.assertNotIn("error", item)
        with Image.open(BytesIO(base64.b64decode(item["base64"]))) as out:
            rgb = out.convert("RGB")
            left = rgb.getpixel((8, 16))
            right = rgb.getpixel((56, 16))
        self.assertLess(max(left), 20, msg=f"transparent leftover leaked: {left}")
        self.assertGreater(right[1], 200, msg=f"opaque green flattened away: {right}")
        self.assertLess(right[0], 40)
        self.assertLess(right[2], 40)

    def test_png_keeps_alpha_so_leftover_rgb_stays_hidden(self):
        img = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
        path = self.data / "alpha.png"
        img.save(path, "PNG")
        item = api.dispatch(
            "blob_batch",
            {"paths": [str(path)], "max_edge": 32, "quality": 80, "format": "png"},
        )["items"][0]
        with Image.open(BytesIO(base64.b64decode(item["base64"]))) as out:
            self.assertEqual(out.mode, "RGBA")
            self.assertEqual(out.getpixel((8, 8))[3], 0)

    def test_quality_changes_hash(self):
        path = _jpeg(self.data / "q.jpg")
        a = api.dispatch(
            "blob_batch",
            {"paths": [str(path)], "max_edge": 256, "quality": 40, "format": "jpeg"},
        )["items"][0]
        b = api.dispatch(
            "blob_batch",
            {"paths": [str(path)], "max_edge": 256, "quality": 90, "format": "jpeg"},
        )["items"][0]
        self.assertNotEqual(a["hash"], b["hash"])

    def test_second_call_is_cache_hit_without_reread(self):
        path = _jpeg(self.data / "cached.jpg")
        first = api.dispatch(
            "blob_batch",
            {"paths": [str(path)], "max_edge": 128, "quality": 80, "format": "jpeg"},
        )["items"][0]
        stat = path.stat()
        path.write_bytes(b"\x00" * stat.st_size)
        os.utime(path, ns=(stat.st_mtime_ns, stat.st_mtime_ns))
        second = api.dispatch(
            "blob_batch",
            {"paths": [str(path)], "max_edge": 128, "quality": 80, "format": "jpeg"},
        )["items"][0]
        self.assertEqual(first["hash"], second["hash"])
        self.assertEqual(second["cache"], "hit")
        self.assertEqual(first["base64"], second["base64"])

    def test_stat_hash_matches_batch(self):
        path = _jpeg(self.data / "stat.jpg")
        params = {"paths": [str(path)], "max_edge": 256, "quality": 80, "format": "jpeg"}
        stat = api.dispatch("blob_stat", params)["items"][0]
        batch = api.dispatch("blob_batch", params)["items"][0]
        self.assertEqual(stat["hash"], batch["hash"])
        self.assertNotIn("base64", stat)
        self.assertIn("base64", batch)

    def test_rewrite_changes_hash(self):
        path = _jpeg(self.data / "rw.jpg", color=(1, 2, 3))
        params = {"paths": [str(path)], "max_edge": 256, "quality": 80, "format": "jpeg"}
        first = api.dispatch("blob_stat", params)["items"][0]["hash"]
        _jpeg(path, color=(200, 10, 10))
        second = api.dispatch("blob_stat", params)["items"][0]["hash"]
        self.assertNotEqual(first, second)

    def test_allowlist_rejects_outside(self):
        outsider = _jpeg(self.root / "nope.jpg")
        item = api.dispatch(
            "blob_batch",
            {"paths": [str(outsider)], "max_edge": 64, "format": "jpeg"},
        )["items"][0]
        self.assertIn("error", item)

    def test_cache_lru_evicts_with_tiny_cap(self):
        os.environ["AXL_BLOB_CACHE_BYTES"] = "2500"
        blobcodec.reset_for_tests()
        a = _jpeg(self.data / "a.jpg", (80, 80), (1, 0, 0))
        b = _jpeg(self.data / "b.jpg", (80, 80), (0, 1, 0))
        first = api.dispatch(
            "blob_batch",
            {"paths": [str(a)], "max_edge": 64, "quality": 40, "format": "jpeg"},
        )["items"][0]
        self.assertGreater(first["bytes"], 0)
        api.dispatch(
            "blob_batch",
            {"paths": [str(b)], "max_edge": 64, "quality": 40, "format": "jpeg"},
        )
        bins = list(self.cache.glob("*.bin"))
        self.assertTrue(bins)
        total = sum(p.stat().st_size for p in bins)
        self.assertLessEqual(total, 2500)

    def test_pool_path_returns_all(self):
        os.environ["AXL_BLOB_WORKERS"] = "2"
        blobcodec.reset_for_tests()
        paths = [_jpeg(self.data / f"{i}.jpg", (64, 48), (i, i, i)) for i in range(4)]
        result = api.dispatch(
            "blob_batch",
            {"paths": [str(p) for p in paths], "max_edge": 32, "quality": 70, "format": "jpeg"},
        )
        self.assertEqual(len(result["items"]), 4)
        for item in result["items"]:
            self.assertNotIn("error", item)
            self.assertEqual(max(item["width"], item["height"]), 32)


class FsRpcTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.out = self.root / "out"
        self.trash = self.root / "trash"
        self.profiles = self.root / "configs"
        self.data.mkdir()
        self.out.mkdir()
        os.environ["AXL_TRASH_DIR"] = str(self.trash)
        self.cfg = {
            "train_data_dir": str(self.data),
            "output_dir": str(self.out),
            "output_name": "rein",
        }
        self._orig = api._train_config_dict
        api._train_config_dict = lambda: dict(self.cfg)

    def tearDown(self):
        api._train_config_dict = self._orig
        self.tmp.cleanup()
        os.environ.pop("AXL_TRASH_DIR", None)

    def test_dataset_list_pairing_and_orphans(self):
        _jpeg(self.data / "keep.jpg")
        (self.data / "keep.txt").write_text("foo, bar", encoding="utf-8")
        Image.new("L", (8, 8), 0).save(self.data / "keep.mask.png")
        (self.data / "lonely.txt").write_text("orphan", encoding="utf-8")
        (self.data / "notes.md").write_text("x", encoding="utf-8")
        result = api.dispatch("dataset_list", {"directory": str(self.data)})
        self.assertEqual(result["orphans"], ["lonely.txt"])
        self.assertEqual(len(result["items"]), 1)
        item = result["items"][0]
        self.assertEqual(item["stem"], "keep")
        self.assertEqual(item["tags"], ["foo", "bar"])
        self.assertTrue(item["has_sidecar_mask"])
        self.assertEqual(item["width"], 2000)

    def test_caption_write_and_mask_roundtrip(self):
        _jpeg(self.data / "pic.jpg", (32, 32))
        api.dispatch(
            "caption_write",
            {"directory": str(self.data), "stem": "pic", "text": "red, hat"},
        )
        self.assertEqual((self.data / "pic.txt").read_text(encoding="utf-8"), "red, hat")
        png = _png_gray_b64()
        api.dispatch(
            "mask_write",
            {"directory": str(self.data), "stem": "pic", "png_base64": png},
        )
        got = api.dispatch("mask_get", {"directory": str(self.data), "stem": "pic"})
        self.assertEqual(got["png_base64"], png)
        api.dispatch("mask_delete", {"directory": str(self.data), "stem": "pic"})
        self.assertFalse((self.data / "pic.mask.png").is_file())

    def test_drop_moves_triplet(self):
        _jpeg(self.data / "gone.jpg", (16, 16))
        (self.data / "gone.txt").write_text("x", encoding="utf-8")
        Image.new("L", (16, 16), 1).save(self.data / "gone.mask.png")
        result = api.dispatch(
            "dataset_drop",
            {"directory": str(self.data), "rate": 1.0, "seed": 1},
        )
        self.assertEqual(result["moved"], 1)
        self.assertFalse((self.data / "gone.jpg").exists())
        self.assertTrue((self.trash / "gone.jpg").is_file())
        self.assertTrue((self.trash / "gone.txt").is_file())
        self.assertTrue((self.trash / "gone.mask.png").is_file())

    def test_drop_respects_stems(self):
        _jpeg(self.data / "keep.jpg", (16, 16))
        (self.data / "keep.txt").write_text("k", encoding="utf-8")
        _jpeg(self.data / "gone.jpg", (16, 16))
        (self.data / "gone.txt").write_text("g", encoding="utf-8")
        api.dispatch(
            "dataset_drop",
            {"directory": str(self.data), "rate": 1.0, "stems": ["gone"]},
        )
        self.assertTrue((self.data / "keep.jpg").is_file())
        self.assertFalse((self.data / "gone.jpg").exists())

    def test_shuffle_keeps_mask_with_image(self):
        for stem, color in (("alpha", (1, 0, 0)), ("beta", (0, 1, 0))):
            _jpeg(self.data / f"{stem}.jpg", (8, 8), color)
            (self.data / f"{stem}.txt").write_text(stem, encoding="utf-8")
            Image.new("L", (8, 8), 9).save(self.data / f"{stem}.mask.png")
        report = api.dispatch("dataset_shuffle", {"directory": str(self.data), "seed": 7})
        self.assertEqual(report["groups"], 2)
        self.assertEqual(report["first_stem"], "0001")
        for stem in ("0001", "0002"):
            caption = (self.data / f"{stem}.txt").read_text(encoding="utf-8")
            mask = self.data / f"{stem}.mask.png"
            image = self.data / f"{stem}.jpg"
            self.assertTrue(image.is_file())
            self.assertTrue(mask.is_file())
            self.assertIn(caption, {"alpha", "beta"})

    def test_shuffle_refuses_orphan_txt(self):
        _jpeg(self.data / "ok.jpg", (8, 8))
        (self.data / "ok.txt").write_text("ok", encoding="utf-8")
        (self.data / "ghost.txt").write_text("nope", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            api.dispatch("dataset_shuffle", {"directory": str(self.data)})
        self.assertIn("ghost.txt", str(ctx.exception))

    def test_directory_outside_train_data_refused(self):
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(ValueError):
            api.dispatch("dataset_list", {"directory": str(other)})

    def test_profile_name_rules_and_roundtrip(self):
        with mock.patch.object(fsrpc, "profile_dir", return_value=self.profiles):
            with self.assertRaises(ValueError):
                api.dispatch("profile_save", {"name": "bad/name", "text": "x = 1", "overwrite": False})
            saved = api.dispatch(
                "profile_save",
                {"name": "Sena", "text": "[training]\nseed = 1\n", "overwrite": False},
            )
            self.assertEqual(saved["name"], "Sena")
            listed = api.dispatch("profile_list", {})
            self.assertEqual(listed["profiles"][0]["name"], "Sena")
            got = api.dispatch("profile_get", {"name": "Sena"})
            self.assertIn("seed = 1", got["text"])
            api.dispatch("profile_delete", {"name": "Sena"})
            self.assertEqual(api.dispatch("profile_list", {})["profiles"], [])

    def test_config_save_rejects_bad_toml(self):
        cfg = self.root / "config.toml"
        cfg.write_text("[training]\nseed = 1\n", encoding="utf-8")
        with mock.patch.object(fsrpc, "config_path", return_value=cfg):
            with self.assertRaises(ValueError):
                api.dispatch("config_save", {"text": "[[[not toml"})
            api.dispatch("config_save", {"text": "[training]\nseed = 2\n"})
            self.assertIn("seed = 2", cfg.read_text(encoding="utf-8"))

    def test_checkpoint_export_same_file_refused(self):
        src = self.out / "run" / "model.safetensors"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"lora")
        with self.assertRaises(ValueError):
            api.dispatch("checkpoint_export", {"source": str(src), "dest": str(src)})
        dest = self.root / "copy.safetensors"
        result = api.dispatch("checkpoint_export", {"source": str(src), "dest": str(dest)})
        self.assertEqual(result["bytes"], 4)
        self.assertEqual(dest.read_bytes(), b"lora")


class NoStdinChannelTest(unittest.TestCase):
    def test_stdin_loop_is_gone(self):
        self.assertFalse(hasattr(api, "run_ipc_loop"))
        self.assertFalse(hasattr(api, "_write"))


class WebsocketPingTest(unittest.TestCase):
    def setUp(self):
        # websockets honors http_proxy; loopback must not go through it.
        self._proxy = {k: os.environ.get(k) for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")}
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"

    def tearDown(self):
        for key, value in self._proxy.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_ping_roundtrip(self):
        from websockets.sync.client import connect
        from websockets.sync.server import serve

        def handler(connection) -> None:
            for raw in connection:
                req = json.loads(raw)
                connection.send(json.dumps(api._reply_for(req)))
                break

        server = serve(handler, "127.0.0.1", 0)
        port = server.socket.getsockname()[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with connect(f"ws://127.0.0.1:{port}", open_timeout=5) as ws:
                ws.send(json.dumps({"id": 1, "method": "ping", "params": {}}))
                reply = json.loads(ws.recv())
            self.assertEqual(reply, {"id": 1, "ok": True, "result": {"status": "ok"}})
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
