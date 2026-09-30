import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import sys

# `python test/test_tagger2.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tagger2.main import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CATEGORIES,
    DEFAULT_THRESHOLD,
    ENGINE,
    MODEL_ID,
    caption_for,
    effective_thresholds,
    list_images,
    load_model_settings,
    main,
    parse_args,
    requested_categories,
    resolve_categories,
    setup_miopen_cache,
    tag_directory,
    tag_text,
)

SETTINGS = {
    "available": True,
    "engine": ENGINE,
    "model": MODEL_ID,
    "model_path": "/models/pixai",
    "cache_dir": "/repo/tagger2/miopen_cache",
    "categories": [
        {"key": "general", "count": 15043, "calibrated": 0.17},
        {"key": "character", "count": 8308, "calibrated": 0.27},
        {"key": "rating", "count": 4, "calibrated": 0.41},
    ],
    "default_categories": ["general"],
    "reason": "",
}

RESULT = {
    "general": {"hair_between_eyes": 0.9, "1girl": 0.8, "masking_tape_(medium)": 0.7},
    "rating": {"rating:g": 0.6},
    "style": {"bkub": 0.99},
}


class FakeTagger:
    """Duck-typed stand-in for the transformers pipeline: one `{"results": …}` per image."""

    def __init__(self, result=None, device="cuda:0", fail_sizes=()):
        self.device = device
        self.result = result if result is not None else RESULT
        self.fail_sizes = set(fail_sizes)
        self.calls = []

    def __call__(self, images, batch_size=1, min_threshold=None):
        self.calls.append((len(images), batch_size, min_threshold))
        if len(images) > 1 and (0, 0) in self.fail_sizes:
            raise RuntimeError("batch boom")
        out = []
        for image in images:
            if image.size in self.fail_sizes:
                raise RuntimeError(f"cannot read {image.size}")
            out.append({"results": self.result})
        return out


class TaggerCliTest(unittest.TestCase):
    def test_defaults(self):
        args = parse_args(["/tmp/alice"])
        self.assertEqual(args.directory, "/tmp/alice")
        self.assertEqual(args.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(args.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(args.categories, "general")
        self.assertEqual(args.model, MODEL_ID)
        self.assertFalse(args.json)
        self.assertFalse(args.cpu)
        self.assertFalse(args.download)
        self.assertFalse(args.info)

    def test_flags(self):
        args = parse_args(["/tmp/alice", "-t", "0.5", "-b", "4", "--categories", "general,rating", "--json", "--cpu", "--download"])
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.categories, "general,rating")
        self.assertTrue(args.json and args.cpu and args.download)

    def test_default_category_is_general(self):
        self.assertEqual(list(DEFAULT_CATEGORIES), ["general"])


class CategoryTest(unittest.TestCase):
    def test_requested_categories_accepts_string_list_and_none(self):
        self.assertEqual(requested_categories(None), ["general"])
        self.assertEqual(requested_categories(" rating , general ,, "), ["rating", "general"])
        self.assertEqual(requested_categories(["meta"]), ["meta"])
        self.assertEqual(requested_categories(""), ["general"])

    def test_resolve_follows_the_models_own_order(self):
        self.assertEqual(resolve_categories("rating,general", SETTINGS), ["general", "rating"])

    def test_resolve_refuses_an_unknown_name(self):
        with self.assertRaises(ValueError) as caught:
            resolve_categories("general,bogus", SETTINGS)
        self.assertIn("bogus", str(caught.exception))
        self.assertIn("general", str(caught.exception))

    def test_resolve_without_model_categories_keeps_the_request(self):
        self.assertEqual(resolve_categories("general,rating", {}), ["general", "rating"])

    def test_thresholds_are_a_floor_over_the_models_calibration(self):
        floors = effective_thresholds(SETTINGS, ["general", "rating"], 0.35)
        self.assertEqual(floors, {"general": 0.35, "rating": 0.41})
        floors = effective_thresholds(SETTINGS, ["general", "rating"], 0.1)
        self.assertEqual(floors, {"general": 0.17, "rating": 0.41})


class CaptionTest(unittest.TestCase):
    def test_orders_by_confidence_and_writes_spaces(self):
        caption = caption_for(RESULT, ["general", "rating"])
        self.assertEqual(caption, "hair between eyes, 1girl, masking tape (medium), rating:g")

    def test_only_the_selected_categories(self):
        self.assertEqual(caption_for(RESULT, ["style"]), "bkub")
        self.assertEqual(caption_for(RESULT, ["rating"]), "rating:g")

    def test_empty_result_writes_an_empty_caption(self):
        self.assertEqual(caption_for({"general": {}}, ["general"]), "")

    def test_tag_text_unwraps_parentheses_too(self):
        self.assertEqual(tag_text("masking_tape_(medium)"), "masking tape (medium)")
        self.assertEqual(tag_text("1girl"), "1girl")


class CacheTest(unittest.TestCase):
    def test_setup_creates_the_directory_and_registers_both_variables(self):
        keys = ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR")
        saved = {key: os.environ.pop(key, None) for key in keys}
        try:
            with tempfile.TemporaryDirectory() as raw:
                target = Path(raw) / "miopen"
                self.assertEqual(setup_miopen_cache(target), target)
                self.assertTrue(target.is_dir())
                self.assertEqual(os.environ["MIOPEN_USER_DB_PATH"], str(target))
                self.assertEqual(os.environ["MIOPEN_CUSTOM_CACHE_DIR"], str(target))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_setup_keeps_a_directory_the_shell_already_pinned(self):
        keys = ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR")
        saved = {key: os.environ.pop(key, None) for key in keys}
        os.environ["MIOPEN_USER_DB_PATH"] = "/tmp/pinned-db"
        try:
            with tempfile.TemporaryDirectory() as raw:
                setup_miopen_cache(Path(raw) / "miopen")
            self.assertEqual(os.environ["MIOPEN_USER_DB_PATH"], "/tmp/pinned-db")
        finally:
            for key in keys:
                os.environ.pop(key, None)
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value


class ModelSettingsTest(unittest.TestCase):
    def test_reads_categories_and_calibrated_thresholds_from_a_local_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            config = {
                "tags_split": [["general", 3], ["rating", 2]],
                "category_best_threshold": {"general": 0.17, "rating": 0.41},
            }
            (Path(raw) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            payload = load_model_settings(raw)

        self.assertTrue(payload["available"])
        self.assertEqual(payload["model_path"], str(Path(raw).resolve()))
        self.assertEqual([item["key"] for item in payload["categories"]], ["general", "rating"])
        self.assertEqual([item["calibrated"] for item in payload["categories"]], [0.17, 0.41])
        self.assertEqual(payload["default_categories"], ["general"])

    def test_a_local_directory_without_tags_split_is_unavailable(self):
        with tempfile.TemporaryDirectory() as raw:
            (Path(raw) / "config.json").write_text("{}", encoding="utf-8")
            payload = load_model_settings(raw)
        self.assertFalse(payload["available"])
        self.assertIn("tags_split", payload["reason"])

    def test_a_repo_id_that_is_not_cached_says_how_to_fetch_it(self):
        payload = load_model_settings("axl-trainer/definitely-not-cached", local_files_only=True)
        self.assertFalse(payload["available"])
        self.assertIn("--download", payload["reason"])
        self.assertEqual(payload["categories"], [])

    def test_info_prints_the_payload_to_stdout(self):
        with tempfile.TemporaryDirectory() as raw:
            config = {"tags_split": [["general", 3]], "category_best_threshold": {"general": 0.2}}
            (Path(raw) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["--info", "--model", raw])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["available"])
        self.assertEqual(payload["categories"][0]["calibrated"], 0.2)


class TagDirectoryTest(unittest.TestCase):
    def _folder(self, root: Path, names=("0001.png", "0002.png", "0003.jpg")) -> None:
        from PIL import Image

        for index, name in enumerate(names):
            Image.new("RGB", (32 + index, 32 + index), color=(10, 20, 30)).save(root / name)
        Image.new("RGB", (8, 8), color=(0, 0, 0)).save(root / "0001.mask.png")
        (root / "orphan.txt").write_text("keep me", encoding="utf-8")

    def test_writes_sidecars_and_leaves_masks_and_orphans_alone(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            self._folder(folder)
            tagger = FakeTagger()
            result = tag_directory(folder, tagger=tagger, categories=["general"])

            self.assertEqual(result["total"], 3)
            self.assertEqual(result["processed"], 3)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["engine"], ENGINE)
            self.assertEqual(result["device"], "cuda:0")
            self.assertEqual(result["categories"], ["general"])
            self.assertEqual(result["thresholds"], {})
            self.assertIn("pixai-tagger-v1.0 on cuda:0", result["provider"])
            self.assertEqual(
                (folder / "0001.txt").read_text(encoding="utf-8"),
                "hair between eyes, 1girl, masking tape (medium)",
            )
            self.assertTrue((folder / "0003.txt").is_file())
            self.assertFalse((folder / "0001.mask.txt").exists())
            self.assertEqual((folder / "orphan.txt").read_text(encoding="utf-8"), "keep me")
            self.assertEqual(tagger.calls[0], (1, 1, DEFAULT_THRESHOLD))

    def test_the_result_payload_is_json_safe(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            self._folder(folder, names=("0001.png",))
            result = tag_directory(folder, tagger=FakeTagger(), categories="general")
        payload = json.loads(json.dumps(result))
        for key in ("total", "processed", "failed"):
            self.assertIsInstance(payload[key], int)
        self.assertEqual(payload["errors"], [])

    def test_uses_the_models_calibrated_floor_and_the_selected_categories(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            self._folder(folder, names=("0001.png",))
            tagger = FakeTagger()
            result = tag_directory(
                folder,
                tagger=tagger,
                settings=SETTINGS,
                categories="rating,general",
                threshold=0.2,
            )
            self.assertEqual(result["categories"], ["general", "rating"])
            self.assertEqual(result["thresholds"], {"general": 0.2, "rating": 0.41})
            self.assertEqual(tagger.calls[0][2], 0.2)
            caption = (folder / "0001.txt").read_text(encoding="utf-8")
            self.assertEqual(caption, "hair between eyes, 1girl, masking tape (medium), rating:g")

    def test_a_bad_category_is_refused_before_any_image_is_read(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            self._folder(folder, names=("0001.png",))
            with self.assertRaises(ValueError):
                tag_directory(folder, tagger=FakeTagger(), settings=SETTINGS, categories="bogus")
            self.assertFalse((folder / "0001.txt").exists())

    def test_one_bad_image_does_not_stop_the_folder(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            self._folder(folder)
            tagger = FakeTagger(fail_sizes={(33, 33)})
            result = tag_directory(folder, tagger=tagger, categories="general")

            self.assertEqual(result["processed"], 2)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["errors"][0]["file"], "0002.png")
            self.assertFalse((folder / "0002.txt").exists())
            self.assertTrue((folder / "0003.txt").is_file())

    def test_a_failed_batch_falls_back_to_one_image_at_a_time(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            self._folder(folder)
            tagger = FakeTagger(fail_sizes={(0, 0)})
            result = tag_directory(folder, tagger=tagger, batch_size=3, categories="general")

            self.assertEqual(result["processed"], 3)
            self.assertEqual(tagger.calls[0][1], 3)
            self.assertEqual([call[1] for call in tagger.calls[1:]], [1, 1, 1])

    def test_missing_directory_and_bad_threshold(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(NotADirectoryError):
                tag_directory(Path(raw) / "missing", tagger=FakeTagger())
            with self.assertRaises(ValueError):
                tag_directory(raw, tagger=FakeTagger(), threshold=1.5)

    def test_list_images_skips_masks_and_foreign_files(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            self._folder(folder)
            (folder / "notes.md").write_text("x", encoding="utf-8")
            names = [path.name for path in list_images(folder)]
        self.assertEqual(names, ["0001.png", "0002.png", "0003.jpg"])
        self.assertNotIn("0001.mask.png", names)


if __name__ == "__main__":
    unittest.main()
