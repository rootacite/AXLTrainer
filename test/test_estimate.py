"""`trainer/estimate.py`: the image counts the Utils → Training step estimate is built from.

Torch-free, GPU-free and PIL-free: the counter reads names, not pixels, so the whole thing is a
directory walk plus the repeat arithmetic.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# `python test/test_estimate.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import TrainDataEntry
from trainer.estimate import IMAGE_EXTENSIONS, count_images, count_train_images


def write_files(folder: Path, names) -> None:
    for name in names:
        (folder / name).write_bytes(b"x")


class CountImagesTest(unittest.TestCase):
    def test_counts_the_trainers_extensions_recursively(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_files(root, ["0001.png", "0002.JPG", "0003.webp", "0004.jpeg", "0005.bmp"])
            write_files(root, ["notes.md", "captions.txt", "weights.pt"])
            nested = root / "sub"
            nested.mkdir()
            write_files(nested, ["0006.png"])
            self.assertEqual(count_images(root), 6)
        self.assertEqual(IMAGE_EXTENSIONS, {".jpg", ".jpeg", ".png", ".webp", ".bmp"})

    def test_skips_mask_sidecars_whatever_their_case(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_files(root, ["0001.png", "0001.mask.png", "0002.MASK.PNG", "0002.png"])
            self.assertEqual(count_images(root), 2)

    def test_an_empty_or_missing_folder_counts_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.assertEqual(count_images(root), 0)
            self.assertEqual(count_images(root / "nope"), 0)


class CountTrainImagesTest(unittest.TestCase):
    def test_reports_each_folder_and_the_per_epoch_total(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "a"
            second = root / "b"
            for folder, names in ((first, ["1.png", "2.png", "3.png"]), (second, ["1.jpg", "2.jpg"])):
                folder.mkdir()
                write_files(folder, names)

            payload = count_train_images(
                [{"path": str(first), "repeat": 3}, {"path": str(second), "repeat": 1}]
            )

        self.assertEqual(payload["entries"][0]["images"], 3)
        self.assertEqual(payload["entries"][0]["repeat"], 3)
        self.assertIsNone(payload["entries"][0]["error"])
        self.assertEqual(payload["images"], 5)
        # 3 images drawn three times plus 2 drawn once: what `total_samples` holds at run time.
        self.assertEqual(payload["samples"], 11)

    def test_a_missing_folder_answers_with_a_reason_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            good = root / "good"
            good.mkdir()
            write_files(good, ["1.png"])
            payload = count_train_images(
                [{"path": str(root / "gone"), "repeat": 1}, {"path": str(good), "repeat": 2}]
            )

        self.assertEqual(payload["entries"][0]["images"], 0)
        self.assertEqual(payload["entries"][0]["error"], "not a directory")
        self.assertEqual(payload["entries"][1]["images"], 1)
        self.assertEqual(payload["images"], 1)
        self.assertEqual(payload["samples"], 2)

    def test_a_missing_path_and_a_bad_repeat_are_tolerated(self):
        payload = count_train_images([{"repeat": 5}, {"path": "/tmp", "repeat": "two"}])
        self.assertEqual(payload["entries"][0]["error"], "no path")
        self.assertEqual(payload["entries"][1]["repeat"], 1)

    def test_train_data_entries_are_accepted_as_they_come_from_config(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_files(root, ["1.png", "2.png"])
            payload = count_train_images([TrainDataEntry(path=str(root), repeat=4)])

        self.assertEqual(payload["entries"][0]["path"], str(root))
        self.assertEqual(payload["entries"][0]["repeat"], 4)
        self.assertEqual(payload["samples"], 8)

    def test_an_empty_list_is_a_zero_total(self):
        self.assertEqual(count_train_images([]), {"entries": [], "images": 0, "samples": 0})

    def test_the_module_stays_torch_free(self):
        """api.py imports it in the helper process, which must not pay for torch."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, trainer.estimate; print('torch' in sys.modules)",
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
