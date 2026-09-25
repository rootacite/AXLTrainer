import contextlib
import hashlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

# `python test/test_mask_blur.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.mask_blur import (
    EMPTY_ALPHA,
    IMAGE_EXTENSIONS,
    MARKER_KEY,
    MASK_SIDECAR_SUFFIX,
    NO_ALPHA,
    OPAQUE_ALPHA,
    PROTECTED,
    WRITTEN,
    alpha_channel_as_l,
    alpha_mask,
    default_jobs,
    image_has_alpha,
    is_mask_sidecar,
    list_images,
    main,
    mask_path_for,
    process_all,
    process_image,
    sidecar_marker,
)

try:  # the loader needs torch; the tool itself does not
    from trainer.utils import load_loss_mask
except Exception:  # pragma: no cover - torch missing outside the `axl` env
    load_loss_mask = None


def _hard_edged_rgba(path: Path, size=(128, 128), box=(32, 32, 95, 95)) -> None:
    """A silhouette on a fully transparent background: a 0/255 alpha step."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle(box, fill=(200, 120, 130, 255))
    img.save(path)


def _solid_rgba(path: Path, size=(64, 64), alpha=255) -> None:
    Image.new("RGBA", size, (10, 20, 30, alpha)).save(path)


def _row(img: Image.Image, y: int) -> list:
    return [img.getpixel((x, y)) for x in range(img.width)]


class NamingTest(unittest.TestCase):
    def test_sidecar_name_rules(self):
        self.assertEqual(MASK_SIDECAR_SUFFIX, ".mask.png")
        self.assertTrue(is_mask_sidecar(Path("x.mask.png")))
        self.assertTrue(is_mask_sidecar(Path("photo.MASK.PNG")))
        self.assertFalse(is_mask_sidecar(Path("x.png")))
        self.assertFalse(is_mask_sidecar(Path("mask.png")))
        self.assertEqual(mask_path_for(Path("/d/cat.jpg")), Path("/d/cat.mask.png"))
        self.assertEqual(mask_path_for(Path("/d/cat.webp")), Path("/d/cat.mask.png"))

    def test_extension_set_matches_the_dataset_contract(self):
        self.assertEqual(IMAGE_EXTENSIONS, {".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class AlphaSourceTest(unittest.TestCase):
    def test_rgb_has_no_alpha(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "flat.jpg"
            Image.new("RGB", (16, 16), (10, 20, 30)).save(path)
            self.assertEqual(alpha_mask(path), (None, NO_ALPHA))

    def test_rgba_alpha_is_extracted_as_l_at_the_source_size(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stand.png"
            _hard_edged_rgba(path, size=(64, 64), box=(16, 16, 47, 47))
            alpha, reason = alpha_mask(path)
            self.assertEqual(reason, "")
            self.assertEqual(alpha.mode, "L")
            self.assertEqual(alpha.size, (64, 64))
            self.assertEqual(set(alpha.getextrema()), {0, 255})

    def test_la_and_palette_alpha(self):
        la = Image.new("LA", (8, 8), (255, 128))
        self.assertTrue(image_has_alpha(la))
        self.assertEqual(alpha_channel_as_l(la).getpixel((0, 0)), 128)

        pal = Image.new("P", (8, 8), 0)
        pal.putpalette([0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254)
        pal.info["transparency"] = 1
        self.assertTrue(image_has_alpha(pal))
        self.assertEqual(alpha_channel_as_l(pal).getpixel((0, 0)), 255)

    def test_uniform_alpha_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            opaque = Path(td) / "opaque.png"
            empty = Path(td) / "empty.png"
            _solid_rgba(opaque, alpha=255)
            _solid_rgba(empty, alpha=0)
            self.assertEqual(alpha_mask(opaque), (None, OPAQUE_ALPHA))
            self.assertEqual(alpha_mask(empty), (None, EMPTY_ALPHA))


class WriteSidecarTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_blur_is_written_beside_the_image_at_the_source_size(self):
        image = self.dir / "stand.png"
        _hard_edged_rgba(image, size=(128, 128))

        outcome = process_image(image, radius=16.0, overwrite=False, dry_run=False)
        self.assertEqual(outcome.kind, WRITTEN)
        self.assertEqual(outcome.sidecar, mask_path_for(image))

        alpha = Image.open(image).getchannel("A")
        with Image.open(mask_path_for(image)) as sidecar:
            self.assertEqual(sidecar.mode, "L")
            self.assertEqual(sidecar.size, alpha.size)
            self.assertEqual(sidecar.info.get(MARKER_KEY), "16")

            binary = _row(alpha, 64)
            softened = _row(sidecar, 64)
        self.assertEqual(len(set(binary)), 2, "the source alpha is a hard 0/255 step")
        self.assertGreater(sum(1 for v in softened if 0 < v < 255), 8)
        # Two facing edges: the row rises into the silhouette and falls out of it.
        left, right = softened[:64], softened[64:]
        self.assertTrue(all(b >= a for a, b in zip(left, left[1:])), "left edge ramps up")
        self.assertTrue(all(b <= a for a, b in zip(right, right[1:])), "right edge ramps down")

    def test_written_sidecar_replaces_the_alpha_in_the_loader(self):
        if load_loss_mask is None:
            self.skipTest("torch is not available")
        import torch

        image = self.dir / "stand.png"
        _hard_edged_rgba(image, size=(256, 256), box=(64, 64, 191, 191))

        # Ratio 1: no resampling softening, so the alpha path is exactly binary.
        before = load_loss_mask(image, 256, 256, 256, 256)
        self.assertEqual(set(torch.unique(before).tolist()), {0.0, 1.0})

        process_image(image, radius=16.0, overwrite=False, dry_run=False)
        after = load_loss_mask(image, 256, 256, 256, 256)
        band = lambda m: int(((m > 0.01) & (m < 0.99)).sum().item())
        self.assertGreater(band(after), 0)
        self.assertEqual(float(after.min()), 0.0)
        self.assertEqual(float(after.max()), 1.0)

    def test_training_image_is_never_touched(self):
        image = self.dir / "stand.png"
        _hard_edged_rgba(image)
        digest = hashlib.sha1(image.read_bytes()).hexdigest()
        mtime = image.stat().st_mtime_ns

        process_image(image, radius=8.0, overwrite=False, dry_run=False)
        self.assertEqual(hashlib.sha1(image.read_bytes()).hexdigest(), digest)
        self.assertEqual(image.stat().st_mtime_ns, mtime)

    def test_no_temporary_file_is_left_behind(self):
        image = self.dir / "stand.png"
        _hard_edged_rgba(image)
        process_image(image, radius=8.0, overwrite=False, dry_run=False)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         ["stand.mask.png", "stand.png"])

    def test_sidecar_is_not_treated_as_a_dataset_image(self):
        image = self.dir / "stand.png"
        _hard_edged_rgba(image)
        process_image(image, radius=8.0, overwrite=False, dry_run=False)
        self.assertEqual(list_images(self.dir, recursive=False), [image])

    def test_dry_run_writes_nothing(self):
        image = self.dir / "stand.png"
        _hard_edged_rgba(image)
        outcome = process_image(image, radius=16.0, overwrite=False, dry_run=True)
        self.assertEqual(outcome.kind, WRITTEN)
        self.assertFalse(mask_path_for(image).exists())


class ProtectionTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        self.image = self.dir / "stand.png"
        _hard_edged_rgba(self.image)

    def tearDown(self):
        self._td.cleanup()

    def test_painted_sidecar_is_protected(self):
        painted = mask_path_for(self.image)
        Image.new("L", (128, 128), 0).save(painted)
        digest = hashlib.sha1(painted.read_bytes()).hexdigest()

        outcome = process_image(self.image, radius=16.0, overwrite=False, dry_run=False)
        self.assertEqual(outcome.kind, PROTECTED)
        self.assertEqual(hashlib.sha1(painted.read_bytes()).hexdigest(), digest)

    def test_overwrite_replaces_a_painted_sidecar(self):
        painted = mask_path_for(self.image)
        Image.new("L", (128, 128), 0).save(painted)

        outcome = process_image(self.image, radius=16.0, overwrite=True, dry_run=False)
        self.assertEqual(outcome.kind, WRITTEN)
        self.assertEqual(sidecar_marker(painted), "16")

    def test_marked_sidecar_is_regenerated_with_the_new_radius(self):
        first = process_image(self.image, radius=8.0, overwrite=False, dry_run=False)
        self.assertEqual(first.kind, WRITTEN)
        sidecar = mask_path_for(self.image)
        self.assertEqual(sidecar_marker(sidecar), "8")

        second = process_image(self.image, radius=24.0, overwrite=False, dry_run=False)
        self.assertEqual(second.kind, WRITTEN)
        self.assertEqual(second.detail, "replaces marker radius 8")
        self.assertEqual(sidecar_marker(sidecar), "24")

    def test_dry_run_reports_a_marked_sidecar_as_writable(self):
        process_image(self.image, radius=8.0, overwrite=False, dry_run=False)
        outcome = process_image(self.image, radius=16.0, overwrite=False, dry_run=True)
        self.assertEqual(outcome.kind, WRITTEN)
        self.assertEqual(sidecar_marker(mask_path_for(self.image)), "8")

    def test_unreadable_sidecar_is_treated_as_protected(self):
        pending = mask_path_for(self.image)
        pending.write_bytes(b"not a png")
        outcome = process_image(self.image, radius=16.0, overwrite=False, dry_run=False)
        self.assertEqual(outcome.kind, PROTECTED)


class ParallelTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _seed(self, folder: Path, count: int = 4) -> None:
        folder.mkdir()
        for i in range(count):
            _hard_edged_rgba(folder / f"img{i}.png", size=(96, 96), box=(16, 16, 79, 79))

    def test_cpu_count_is_the_default_worker_count(self):
        self.assertGreaterEqual(default_jobs(), 1)

    def test_process_all_keeps_the_input_order(self):
        folder = self.dir / "ds"
        self._seed(folder, count=5)
        images = list_images(folder, recursive=False)

        outcomes = process_all(images, radius=8.0, overwrite=False, dry_run=True, jobs=3)
        self.assertEqual([o.image for o in outcomes], images)
        self.assertEqual({o.kind for o in outcomes}, {WRITTEN})

    def test_process_all_with_one_worker_runs_in_this_process(self):
        folder = self.dir / "ds"
        self._seed(folder, count=1)
        outcome = process_all(list_images(folder, recursive=False), 8.0, False, False, jobs=1)[0]
        self.assertEqual(outcome.kind, WRITTEN)
        self.assertTrue(outcome.sidecar.is_file())

    def test_a_single_image_never_starts_a_pool(self):
        # jobs > 1 but one image: the in-process path, so a one-file dataset pays no overhead.
        folder = self.dir / "ds"
        self._seed(folder, count=1)
        outcome = process_all(list_images(folder, recursive=False), 8.0, False, False, jobs=8)[0]
        self.assertEqual(outcome.kind, WRITTEN)


class CliTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_parallel_and_sequential_runs_agree(self):
        source = self.dir / "source"
        source.mkdir()
        for i in range(4):
            _hard_edged_rgba(source / f"img{i}.png", size=(96, 96), box=(16, 16, 79, 79))
        sequential, parallel = self.dir / "seq", self.dir / "par"
        for folder in (sequential, parallel):
            folder.mkdir()
            for p in source.iterdir():
                shutil.copy(p, folder / p.name)

        code, _, err = self._run([str(sequential), "--radius", "12", "--jobs", "1"])
        self.assertEqual(code, 0, err)
        code, out, err = self._run([str(parallel), "--radius", "12", "--jobs", "3"])
        self.assertEqual(code, 0, err)
        self.assertIn("3 worker(s)", out)

        sidecars = sorted(sequential.glob("*.mask.png"))
        self.assertEqual(len(sidecars), 4)
        for p in sidecars:
            self.assertEqual(hashlib.sha1(p.read_bytes()).hexdigest(),
                             hashlib.sha1((parallel / p.name).read_bytes()).hexdigest())

    def test_negative_jobs_is_a_usage_error(self):
        _hard_edged_rgba(self.dir / "stand.png")
        code, _, err = self._run([str(self.dir), "--jobs", "-1"])
        self.assertEqual(code, 2)
        self.assertIn("--jobs must be", err)

    def test_end_to_end_over_a_folder(self):
        _hard_edged_rgba(self.dir / "stand.png")
        Image.new("RGB", (32, 32), (1, 2, 3)).save(self.dir / "flat.jpg")

        code, out, err = self._run([str(self.dir), "--radius", "12"])
        self.assertEqual(code, 0, err)
        self.assertTrue((self.dir / "stand.mask.png").is_file())
        self.assertFalse((self.dir / "flat.jpg.mask.png").exists())
        self.assertIn("wrote 1 sidecar", out)
        self.assertIn("1 no alpha channel", out)

    def test_verbose_lists_every_image(self):
        _hard_edged_rgba(self.dir / "stand.png")
        code, out, _ = self._run([str(self.dir), "--radius", "12", "--verbose"])
        self.assertEqual(code, 0)
        self.assertIn("written", out)
        self.assertIn("stand.png", out)

    def test_dry_run_flag_writes_nothing(self):
        _hard_edged_rgba(self.dir / "stand.png")
        code, out, _ = self._run([str(self.dir), "--radius", "12", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("dry run", out)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["stand.png"])

    def test_subfolders_need_the_recursive_flag(self):
        sub = self.dir / "sub"
        sub.mkdir()
        _hard_edged_rgba(sub / "stand.png")

        self._run([str(self.dir), "--radius", "12"])
        self.assertFalse((sub / "stand.mask.png").exists())

        self._run([str(self.dir), "--radius", "12", "--recursive"])
        self.assertTrue((sub / "stand.mask.png").is_file())

    def test_radius_outside_the_hint_range_still_runs_with_a_note(self):
        _hard_edged_rgba(self.dir / "stand.png")
        code, _, err = self._run([str(self.dir), "--radius", "4"])
        self.assertEqual(code, 0)
        self.assertIn("outside the usual", err)
        self.assertTrue((self.dir / "stand.mask.png").is_file())

    def test_invalid_radius_is_a_usage_error(self):
        code, _, err = self._run([str(self.dir), "--radius", "0"])
        self.assertEqual(code, 2)
        self.assertIn("--radius must be", err)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_missing_directory_fails(self):
        code, _, err = self._run([str(self.dir / "nope")])
        self.assertEqual(code, 1)
        self.assertIn("not a directory", err)

    def test_unreadable_image_is_reported_and_fails_the_run(self):
        with tempfile.TemporaryDirectory() as td:
            broken = Path(td) / "broken.png"
            broken.write_bytes(b"not a png at all")
            _hard_edged_rgba(self.dir / "stand.png")

            code, out, _ = self._run([td, str(self.dir), "--radius", "12"])
            self.assertEqual(code, 1)
            self.assertIn("1 failed", out)
            self.assertTrue((self.dir / "stand.mask.png").is_file(), "a bad file must not stop the batch")


if __name__ == "__main__":
    unittest.main()
