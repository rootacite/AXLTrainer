import os
import tempfile
import unittest
from unittest import mock

import torch

import api
from trainer.config import TrainConfig
from trainer.family import (
    CATALOG,
    ModelSpecError,
    UnsupportedFamilyError,
    lookup_spec,
    require_matching_spec,
    require_trainable,
    resolve_family,
)
from trainer.models import build_kohya_metadata
from trainer.setup import build_train_objects


def _sd35_kwargs() -> dict:
    spec = CATALOG["sd3.5-large"]
    return {
        "base_model_version": spec.base_model_version,
        "modelspec_architecture": spec.architecture,
        "modelspec_implementation": spec.implementation,
        "modelspec_sai_model_spec": spec.sai_model_spec,
    }


class FamilyCatalogTest(unittest.TestCase):
    def test_default_config_resolves_trainable_sdxl(self):
        family = resolve_family(TrainConfig())
        self.assertEqual(family.spec.family_id, "sdxl")
        self.assertTrue(family.trainable)
        self.assertEqual(family.prediction_type(TrainConfig()), "epsilon")

    def test_unknown_version_raises(self):
        with self.assertRaises(ModelSpecError):
            lookup_spec("sd1.5")

    def test_mismatched_architecture_raises(self):
        with self.assertRaises(ModelSpecError):
            TrainConfig(modelspec_architecture="not-a-real-arch")

    def test_require_matching_spec_accepts_catalog_row(self):
        spec = CATALOG["sdxl_base_v1-0"]
        got = require_matching_spec(
            spec.base_model_version,
            spec.architecture,
            spec.implementation,
            spec.sai_model_spec,
        )
        self.assertEqual(got, spec)

    def test_sd35_resolves_but_is_not_trainable(self):
        cfg = TrainConfig(**_sd35_kwargs())
        family = resolve_family(cfg)
        self.assertEqual(family.spec.family_id, "sd3.5")
        self.assertFalse(family.trainable)
        with self.assertRaises(UnsupportedFamilyError) as ctx:
            require_trainable(family)
        self.assertIn("SD 3.5", str(ctx.exception))

    def test_build_train_objects_rejects_sd35_before_diffusers(self):
        cfg = TrainConfig(**_sd35_kwargs())
        with mock.patch("trainer.family_sdxl.load_sdxl_pipeline") as loader:
            with self.assertRaises(UnsupportedFamilyError):
                build_train_objects(cfg)
            loader.assert_not_called()

    def test_sdxl_extra_cond_is_six_time_ids(self):
        family = resolve_family(TrainConfig())
        extra = family.extra_cond(
            src_wh=(1024, 768),
            bucket_wh=(1280, 768),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(extra["time_ids"].shape), (6,))
        self.assertEqual(
            extra["time_ids"].tolist(),
            [768.0, 1024.0, 0.0, 0.0, 768.0, 1280.0],
        )


class MetadataPredictionTypeTest(unittest.TestCase):
    def test_epsilon_when_not_vpred(self):
        cfg = TrainConfig(is_vpred=False)
        family = resolve_family(cfg)
        meta = build_kohya_metadata(
            cfg, 1, None, False, prediction_type=family.prediction_type(cfg)
        )
        self.assertEqual(meta["modelspec.prediction_type"], "epsilon")
        self.assertEqual(meta["ss_v_pred"], "0")
        self.assertEqual(meta["ss_base_model_version"], cfg.base_model_version)

    def test_v_prediction_when_vpred(self):
        cfg = TrainConfig(is_vpred=True)
        family = resolve_family(cfg)
        self.assertEqual(family.prediction_type(cfg), "v_prediction")
        meta = build_kohya_metadata(
            cfg, 1, None, False, prediction_type=family.prediction_type(cfg)
        )
        self.assertEqual(meta["modelspec.prediction_type"], "v_prediction")
        self.assertEqual(meta["ss_v_pred"], "1")


class TrainStartFamilyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_train_start_rejects_sd35_without_spawn(self):
        cfg = TrainConfig(**_sd35_kwargs())
        with mock.patch("api.TrainConfig", return_value=cfg):
            with mock.patch("subprocess.Popen") as popen:
                with self.assertRaises(ValueError) as ctx:
                    api.handle_train_start({})
                self.assertIn("SD 3.5", str(ctx.exception))
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
