"""Both of the trainer's optimizers: Schedule-Free AdamW, one per parameter group side.

The UNet and the text encoders used to run different optimizer types; they now share
`AdamWScheduleFree`, so the TE keeps only its own hyperparameters — including the warmup that used
to live in `[training]` as `lr_warmup_steps`.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# `python test/test_optimizers.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schedulefree import AdamWScheduleFree

from trainer.models import build_te_optimizer, build_unet_optimizer
from trainer.setup import TrainArtifacts


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        unet_learning_rate=7e-5,
        unet_weight_decay=0.01,
        unet_betas_1=0.9,
        unet_betas_2=0.99,
        unet_warmup_steps=100,
        te_learning_rate=7e-6,
        te_weight_decay=0.01,
        te_betas_1=0.9,
        te_betas_2=0.99,
        te_warmup_steps=100,
    )


def _params():
    return [torch.nn.Parameter(torch.zeros(2))]


class ScheduleFreeOptimizersTest(unittest.TestCase):
    def test_both_optimizers_are_schedule_free(self):
        cfg = _cfg()
        unet = build_unet_optimizer(cfg, _params())
        te = build_te_optimizer(cfg, _params())
        self.assertIsInstance(unet, AdamWScheduleFree)
        self.assertIsInstance(te, AdamWScheduleFree)

    def test_the_unet_keeps_its_own_hyperparameters(self):
        cfg = _cfg()
        group = build_unet_optimizer(cfg, _params()).param_groups[0]
        self.assertEqual(cfg.unet_learning_rate, group["lr"])
        self.assertEqual((cfg.unet_betas_1, cfg.unet_betas_2), group["betas"])
        self.assertEqual(cfg.unet_weight_decay, group["weight_decay"])
        self.assertEqual(cfg.unet_warmup_steps, group["warmup_steps"])
        # Neither optimizer has an `eps` key: both keep the library default.
        self.assertEqual(1e-8, group["eps"])

    def test_the_te_takes_its_own_hyperparameters_and_warmup(self):
        cfg = _cfg()
        group = build_te_optimizer(cfg, _params()).param_groups[0]
        self.assertEqual(cfg.te_learning_rate, group["lr"])
        self.assertEqual((cfg.te_betas_1, cfg.te_betas_2), group["betas"])
        self.assertEqual(cfg.te_weight_decay, group["weight_decay"])
        self.assertEqual(cfg.te_warmup_steps, group["warmup_steps"])

    def test_a_schedule_free_optimizer_steps_once_it_is_in_train_mode(self):
        cfg = _cfg()
        param = torch.nn.Parameter(torch.ones(2))
        optimizer = build_te_optimizer(cfg, [param])
        optimizer.train()
        param.grad = torch.ones_like(param)
        optimizer.step()
        self.assertNotEqual(0.0, optimizer.param_groups[0]["scheduled_lr"])

    def test_the_swap_shape_no_longer_carries_a_te_scheduler(self):
        fields = getattr(TrainArtifacts, "__dataclass_fields__")
        self.assertNotIn("te_scheduler", fields)
        self.assertIn("denoise_optimizer", fields)
        self.assertIn("te_optimizer", fields)


class PrepareSlotsTest(unittest.TestCase):
    """`main._prepare_artifacts` slices the prepare list by hand, so the slots are worth pinning."""

    def test_the_optimizers_come_back_in_their_own_slots(self):
        from accelerate import Accelerator

        cfg = _cfg()
        denoise = torch.nn.Linear(4, 4)
        text_encoders = [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
        denoise_optimizer = build_unet_optimizer(cfg, denoise.parameters())
        te_optimizer = build_te_optimizer(
            cfg,
            [p for te in text_encoders for p in te.parameters()],
        )

        accelerator = Accelerator(mixed_precision="no", device_placement=False)
        n_te = len(text_encoders)
        prepared = accelerator.prepare(
            denoise,
            *text_encoders,
            denoise_optimizer,
            te_optimizer,
        )

        self.assertEqual(1 + n_te + 2, len(prepared))
        self.assertEqual(cfg.unet_learning_rate, prepared[1 + n_te].param_groups[0]["lr"])
        self.assertEqual(cfg.te_learning_rate, prepared[2 + n_te].param_groups[0]["lr"])


class TeWarmupStepsConfigTest(unittest.TestCase):
    """`te_warmup_steps` is the key; the `[training].lr_warmup_steps` it replaced still works."""

    def _load(self, toml_text: str):
        """`TrainConfig` as it resolves against a config.toml holding exactly [toml_text]."""
        config_py = Path(__file__).resolve().parent.parent / "trainer" / "config.py"
        with tempfile.TemporaryDirectory(prefix="axl-te-warmup-") as tmp:
            Path(tmp, "config.toml").write_text(toml_text, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("_test_te_warmup_config", config_py)
            module = importlib.util.module_from_spec(spec)
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                spec.loader.exec_module(module)
            finally:
                os.chdir(cwd)
        return module.TrainConfig()

    def test_the_new_key_wins_over_the_old_one(self):
        cfg = self._load("[te_optimizer]\nte_warmup_steps = 40\n\n[training]\nlr_warmup_steps = 100\n")
        self.assertEqual(40, cfg.te_warmup_steps)

    def test_a_config_that_still_only_has_the_old_key_keeps_its_warmup(self):
        cfg = self._load("[training]\nlr_warmup_steps = 77\n")
        self.assertEqual(77, cfg.te_warmup_steps)

    def test_neither_key_falls_back_to_the_default(self):
        cfg = self._load("# no optimizer keys at all\n")
        self.assertEqual(100, cfg.te_warmup_steps)


if __name__ == "__main__":
    unittest.main()
