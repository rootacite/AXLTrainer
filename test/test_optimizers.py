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

from trainer import control, loop
from trainer.config import TrainConfig
from trainer.control import LiveSettings
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


class _StubAccelerator:
    is_main_process = True

    def log(self, payload, step=None):
        pass


class _RecordingFamily:
    """`save_lora` / `generate_sample`, recording what the weights were when they were called.

    This is the whole point of the invariant: the cadence save and the sample pass read the
    parameters, so they have to see the averaged `x`, and the training iterate afterwards.
    """

    def __init__(self, optimizers, params):
        self.optimizers = optimizers
        self.params = params
        self.seen: list[tuple[str, dict]] = []

    def _snapshot(self, kind: str) -> None:
        self.seen.append(
            (
                kind,
                {
                    "params": {name: param.detach().clone() for name, param in self.params.items()},
                    "train_mode": {
                        name: bool(optimizer.param_groups[0]["train_mode"])
                        for name, optimizer in self.optimizers.items()
                    },
                },
            )
        )

    def save_lora(self, accelerator, modules, cfg, global_step, **kwargs):
        self._snapshot("save")

    def generate_sample(self, **kwargs):
        self._snapshot("sample")


def expected_x(optimizer, param) -> torch.Tensor:
    """What `AdamWScheduleFree.eval()` leaves in the parameter: `p + (1 - 1/beta1)(z - p)`."""
    beta1 = optimizer.param_groups[0]["betas"][0]
    z = optimizer.state[param]["z"].detach()
    return param.detach() + (1 - 1 / beta1) * (z - param.detach())


class CheckpointAndSamplesUseTheAveragedWeightsTest(unittest.TestCase):
    """`loop.optimizers_eval`, as the step cadence save and `main.py`'s final save both use it.

    Schedule-Free trains at `y` and averages into `x`; a checkpoint (and the samples drawn from it)
    that is written without `eval()` silently holds `y` instead — and one that is left in eval mode
    makes the next `step()` raise. Both directions are pinned here, with the value hand-computed
    from the optimizer's own state.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        control.release_lock()

    def tearDown(self):
        control.release_lock()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def _optimizers(self):
        # A deliberately coarse step: `x` and `y` are only 1.1% of `z - p` apart, so a fine lr puts
        # them inside one float32 ulp and the test could not tell the two apart at all.
        cfg = SimpleNamespace(
            unet_learning_rate=0.05, unet_weight_decay=0.0, unet_betas_1=0.9, unet_betas_2=0.99,
            unet_warmup_steps=1,
            te_learning_rate=0.05, te_weight_decay=0.0, te_betas_1=0.9, te_betas_2=0.99,
            te_warmup_steps=1,
        )
        params = {
            "denoise": torch.nn.Parameter(torch.linspace(-0.01, 0.01, 4)),
            "te": torch.nn.Parameter(torch.linspace(0.01, -0.01, 4)),
        }
        optimizers = {
            "denoise": build_unet_optimizer(cfg, [params["denoise"]]),
            "te": build_te_optimizer(cfg, [params["te"]]),
        }
        for optimizer in optimizers.values():
            optimizer.train()
        for _ in range(5):
            for param in params.values():
                param.grad = torch.full_like(param, 0.25)
            for optimizer in optimizers.values():
                optimizer.step()
        return optimizers, params

    def _artifacts(self, optimizers, params, *, sampling_enabled: bool):
        family = _RecordingFamily(optimizers, params)
        return SimpleNamespace(
            accelerator=_StubAccelerator(),
            denoise_optimizer=optimizers["denoise"],
            te_optimizer=optimizers["te"],
            settings=LiveSettings(
                save_every_n_steps=1,
                sampling_enabled=sampling_enabled,
                next_save_step=1,
            ),
            family=family,
            modules=SimpleNamespace(),
            device=torch.device("cpu"),
            weight_dtype=torch.bfloat16,
        ), family

    def _assert_snapshot_is_x(self, snapshot: dict, optimizers, params, expected: dict) -> None:
        for name, param in params.items():
            torch.testing.assert_close(
                snapshot["params"][name],
                expected[name],
                rtol=0.0,
                atol=1e-6,
                msg=f"{name}: the weights written were not the averaged x",
            )
            self.assertFalse(
                snapshot["train_mode"][name],
                f"{name}: the optimizer was still in train mode, so the save/sample read y",
            )

    def test_a_cadence_save_and_its_samples_read_x_and_restore_y(self):
        optimizers, params = self._optimizers()
        expected = {name: expected_x(optimizers[name], param) for name, param in params.items()}
        y = {name: param.detach().clone() for name, param in params.items()}
        # x and y really are different values here, well beyond the tolerance below, so reading the
        # wrong one cannot pass unnoticed.
        for name in params:
            gap = float((expected[name] - y[name]).abs().max())
            self.assertGreater(gap, 1e-3, f"{name}: x and y are {gap:g} apart, too close to test")
        artifacts, family = self._artifacts(optimizers, params, sampling_enabled=True)

        loop._maybe_log_and_sample(artifacts=artifacts, cfg=TrainConfig(), global_step=1)

        self.assertEqual(["save", "sample"], [kind for kind, _ in family.seen])
        for _kind, snapshot in family.seen:
            self._assert_snapshot_is_x(snapshot, optimizers, params, expected)
        for name, param in params.items():
            self.assertTrue(torch.equal(param.detach(), y[name]), f"{name}: y was not restored")
            self.assertTrue(optimizers[name].param_groups[0]["train_mode"])
        # …and the run can go on stepping, which is what a left-behind eval mode would break.
        for _ in range(2):
            for param in params.values():
                param.grad = torch.full_like(param, 0.25)
            for optimizer in optimizers.values():
                optimizer.step()

    def test_a_save_without_sampling_reads_x_as_well(self):
        optimizers, params = self._optimizers()
        expected = {name: expected_x(optimizers[name], param) for name, param in params.items()}
        artifacts, family = self._artifacts(optimizers, params, sampling_enabled=False)

        loop._maybe_log_and_sample(artifacts=artifacts, cfg=TrainConfig(), global_step=1)

        self.assertEqual(["save"], [kind for kind, _ in family.seen])
        self._assert_snapshot_is_x(family.seen[0][1], optimizers, params, expected)

    def test_the_shared_helper_restores_the_modes_even_when_the_body_raises(self):
        optimizers, params = self._optimizers()

        with self.assertRaises(RuntimeError):
            with loop.optimizers_eval(optimizers.values()):
                self.assertFalse(optimizers["te"].param_groups[0]["train_mode"])
                raise RuntimeError("a save that blows up")

        for optimizer in optimizers.values():
            self.assertTrue(optimizer.param_groups[0]["train_mode"])

    def test_the_early_stop_save_reads_x_too(self):
        """`loop.save_stopped_lora`, the third save site: a stop during training owes a checkpoint."""
        optimizers, params = self._optimizers()
        expected = {name: expected_x(optimizers[name], param) for name, param in params.items()}
        y = {name: param.detach().clone() for name, param in params.items()}
        artifacts, family = self._artifacts(optimizers, params, sampling_enabled=False)
        cfg = TrainConfig()
        # A run directory of its own, so the guard below sees no checkpoint for the step.
        cfg.output_dir = self.tmp.name
        cfg.logging_dir = self.tmp.name
        cfg.run_dir = ""
        cfg.output_name = "stopped"

        self.assertTrue(loop.save_stopped_lora(artifacts, cfg, global_step=1))
        self.assertEqual(["save"], [kind for kind, _ in family.seen])
        self._assert_snapshot_is_x(family.seen[0][1], optimizers, params, expected)
        for name, param in params.items():
            self.assertTrue(torch.equal(param.detach(), y[name]), f"{name}: y was not restored")
            self.assertTrue(optimizers[name].param_groups[0]["train_mode"])
        for _ in range(2):  # the run could keep stepping; a left-behind eval mode would break this
            for param in params.values():
                param.grad = torch.full_like(param, 0.25)
            for optimizer in optimizers.values():
                optimizer.step()

        # The two ways it owes nothing: the step already has a checkpoint, and step 0.
        written = loop.lora_checkpoint_file(cfg, 1)
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_bytes(b"")
        before = len(family.seen)
        self.assertFalse(loop.save_stopped_lora(artifacts, cfg, global_step=1))
        self.assertFalse(loop.save_stopped_lora(artifacts, cfg, global_step=0))
        self.assertEqual(before, len(family.seen))

    def test_a_plain_optimizer_without_eval_or_train_is_left_alone(self):
        param = torch.nn.Parameter(torch.zeros(2))
        plain = torch.optim.AdamW([param])

        with loop.optimizers_eval((plain,)):
            param.grad = torch.ones_like(param)
            plain.step()

        self.assertEqual(2, param.numel())


if __name__ == "__main__":
    unittest.main()
