import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from diffusers.configuration_utils import FrozenDict
from safetensors.torch import save_file

import sys

# `python test/test_min_snr.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import TrainConfig
from trainer.family import resolve_family
from trainer.utils import apply_loss_mask
from trainer.family_sdxl import min_snr_weight

# The pipeline config a single-file SDXL base resolves to: EulerDiscreteScheduler with SDXL's own
# betas, epsilon and no zero-terminal-SNR rescale. `build_noise_scheduler` turns this into the DDIM
# scheduler training uses, exactly as the trainer does.
BASE_CONFIG = {
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": False,
    "timestep_spacing": "linspace",
    "steps_offset": 1,
    "set_alpha_to_one": False,
    "clip_sample": False,
    "thresholding": False,
    "trained_betas": None,
    "skip_prk_steps": True,
    "interpolation_type": "linear",
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "sigma_min": None,
    "sigma_max": None,
    "timestep_type": "discrete",
    "final_sigmas_type": "zero",
}


def _marked_base(root: Path, name: str, tensors: dict) -> Path:
    path = root / f"{name}.safetensors"
    save_file(tensors, str(path))
    return path


def _cfg(base: Path, **overrides) -> TrainConfig:
    cfg = TrainConfig(pretrained_model_name_or_path=str(base), noise_offset=0.0, **overrides)
    return cfg


def _stub_pipe(config: dict) -> SimpleNamespace:
    return SimpleNamespace(scheduler=SimpleNamespace(config=FrozenDict(config)))


def _training_scheduler(cfg: TrainConfig, config: dict | None = None) -> object:
    """The real training scheduler for `cfg`, built the way the trainer builds it."""
    family = resolve_family(cfg)
    return family.build_noise_scheduler(_stub_pipe(config or BASE_CONFIG), cfg)


def _reference_weight(alphas_cumprod: torch.Tensor, timesteps: torch.Tensor, gamma: float) -> torch.Tensor:
    """`min(snr, gamma) / snr` written out independently of the production helper."""
    alphas = alphas_cumprod.to(torch.float32)[timesteps]
    snr = alphas / (1.0 - alphas)
    weight = torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr
    weight = torch.where(torch.isnan(weight), torch.ones_like(weight), weight)
    return weight.clamp(max=1.0)


class StubDenoise:
    """Stands in for the UNet: returns whatever it was handed, ignoring its inputs."""

    def __init__(self, prediction: torch.Tensor):
        self.prediction = prediction

    def __call__(self, noisy_latents, timesteps, **kwargs):
        return (self.prediction,)


class MinSnrWeightTest(unittest.TestCase):
    def test_the_weight_table_follows_the_paper(self):
        with tempfile.TemporaryDirectory() as raw:
            cfg = _cfg(_marked_base(Path(raw), "plain", {"x": torch.zeros(1)}))
        scheduler = _training_scheduler(cfg)
        timesteps = torch.tensor([0, 10, 50, 100, 200, 400, 999])
        weight = min_snr_weight(timesteps, scheduler, 5.0)
        expected = _reference_weight(scheduler.alphas_cumprod, timesteps, 5.0)

        self.assertTrue(torch.allclose(weight, expected, atol=1e-7))
        # SDXL's epsilon schedule: snr 1175.4 at t=0 (down to 1/235), 3.06 at t=200 (kept).
        self.assertAlmostEqual(float(weight[0]), 0.00425, places=5)
        self.assertAlmostEqual(float(weight[4]), 1.0, places=6)
        self.assertTrue(torch.all(weight <= 1.0))
        self.assertTrue(torch.all(weight[1:] >= weight[:-1]))  # lower noise is weighted less

    def test_a_snr_below_gamma_keeps_the_full_weight(self):
        # Below gamma the ratio is snr/snr, i.e. no down-weighting at the high-noise end.
        scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([0.1, 0.2, 0.5]))
        timesteps = torch.tensor([0, 1, 2])
        weight = min_snr_weight(timesteps, scheduler, 5.0)
        self.assertTrue(torch.allclose(weight, torch.ones(3), atol=1e-6))
        # The same table with gamma below those snr values scales each step by gamma/snr:
        # a=0.1 → snr=1/9, a=0.2 → snr=1/4, a=0.5 → snr=1.
        tighter = min_snr_weight(timesteps, scheduler, 0.1)
        self.assertAlmostEqual(float(tighter[0]), 0.1 * 9, places=5)
        self.assertAlmostEqual(float(tighter[1]), 0.1 * 4, places=5)
        self.assertAlmostEqual(float(tighter[2]), 0.1, places=6)

    def test_the_table_comes_from_the_scheduler_it_is_given(self):
        timesteps = torch.tensor([0])
        low = SimpleNamespace(alphas_cumprod=torch.tensor([0.5]))
        high = SimpleNamespace(alphas_cumprod=torch.tensor([0.9]))
        self.assertAlmostEqual(float(min_snr_weight(timesteps, low, 5.0)[0]), 1.0, places=6)
        self.assertAlmostEqual(float(min_snr_weight(timesteps, high, 5.0)[0]), 5.0 / 9.0, places=6)

    def test_a_degenerate_schedule_does_not_produce_nan(self):
        timesteps = torch.tensor([0, 1])
        scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([0.0, 0.5]))
        weight = min_snr_weight(timesteps, scheduler, 5.0)
        self.assertTrue(torch.isfinite(weight).all())
        self.assertAlmostEqual(float(weight[0]), 1.0, places=6)


class MinSnrLossTest(unittest.TestCase):
    BATCH = 2
    SHAPE = (2, 4, 8, 8)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.epsilon_base = _marked_base(root, "epsilon", {"x": torch.zeros(1)})
        self.vpred_base = _marked_base(
            root, "vpred", {"v_pred": torch.tensor([]), "ztsnr": torch.tensor([])}
        )
        torch.manual_seed(1919)
        self.latents = torch.randn(*self.SHAPE)
        self.noise = torch.randn(*self.SHAPE)
        self.prediction = torch.randn(*self.SHAPE)
        self.timesteps = torch.tensor([10, 900])

    def tearDown(self):
        self.tmp.cleanup()

    def _loss(self, cfg: TrainConfig, scheduler=None, mask=None):
        modules = SimpleNamespace(
            noise_scheduler=scheduler or _training_scheduler(cfg),
            denoise=StubDenoise(self.prediction),
        )
        extra = {"time_ids": torch.zeros(6), "loss_mask": mask}
        family = resolve_family(cfg)
        with mock.patch("torch.randn_like", return_value=self.noise), mock.patch(
            "torch.randint", return_value=self.timesteps
        ):
            return family.denoise_loss(
                latents=self.latents,
                encoded=(torch.zeros(1), torch.zeros(1)),
                extra=extra,
                modules=modules,
                cfg=cfg,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def _reference(self, scheduler, *, weighted: bool, gamma: float = 5.0, mask=None):
        target = (
            scheduler.get_velocity(self.latents, self.noise, self.timesteps)
            if scheduler.config.prediction_type == "v_prediction"
            else self.noise
        )
        err = (self.prediction.float() - target.float()) ** 2
        err = apply_loss_mask(err, mask)
        per_sample = err.mean(dim=(1, 2, 3))
        if weighted:
            per_sample = per_sample * _reference_weight(
                scheduler.alphas_cumprod, self.timesteps, gamma
            )
        return per_sample.mean()

    def test_a_gamma_of_zero_leaves_the_loss_untouched(self):
        cfg = _cfg(self.epsilon_base, min_snr_gamma=0.0)
        scheduler = _training_scheduler(cfg)
        loss = self._loss(cfg, scheduler)
        self.assertTrue(torch.equal(loss, self._reference(scheduler, weighted=False)))

    def test_a_positive_gamma_scales_by_the_weight(self):
        cfg = _cfg(self.epsilon_base, min_snr_gamma=5.0)
        scheduler = _training_scheduler(cfg)
        loss = self._loss(cfg, scheduler)
        self.assertTrue(torch.equal(loss, self._reference(scheduler, weighted=True)))

        relaxed = self._loss(_cfg(self.epsilon_base, min_snr_gamma=0.5), scheduler)
        self.assertLess(float(relaxed), float(loss))  # a smaller gamma bites the low-noise step

    def test_the_weight_is_per_sample(self):
        # t=10 is weighted far below t=900, so the batch's weighting is not a single factor.
        cfg = _cfg(self.epsilon_base, min_snr_gamma=5.0)
        scheduler = _training_scheduler(cfg)
        weight = _reference_weight(scheduler.alphas_cumprod, self.timesteps, 5.0)
        self.assertLess(float(weight[0]), 0.5)
        self.assertAlmostEqual(float(weight[1]), 1.0, places=6)

        loss = self._loss(cfg, scheduler)
        unweighted = self._reference(scheduler, weighted=False)
        plain = (self.prediction.float() - self.noise) ** 2
        per_sample = plain.mean(dim=(1, 2, 3))
        self.assertAlmostEqual(float(loss / unweighted), float((per_sample * weight).mean() / per_sample.mean()), places=6)

    def test_the_mask_still_zeroes_weighted_pixels(self):
        cfg = _cfg(self.epsilon_base, min_snr_gamma=5.0)
        scheduler = _training_scheduler(cfg)
        mask = torch.ones(2, 1, 8, 8)
        mask[1] = 0.0  # the second sample is entirely masked out
        loss = self._loss(cfg, scheduler, mask)
        self.assertEqual(float(loss), float(self._reference(scheduler, weighted=True, mask=mask)))
        # ... and it contributes nothing: the weighted mean is the first sample's half of the batch.
        first = self._reference(scheduler, weighted=True, mask=mask)
        mask_one = torch.ones(2, 1, 8, 8)
        with_one = self._loss(cfg, scheduler, mask_one)
        self.assertGreater(float(with_one), float(first))

    def test_a_vpred_base_silently_ignores_it(self):
        cfg = _cfg(self.vpred_base, min_snr_gamma=5.0)
        self.assertEqual(cfg.prediction_type, "v_prediction")
        self.assertEqual(cfg.min_snr_gamma, 0.0)  # derived away, silently
        scheduler = _training_scheduler(cfg)
        self.assertTrue(scheduler.config.rescale_betas_zero_snr)
        loss = self._loss(cfg, scheduler)
        # The velocity target, unweighted: same number the loss would report with gamma unset.
        self.assertTrue(torch.equal(loss, self._reference(scheduler, weighted=False)))
        self.assertTrue(torch.isfinite(loss))

    def test_an_epsilon_base_keeps_the_configured_value(self):
        cfg = _cfg(self.epsilon_base, min_snr_gamma=5.0)
        self.assertEqual(cfg.prediction_type, "epsilon")
        self.assertEqual(cfg.min_snr_gamma, 5.0)
        self.assertEqual(TrainConfig().min_snr_gamma >= 0.0, True)
        # The gate is the prediction type, not the marker's absence: a file with v_pred only is
        # still v-prediction and still reads 0.
        only_v = _marked_base(Path(self.tmp.name), "only_v", {"v_pred": torch.tensor([])})
        self.assertEqual(_cfg(only_v, min_snr_gamma=5.0).min_snr_gamma, 0.0)

    def test_a_diffusers_directory_base_keeps_the_configured_value(self):
        root = Path(self.tmp.name) / "diffusers"
        (root / "scheduler").mkdir(parents=True)
        (root / "scheduler" / "scheduler_config.json").write_text(
            '{"prediction_type": "epsilon", "rescale_betas_zero_snr": false}', encoding="utf-8"
        )
        cfg = _cfg(root, min_snr_gamma=5.0)
        self.assertEqual(cfg.prediction_type, "epsilon")
        self.assertEqual(cfg.min_snr_gamma, 5.0)


if __name__ == "__main__":
    unittest.main()
