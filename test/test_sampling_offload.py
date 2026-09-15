import os
import tempfile
import unittest

import torch
from torch import nn

import sys
from pathlib import Path

# `python test/test_sampling_offload.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer import control
from trainer.device_swap import SwapContext, run_pause, run_resume
from trainer.cache import prepare_encoding_devices
from trainer.sampling import (
    _offload_text_encoders,
    _prepare_decode_devices,
    _prepare_denoise_device,
    _reify_autograd_tensors,
    _restore_train_modules,
)


def _device_of(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


class SamplingOffloadHelperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.unet = nn.Linear(8, 8)
        self.te1 = nn.Linear(8, 8)
        self.te2 = nn.Linear(8, 8)
        self.vae = nn.Linear(8, 8)
        self.unet.train()
        self.te1.train()
        self.te2.train()
        self.vae.eval()

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_decode_stage_leaves_unet_and_tes_on_cpu(self):
        _offload_text_encoders(self.te1, self.te2)
        _prepare_denoise_device(self.unet, self.device, self.te1, self.te2)
        _prepare_decode_devices(self.unet, self.vae, self.device)

        self.assertEqual(_device_of(self.unet).type, "cpu")
        self.assertEqual(_device_of(self.te1).type, "cpu")
        self.assertEqual(_device_of(self.te2).type, "cpu")
        self.assertEqual(_device_of(self.vae).type, "cpu")

    def test_restore_returns_unet_and_tes_and_offloads_vae(self):
        self.unet.eval()
        self.te1.eval()
        _prepare_decode_devices(self.unet, self.vae, self.device)
        _restore_train_modules(
            unet=self.unet,
            te1=self.te1,
            te2=self.te2,
            vae=self.vae,
            device=self.device,
        )
        self.assertEqual(_device_of(self.unet), self.device)
        self.assertEqual(_device_of(self.te1), self.device)
        self.assertEqual(_device_of(self.te2), self.device)
        self.assertEqual(_device_of(self.vae).type, "cpu")

    def test_resume_sampling_then_prepare_denoise_keeps_tes_off_gpu(self):
        ctx = SwapContext(
            device=self.device,
            vae=self.vae,
            denoise=self.unet,
            text_encoders=[self.te1, self.te2],
        )
        _offload_text_encoders(self.te1, self.te2)
        run_pause(ctx, "sampling")
        run_resume(ctx, "sampling")
        _prepare_denoise_device(self.unet, self.device, self.te1, self.te2)
        self.assertEqual(_device_of(self.te1).type, "cpu")
        self.assertEqual(_device_of(self.te2).type, "cpu")
        self.assertEqual(_device_of(self.unet), self.device)

    def test_prepare_encoding_devices_offloads_unet_and_tes(self):
        ctx = SwapContext(
            device=self.device,
            vae=self.vae,
            denoise=self.unet,
            text_encoders=[self.te1, self.te2],
        )
        prepare_encoding_devices(self.vae, self.device, ctx)
        self.assertEqual(_device_of(self.unet).type, "cpu")
        self.assertEqual(_device_of(self.te1).type, "cpu")
        self.assertEqual(_device_of(self.te2).type, "cpu")
        self.assertEqual(_device_of(self.vae), self.device)

    def test_restore_preserves_mixed_parameter_dtypes(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
                self.lora = nn.Parameter(torch.randn(4, 4, dtype=torch.float32))

        unet = Mixed()
        te1 = Mixed()
        te2 = Mixed()
        vae = nn.Linear(4, 4)
        _prepare_decode_devices(unet, vae, self.device)
        _restore_train_modules(unet=unet, te1=te1, te2=te2, vae=vae, device=self.device)
        self.assertEqual(unet.base.dtype, torch.bfloat16)
        self.assertEqual(unet.lora.dtype, torch.float32)
        self.assertEqual(te1.base.dtype, torch.bfloat16)
        self.assertEqual(te1.lora.dtype, torch.float32)

    def test_reify_after_inference_mode_move_allows_backward(self):
        module = nn.Linear(4, 4)
        with torch.inference_mode():
            module.weight.data = torch.randn_like(module.weight)
            module.bias.data = torch.randn_like(module.bias)
        self.assertTrue(any(torch.is_inference(p) for p in module.parameters()))
        _reify_autograd_tensors(module)
        self.assertFalse(any(torch.is_inference(p) for p in module.parameters()))
        module.train()
        loss = module(torch.randn(2, 4)).sum()
        loss.backward()
        self.assertIsNotNone(module.weight.grad)


if __name__ == "__main__":
    unittest.main()
