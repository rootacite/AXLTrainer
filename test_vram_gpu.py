import os
import tempfile
import unittest

import torch
from torch import nn

from trainer import control
from trainer.device_swap import SwapContext, run_pause, run_resume
from trainer.cache import prepare_encoding_devices
from trainer.family_sdxl import enable_te_gradient_checkpointing
from trainer.sampling import (
    _offload_text_encoders,
    _prepare_decode_devices,
    _prepare_denoise_device,
    _restore_train_modules,
)

_HAS_CUDA = torch.cuda.is_available()


def _device_of(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


class FatLinear(nn.Module):
    def __init__(self, size: int = 2048):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, size))


@unittest.skipUnless(_HAS_CUDA, "CUDA/ROCm GPU required")
class TeCheckpointGpuTest(unittest.TestCase):
    def test_peft_clip_backward_with_checkpointing(self):
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPTextConfig, CLIPTextModel

        config = CLIPTextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=77,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        )
        base = CLIPTextModel(config)
        base.requires_grad_(False)
        te = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=4,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            ),
        )
        enable_te_gradient_checkpointing(te)
        te.train()
        te.to("cuda")

        input_ids = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
        out = te(input_ids, output_hidden_states=True, return_dict=True)
        # last_hidden_state keeps every layer on the graph so checkpointing
        # must recompute through all LoRA blocks (clip_skip indexing is T2).
        loss = out.last_hidden_state.float().pow(2).mean()
        loss.backward()

        lora_with_grad = 0
        for name, param in te.named_parameters():
            if not param.requires_grad:
                continue
            self.assertIsNotNone(param.grad, msg=name)
            self.assertTrue(torch.isfinite(param.grad).all(), msg=name)
            lora_with_grad += 1
        self.assertGreater(lora_with_grad, 0)


@unittest.skipUnless(_HAS_CUDA, "CUDA/ROCm GPU required")
class SamplingOffloadGpuTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AXL_RUNTIME_DIR"] = self.tmp.name
        control._state = {}
        control._last_cmd_seq = 0
        control._ended = False
        control._last_write_mono = 0.0
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self.unet = FatLinear().to(self.device)
        self.te1 = FatLinear().to(self.device)
        self.te2 = FatLinear().to(self.device)
        self.vae = nn.Linear(8, 8)
        torch.cuda.empty_cache()

    def tearDown(self):
        del self.unet, self.te1, self.te2, self.vae
        torch.cuda.empty_cache()
        self.tmp.cleanup()
        os.environ.pop("AXL_RUNTIME_DIR", None)

    def test_decode_offload_drops_allocated_and_restore_returns_cuda(self):
        _offload_text_encoders(self.te1, self.te2)
        _prepare_denoise_device(self.unet, self.device, self.te1, self.te2)
        after_denoise = torch.cuda.memory_allocated()

        _prepare_decode_devices(self.unet, self.vae, self.device)
        after_decode_prep = torch.cuda.memory_allocated()
        self.assertEqual(_device_of(self.unet).type, "cpu")
        self.assertEqual(_device_of(self.te1).type, "cpu")
        self.assertEqual(_device_of(self.te2).type, "cpu")
        self.assertEqual(_device_of(self.vae).type, "cuda")
        self.assertLess(after_decode_prep, after_denoise)

        _restore_train_modules(
            unet=self.unet,
            te1=self.te1,
            te2=self.te2,
            vae=self.vae,
            device=self.device,
        )
        self.assertEqual(_device_of(self.unet).type, "cuda")
        self.assertEqual(_device_of(self.te1).type, "cuda")
        self.assertEqual(_device_of(self.te2).type, "cuda")
        self.assertEqual(_device_of(self.vae).type, "cpu")

    def test_pause_resume_sampling_reoffloads_tes(self):
        ctx = SwapContext(
            device=self.device,
            vae=self.vae,
            denoise=self.unet,
            text_encoders=[self.te1, self.te2],
        )
        _offload_text_encoders(self.te1, self.te2)
        run_pause(ctx, "sampling")
        run_resume(ctx, "sampling")
        self.assertEqual(_device_of(self.te1).type, "cuda")
        self.assertEqual(_device_of(self.te2).type, "cuda")

        _prepare_denoise_device(self.unet, self.device, self.te1, self.te2)
        self.assertEqual(_device_of(self.te1).type, "cpu")
        self.assertEqual(_device_of(self.te2).type, "cpu")
        self.assertEqual(_device_of(self.unet).type, "cuda")

    def test_prepare_encoding_devices_leaves_only_vae_on_cuda(self):
        ctx = SwapContext(
            device=self.device,
            vae=self.vae,
            denoise=self.unet,
            text_encoders=[self.te1, self.te2],
        )
        before = torch.cuda.memory_allocated()
        prepare_encoding_devices(self.vae, self.device, ctx)
        after = torch.cuda.memory_allocated()
        self.assertEqual(_device_of(self.unet).type, "cpu")
        self.assertEqual(_device_of(self.te1).type, "cpu")
        self.assertEqual(_device_of(self.te2).type, "cpu")
        self.assertEqual(_device_of(self.vae).type, "cuda")
        self.assertLess(after, before)

    def test_restore_after_inference_offload_allows_backward(self):
        linear = nn.Linear(8, 8, device=self.device)
        linear.train()
        with torch.inference_mode():
            _offload_text_encoders(self.te1, self.te2)
            _prepare_denoise_device(self.unet, self.device, self.te1, self.te2)
            _prepare_decode_devices(self.unet, self.vae, self.device)
            _restore_train_modules(
                unet=self.unet,
                te1=linear,
                te2=self.te2,
                vae=self.vae,
                device=self.device,
            )
        loss = linear(torch.randn(2, 8, device=self.device)).sum()
        loss.backward()
        self.assertIsNotNone(linear.weight.grad)


if __name__ == "__main__":
    unittest.main()
