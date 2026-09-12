import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel

import api
from trainer.checkpoints import read_lora_metadata, resolve_resume_path
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
from trainer.family import FamilyModules
from trainer.family_sdxl import (
    SdxlFamily,
    _convert_peft_to_kohya_bf16,
    build_kohya_to_peft_map,
    enable_te_gradient_checkpointing,
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


class TeCheckpointHelperTest(unittest.TestCase):
    def test_calls_transformers_and_input_grad_hooks(self):
        class Stub:
            def __init__(self):
                self.checkpoint = 0
                self.input_grads = 0

            def gradient_checkpointing_enable(self):
                self.checkpoint += 1

            def enable_input_require_grads(self):
                self.input_grads += 1

        stub = Stub()
        enable_te_gradient_checkpointing(stub)
        self.assertEqual(stub.checkpoint, 1)
        self.assertEqual(stub.input_grads, 1)

    def test_falls_back_to_diffusers_enable_name(self):
        class Stub:
            def __init__(self):
                self.checkpoint = 0

            def enable_gradient_checkpointing(self):
                self.checkpoint += 1

        stub = Stub()
        enable_te_gradient_checkpointing(stub)
        self.assertEqual(stub.checkpoint, 1)

    def test_plain_linear_is_a_noop(self):
        module = torch.nn.Linear(4, 4)
        enable_te_gradient_checkpointing(module)


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


class KohyaKeyMapTest(unittest.TestCase):
    """The resume map must invert exactly what save_lora writes."""

    def test_sdxl_unet_paths_round_trip(self):
        adapter = {
            "base_model.model.down_blocks.1.attentions.0.to_q.lora_A.default.weight": torch.zeros(4, 8),
            "base_model.model.down_blocks.1.attentions.0.to_q.lora_B.default.weight": torch.zeros(8, 4),
            "base_model.model.mid_block.attentions.0.to_v.lora_A.default.weight": torch.zeros(4, 8),
            "base_model.model.mid_block.attentions.0.to_v.lora_B.default.weight": torch.zeros(8, 4),
            "base_model.model.up_blocks.1.attentions.2.to_out.0.lora_A.default.weight": torch.zeros(4, 8),
            "base_model.model.up_blocks.1.attentions.2.to_out.0.lora_B.default.weight": torch.zeros(8, 4),
        }
        mapping = build_kohya_to_peft_map(adapter, "unet", 24)
        self.assertEqual(
            mapping["lora_unet_input_blocks_4_1_to_q.lora_down.weight"],
            "base_model.model.down_blocks.1.attentions.0.to_q.lora_A.default.weight",
        )
        self.assertEqual(
            mapping["lora_unet_middle_block_1_to_v.lora_up.weight"],
            "base_model.model.mid_block.attentions.0.to_v.lora_B.default.weight",
        )
        self.assertEqual(
            mapping["lora_unet_output_blocks_5_1_to_out_0.lora_down.weight"],
            "base_model.model.up_blocks.1.attentions.2.to_out.0.lora_A.default.weight",
        )
        self.assertEqual(len(mapping), 6)

    def test_te_paths_round_trip(self):
        adapter = {
            "base_model.model.text_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(4, 8),
            "base_model.model.text_model.encoder.layers.0.self_attn.q_proj.lora_B.default.weight": torch.zeros(8, 4),
        }
        mapping = build_kohya_to_peft_map(adapter, "te1", 24)
        self.assertEqual(
            mapping["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"],
            "base_model.model.text_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight",
        )
        self.assertEqual(len(mapping), 2)

    def test_alpha_keys_are_skipped(self):
        adapter = {"base_model.model.down_blocks.1.attentions.0.to_q.lora_A.default.weight": torch.zeros(4, 8)}
        mapping = build_kohya_to_peft_map(adapter, "unet", 24)
        self.assertFalse(any(key.endswith(".alpha") for key in mapping))

    def test_base_weights_are_not_mapped(self):
        mapping = build_kohya_to_peft_map(
            {"base_model.model.down_blocks.1.attentions.0.to_q.weight": torch.zeros(8, 8)},
            "unet",
            24,
        )
        self.assertEqual(mapping, {})


def _tiny_text_encoder(rank: int, alpha: int):
    from peft import LoraConfig, get_peft_model

    config = CLIPTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=100,
        max_position_embeddings=77,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    model = CLIPTextModel(config)
    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ),
    )


class ResumeLoadTest(unittest.TestCase):
    """Weights-only resume: kohya file → PEFT modules, no optimizers, no counters."""

    RANK = 4
    ALPHA = 8

    def setUp(self):
        from peft import get_peft_model_state_dict

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.source = _tiny_text_encoder(self.RANK, self.ALPHA)
        self.checkpoint = self.dir / "rein.safetensors"
        converted = _convert_peft_to_kohya_bf16(
            get_peft_model_state_dict(self.source), "te1", float(self.ALPHA)
        )
        # An unmapped UNet-style key must be reported as skipped, not silently dropped.
        converted["lora_unet_down_blocks_1_attentions_0_to_q.lora_down.weight"] = torch.zeros(
            4, 8, dtype=torch.bfloat16
        )
        save_file(
            converted,
            str(self.checkpoint),
            metadata={
                "ss_steps": "300",
                "ss_epoch": "7",
                "ss_network_dim": str(self.RANK),
                "ss_network_alpha": str(self.ALPHA),
            },
        )

    def _family_and_modules(self, rank: int = RANK, alpha: int = ALPHA):
        target = _tiny_text_encoder(rank, alpha)
        stand_in = _tiny_text_encoder(rank, alpha)
        modules = FamilyModules(
            pipe=None,
            vae=None,
            denoise=stand_in,
            tokenizers=[],
            text_encoders=[target, _tiny_text_encoder(rank, alpha)],
        )
        return SdxlFamily(CATALOG["sdxl_base_v1-0"]), modules, target

    def _cfg(self, **overrides):
        values = {
            "resume_lora_path": str(self.checkpoint),
            "network_dim": self.RANK,
            "network_alpha": self.ALPHA,
        }
        values.update(overrides)
        return TrainConfig(**values)

    def test_loads_weights_and_reports_metadata(self):
        from peft import get_peft_model_state_dict

        family, modules, target = self._family_and_modules()
        info = family.load_lora(self._cfg(), modules)

        self.assertEqual(info["path"], str(self.checkpoint))
        self.assertEqual(info["filename"], "rein.safetensors")
        self.assertEqual(info["step"], 300)
        self.assertEqual(info["epoch"], 7)
        self.assertEqual(info["skipped"], 1)
        self.assertGreater(info["loaded"], 0)

        expected = get_peft_model_state_dict(self.source)
        actual = get_peft_model_state_dict(target)
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self.assertTrue(
                torch.equal(actual[key], expected[key].to(torch.bfloat16).to(actual[key].dtype)),
                f"weights differ for {key}",
            )

    def test_empty_path_is_a_noop(self):
        family, modules, _target = self._family_and_modules()
        self.assertEqual(family.load_lora(self._cfg(resume_lora_path=""), modules), {})

    def test_rank_mismatch_raises_with_hint(self):
        family, modules, _target = self._family_and_modules(rank=8, alpha=self.ALPHA)
        with self.assertRaises(ValueError) as ctx:
            family.load_lora(self._cfg(), modules)
        message = str(ctx.exception)
        self.assertIn("network_dim", message)

    def test_alpha_mismatch_warns(self):
        family, modules, _target = self._family_and_modules()
        with self.assertLogs("trainer.family_sdxl", level="WARNING") as logs:
            family.load_lora(self._cfg(network_alpha=99), modules)
        self.assertTrue(any("alpha" in line for line in logs.output))

    def test_checkpoint_with_only_unmapped_keys_raises(self):
        family, modules, _target = self._family_and_modules()
        foreign = self.dir / "foreign.safetensors"
        save_file(
            {"lora_unet_down_blocks_1_attentions_0_to_q.lora_down.weight": torch.zeros(4, 8)},
            str(foreign),
        )
        with self.assertRaises(ValueError) as ctx:
            family.load_lora(self._cfg(resume_lora_path=str(foreign)), modules)
        self.assertIn("no LoRA tensors", str(ctx.exception))

    def test_missing_file_raises(self):
        family, modules, _target = self._family_and_modules()
        cfg = self._cfg(resume_lora_path=str(self.dir / "nope.safetensors"))
        with self.assertRaises(ValueError) as ctx:
            family.load_lora(cfg, modules)
        self.assertIn("not found", str(ctx.exception))

    def test_sd35_refuses_resume(self):
        family = resolve_family(TrainConfig(**_sd35_kwargs()))
        with self.assertRaises(UnsupportedFamilyError):
            family.load_lora(self._cfg(), None)
        self.assertEqual(family.load_lora(TrainConfig(**_sd35_kwargs()), None), {})


class ResolveResumePathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_accepts_file(self):
        target = self.dir / "rein.safetensors"
        target.write_bytes(b"x")
        self.assertEqual(resolve_resume_path(str(target)), target)
        self.assertEqual(resolve_resume_path(target), target)

    def test_accepts_directory_with_one_file(self):
        target = self.dir / "rein_final"
        target.mkdir()
        expected = target / "rein.safetensors"
        expected.write_bytes(b"x")
        self.assertEqual(resolve_resume_path(str(target)), expected)

    def test_rejects_empty_or_ambiguous_directory(self):
        empty = self.dir / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError) as ctx:
            resolve_resume_path(str(empty))
        self.assertIn("no .safetensors", str(ctx.exception))

        both = self.dir / "both"
        both.mkdir()
        (both / "a.safetensors").write_bytes(b"x")
        (both / "b.safetensors").write_bytes(b"x")
        with self.assertRaises(ValueError) as ctx:
            resolve_resume_path(str(both))
        self.assertIn("point resume_lora_path at the file", str(ctx.exception))

    def test_rejects_missing_and_wrong_suffix(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_resume_path(str(self.dir / "nope.safetensors"))
        self.assertIn("not found", str(ctx.exception))

        wrong = self.dir / "rein.ckpt"
        wrong.write_bytes(b"x")
        with self.assertRaises(ValueError) as ctx:
            resolve_resume_path(str(wrong))
        self.assertIn(".safetensors", str(ctx.exception))

    def test_empty_value_raises(self):
        with self.assertRaises(ValueError):
            resolve_resume_path("   ")

    def test_read_lora_metadata_round_trip(self):
        target = self.dir / "rein.safetensors"
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(target),
            metadata={"ss_steps": "42"},
        )
        self.assertEqual(read_lora_metadata(target)["ss_steps"], "42")


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

    def test_train_start_rejects_missing_resume_checkpoint(self):
        cfg = TrainConfig(resume_lora_path="/tmp/axl-missing-resume.safetensors")
        with mock.patch("api.TrainConfig", return_value=cfg):
            with mock.patch("subprocess.Popen") as popen:
                with self.assertRaises(ValueError) as ctx:
                    api.handle_train_start({})
                self.assertIn("not found", str(ctx.exception))
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
