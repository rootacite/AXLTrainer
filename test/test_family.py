import atexit
import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel

import sys

# `python test/test_family.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TrainConfig takes its field defaults from the config.toml in the cwd
# (trainer/config.py), and the repo's own file selects the author's local locon
# network. These tests assert the code defaults, so they import a copy of
# trainer.config loaded against a fixture that sets no keys; the repo's module keeps
# the values the other suites and the GUI read.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_DIR = Path(tempfile.mkdtemp(prefix="axl-test-family-config-"))
_FIXTURE_DIR.joinpath("config.toml").write_text(
    "# Fixture for test_family.py: no keys, so TrainConfig keeps its code defaults.\n",
    encoding="utf-8",
)
_original_cwd = os.getcwd()
try:
    os.chdir(_FIXTURE_DIR)
    _config_spec = importlib.util.spec_from_file_location(
        "_test_family_config", _REPO_ROOT / "trainer" / "config.py"
    )
    _fixture_config = importlib.util.module_from_spec(_config_spec)
    _config_spec.loader.exec_module(_fixture_config)
finally:
    os.chdir(_original_cwd)
atexit.register(shutil.rmtree, _FIXTURE_DIR, ignore_errors=True)

TrainConfig = _fixture_config.TrainConfig

import api
from trainer.checkpoints import (
    infer_network_type,
    read_lora_metadata,
    require_resume_network_type,
    resolve_resume_path,
)
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
    _CONV_ADAPTER_NAME,
    _LOCON_TE_LORA_TARGETS,
    _LOCON_UNET_CONV_TARGETS,
    _LOCON_UNET_LINEAR_TARGETS,
    _LOCON_UNET_LORA_TARGETS,
    _STANDARD_UNET_LORA_TARGETS,
    _TE_LORA_TARGETS,
    _convert_peft_to_kohya_bf16,
    build_kohya_to_peft_map,
    enable_te_gradient_checkpointing,
    te_lora_targets,
    unet_lora_targets,
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

    def test_output_name_has_to_be_filename_safe(self):
        for name in ("re in", "re/in", "rein "):
            with self.assertRaises(ValueError) as ctx:
                TrainConfig(output_name=name)
            self.assertIn("output_name must be", str(ctx.exception))
        self.assertEqual(TrainConfig(output_name="re_in_v2").output_name, "re_in_v2")

    def test_a_config_file_with_a_space_in_the_name_fails_at_startup(self):
        # The startup path end to end: config.toml → flattened keys → TrainConfig, in a fresh
        # module load, because a module reads its file once at import (which is why the trainer
        # sees the change and a long-running process does not).
        directory = tempfile.mkdtemp(prefix="axl-test-bad-name-")
        try:
            Path(directory, "config.toml").write_text(
                '[environment]\noutput_name = "re in"\n', encoding="utf-8"
            )
            spec = importlib.util.spec_from_file_location(
                "_test_bad_name_config", _REPO_ROOT / "trainer" / "config.py"
            )
            module = importlib.util.module_from_spec(spec)
            previous_cwd = os.getcwd()
            try:
                os.chdir(directory)
                spec.loader.exec_module(module)
                with self.assertRaises(ValueError) as ctx:
                    module.TrainConfig()
            finally:
                os.chdir(previous_cwd)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        self.assertIn("output_name must be", str(ctx.exception))

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

    def test_apply_lora_skips_checkpointing_when_disabled(self):
        family = SdxlFamily(CATALOG["sdxl_base_v1-0"])
        te = mock.Mock(name="te")
        denoise = mock.Mock(name="denoise")
        modules = FamilyModules(
            pipe=None,
            vae=None,
            denoise=denoise,
            tokenizers=[],
            text_encoders=[te],
        )
        cfg = TrainConfig(
            gradient_checkpointing_unet=False,
            gradient_checkpointing_te=False,
        )
        with mock.patch("trainer.family_sdxl.get_peft_model", side_effect=lambda module, _cfg: module), \
             mock.patch("trainer.family_sdxl.enable_flash_attention"), \
             mock.patch("trainer.family_sdxl.enable_te_gradient_checkpointing") as te_ckpt:
            family.apply_lora(cfg, modules)
        te_ckpt.assert_not_called()
        denoise.enable_gradient_checkpointing.assert_not_called()

    def test_apply_lora_enables_checkpointing_by_default(self):
        family = SdxlFamily(CATALOG["sdxl_base_v1-0"])
        te = mock.Mock(name="te")
        denoise = mock.Mock(name="denoise")
        modules = FamilyModules(
            pipe=None,
            vae=None,
            denoise=denoise,
            tokenizers=[],
            text_encoders=[te],
        )
        cfg = TrainConfig(
            gradient_checkpointing_unet=True,
            gradient_checkpointing_te=True,
        )
        with mock.patch("trainer.family_sdxl.get_peft_model", side_effect=lambda module, _cfg: module), \
             mock.patch("trainer.family_sdxl.enable_flash_attention"), \
             mock.patch("trainer.family_sdxl.enable_te_gradient_checkpointing") as te_ckpt:
            family.apply_lora(cfg, modules)
        te_ckpt.assert_called_once_with(te)
        denoise.enable_gradient_checkpointing.assert_called_once_with()


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
        self.assertEqual(meta["ss_network_type"], "standard")
        self.assertNotIn("ss_network_args", meta)

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
        self.assertEqual(
            mapping["lora_te1_encoder_layers_0_self_attn_q_proj.lora_down.weight"],
            mapping["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"],
        )

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

    def test_read_lora_metadata_rerereads_a_changed_file(self):
        """The cache is the dashboard's repeated scan; a rewritten checkpoint must not go stale."""
        target = self.dir / "rein.safetensors"
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(target),
            metadata={"ss_steps": "42"},
        )
        self.assertEqual(read_lora_metadata(target)["ss_steps"], "42")
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(8, 2)},
            str(target),
            metadata={"ss_steps": "77"},
        )
        self.assertEqual(read_lora_metadata(target)["ss_steps"], "77")

    def test_a_caller_cannot_corrupt_the_metadata_cache(self):
        target = self.dir / "rein.safetensors"
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(target),
            metadata={"ss_steps": "42"},
        )
        first = read_lora_metadata(target)
        first["ss_steps"] = "0"
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

    def test_train_start_rejects_locon_resume_on_standard_config(self):
        path = Path(self.tmp.name) / "locon.safetensors"
        save_file(
            {"lora_unet_x.lora_down.weight": torch.zeros(4, 2)},
            str(path),
            metadata={
                "ss_network_type": "locon",
                "ss_network_dim": "4",
                "ss_network_alpha": "2",
            },
        )
        cfg = TrainConfig(resume_lora_path=str(path), network_type="standard")
        with mock.patch("api.TrainConfig", return_value=cfg):
            with mock.patch("subprocess.Popen") as popen:
                with self.assertRaises(ValueError) as ctx:
                    api.handle_train_start({})
                message = str(ctx.exception)
                self.assertIn("locon", message)
                self.assertIn("standard", message)
                popen.assert_not_called()


class NetworkTypeConfigTest(unittest.TestCase):
    def test_default_is_standard(self):
        self.assertEqual(TrainConfig().network_type, "standard")

    def test_locon_requires_conv_dim(self):
        with self.assertRaises(ValueError) as ctx:
            TrainConfig(network_type="locon")
        self.assertIn("conv_dim", str(ctx.exception))

    def test_locon_allows_unequal_rank(self):
        cfg = TrainConfig(
            network_type="locon",
            network_dim=16,
            network_alpha=8,
            conv_dim=8,
            conv_alpha=4,
        )
        self.assertEqual(cfg.conv_dim, 8)
        self.assertEqual(cfg.conv_alpha, 4)
        self.assertEqual(cfg.network_dim, 16)

    def test_locon_accepts_matching_rank(self):
        cfg = TrainConfig(
            network_type="locon",
            network_dim=16,
            network_alpha=8,
            conv_dim=16,
            conv_alpha=8,
        )
        self.assertEqual(cfg.network_type, "locon")
        self.assertEqual(cfg.conv_dim, 16)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            TrainConfig(network_type="lycoris")


class NetworkTypeInferTest(unittest.TestCase):
    def test_missing_type_is_standard(self):
        self.assertEqual(infer_network_type({}), "standard")
        self.assertEqual(
            infer_network_type({"ss_network_args": "conv_dim=0 conv_alpha=0"}),
            "standard",
        )

    def test_missing_type_with_conv_dim_is_locon(self):
        self.assertEqual(
            infer_network_type({"ss_network_args": "conv_dim=8 conv_alpha=8"}),
            "locon",
        )
        self.assertEqual(
            infer_network_type({"ss_network_args": '{"conv_dim": 8, "conv_alpha": 8}'}),
            "locon",
        )

    def test_explicit_type_wins(self):
        self.assertEqual(
            infer_network_type(
                {"ss_network_type": "standard", "ss_network_args": "conv_dim=8"}
            ),
            "standard",
        )

    def test_require_mismatch_names_both_types(self):
        with self.assertRaises(ValueError) as ctx:
            require_resume_network_type(
                TrainConfig(network_type="standard"),
                {"ss_network_type": "locon"},
                "/tmp/x.safetensors",
            )
        message = str(ctx.exception)
        self.assertIn("locon", message)
        self.assertIn("standard", message)
        self.assertIn("/tmp/x.safetensors", message)


class LoconTargetsAndRemapTest(unittest.TestCase):
    def test_standard_targets_unchanged(self):
        self.assertEqual(
            unet_lora_targets("standard"),
            ("to_q", "to_k", "to_v", "to_out.0"),
        )
        self.assertEqual(unet_lora_targets("standard"), _STANDARD_UNET_LORA_TARGETS)
        self.assertEqual(_TE_LORA_TARGETS, ("q_proj", "k_proj", "v_proj", "out_proj"))
        self.assertEqual(te_lora_targets("standard"), _TE_LORA_TARGETS)
        self.assertEqual(te_lora_targets("locon"), _LOCON_TE_LORA_TARGETS)
        self.assertIn("fc1", te_lora_targets("locon"))
        self.assertIn("fc2", te_lora_targets("locon"))
        self.assertNotIn("fc1", te_lora_targets("standard"))

    def test_locon_targets_include_c3lier_and_not_conv_in_out(self):
        targets = unet_lora_targets("locon")
        self.assertEqual(targets, _LOCON_UNET_LORA_TARGETS)
        for name in (
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "time_emb_proj",
            "conv",
        ):
            self.assertIn(name, targets)
        self.assertNotIn("conv_in", targets)
        self.assertNotIn("conv_out", targets)
        self.assertFalse(any(name.endswith("conv") and name != "conv" for name in targets))

    def test_apply_lora_standard_keeps_attention_targets(self):
        family = SdxlFamily(CATALOG["sdxl_base_v1-0"])
        captured = []

        def capture(module, lora_cfg):
            captured.append(lora_cfg)
            return module

        modules = FamilyModules(
            pipe=None,
            vae=None,
            denoise=mock.Mock(name="denoise"),
            tokenizers=[],
            text_encoders=[mock.Mock(name="te")],
        )
        cfg = TrainConfig(
            network_type="standard",
            gradient_checkpointing_unet=False,
            gradient_checkpointing_te=False,
        )
        with mock.patch("trainer.family_sdxl.get_peft_model", side_effect=capture), \
             mock.patch("trainer.family_sdxl.enable_flash_attention"):
            family.apply_lora(cfg, modules)
        te_cfg, unet_cfg = captured[0], captured[-1]
        self.assertEqual(set(te_cfg.target_modules), set(_TE_LORA_TARGETS))
        self.assertEqual(set(unet_cfg.target_modules), set(_STANDARD_UNET_LORA_TARGETS))
        modules.denoise.add_adapter.assert_not_called()

    def test_apply_lora_locon_uses_two_adapters_and_te_mlp(self):
        family = SdxlFamily(CATALOG["sdxl_base_v1-0"])
        captured = []

        def capture(module, lora_cfg):
            captured.append(lora_cfg)
            return module

        denoise = mock.Mock(name="denoise")
        modules = FamilyModules(
            pipe=None,
            vae=None,
            denoise=denoise,
            tokenizers=[],
            text_encoders=[mock.Mock(name="te")],
        )
        cfg = TrainConfig(
            network_type="locon",
            network_dim=16,
            network_alpha=8,
            conv_dim=8,
            conv_alpha=4,
            gradient_checkpointing_unet=False,
            gradient_checkpointing_te=False,
        )
        with mock.patch("trainer.family_sdxl.get_peft_model", side_effect=capture), \
             mock.patch("trainer.family_sdxl.enable_flash_attention"):
            family.apply_lora(cfg, modules)
        te_cfg, unet_cfg = captured[0], captured[-1]
        self.assertEqual(set(te_cfg.target_modules), set(_LOCON_TE_LORA_TARGETS))
        self.assertEqual(set(unet_cfg.target_modules), set(_LOCON_UNET_LINEAR_TARGETS))
        self.assertEqual(unet_cfg.r, 16)
        denoise.add_adapter.assert_called_once()
        name, conv_cfg = denoise.add_adapter.call_args.args
        self.assertEqual(name, _CONV_ADAPTER_NAME)
        self.assertEqual(set(conv_cfg.target_modules), set(_LOCON_UNET_CONV_TARGETS))
        self.assertEqual(conv_cfg.r, 8)
        self.assertEqual(conv_cfg.lora_alpha, 4)
        denoise.base_model.set_adapter.assert_called_once_with(
            ["default", _CONV_ADAPTER_NAME]
        )
        self.assertEqual(
            set(_LOCON_UNET_LINEAR_TARGETS) | set(_LOCON_UNET_CONV_TARGETS),
            set(_LOCON_UNET_LORA_TARGETS),
        )

    def test_conv1_remaps_to_ldm_in_layers(self):
        adapter = {
            "base_model.model.down_blocks.0.resnets.0.conv1.lora_A.default.weight": torch.zeros(
                4, 8, 3, 3
            ),
            "base_model.model.down_blocks.0.resnets.0.conv1.lora_B.default.weight": torch.zeros(
                8, 4, 1, 1
            ),
        }
        mapping = build_kohya_to_peft_map(adapter, "unet", 4)
        self.assertEqual(
            mapping["lora_unet_input_blocks_1_0_in_layers_2.lora_down.weight"],
            "base_model.model.down_blocks.0.resnets.0.conv1.lora_A.default.weight",
        )

    def test_conv_4d_round_trip(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(8, 8, 3, padding=1)

        wrapped = get_peft_model(
            Block(),
            LoraConfig(
                r=4,
                lora_alpha=4,
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=["conv1"],
            ),
        )
        converted = _convert_peft_to_kohya_bf16(
            get_peft_model_state_dict(wrapped), "unet", 4.0
        )
        down = converted["lora_unet_conv1.lora_down.weight"]
        up = converted["lora_unet_conv1.lora_up.weight"]
        self.assertEqual(down.ndim, 4)
        self.assertEqual(tuple(up.shape[-2:]), (1, 1))

    def test_conv_4d_uses_conv_rank(self):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(8, 8, 3, padding=1)

        wrapped = get_peft_model(
            Block(),
            LoraConfig(
                r=8,
                lora_alpha=4,
                lora_dropout=0.0,
                init_lora_weights="gaussian",
                target_modules=["conv1"],
            ),
        )
        converted = _convert_peft_to_kohya_bf16(
            get_peft_model_state_dict(wrapped), "unet", 4.0
        )
        down = converted["lora_unet_conv1.lora_down.weight"]
        self.assertEqual(down.shape[0], 8)
        self.assertEqual(float(converted["lora_unet_conv1.alpha"]), 4.0)

    def test_linear_and_conv_alphas_stay_separate(self):
        linear = {
            "base_model.model.down_blocks.1.attentions.0.to_q.lora_A.default.weight": torch.zeros(
                16, 8
            ),
        }
        conv = {
            "base_model.model.down_blocks.0.resnets.0.conv1.lora_A.default.weight": torch.zeros(
                8, 8, 3, 3
            ),
        }
        lin = _convert_peft_to_kohya_bf16(linear, "unet", 8.0)
        convd = _convert_peft_to_kohya_bf16(conv, "unet", 4.0)
        self.assertEqual(float(lin["lora_unet_input_blocks_4_1_to_q.alpha"]), 8.0)
        self.assertEqual(
            float(convd["lora_unet_input_blocks_1_0_in_layers_2.alpha"]), 4.0
        )

    def test_te1_encoder_layers_save_inserts_text_model(self):
        adapter = {
            "base_model.model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight": torch.zeros(
                4, 8
            ),
        }
        converted = _convert_peft_to_kohya_bf16(adapter, "te1", 24)
        self.assertIn(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            converted,
        )
        self.assertNotIn(
            "lora_te1_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            converted,
        )

    def test_locon_metadata_writes_args(self):
        cfg = TrainConfig(
            network_type="locon",
            network_dim=16,
            network_alpha=8,
            conv_dim=16,
            conv_alpha=8,
            is_vpred=False,
        )
        family = resolve_family(cfg)
        meta = build_kohya_metadata(
            cfg, 1, None, False, prediction_type=family.prediction_type(cfg)
        )
        self.assertEqual(meta["ss_network_type"], "locon")
        self.assertEqual(meta["ss_network_args"], "conv_dim=16 conv_alpha=8")
        self.assertEqual(meta["ss_network_module"], "networks.lora")

    def test_locon_metadata_writes_unequal_conv_rank(self):
        cfg = TrainConfig(
            network_type="locon",
            network_dim=16,
            network_alpha=8,
            conv_dim=8,
            conv_alpha=4,
            is_vpred=False,
        )
        family = resolve_family(cfg)
        meta = build_kohya_metadata(
            cfg, 1, None, False, prediction_type=family.prediction_type(cfg)
        )
        self.assertEqual(meta["ss_network_args"], "conv_dim=8 conv_alpha=4")
        self.assertEqual(meta["ss_network_dim"], "16")
        self.assertEqual(meta["ss_network_alpha"], "8")


class ResumeNetworkTypeTest(ResumeLoadTest):
    def test_locon_tag_refuses_standard_wrap(self):
        family, modules, _target = self._family_and_modules()
        tagged = self.dir / "locon_tagged.safetensors"
        save_file(
            {"lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": torch.zeros(4, 8)},
            str(tagged),
            metadata={"ss_network_type": "locon", "ss_network_dim": str(self.RANK)},
        )
        with self.assertRaises(ValueError) as ctx:
            family.load_lora(self._cfg(resume_lora_path=str(tagged)), modules)
        self.assertIn("locon", str(ctx.exception))

    def test_conv_dim_args_without_type_is_locon(self):
        family, modules, _target = self._family_and_modules()
        tagged = self.dir / "kohya_locon.safetensors"
        save_file(
            {"lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": torch.zeros(4, 8)},
            str(tagged),
            metadata={"ss_network_args": "conv_dim=4 conv_alpha=8", "ss_network_dim": str(self.RANK)},
        )
        with self.assertRaises(ValueError) as ctx:
            family.load_lora(self._cfg(resume_lora_path=str(tagged)), modules)
        self.assertIn("locon", str(ctx.exception))

    def test_locon_without_te_mlp_refuses(self):
        family, modules, _target = self._family_and_modules()
        tagged = self.dir / "locon_no_mlp.safetensors"
        save_file(
            {
                "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": torch.zeros(
                    4, 8
                )
            },
            str(tagged),
            metadata={
                "ss_network_type": "locon",
                "ss_network_args": "conv_dim=4 conv_alpha=8",
                "ss_network_dim": str(self.RANK),
            },
        )
        cfg = self._cfg(
            resume_lora_path=str(tagged),
            network_type="locon",
            conv_dim=4,
            conv_alpha=8,
        )
        with self.assertRaises(ValueError) as ctx:
            family.load_lora(cfg, modules)
        self.assertIn("fc1", str(ctx.exception))

    def test_locon_conv_dim_mismatch_refuses(self):
        family, modules, _target = self._family_and_modules()
        tagged = self.dir / "locon_conv.safetensors"
        save_file(
            {
                "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": torch.zeros(
                    4, 8
                )
            },
            str(tagged),
            metadata={
                "ss_network_type": "locon",
                "ss_network_args": "conv_dim=16 conv_alpha=8",
                "ss_network_dim": str(self.RANK),
            },
        )
        cfg = self._cfg(
            resume_lora_path=str(tagged),
            network_type="locon",
            conv_dim=4,
            conv_alpha=8,
        )
        with self.assertRaises(ValueError) as ctx:
            family.load_lora(cfg, modules)
        self.assertIn("conv_dim", str(ctx.exception))

    def test_missing_type_loads_as_standard(self):
        family, modules, _target = self._family_and_modules()
        info = family.load_lora(self._cfg(), modules)
        self.assertGreater(info["loaded"], 0)

    def test_te1_old_spelling_still_loads(self):
        from peft import get_peft_model_state_dict

        family, modules, target = self._family_and_modules()
        converted = _convert_peft_to_kohya_bf16(
            get_peft_model_state_dict(self.source), "te1", float(self.ALPHA)
        )
        old = {
            key.replace(
                "lora_te1_text_model_encoder_layers_", "lora_te1_encoder_layers_"
            ): tensor
            for key, tensor in converted.items()
        }
        path = self.dir / "old_te1.safetensors"
        save_file(
            old,
            str(path),
            metadata={
                "ss_network_dim": str(self.RANK),
                "ss_network_alpha": str(self.ALPHA),
            },
        )
        info = family.load_lora(self._cfg(resume_lora_path=str(path)), modules)
        self.assertGreater(info["loaded"], 0)
        expected = get_peft_model_state_dict(self.source)
        actual = get_peft_model_state_dict(target)
        for key in expected:
            self.assertTrue(
                torch.equal(actual[key], expected[key].to(torch.bfloat16).to(actual[key].dtype)),
                f"weights differ for {key}",
            )


if __name__ == "__main__":
    unittest.main()
