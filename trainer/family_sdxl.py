from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file

try:
    from checkpoints import (
        conv_dim_alpha_from_metadata,
        read_lora_metadata,
        require_resume_network_type,
        resolve_resume_path,
    )
    from family import FamilyModules, FamilySpec
    from models import (
        build_kohya_metadata,
        enable_flash_attention,
        lora_checkpoint_file,
    )
    from utils import apply_loss_mask, build_time_ids
except ImportError:
    from trainer.checkpoints import (
        conv_dim_alpha_from_metadata,
        read_lora_metadata,
        require_resume_network_type,
        resolve_resume_path,
    )
    from trainer.family import FamilyModules, FamilySpec
    from trainer.models import (
        build_kohya_metadata,
        enable_flash_attention,
        lora_checkpoint_file,
    )
    from trainer.utils import apply_loss_mask, build_time_ids

base_dir = os.getcwd()
if base_dir not in sys.path:
    sys.path.append(base_dir)

from text_processing import encode_prompt_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDXL UNet: diffusers path  →  kohya / ComfyUI ldm path.
# Block prefixes from unet_to_diffusers (SDXL). Longest prefix wins; then
# ResNet leaves (conv1 → in_layers.2, …).
# ---------------------------------------------------------------------------
_UNET_ATTN_MAP: dict[str, str] = {
    "down_blocks.1.attentions.0": "input_blocks.4.1",
    "down_blocks.1.attentions.1": "input_blocks.5.1",
    "down_blocks.2.attentions.0": "input_blocks.7.1",
    "down_blocks.2.attentions.1": "input_blocks.8.1",
    "mid_block.attentions.0": "middle_block.1",
    "up_blocks.0.attentions.0": "output_blocks.0.1",
    "up_blocks.0.attentions.1": "output_blocks.1.1",
    "up_blocks.0.attentions.2": "output_blocks.2.1",
    "up_blocks.1.attentions.0": "output_blocks.3.1",
    "up_blocks.1.attentions.1": "output_blocks.4.1",
    "up_blocks.1.attentions.2": "output_blocks.5.1",
}

_UNET_RESNET_BLOCK_MAP: dict[str, str] = {
    "down_blocks.0.resnets.0": "input_blocks.1.0",
    "down_blocks.0.resnets.1": "input_blocks.2.0",
    "down_blocks.1.resnets.0": "input_blocks.4.0",
    "down_blocks.1.resnets.1": "input_blocks.5.0",
    "down_blocks.2.resnets.0": "input_blocks.7.0",
    "down_blocks.2.resnets.1": "input_blocks.8.0",
    "mid_block.resnets.0": "middle_block.0",
    "mid_block.resnets.1": "middle_block.2",
    "up_blocks.0.resnets.0": "output_blocks.0.0",
    "up_blocks.0.resnets.1": "output_blocks.1.0",
    "up_blocks.0.resnets.2": "output_blocks.2.0",
    "up_blocks.1.resnets.0": "output_blocks.3.0",
    "up_blocks.1.resnets.1": "output_blocks.4.0",
    "up_blocks.1.resnets.2": "output_blocks.5.0",
    "up_blocks.2.resnets.0": "output_blocks.6.0",
    "up_blocks.2.resnets.1": "output_blocks.7.0",
    "up_blocks.2.resnets.2": "output_blocks.8.0",
}

_UNET_SAMPLER_MAP: dict[str, str] = {
    "down_blocks.0.downsamplers.0.conv": "input_blocks.3.0.op",
    "down_blocks.1.downsamplers.0.conv": "input_blocks.6.0.op",
    "up_blocks.0.upsamplers.0.conv": "output_blocks.2.2.conv",
    "up_blocks.1.upsamplers.0.conv": "output_blocks.5.2.conv",
    "up_blocks.2.upsamplers.0.conv": "output_blocks.8.1.conv",
}

_UNET_EMBED_MAP: dict[str, str] = {
    "class_embedding.linear_1": "label_emb.0.0",
    "class_embedding.linear_2": "label_emb.0.2",
    "add_embedding.linear_1": "label_emb.0.0",
    "add_embedding.linear_2": "label_emb.0.2",
}

_UNET_RESNET_LEAF: dict[str, str] = {
    "conv1": "in_layers.2",
    "conv2": "out_layers.3",
    "time_emb_proj": "emb_layers.1",
    "conv_shortcut": "skip_connection",
}

_UNET_BLOCK_MAP: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            **_UNET_RESNET_BLOCK_MAP,
            **_UNET_SAMPLER_MAP,
            **_UNET_EMBED_MAP,
            **_UNET_ATTN_MAP,
        }.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_TE_PATH_REPLACEMENTS: list[tuple[str, str]] = [
    ("text_model.encoder.layers.", "text_model_encoder_layers_"),
    ("text_model.embeddings.", "text_model_embeddings_"),
    ("text_projection.", "text_projection_"),
    ("text_model.final_layer_norm.", "text_model_final_layer_norm_"),
    ("final_layer_norm.", "final_layer_norm_"),
]

_TE1_NEW_PREFIX = "lora_te1_text_model_encoder_layers_"
_TE1_OLD_PREFIX = "lora_te1_encoder_layers_"

_TE_LORA_TARGETS: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")
_TE_MLP_TARGETS: tuple[str, ...] = ("fc1", "fc2")
_LOCON_TE_LORA_TARGETS: tuple[str, ...] = _TE_LORA_TARGETS + _TE_MLP_TARGETS
_STANDARD_UNET_LORA_TARGETS: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
_LOCON_UNET_LINEAR_TARGETS: tuple[str, ...] = _STANDARD_UNET_LORA_TARGETS + (
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
    "time_emb_proj",
)
_LOCON_UNET_CONV_TARGETS: tuple[str, ...] = (
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv",
)
_LOCON_UNET_LORA_TARGETS: tuple[str, ...] = (
    _LOCON_UNET_LINEAR_TARGETS + _LOCON_UNET_CONV_TARGETS
)
_CONV_ADAPTER_NAME = "conv"


def unet_lora_targets(network_type: str) -> tuple[str, ...]:
    kind = str(network_type or "standard").strip().lower()
    if kind == "standard":
        return _STANDARD_UNET_LORA_TARGETS
    if kind == "locon":
        return _LOCON_UNET_LORA_TARGETS
    raise ValueError(f"network_type must be 'standard' or 'locon', not {network_type!r}")


def te_lora_targets(network_type: str) -> tuple[str, ...]:
    kind = str(network_type or "standard").strip().lower()
    if kind == "locon":
        return _LOCON_TE_LORA_TARGETS
    if kind == "standard":
        return _TE_LORA_TARGETS
    raise ValueError(f"network_type must be 'standard' or 'locon', not {network_type!r}")


def _activate_locon_adapters(module: Any) -> None:
    """PEFT 0.20 PeftModel.set_adapter takes one name; the LoRA tuner accepts both."""
    tuner = getattr(module, "base_model", None)
    setter = getattr(tuner, "set_adapter", None)
    if callable(setter):
        setter(["default", _CONV_ADAPTER_NAME])


def _remap_unet_path(key: str) -> str:
    key = key.replace("unet.", "", 1)
    for src, dst in _UNET_BLOCK_MAP:
        if src in key:
            key = key.replace(src, dst, 1)
            break
    for src, dst in _UNET_RESNET_LEAF.items():
        token = f".{src}"
        if key.endswith(token) or f"{token}." in key:
            key = key.replace(src, dst, 1)
            break
    return key


def _remap_te_path(key: str) -> str:
    if key.startswith("encoder."):
        key = "text_model." + key
    for src, dst in _TE_PATH_REPLACEMENTS:
        key = key.replace(src, dst)
    return key


def _remap_core_path(raw_key: str, prefix: str) -> str:
    key = raw_key.replace("base_model.model.", "")
    if prefix == "unet":
        key = _remap_unet_path(key)
    elif prefix in ("te1", "te2"):
        key = _remap_te_path(key)
    return key.replace(".", "_")


def enable_te_gradient_checkpointing(module: Any) -> None:
    """Turn on TE checkpointing after PEFT wrap. Frozen embeddings need input grads."""
    candidates: list[Any] = [module]
    getter = getattr(module, "get_base_model", None)
    if callable(getter):
        try:
            candidates.append(getter())
        except Exception:
            pass
    inner = getattr(module, "base_model", None)
    if inner is not None:
        candidates.append(inner)

    seen: list[Any] = []
    for candidate in candidates:
        if candidate is None or any(candidate is item for item in seen):
            continue
        seen.append(candidate)

    for candidate in seen:
        fn = getattr(candidate, "gradient_checkpointing_enable", None)
        if not callable(fn):
            fn = getattr(candidate, "enable_gradient_checkpointing", None)
        if callable(fn):
            fn()
            break

    for candidate in seen:
        fn = getattr(candidate, "enable_input_require_grads", None)
        if callable(fn):
            fn()
            break


def _convert_peft_to_kohya_bf16(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    alpha: float,
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        key = k.replace("base_model.model.", "")
        if ".lora_A." in key:
            suffix = "lora_down.weight"
            core_raw = key.split(".lora_A.")[0]
        elif ".lora_B." in key:
            suffix = "lora_up.weight"
            core_raw = key.split(".lora_B.")[0]
        else:
            logger.debug("Skipping non-AB LoRA key: %s", k)
            continue
        core_path = _remap_core_path(core_raw, prefix)
        weight_key = f"lora_{prefix}_{core_path}.{suffix}"
        converted[weight_key] = v.detach().cpu().to(dtype=torch.bfloat16)
        alpha_key = f"lora_{prefix}_{core_path}.alpha"
        if alpha_key not in converted:
            converted[alpha_key] = torch.tensor(alpha, dtype=torch.bfloat16)
    return converted


def build_kohya_to_peft_map(
    adapter_state_dict: dict[str, torch.Tensor],
    prefix: str,
    alpha: float,
) -> dict[str, str]:
    """Invert the save-time remap: kohya key → PEFT parameter name.

    Derived from the model's own adapter tensors so it stays exactly in sync with
    `_convert_peft_to_kohya_bf16` (no fragile un-flattening of the kohya path).
    """
    mapping: dict[str, str] = {}
    for peft_key, tensor in adapter_state_dict.items():
        for kohya_key in _convert_peft_to_kohya_bf16({peft_key: tensor}, prefix, alpha):
            if kohya_key.endswith(".alpha"):
                continue
            mapping[kohya_key] = peft_key
    if prefix == "te1":
        extra: dict[str, str] = {}
        for kohya_key, peft_key in mapping.items():
            if kohya_key.startswith(_TE1_NEW_PREFIX):
                extra[_TE1_OLD_PREFIX + kohya_key[len(_TE1_NEW_PREFIX) :]] = peft_key
        mapping.update(extra)
    return mapping


def adapter_parameter_names(module: Any) -> dict[str, torch.Tensor]:
    """Adapter tensors keyed by the exact name `load_state_dict` expects.

    `get_peft_model_state_dict` strips the adapter name (`.default`) from its keys,
    so the resume path must read `named_parameters()` instead.
    """
    return {
        name: param
        for name, param in module.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _load_into_module(
    module: Any,
    updates: dict[str, torch.Tensor],
    *,
    source: Path,
) -> int:
    shapes = {name: tuple(param.shape) for name, param in module.named_parameters()}
    checked: dict[str, torch.Tensor] = {}
    for peft_key, tensor in updates.items():
        expected = shapes.get(peft_key)
        if expected is None:
            continue
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"checkpoint tensor {peft_key} has shape {tuple(tensor.shape)} but this LoRA "
                f"expects {expected}; set network_dim/network_alpha to match the checkpoint "
                f"({source})"
            )
        checked[peft_key] = tensor
    if checked:
        module.load_state_dict(checked, strict=False)
    return len(checked)


def load_sdxl_pipeline(path: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    path_obj = Path(path)
    loader_func = (
        StableDiffusionXLPipeline.from_single_file
        if path_obj.is_file()
        else StableDiffusionXLPipeline.from_pretrained
    )
    # diffusers 0.40 lazy-imports guiders → kornia.geometry, which still uses @torch.jit.script
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*torch\.jit\.script.*deprecated.*",
        )
        return loader_func(
            str(path_obj),
            dtype=dtype,
            feature_extractor=None,
        )


class SdxlFamily:
    def __init__(self, spec: FamilySpec):
        self.spec = spec
        self.trainable = True

    def load_pipeline(self, path: str, dtype: torch.dtype) -> Any:
        return load_sdxl_pipeline(path, dtype)

    def unpack(self, pipe: Any) -> FamilyModules:
        return FamilyModules(
            pipe=pipe,
            vae=pipe.vae,
            denoise=pipe.unet,
            tokenizers=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoders=[pipe.text_encoder, pipe.text_encoder_2],
        )

    def apply_lora(self, cfg: Any, modules: FamilyModules) -> FamilyModules:
        network_type = str(getattr(cfg, "network_type", "standard") or "standard").strip().lower()
        te_targets = te_lora_targets(network_type)

        def te_lora_config() -> LoraConfig:
            return LoraConfig(
                r=cfg.network_dim,
                lora_alpha=cfg.network_alpha,
                lora_dropout=cfg.network_dropout,
                init_lora_weights="gaussian",
                target_modules=list(te_targets),
            )

        unet_lora_config = LoraConfig(
            r=cfg.network_dim,
            lora_alpha=cfg.network_alpha,
            lora_dropout=cfg.network_dropout,
            init_lora_weights="gaussian",
            target_modules=list(
                _LOCON_UNET_LINEAR_TARGETS
                if network_type == "locon"
                else _STANDARD_UNET_LORA_TARGETS
            ),
        )
        tes = [get_peft_model(te, te_lora_config()) for te in modules.text_encoders]
        if bool(getattr(cfg, "gradient_checkpointing_te", True)):
            for te in tes:
                enable_te_gradient_checkpointing(te)
        denoise = get_peft_model(modules.denoise, unet_lora_config)
        if network_type == "locon":
            conv_config = LoraConfig(
                r=int(cfg.conv_dim),
                lora_alpha=int(cfg.conv_alpha),
                lora_dropout=cfg.network_dropout,
                init_lora_weights="gaussian",
                target_modules=list(_LOCON_UNET_CONV_TARGETS),
            )
            denoise.add_adapter(_CONV_ADAPTER_NAME, conv_config)
            _activate_locon_adapters(denoise)
        if bool(getattr(cfg, "gradient_checkpointing_unet", True)):
            denoise.enable_gradient_checkpointing()
        enable_flash_attention(denoise)
        modules.denoise = denoise
        modules.text_encoders = tes
        return modules

    def load_lora(self, cfg: Any, modules: FamilyModules) -> dict[str, Any]:
        """Load a kohya LoRA checkpoint into the wrapped UNet / text encoders.

        Weights only: the optimizer state, LR schedule, and step/epoch counters
        all start from zero for the new run.
        """
        raw = str(getattr(cfg, "resume_lora_path", "") or "").strip()
        if not raw:
            return {}

        source = resolve_resume_path(raw)
        metadata = read_lora_metadata(source)
        require_resume_network_type(cfg, metadata, source)
        state = load_file(str(source))
        network_type = str(getattr(cfg, "network_type", "standard") or "standard").strip().lower()
        if network_type == "locon":
            if not any("mlp_fc1" in key or "mlp_fc2" in key for key in state):
                raise ValueError(
                    f"checkpoint {source} is locon without TE MLP (fc1/fc2); "
                    "this trainer wraps CLIPMLP on locon — train a new locon file"
                )
            file_conv_dim, _file_conv_alpha = conv_dim_alpha_from_metadata(metadata)
            if file_conv_dim >= 1 and int(cfg.conv_dim) != file_conv_dim:
                raise ValueError(
                    f"checkpoint {source} conv_dim={file_conv_dim} but this config is "
                    f"conv_dim={cfg.conv_dim}"
                )

        targets: list[tuple[str, Any]] = [("unet", modules.denoise)]
        targets += [(prefix, te) for prefix, te in zip(("te1", "te2"), modules.text_encoders)]

        key_maps = {
            prefix: build_kohya_to_peft_map(
                adapter_parameter_names(module), prefix, cfg.network_alpha
            )
            for prefix, module in targets
        }

        buckets: dict[str, dict[str, torch.Tensor]] = {prefix: {} for prefix, _ in targets}
        skipped: list[str] = []
        for kohya_key, tensor in state.items():
            if kohya_key.endswith(".alpha"):
                continue
            for prefix, _module in targets:
                peft_key = key_maps[prefix].get(kohya_key)
                if peft_key is None:
                    continue
                buckets[prefix][peft_key] = tensor
                break
            else:
                skipped.append(kohya_key)

        if not any(buckets.values()):
            raise ValueError(
                f"no LoRA tensors in {source} match this SDXL LoRA layout "
                f"(network_dim={cfg.network_dim}); is it a LoRA for a different base model?"
            )

        loaded = 0
        for prefix, module in targets:
            loaded += _load_into_module(module, buckets[prefix], source=source)
        if loaded == 0:
            raise ValueError(
                f"checkpoint {source} has LoRA tensors but none map onto this model's "
                f"adapter parameters; refusing to continue with unloaded weights"
            )

        if skipped:
            preview = ", ".join(sorted(skipped)[:5])
            logger.warning(
                "Resume: skipped %d checkpoint tensors that this LoRA does not use (e.g. %s)",
                len(skipped),
                preview,
            )

        checkpoint_alpha = _parse_int(metadata.get("ss_network_alpha"))
        if checkpoint_alpha is not None and checkpoint_alpha != int(cfg.network_alpha):
            logger.warning(
                "Resume: checkpoint alpha=%s differs from network_alpha=%s; the checkpoint's "
                "alpha scalars are ignored, this run uses network_alpha.",
                checkpoint_alpha,
                cfg.network_alpha,
            )

        step = _parse_int(metadata.get("ss_steps"))
        epoch = _parse_int(metadata.get("ss_epoch"))
        # stderr: start_train.sh keeps stdout free for the IPC-less launcher filter.
        print(
            f"Resume: loaded {loaded} tensors from {source} "
            f"(skipped={len(skipped)}, checkpoint step={step} epoch={epoch}). "
            f"This run restarts step/epoch counting at 0.",
            file=sys.stderr,
        )
        return {
            "path": str(source),
            "filename": source.name,
            "step": step,
            "epoch": epoch,
            "loaded": loaded,
            "skipped": len(skipped),
        }

    def build_noise_scheduler(self, pipe: Any, cfg: Any) -> Any:
        # The base checkpoint decides what the training target is (`TrainConfig.prediction_type`
        # / `zero_terminal_snr`, resolved from its own marker tensors); training epsilons against
        # a velocity model (or the reverse) never converges.
        return DDIMScheduler.from_config(
            pipe.scheduler.config,
            prediction_type=cfg.prediction_type,
            rescale_betas_zero_snr=bool(cfg.zero_terminal_snr)
            and cfg.prediction_type == "v_prediction",
        )

    def extra_cond(
        self,
        *,
        src_wh: tuple[int, int],
        bucket_wh: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        src_w, src_h = src_wh
        bucket_w, bucket_h = bucket_wh
        return {
            "time_ids": build_time_ids(
                original_size=(src_h, src_w),
                crop_top_left=(0, 0),
                target_size=(bucket_h, bucket_w),
                device=device,
                dtype=dtype,
            )
        }

    def encode_prompts(
        self,
        prompts: list[str],
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer_1, tokenizer_2 = modules.tokenizers
        text_encoder_1, text_encoder_2 = modules.text_encoders
        prompt_embeds, pooled_prompt_embeds, _ = encode_prompt_batch(
            prompts=prompts,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            clip_skip=cfg.clip_skip,
            max_token_length=cfg.max_token_length,
            device=device,
            dtype=dtype,
        )
        return prompt_embeds, pooled_prompt_embeds

    def prediction_type(self, cfg: Any) -> str:
        return cfg.prediction_type

    def denoise_loss(
        self,
        *,
        latents: torch.Tensor,
        encoded: Any,
        extra: dict[str, torch.Tensor],
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prompt_embeds, pooled_prompt_embeds = encoded
        noise_scheduler = modules.noise_scheduler
        noise = torch.randn_like(latents)
        if cfg.noise_offset > 0:
            offset = cfg.noise_offset * torch.randn(
                latents.shape[0],
                latents.shape[1],
                1,
                1,
                device=device,
                dtype=dtype,
            )
            noise = noise + offset

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = modules.denoise(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": extra["time_ids"],
            },
            return_dict=False,
        )[0]
        if cfg.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        err = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        err = apply_loss_mask(err, extra.get("loss_mask"))
        return err.mean()

    def compute_loss(
        self,
        *,
        prompts: list[str],
        latents: torch.Tensor,
        extra: dict[str, torch.Tensor],
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        encoded = self.encode_prompts(prompts, modules, cfg, device, dtype)
        return self.denoise_loss(
            latents=latents,
            encoded=encoded,
            extra=extra,
            modules=modules,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )

    def save_lora(
        self,
        accelerator: Any,
        modules: FamilyModules,
        cfg: Any,
        global_step: int,
        epoch: Optional[int] = None,
        final: bool = False,
    ) -> None:
        if not accelerator.is_main_process:
            return

        unwrapped_unet = accelerator.unwrap_model(modules.denoise)
        tes = [accelerator.unwrap_model(te) for te in modules.text_encoders]
        unet_state = _convert_peft_to_kohya_bf16(
            get_peft_model_state_dict(unwrapped_unet, adapter_name="default"),
            "unet",
            cfg.network_alpha,
        )
        peft_config = getattr(unwrapped_unet, "peft_config", None) or {}
        if _CONV_ADAPTER_NAME in peft_config:
            unet_state.update(
                _convert_peft_to_kohya_bf16(
                    get_peft_model_state_dict(
                        unwrapped_unet, adapter_name=_CONV_ADAPTER_NAME
                    ),
                    "unet",
                    cfg.conv_alpha,
                )
            )
        te_states = [
            _convert_peft_to_kohya_bf16(
                get_peft_model_state_dict(te), prefix, cfg.network_alpha
            )
            for te, prefix in zip(tes, ("te1", "te2"))
        ]
        all_keys = list(unet_state) + [k for state in te_states for k in state]
        if len(all_keys) != len(set(all_keys)):
            logger.warning("Duplicate keys detected when merging LoRA state dicts!")

        merged: dict[str, torch.Tensor] = {**unet_state}
        for state in te_states:
            merged.update(state)

        out_file = lora_checkpoint_file(cfg, global_step, epoch=epoch, final=final)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        metadata = build_kohya_metadata(
            cfg,
            global_step,
            epoch,
            final,
            prediction_type=self.prediction_type(cfg),
        )
        save_file(merged, str(out_file), metadata=metadata)

        te_counts = "  ".join(
            f"{prefix}={len(state) // 3}"
            for prefix, state in zip(("te1", "te2"), te_states)
        )
        logger.info(
            "Saved LoRA checkpoint → %s  |  unet=%d  %s layers  dtype=bf16",
            out_file,
            len(unet_state) // 3,
            te_counts,
        )

    def generate_sample(
        self,
        *,
        accelerator: Any,
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
        global_step: int,
        output_dir_base: Any,
        swap_ctx: Any = None,
    ) -> None:
        try:
            from sampling import generate_sample_image
        except ImportError:
            from trainer.sampling import generate_sample_image

        tes = [accelerator.unwrap_model(te) for te in modules.text_encoders]
        generate_sample_image(
            accelerator=accelerator,
            pipe=modules.pipe,
            trained_unet=accelerator.unwrap_model(modules.denoise),
            trained_te1=tes[0],
            trained_te2=tes[1],
            cfg=cfg,
            device=device,
            dtype=dtype,
            global_step=global_step,
            output_dir_base=Path(output_dir_base),
            swap_ctx=swap_ctx,
        )
