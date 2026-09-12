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
from safetensors.torch import save_file

try:
    from family import FamilyModules, FamilySpec
    from models import (
        build_kohya_metadata,
        enable_flash_attention,
        lora_checkpoint_file,
    )
    from utils import build_time_ids
except ImportError:
    from trainer.family import FamilyModules, FamilySpec
    from trainer.models import (
        build_kohya_metadata,
        enable_flash_attention,
        lora_checkpoint_file,
    )
    from trainer.utils import build_time_ids

base_dir = os.getcwd()
if base_dir not in sys.path:
    sys.path.append(base_dir)

from text_processing import encode_prompt_batch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDXL UNet: diffusers path  →  kohya / ComfyUI ldm path
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

_UNET_RESNET_MAP: dict[str, str] = {
    "down_blocks.0.resnets.0.conv_shortcut": "input_blocks.1.0.skip_connection",
    "down_blocks.0.resnets.1.conv_shortcut": "input_blocks.2.0.skip_connection",
    "down_blocks.1.resnets.1.conv_shortcut": "input_blocks.5.0.skip_connection",
    "down_blocks.2.resnets.1.conv_shortcut": "input_blocks.8.0.skip_connection",
    "mid_block.resnets.0.conv_shortcut": "middle_block.0.skip_connection",
    "mid_block.resnets.1.conv_shortcut": "middle_block.2.skip_connection",
}

_UNET_SAMPLER_MAP: dict[str, str] = {
    "down_blocks.2.downsamplers.0.conv": "input_blocks.9.0.op",
    "up_blocks.2.upsamplers.0.conv": "output_blocks.8.1.conv",
}

_UNET_EMBED_MAP: dict[str, str] = {
    "class_embedding.linear_1": "label_emb.0.0",
    "class_embedding.linear_2": "label_emb.0.2",
    "add_embedding.linear_1": "label_emb.0.0",
    "add_embedding.linear_2": "label_emb.0.2",
}

_UNET_PATH_MAP: dict[str, str] = {
    **_UNET_RESNET_MAP,
    **_UNET_SAMPLER_MAP,
    **_UNET_EMBED_MAP,
    **_UNET_ATTN_MAP,
}

_TE_PATH_REPLACEMENTS: list[tuple[str, str]] = [
    ("text_model.encoder.layers.", "text_model_encoder_layers_"),
    ("text_model.embeddings.", "text_model_embeddings_"),
    ("text_projection.", "text_projection_"),
    ("text_model.final_layer_norm.", "text_model_final_layer_norm_"),
    ("final_layer_norm.", "final_layer_norm_"),
]


def _remap_unet_path(key: str) -> str:
    key = key.replace("unet.", "", 1)
    for src, dst in _UNET_PATH_MAP.items():
        if src in key:
            key = key.replace(src, dst, 1)
            break
    return key


def _remap_te_path(key: str) -> str:
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
            torch_dtype=dtype,
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
        def te_lora_config() -> LoraConfig:
            return LoraConfig(
                r=cfg.network_dim,
                lora_alpha=cfg.network_alpha,
                lora_dropout=cfg.network_dropout,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )

        unet_lora_config = LoraConfig(
            r=cfg.network_dim,
            lora_alpha=cfg.network_alpha,
            lora_dropout=cfg.network_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        tes = [get_peft_model(te, te_lora_config()) for te in modules.text_encoders]
        for te in tes:
            enable_te_gradient_checkpointing(te)
        denoise = get_peft_model(modules.denoise, unet_lora_config)
        denoise.enable_gradient_checkpointing()
        enable_flash_attention(denoise)
        modules.denoise = denoise
        modules.text_encoders = tes
        return modules

    def build_noise_scheduler(self, pipe: Any, cfg: Any) -> Any:
        scheduler_kwargs: dict[str, Any] = {}
        if cfg.is_vpred:
            scheduler_kwargs["prediction_type"] = "v_prediction"
            scheduler_kwargs["rescale_betas_zero_snr"] = True
        else:
            scheduler_kwargs["prediction_type"] = "epsilon"
        return DDIMScheduler.from_config(pipe.scheduler.config, **scheduler_kwargs)

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
        return "v_prediction" if cfg.is_vpred else "epsilon"

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
        if cfg.is_vpred:
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

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
            get_peft_model_state_dict(unwrapped_unet), "unet", cfg.network_alpha
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
