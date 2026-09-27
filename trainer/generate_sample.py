"""Generate one extra sample image from a LoRA checkpoint, outside any training run.

    python -u trainer/generate_sample.py --spec <job.json>

api.py writes the spec (a job record, see `trainer/genjob.py`), spawns this
script detached, and Ranko follows the job file while it runs. The image is
written next to its spec under `{name}_samples/generated/`, so it sits beside the
run's own samples without entering their `_<step>_<repeat>.png` namespace.

The settings come from the checkpoint's own kohya metadata where possible
(network_type, network_dim/alpha, conv_dim/alpha, clip_skip, max_token_length,
base model), so a sample of an old checkpoint is reproduced with the settings
it was trained with rather than with whatever config.toml says today.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# `python trainer/generate_sample.py` puts trainer/ on sys.path, `import api` does
# not; support both (see AGENT.md "Import dualism").
try:
    import genjob
    from checkpoints import (
        conv_dim_alpha_from_metadata,
        infer_network_type,
        read_lora_metadata,
        resolve_resume_path,
    )
    from config import TrainConfig
    from env import flush_memory, setup_migraphx_cache
    from family import require_trainable, resolve_family
    from models import enable_flash_attention
except ImportError:
    from trainer import genjob
    from trainer.checkpoints import (
        conv_dim_alpha_from_metadata,
        infer_network_type,
        read_lora_metadata,
        resolve_resume_path,
    )
    from trainer.config import TrainConfig
    from trainer.env import flush_memory, setup_migraphx_cache
    from trainer.family import require_trainable, resolve_family
    from trainer.models import enable_flash_attention

from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from text_processing import encode_prompt_batch


def _log(message: str) -> None:
    print(f"[generate_sample] {message}", flush=True)


def _meta_int(metadata: dict[str, str], key: str, fallback: int) -> int:
    raw = str(metadata.get(key) or "").strip()
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return fallback


def _build_config(metadata: dict[str, str], checkpoint: Path) -> TrainConfig:
    """config.toml, corrected with the settings the checkpoint was trained with."""
    cfg = TrainConfig()

    version = str(metadata.get("ss_base_model_version") or "").strip()
    if version and version != cfg.base_model_version:
        raise RuntimeError(
            f"checkpoint {checkpoint} was trained on {version!r} but config.toml targets "
            f"{cfg.base_model_version!r}; point [model_spec] at {version!r} to sample from it"
        )

    network_dim = _meta_int(metadata, "ss_network_dim", cfg.network_dim)
    network_alpha = _meta_int(metadata, "ss_network_alpha", cfg.network_alpha)
    network_type = infer_network_type(metadata)
    conv_dim, conv_alpha = conv_dim_alpha_from_metadata(metadata)
    if network_type == "locon":
        if conv_dim < 1:
            conv_dim = network_dim
        if conv_alpha < 1:
            conv_alpha = network_alpha
    else:
        conv_dim, conv_alpha = 0, 0

    cfg = replace(
        cfg,
        resume_lora_path=str(checkpoint),
        network_type=network_type,
        network_dim=network_dim,
        network_alpha=network_alpha,
        conv_dim=conv_dim,
        conv_alpha=conv_alpha,
        clip_skip=_meta_int(metadata, "ss_clip_skip", cfg.clip_skip),
        max_token_length=_meta_int(metadata, "ss_max_token_length", cfg.max_token_length),
        # Inference only: checkpointing and its input-require-grads hooks are pure overhead here.
        gradient_checkpointing_unet=False,
        gradient_checkpointing_te=False,
    )

    trained_base = str(metadata.get("ss_pretrained_model_name_or_path") or "").strip()
    if trained_base and trained_base != cfg.pretrained_model_name_or_path:
        if Path(trained_base).exists():
            _log(f"base model from the checkpoint metadata: {trained_base}")
            cfg = replace(cfg, pretrained_model_name_or_path=trained_base)
        else:
            _log(
                f"checkpoint base model {trained_base} is gone; using config.toml's "
                f"{cfg.pretrained_model_name_or_path}"
            )
    return cfg


def _prepare_scheduler(pipe, steps: int, device: torch.device) -> None:
    """Same scheduler and sigma spacing as trainer/sampling.py, so images stay comparable."""
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
    )
    sigmas = np.linspace(pipe.scheduler.config.num_train_timesteps - 1, 0, steps)
    sigmas = np.append(sigmas, 0.0).astype(np.float32)
    pipe.scheduler.sigmas = torch.from_numpy(sigmas).to(device)
    pipe.scheduler.num_inference_steps = steps


def _decode(pipe, latents: torch.Tensor, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    if hasattr(pipe.vae.config, "force_upcast"):
        pipe.vae.config.force_upcast = False
    pipe.vae.to(device=device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    decoded = pipe.vae.decode(latents.to(device=device, dtype=dtype), return_dict=False)[0]
    image = (decoded / 2 + 0.5).clamp(0, 1)
    image = image[0].permute(1, 2, 0).detach().float().cpu().numpy()
    return (image * 255).round().astype("uint8")


@torch.no_grad()
def run_generation(spec: dict, generated: Path) -> None:
    job_id = str(spec["id"])
    steps = int(spec["steps"])
    prompt = str(spec["prompt"])
    negative_prompt = str(spec.get("negative_prompt") or "")
    guidance_scale = float(spec["cfg"])
    requested_seed = int(spec["seed"])
    width, height = int(spec["width"]), int(spec["height"])

    genjob.update_job(generated, job_id, state=genjob.STATE_RUNNING, pid=os.getpid(), current_step=0)

    checkpoint = resolve_resume_path(spec["checkpoint"])
    metadata = read_lora_metadata(checkpoint)
    cfg = _build_config(metadata, checkpoint)
    family = resolve_family(cfg)
    require_trainable(family)

    dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(
        f"checkpoint={checkpoint} type={cfg.network_type} dim={cfg.network_dim} "
        f"alpha={cfg.network_alpha} "
        f"steps={steps} cfg={guidance_scale} seed={requested_seed} {width}x{height} on {device}"
    )

    setup_migraphx_cache()
    pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, dtype)
    modules = family.unpack(pipe)
    modules = family.apply_lora(cfg, modules)
    family.load_lora(cfg, modules)

    trained_unet = modules.denoise.eval()
    te1, te2 = (te.eval() for te in modules.text_encoders)
    enable_flash_attention(trained_unet)

    pipe.unet = trained_unet
    pipe.text_encoder = te1
    pipe.text_encoder_2 = te2

    _prepare_scheduler(pipe, steps, device)

    unet_was_gpu = False
    try:
        for module in (trained_unet, te1, te2):
            module.to(device=device)
        unet_was_gpu = True

        prompt_embeds, pooled_prompt_embeds, num_chunks = encode_prompt_batch(
            prompts=[prompt],
            tokenizer_1=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            text_encoder_1=te1,
            text_encoder_2=te2,
            clip_skip=cfg.clip_skip,
            max_token_length=cfg.max_token_length,
            device=device,
            dtype=dtype,
        )
        negative_embeds, negative_pooled, _ = encode_prompt_batch(
            prompts=[negative_prompt],
            tokenizer_1=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            text_encoder_1=te1,
            text_encoder_2=te2,
            clip_skip=cfg.clip_skip,
            max_token_length=cfg.max_token_length,
            device=device,
            dtype=dtype,
            target_num_chunks=num_chunks,
        )

        for module in (te1, te2):
            module.to("cpu")
        flush_memory(device)

        generator = torch.Generator(device="cpu")
        seed = requested_seed
        if seed == 0:
            seed = int(torch.randint(0, 2**32, (1,)).item())
            _log(f"random seed: {seed}")
        generator.manual_seed(seed)

        def _on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
            genjob.update_job(generated, job_id, current_step=int(step_index) + 1)
            return callback_kwargs

        latent_result = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
            callback_on_step_end=_on_step_end,
        )

        trained_unet.to("cpu")
        flush_memory(device)

        latents = latent_result.images / pipe.vae.config.scaling_factor
        image = _decode(pipe, latents, device, torch.bfloat16)

        target = genjob.image_path(generated, job_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(target)
    finally:
        if unet_was_gpu:
            for module in (trained_unet, te1, te2):
                module.to("cpu")
        flush_memory(device)

    genjob.update_job(
        generated,
        job_id,
        state=genjob.STATE_DONE,
        seed=seed,
        current_step=steps,
        image_path=str(target),
        error=None,
    )
    _log(f"saved {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one sample image from a LoRA checkpoint")
    parser.add_argument("--spec", required=True, help="job JSON written by api.py")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    generated = spec_path.parent
    spec = genjob.read_job(spec_path)
    if spec is None:
        print(f"[generate_sample] unreadable spec: {spec_path}", file=sys.stderr)
        return 2

    job_id = str(spec.get("id") or spec_path.stem)
    try:
        run_generation(spec, generated)
    except Exception as exc:  # noqa: BLE001 - the panel shows the message, the log the traceback
        traceback.print_exc()
        genjob.update_job(
            generated,
            job_id,
            state=genjob.STATE_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
