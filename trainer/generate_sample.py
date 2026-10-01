"""Generate one extra sample image from a LoRA checkpoint, outside any training run, or evaluate one.

    python -u trainer/generate_sample.py --spec <job.json>

api.py writes the spec (a job record, see `trainer/genjob.py`), spawns this
script detached, and Ranko follows the job file while it runs. A generated image
is written next to its spec under `{name}_samples/generated/`, so it sits beside
the run's own samples without entering their `_<step>_<repeat>.png` namespace.

The settings come from the checkpoint's own kohya metadata where possible
(network_type, network_dim/alpha, conv_dim/alpha, clip_skip, max_token_length,
base model), so a sample of an old checkpoint is reproduced with the settings
it was trained with rather than with whatever config.toml says today. An
evaluation (`mode: evaluate`, see `run_evaluation`) adds one more source: the
config the run saved beside its logs, which is what decides the prompts and the
sampling values its images are held to.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

# `python trainer/generate_sample.py` puts trainer/ on sys.path, `import api` does
# not; support both (see AGENT.md "Import dualism").
try:
    import evaluation
    import genjob
    from checkpoints import (
        conv_dim_alpha_from_metadata,
        infer_network_type,
        read_lora_metadata,
        resolve_resume_path,
    )
    from config import SampleSet, TrainConfig, resolve_sample_sets, run_config_mapping
    from env import flush_memory, setup_migraphx_cache
    from family import require_trainable, resolve_family
    from models import enable_flash_attention, sample_scheduler_kwargs
except ImportError:
    from trainer import evaluation, genjob
    from trainer.checkpoints import (
        conv_dim_alpha_from_metadata,
        infer_network_type,
        read_lora_metadata,
        resolve_resume_path,
    )
    from trainer.config import SampleSet, TrainConfig, resolve_sample_sets, run_config_mapping
    from trainer.env import flush_memory, setup_migraphx_cache
    from trainer.family import require_trainable, resolve_family
    from trainer.models import enable_flash_attention, sample_scheduler_kwargs

from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from text_processing import encode_prompt_batch


def _log(message: str) -> None:
    print(f"[generate_sample] {message}", flush=True)


class _Cancelled(Exception):
    """`cancel_generation` asked this job to stop; raised at the next check point."""


# Set from the SIGTERM handler, read between images and inside the denoise callback, so a cancel
# lands within one step instead of at the end of the job.
_cancel = {"asked": False}


def _cancel_asked() -> bool:
    return bool(_cancel["asked"])


def _install_signal_handler() -> None:
    def _handler(signum, _frame):
        if _cancel["asked"]:
            _log("cancel asked again; leaving now")
            os._exit(1)
        _cancel["asked"] = True
        _log("cancel asked; stopping at the next step")

    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, name, None)
        if signum is not None:
            try:
                signal.signal(signum, _handler)
            except (ValueError, OSError):
                pass


def _meta_int(metadata: dict[str, str], key: str, fallback: int) -> int:
    raw = str(metadata.get(key) or "").strip()
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return fallback


def _build_config(
    metadata: dict[str, str],
    checkpoint: Path,
    base_cfg: Optional[TrainConfig] = None,
) -> TrainConfig:
    """`base_cfg` (config.toml, or a run's own saved config) corrected with the settings the
    checkpoint was trained with."""
    cfg = base_cfg if base_cfg is not None else TrainConfig()

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


def _prepare_scheduler(pipe, steps: int, device: torch.device, scheduler_kwargs: dict) -> None:
    """Same sampler (Euler a, linspace = ComfyUI's normal) and prediction type as
    trainer/sampling.py, so images stay comparable."""
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="linspace",
        **scheduler_kwargs,
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
    # Single mode takes the rescale from config.toml: the job spec carries the prompt-shaped
    # settings, while the sets and batch modes get their per-set value from `resolve_sample_sets`.
    guidance_rescale = float(cfg.guidance_rescale)
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

    _prepare_scheduler(pipe, steps, device, sample_scheduler_kwargs(cfg, pipe.scheduler.config))

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
            guidance_rescale=guidance_rescale,
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


def _seed_for(sample_set, repeat_idx: int) -> int:
    """sampling.py's rule: 0 draws a fresh random seed per image, else `seed + repeat_idx`."""
    if sample_set.seed == 0:
        seed = int(torch.randint(0, 2**32, (1,)).item())
        _log(f"random seed for {sample_set.name}.{repeat_idx}: {seed}")
        return seed
    return sample_set.seed + repeat_idx


def _load_family(cfg, dtype: torch.dtype):
    """The base pipeline with `cfg`'s LoRA wrap applied, weights loaded and ready to render."""
    family = resolve_family(cfg)
    require_trainable(family)
    setup_migraphx_cache()
    pipe = family.load_pipeline(cfg.pretrained_model_name_or_path, dtype)
    modules = family.unpack(pipe)
    modules = family.apply_lora(cfg, modules)
    family.load_lora(cfg, modules)

    modules.denoise.eval()
    for text_encoder in modules.text_encoders:
        text_encoder.eval()
    enable_flash_attention(modules.denoise)
    pipe.unet = modules.denoise
    pipe.text_encoder = modules.text_encoders[0]
    pipe.text_encoder_2 = modules.text_encoders[1]
    return pipe, modules


def _plan_slots(
    sets: Sequence[SampleSet],
    plan: Optional[Sequence[tuple[int, int]]] = None,
) -> dict[int, list[int]]:
    """Which repeat indices each set gets: the whole pass when `plan` is None, else exactly the
    slots it lists. Grouped per set with the repeats ascending, and a set with nothing to draw is
    absent, so an evaluation's top-up encodes only the prompts it is actually going to use."""
    slots: dict[int, list[int]] = {}
    for set_index, repeat_idx in plan if plan is not None else (
        (index, repeat) for index, entry in enumerate(sets) for repeat in range(entry.repeat)
    ):
        slots.setdefault(int(set_index), []).append(int(repeat_idx))
    return {set_index: sorted(repeats) for set_index, repeats in slots.items()}


def _shape_key(cfg) -> tuple:
    """What the LoRA wrap is built from, so a batch knows when it needs a new pipeline.

    The weights themselves are not part of it: a checkpoint that only differs in its trained
    values loads into the same wrap (that is what `load_lora` does). A different base model, LoRA
    kind or rank/alpha does need a rebuild.
    """
    return (
        str(cfg.pretrained_model_name_or_path),
        str(cfg.network_type),
        int(cfg.network_dim),
        int(cfg.network_alpha),
        int(getattr(cfg, "conv_dim", 0) or 0),
        int(getattr(cfg, "conv_alpha", 0) or 0),
    )


@torch.no_grad()
def _render_sets(
    *,
    pipe,
    modules,
    cfg,
    sets,
    generated: Path,
    job_id: str,
    device: torch.device,
    dtype: torch.dtype,
    plan: Optional[Sequence[tuple[int, int]]] = None,
) -> list[tuple[int, int, str]]:
    """Render the slots of `sets` for the checkpoint already loaded in `pipe` / `modules`.

    One image per `(set, repeat)`, named `{job_id}_p{set}_{repeat}.png` in `generated/`, with the
    job record's progress updated as it goes. `plan` lists exactly which slots to render — the whole
    sets pass when it is None, and an evaluation's missing positions otherwise, so a set with
    nothing to draw costs no encode. Returns `(set_index, repeat_idx, path)` in render order.
    """
    slots = _plan_slots(sets, plan)
    total_images = sum(len(repeats) for repeats in slots.values())

    trained_unet = modules.denoise
    te1, te2 = modules.text_encoders[0], modules.text_encoders[1]
    rendered: list[tuple[int, int, str]] = []
    scheduler_kwargs = sample_scheduler_kwargs(cfg, pipe.scheduler.config)

    for set_index, sample_set in enumerate(sets):
        repeats = slots.get(set_index, [])
        if not repeats:
            continue
        if _cancel_asked():
            raise _Cancelled()
        _prepare_scheduler(pipe, sample_set.steps, device, scheduler_kwargs)
        # Each set encodes on its own, so one set's prompt length never pads another's.
        for module in (te1, te2):
            module.to(device=device)
        prompt_embeds, pooled_prompt_embeds, num_chunks = encode_prompt_batch(
            prompts=[sample_set.prompt],
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
            prompts=[sample_set.negative],
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
        _log(
            f"set {set_index + 1}/{len(sets)} {sample_set.name}: {len(repeats)} image(s), "
            f"{sample_set.width}x{sample_set.height}, {sample_set.steps} steps, "
            f"cfg {sample_set.guidance_scale}, seed {sample_set.seed}"
        )

        for repeat_idx in repeats:
            generator = torch.Generator(device="cpu")
            seed = _seed_for(sample_set, repeat_idx)
            generator.manual_seed(seed)

            def _on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
                if _cancel_asked():
                    raise _Cancelled()
                genjob.update_job(
                    generated,
                    job_id,
                    current_step=int(step_index) + 1,
                    total_steps=sample_set.steps,
                    images_done=len(rendered),
                    total_images=total_images,
                    current_set=set_index + 1,
                    total_sets=len(sets),
                )
                return callback_kwargs

            trained_unet.to(device=device)
            flush_memory(device)
            latent_result = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                width=sample_set.width,
                height=sample_set.height,
                num_inference_steps=sample_set.steps,
                guidance_scale=sample_set.guidance_scale,
                guidance_rescale=sample_set.guidance_rescale,
                generator=generator,
                output_type="latent",
                callback_on_step_end=_on_step_end,
            )
            trained_unet.to("cpu")
            flush_memory(device)

            latents = latent_result.images / pipe.vae.config.scaling_factor
            # bf16 for the decode, as the single-image path has always done whatever the train dtype.
            image = _decode(pipe, latents, device, torch.bfloat16)
            pipe.vae.to("cpu")

            target = genjob.set_image_path(generated, job_id, set_index, repeat_idx)
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(target)
            rendered.append((set_index, repeat_idx, str(target)))
            genjob.update_job(
                generated,
                job_id,
                files=[path for _set, _repeat, path in rendered],
                images_done=len(rendered),
                total_images=total_images,
                current_step=0,
                total_steps=sample_set.steps,
                current_set=set_index + 1,
                total_sets=len(sets),
                seed=seed,
            )
            _log(f"saved {target} ({len(rendered)}/{total_images})")
    return rendered


@torch.no_grad()
def run_sample_sets(spec: dict, generated: Path) -> None:
    """Render every `[[validation.samples]]` set for this checkpoint.

    The prompt sets, sizes, steps, CFG, seeds and repeats come from `config.toml` — the same
    logic the trainer's own sample points use — while the model side (network type / dim /
    alpha, `clip_skip`, `max_token_length`, base model) comes from the checkpoint's metadata,
    through `_build_config`. Images go to `{name}_samples/generated/`, named
    `{job_id}_p{set}_{repeat}.png`; the run's own samples are never touched.
    """
    job_id = str(spec["id"])
    checkpoint = resolve_resume_path(spec["checkpoint"])
    metadata = read_lora_metadata(checkpoint)
    cfg = _build_config(metadata, checkpoint)
    sets = resolve_sample_sets(cfg)
    if not sets:
        raise RuntimeError("config.toml has no [[validation.samples]] sets to render")
    total_images = sum(sample_set.repeat for sample_set in sets)

    dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(
        f"checkpoint={checkpoint} type={cfg.network_type} dim={cfg.network_dim} "
        f"alpha={cfg.network_alpha} sets={len(sets)} images={total_images} on {device}"
    )

    genjob.update_job(
        generated,
        job_id,
        state=genjob.STATE_RUNNING,
        pid=os.getpid(),
        current_step=0,
        images_done=0,
        total_images=total_images,
        current_set=0,
        total_sets=len(sets),
    )

    pipe, modules = _load_family(cfg, dtype)
    try:
        rendered = _render_sets(
            pipe=pipe,
            modules=modules,
            cfg=cfg,
            sets=sets,
            generated=generated,
            job_id=job_id,
            device=device,
            dtype=dtype,
        )
    finally:
        for module in (modules.denoise, *modules.text_encoders):
            module.to("cpu")
        flush_memory(device)

    files = [path for _set, _repeat, path in rendered]
    genjob.update_job(
        generated,
        job_id,
        state=genjob.STATE_DONE,
        files=files,
        images_done=len(files),
        total_images=total_images,
        current_step=0,
        error=None,
    )
    _log(f"finished {len(files)} image(s) for {checkpoint}")


@torch.no_grad()
def run_sample_batch(spec: dict, generated: Path) -> None:
    """Run a `sets` pass for every checkpoint of one step range, oldest step first.

    The batch record (`genjob.new_batch_job`) is the plan: this process walks it and gives each
    checkpoint its own `sets` job, so its images are named, shown under its card and followed
    exactly as a manual pass from that checkpoint would be. The pipeline is built once for the
    whole run and rebuilt only when a checkpoint's LoRA shape (base model, kind, rank) differs, so
    a range of one run's checkpoints pays the model load once. A checkpoint that fails is recorded
    on its own job and in the batch's `failed` list, and the batch carries on with the rest.
    """
    batch_id = str(spec["id"])
    entries = list(spec.get("checkpoints") or [])
    if not entries:
        raise RuntimeError("this batch has no checkpoints to render")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"batch {batch_id}: {len(entries)} checkpoint(s) on {device}")

    loaded_key: tuple | None = None
    pipe = None
    modules = None
    images_done = 0
    checkpoints_done = 0
    failed: list[dict] = []
    job_ids: list[str] = []

    genjob.update_job(
        generated,
        batch_id,
        state=genjob.STATE_RUNNING,
        pid=os.getpid(),
        current_checkpoint=None,
        checkpoint_index=0,
        images_done=0,
        failed=[],
        job_ids=[],
    )

    cancelled = False
    stopped_at = 0
    try:
        for index, entry in enumerate(entries, start=1):
            stopped_at = index - 1
            if _cancel_asked():
                cancelled = True
                break
            job_id = ""
            checkpoint = None
            label = Path(str(entry.get("path") or "")).name or "checkpoint"
            genjob.update_job(
                generated,
                batch_id,
                current_checkpoint=str(entry.get("path") or ""),
                checkpoint_index=index,
            )
            try:
                # Inside the try: an entry that cannot even be resolved is this checkpoint's
                # failure, and the rest of the range still gets its turn.
                checkpoint = resolve_resume_path(entry["path"])
                label = checkpoint.name
                genjob.update_job(
                    generated,
                    batch_id,
                    current_checkpoint=str(checkpoint),
                    checkpoint_index=index,
                )
                metadata = read_lora_metadata(checkpoint)
                cfg = _build_config(metadata, checkpoint)
                sets = resolve_sample_sets(cfg)
                if not sets:
                    raise RuntimeError("config.toml has no [[validation.samples]] sets to render")

                job = genjob.new_job(
                    {"step": entry.get("step")},
                    run_id=str(spec.get("run_id") or ""),
                    output_name=str(spec.get("output_name") or ""),
                    checkpoint=str(checkpoint),
                    mode=genjob.MODE_SETS,
                    total_images=sum(sample_set.repeat for sample_set in sets),
                    extra={"batch_id": batch_id, "batch_index": index, "batch_total": len(entries)},
                )
                job_id = str(job["id"])
                # Written before the render, so this checkpoint's card shows it is being done.
                genjob.write_job(generated, job)
                job_ids.append(job_id)
                genjob.update_job(generated, batch_id, job_ids=job_ids)
                genjob.update_job(
                    generated,
                    job_id,
                    state=genjob.STATE_RUNNING,
                    pid=os.getpid(),
                    current_step=0,
                    images_done=0,
                    total_images=sum(sample_set.repeat for sample_set in sets),
                    current_set=0,
                    total_sets=len(sets),
                )

                dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
                shape = (_shape_key(cfg), dtype)
                if shape != loaded_key:
                    _log(f"[{index}/{len(entries)}] loading {cfg.pretrained_model_name_or_path} ({cfg.network_type})")
                    pipe, modules = _load_family(cfg, dtype)
                    loaded_key = shape
                else:
                    _log(f"[{index}/{len(entries)}] loading the LoRA weights of {label}")
                    resolve_family(cfg).load_lora(cfg, modules)

                rendered = _render_sets(
                    pipe=pipe,
                    modules=modules,
                    cfg=cfg,
                    sets=sets,
                    generated=generated,
                    job_id=job_id,
                    device=device,
                    dtype=dtype,
                )
                files = [path for _set, _repeat, path in rendered]
                images_done += len(files)
                checkpoints_done += 1
                genjob.update_job(
                    generated,
                    job_id,
                    state=genjob.STATE_DONE,
                    files=files,
                    images_done=len(files),
                    total_images=len(files),
                    current_step=0,
                    error=None,
                )
                _log(f"[{index}/{len(entries)}] finished {len(files)} image(s) for {label}")
            except _Cancelled:
                # The checkpoint's own job keeps the images it managed to write; a cancel is not a
                # failure, so it is neither an error nor a `failed` entry.
                genjob.update_job(
                    generated,
                    job_id,
                    state=genjob.STATE_CANCELLED,
                    cancel_requested=True,
                    error=None,
                )
                _log(f"[{index}/{len(entries)}] cancelled during {label}")
                cancelled = True
                stopped_at = index
                break
            except Exception as exc:  # noqa: BLE001 - one bad checkpoint must not stop the range
                traceback.print_exc()
                message = f"{type(exc).__name__}: {exc}"
                failed.append({"checkpoint": str(entry.get("path") or ""), "error": message})
                if job_id:
                    genjob.update_job(generated, job_id, state=genjob.STATE_ERROR, error=message)
                _log(f"[{index}/{len(entries)}] {label}: {message}")
            genjob.update_job(
                generated,
                batch_id,
                images_done=images_done,
                failed=failed,
                job_ids=job_ids,
                checkpoint_index=index,
            )
    finally:
        if modules is not None:
            for module in (modules.denoise, *modules.text_encoders):
                module.to("cpu")
            flush_memory(device)

    if cancelled:
        state = genjob.STATE_CANCELLED
        error = None
    elif checkpoints_done:
        state = genjob.STATE_DONE
        error = None
    else:
        state = genjob.STATE_ERROR
        error = f"none of the {len(entries)} checkpoint(s) of the range could be rendered"
    genjob.update_job(
        generated,
        batch_id,
        state=state,
        cancel_requested=cancelled,
        images_done=images_done,
        failed=failed,
        job_ids=job_ids,
        current_checkpoint=None,
        checkpoint_index=stopped_at if cancelled else len(entries),
        error=error,
    )
    _log(
        f"batch {batch_id}: {checkpoints_done}/{len(entries)} checkpoint(s), {images_done} image(s)"
        + (" (cancelled)" if cancelled else "")
    )

def _record_config(spec: dict) -> TrainConfig:
    """The config an evaluation is held to, resolved the way api.py resolved it when it planned the
    pass: the run's own `config.toml` copy, the hparams it recorded at startup when it predates
    those copies, else today's `config.toml`. `config_source` in the spec says which one that was;
    `config_log_dir` is what the resolution runs on again here, so one function decides."""
    log_dir = str(spec.get("config_log_dir") or "")
    if not log_dir:
        _log("no run log directory on this spec; using config.toml")
        return TrainConfig()
    mapping, source = run_config_mapping(log_dir)
    _log(f"config from {source}")
    return TrainConfig.from_mapping(mapping)


def _record_sets(spec: dict) -> list[SampleSet]:
    """The prompt sets the plan was built from. Empty when the record carries none (a hand-written
    spec), which sends the caller back to `resolve_sample_sets`."""
    raw = spec.get("sample_sets")
    if not isinstance(raw, list):
        return []
    sets: list[SampleSet] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            sets.append(SampleSet(**entry))
        except TypeError:
            continue
    return sets


@torch.no_grad()
def run_evaluation(spec: dict, generated: Path) -> None:
    """Evaluate one checkpoint: top its sample count up to the spec's depth, tag every image, score.

    The record api.py wrote is the work order. `plan` lists the `(set, repeat)` slots that are
    missing (none at all when the checkpoint already holds `depth` images), `sample_sets` is what to
    render them with, and `images` is the set to tag and score — the run's own samples at the
    checkpoint's step included, each already carrying the prompt it was rendered from. Rendering
    happens first and only when there is something to draw, so a re-evaluation of a checkpoint that
    already has enough images never loads the diffusion model; the tagger is the only model it
    touches. Both scoreboards go back into the same record (`evaluation.score_images`).
    """
    job_id = str(spec["id"])
    checkpoint = resolve_resume_path(spec["checkpoint"])
    metadata = read_lora_metadata(checkpoint)
    cfg = _build_config(metadata, checkpoint, base_cfg=_record_config(spec))
    plan = dict(spec.get("plan") or {})
    slots = evaluation.render_slots(plan)
    images = evaluation.images_from_payload(spec.get("images"))
    raw_threshold = spec.get("threshold")
    threshold = float(raw_threshold) if raw_threshold is not None else 0.35
    categories = [str(item) for item in spec.get("categories") or ["general"]]

    genjob.update_job(
        generated,
        job_id,
        state=genjob.STATE_RUNNING,
        pid=os.getpid(),
        error=None,
        cancel_requested=False,
    )
    _log(
        f"checkpoint={checkpoint} depth={spec.get('depth')} images={len(images)} "
        f"to render={len(slots)} threshold={threshold:g} categories={','.join(categories)}"
    )

    if slots:
        sets = _record_sets(spec) or resolve_sample_sets(cfg)
        dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe, modules = _load_family(cfg, dtype)
        try:
            rendered = _render_sets(
                pipe=pipe,
                modules=modules,
                cfg=cfg,
                sets=sets,
                generated=generated,
                job_id=job_id,
                device=device,
                dtype=dtype,
                plan=slots,
            )
        finally:
            for module in (modules.denoise, *modules.text_encoders):
                module.to("cpu")
            flush_memory(device)

        for set_index, repeat_idx, path in rendered:
            images.append(
                evaluation.ImageRef(
                    path=str(path),
                    set_index=int(set_index),
                    repeat_idx=int(repeat_idx),
                    source=evaluation.SOURCE_GENERATED,
                    prompt=evaluation.sample_set_prompt(sets[set_index]) if set_index < len(sets) else "",
                )
            )

    total_images = len(images)
    tagged: list[evaluation.ImageRef] = []
    # Matched on the normalized path: the tagger answers with `str(Path(path))`, which can differ
    # from the recorded string in a slash or a `.` while naming the same file.
    by_path = {os.path.normpath(image.path): image for image in images}

    def _on_entry(entry: dict[str, Any]) -> None:
        """Persist one tagged image and stop the pass when a cancel has arrived."""
        image = by_path.get(os.path.normpath(str(entry.get("path") or "")))
        if image is not None:
            image.tags = [str(tag) for tag in entry.get("tags") or []]
            image.error = None if entry.get("error") is None else str(entry["error"])
            tagged.append(image)
            genjob.update_job(
                generated,
                job_id,
                phase=genjob.PHASE_TAGGING,
                images=[item.to_dict() for item in tagged],
                images_done=len(tagged),
                total_images=total_images,
            )
        if _cancel_asked():
            raise _Cancelled()

    genjob.update_job(
        generated,
        job_id,
        phase=genjob.PHASE_TAGGING,
        images=[image.to_dict() for image in images],
        images_done=0,
        total_images=total_images,
    )

    import tagger2.main as tagger

    tagger.tag_paths(
        [image.path for image in images],
        threshold=threshold,
        categories=categories,
        on_entry=_on_entry,
    )

    genjob.update_job(generated, job_id, phase=genjob.PHASE_SCORING)
    scores = evaluation.score_images(images)
    genjob.update_job(
        generated,
        job_id,
        state=genjob.STATE_DONE,
        phase=genjob.PHASE_DONE,
        scores=scores,
        images=[image.to_dict() for image in images],
        images_done=total_images,
        total_images=total_images,
        error=None,
    )
    _log(f"scored {scores['images_scored']} image(s): F1 {scores['f1']} (micro), {scores['union_f1']} (union)")


def main() -> int:
    _install_signal_handler()
    parser = argparse.ArgumentParser(description="Generate sample images from a LoRA checkpoint")
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
        mode = str(spec.get("mode") or genjob.MODE_SINGLE)
        if mode == genjob.MODE_BATCH:
            run_sample_batch(spec, generated)
        elif mode == genjob.MODE_SETS:
            run_sample_sets(spec, generated)
        elif mode == genjob.MODE_EVALUATE:
            run_evaluation(spec, generated)
        else:
            run_generation(spec, generated)
    except _Cancelled:
        # The images already written stay in the job's `files`; this is not a failure.
        genjob.update_job(
            generated,
            job_id,
            state=genjob.STATE_CANCELLED,
            cancel_requested=True,
            error=None,
        )
        _log(f"cancelled {job_id}")
        return 0
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
