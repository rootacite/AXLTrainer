import asyncio

import io
import os
import re
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer.config import TrainConfig
import generate.config as gen_config
import generate.detailer as gen_detailer
import generate.model_loader as gen_model_loader
import generate.pipeline as gen_pipeline
import generate.upscaler as gen_upscaler

import torch
import gc

# ==========================================
# Lifespan
# ==========================================

@asynccontextmanager
async def lifespan(lapp: FastAPI):
    lapp.state.runtime_pipelines = None
    lapp.state.generation_lock = asyncio.Lock()
    yield


app = FastAPI(
    title="LoRA Training & Generation API",
    version="1.0.0",
    description="Backend API for training dashboard and image generation pipeline.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Data Models
# ==========================================

class RefinementPassOverride(BaseModel):
    name: str
    model: str
    denoise: float
    guide_size: int


class PromptOverrides(BaseModel):
    positive_prompt: Optional[str] = Field(default=None)
    negative_prompt: Optional[str] = Field(default=None)
    base_model_path: Optional[str] = Field(default=None)
    lora_path: Optional[str] = Field(default=None)
    lora_scale: Optional[float] = Field(default=None)
    realesrgan_model_path: Optional[str] = Field(default=None)
    max_token_length: Optional[int] = Field(default=None)
    clip_skip: Optional[int] = Field(default=None)
    output_filename_prefix: Optional[str] = Field(default=None)
    refinement_passes: Optional[List[RefinementPassOverride]] = Field(default=None)


@dataclass
class RuntimePipelines:
    base_pipe: Any
    inpaint_pipe: Any
    img2img_pipe: Any
    load_key: Tuple[Any, ...]


def _to_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _normalize_refinement_passes(
    refinement_passes: Optional[List[RefinementPassOverride]],
) -> Optional[List[Dict[str, Any]]]:
    if refinement_passes is None:
        return None
    return [item.model_dump() for item in refinement_passes]


def _validate_positive(name: str, value: Optional[int]) -> None:
    if value is not None and value <= 0:
        raise HTTPException(status_code=400, detail=f"{name} must be a positive integer.")


def _pipeline_load_key() -> Tuple[Any, ...]:
    return (
        gen_config.BASE_MODEL_PATH,
        gen_config.LORA_PATH,
        gen_config.LORA_SCALE,
        gen_config.REALESRGAN_MODEL_PATH,
        gen_config.DEVICE,
        str(gen_config.TORCH_DTYPE),
    )


def _get_tensorboard_metrics(
    log_dir: str,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
) -> dict:
    if not os.path.exists(log_dir):
        return {}

    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        return {}

    latest_log_dir = str(max(event_files, key=os.path.getmtime).parent)
    ea = EventAccumulator(latest_log_dir, size_guidance={"scalars": 0})
    ea.Reload()

    metrics: dict = {}
    if "scalars" in ea.Tags():
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            filtered = [
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in events
                if (start_step is None or e.step >= start_step)
                and (end_step is None or e.step <= end_step)
            ]
            metrics[tag] = filtered

    return metrics


def _sync_ensure_runtime_pipelines(state) -> RuntimePipelines | None:
    desired_key = _pipeline_load_key()
    runtime: Optional[RuntimePipelines] = getattr(state, "runtime_pipelines", None)

    if runtime is not None and runtime.load_key == desired_key:
        return runtime

    try:
        gen_upscaler.get_realesrgan_upsampler.cache_clear()
    except Exception:
        pass

    base_pipe = gen_model_loader.load_base_pipeline()
    inpaint_pipe = gen_model_loader.load_inpaint_pipeline_from_base(base_pipe)
    img2img_pipe = gen_model_loader.load_img2img_pipeline_from_base(base_pipe)

    runtime = RuntimePipelines(
        base_pipe=base_pipe,
        inpaint_pipe=inpaint_pipe,
        img2img_pipe=img2img_pipe,
        load_key=desired_key,
    )
    state.runtime_pipelines = runtime
    return runtime


@contextmanager
def _temporary_config_overrides(
    overrides: PromptOverrides,
    numeric_overrides: Dict[str, Any],
) -> Iterator[None]:
    original_values: Dict[str, Any] = {}

    def set_if_present(_attr: str, _value: Any) -> None:
        if _value is None:
            return
        original_values[_attr] = getattr(gen_config, _attr)
        setattr(gen_config, _attr, _value)

    try:
        set_if_present("POSITIVE_PROMPT", overrides.positive_prompt)
        set_if_present("NEGATIVE_PROMPT", overrides.negative_prompt)
        set_if_present("BASE_MODEL_PATH", overrides.base_model_path)
        set_if_present("LORA_PATH", overrides.lora_path)
        set_if_present("LORA_SCALE", overrides.lora_scale)
        set_if_present("REALESRGAN_MODEL_PATH", overrides.realesrgan_model_path)
        set_if_present("max_token_length", overrides.max_token_length)
        set_if_present("clip_skip", overrides.clip_skip)
        set_if_present("OUTPUT_FILENAME_PREFIX", overrides.output_filename_prefix)

        if overrides.refinement_passes is not None:
            original_values["REFINEMENT_PASSES"] = getattr(gen_config, "REFINEMENT_PASSES")
            setattr(
                gen_config,
                "REFINEMENT_PASSES",
                _normalize_refinement_passes(overrides.refinement_passes),
            )

        for key, value in numeric_overrides.items():
            if value is None or not hasattr(gen_config, key):
                continue
            original_values.setdefault(key, getattr(gen_config, key))
            setattr(gen_config, key, value)

        yield
    finally:
        for attr, value in reversed(list(original_values.items())):
            setattr(gen_config, attr, value)



def _sync_quick(
    state,
    overrides: PromptOverrides,
    numeric_overrides: Dict[str, Any],
    seed: Optional[int],
    steps: Optional[int],
    cfg_scale: Optional[float],
) -> bytes:
    with _temporary_config_overrides(overrides, numeric_overrides):
        runtime = _sync_ensure_runtime_pipelines(state)
        image, _ = gen_pipeline.generate_base_image(
            runtime.base_pipe,
            seed_override=seed,
            steps_override=steps,
            cfg_scale_override=cfg_scale,
            return_seed=True,
        )

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        gc.collect()

    return _to_png_bytes(image)


def _sync_generate(
    state,
    overrides: PromptOverrides,
    numeric_overrides: Dict[str, Any],
    stages: int,
    seed: Optional[int],
    steps: Optional[int],
    cfg_scale: Optional[float],
) -> bytes:
    with _temporary_config_overrides(overrides, numeric_overrides):
        runtime = _sync_ensure_runtime_pipelines(state)

        raw_image, actual_seed = gen_pipeline.generate_base_image(
            runtime.base_pipe,
            seed_override=seed,
            steps_override=steps,
            cfg_scale_override=cfg_scale,
            return_seed=True,
        )
        if stages == 1:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            gc.collect()
            return _to_png_bytes(raw_image)

        upscaled = gen_upscaler.ultimate_sd_upscale(
            image=raw_image,
            img2img_pipe=runtime.img2img_pipe,
            upscale_factor=2.0,
            tile_size=480,
            overlap=64,
            denoise_strength=0.1,
            seed_override=actual_seed,
            steps_override=steps,
            cfg_scale_override=cfg_scale,
        )
        if stages == 2:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            gc.collect()
            return _to_png_bytes(upscaled)

        final = gen_detailer.run_detailer_pipeline(
            upscaled,
            runtime.inpaint_pipe,
            seed_override=actual_seed,
            steps_override=steps,
            cfg_scale_override=cfg_scale,
        )
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        gc.collect()

        return _to_png_bytes(final)


# ==========================================
# Dashboard & Samples Endpoints
# ==========================================

@app.get("/api/dashboard")
async def get_dashboard_data(
    name: Optional[str] = Query(default=None),
    start_step: Optional[int] = Query(default=None),
    end_step: Optional[int] = Query(default=None),
):
    cfg = await asyncio.to_thread(lambda: vars(TrainConfig()))
    output_name = name or cfg.get("output_name", "default")
    logging_dir = cfg.get("logging_dir", "./logs")
    target_log_dir = os.path.join(logging_dir, output_name)

    metrics = await asyncio.to_thread(
        _get_tensorboard_metrics, target_log_dir, start_step, end_step
    )

    latest_stats: Dict[str, Any] = {}
    for tag, data in metrics.items():
        if data:
            latest_stats[tag] = data[-1]["value"]
            latest_stats["current_step"] = data[-1]["step"]

    return {"config": cfg, "latest_stats": latest_stats, "metrics": metrics}


@app.get("/api/samples")
async def list_samples(name: Optional[str] = Query(default=None)):
    cfg = await asyncio.to_thread(lambda: vars(TrainConfig()))
    output_dir = cfg.get("output_dir", "./output")
    output_name = name or cfg.get("output_name", "default")

    def _scan() -> dict[int, list] | dict[Any, Any]:
        sample_dir = Path(output_dir) / f"{output_name}_samples"
        if not sample_dir.exists():
            return {}

        pattern = re.compile(r"_(\d+)_(\d+)\.png$")
        grouped: Dict[int, list] = defaultdict(list)

        for img_path in sample_dir.glob("*.png"):
            match = pattern.search(img_path.name)
            if match:
                step, repeat_idx = int(match.group(1)), int(match.group(2))
            else:
                step, repeat_idx = -1, 0
            grouped[step].append({"filename": img_path.name, "repeat_idx": repeat_idx})

        for step in grouped:
            grouped[step] = sorted(grouped[step], key=lambda x: x["repeat_idx"])

        return {k: grouped[k] for k in sorted(grouped.keys(), reverse=True)}

    samples = await asyncio.to_thread(_scan)
    return {"samples": samples}


@app.get("/api/samples/{filename}")
async def get_sample_image(
    filename: str, name: Optional[str] = Query(default=None)
):
    cfg = await asyncio.to_thread(lambda: vars(TrainConfig()))
    output_dir = cfg.get("output_dir", "./output")
    output_name = name or cfg.get("output_name", "default")

    sample_dir = Path(output_dir) / f"{output_name}_samples"
    file_path = sample_dir / filename

    exists = await asyncio.to_thread(lambda: file_path.exists() and file_path.is_file())
    if not exists:
        raise HTTPException(status_code=404, detail="Sample image not found")

    return FileResponse(str(file_path), media_type="image/png")


# ==========================================
# Generation Endpoints
# ==========================================

@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/quick")
async def api_quick(
    body: Optional[PromptOverrides] = Body(default=None),
    seed: Optional[int] = Query(default=None),
    steps: Optional[int] = Query(default=None),
    cfg_scale: Optional[float] = Query(default=None, alias="cfg_scale"),
    width: Optional[int] = Query(default=None),
    height: Optional[int] = Query(default=None),
):
    body = body or PromptOverrides()
    _validate_positive("steps", steps)
    _validate_positive("width", width)
    _validate_positive("height", height)

    numeric_overrides = {
        "SEED": seed, "STEPS": steps, "CFG_SCALE": cfg_scale,
        "WIDTH": width, "HEIGHT": height,
    }

    lock: asyncio.Lock = app.state.generation_lock
    async with lock:
        try:
            png_bytes = await asyncio.to_thread(
                _sync_quick, app.state, body, numeric_overrides,
                seed, steps, cfg_scale,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png")


@app.post("/api/generate")
async def api_generate(
    body: Optional[PromptOverrides] = Body(default=None),
    stages: int = Query(default=3, ge=1, le=3),
    seed: Optional[int] = Query(default=None),
    steps: Optional[int] = Query(default=None),
    cfg_scale: Optional[float] = Query(default=None, alias="cfg_scale"),
    width: Optional[int] = Query(default=None),
    height: Optional[int] = Query(default=None),
):
    body = body or PromptOverrides()
    _validate_positive("steps", steps)
    _validate_positive("width", width)
    _validate_positive("height", height)

    numeric_overrides = {
        "SEED": seed, "STEPS": steps, "CFG_SCALE": cfg_scale,
        "WIDTH": width, "HEIGHT": height,
    }

    lock: asyncio.Lock = app.state.generation_lock
    async with lock:
        try:
            png_bytes = await asyncio.to_thread(
                _sync_generate, app.state, body, numeric_overrides,
                stages, seed, steps, cfg_scale,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)