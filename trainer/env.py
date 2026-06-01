from __future__ import annotations

import gc
import os
from pathlib import Path

import torch


def setup_migraphx_cache(cache_dir_name: str = "migraphx_cache") -> Path:
    """Create and register the MIGraphX cache directory."""
    cache_dir = Path.cwd() / cache_dir_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ORT_MIGRAPHX_MODEL_CACHE_PATH"] = str(cache_dir)
    os.environ["ORT_MIGRAPHX_CACHE_PATH"] = str(cache_dir)
    return cache_dir


def flush_memory(device: torch.device) -> None:
    """Release Python and GPU-side cached memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "hip") and torch.hip.is_available():
        torch.hip.empty_cache()