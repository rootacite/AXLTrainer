from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

try:
    import control
    from env import flush_memory
except ImportError:
    from trainer import control
    from trainer.env import flush_memory


@dataclass
class SwapContext:
    device: torch.device
    vae: Any = None
    denoise: Any = None
    text_encoders: list = field(default_factory=list)
    denoise_optimizer: Any = None
    te_optimizer: Any = None
    saved_modes: dict[str, Any] = field(default_factory=dict)


def optimizer_tensors_to(optimizer: Any, device) -> None:
    if optimizer is None:
        return
    for group in getattr(optimizer, "param_groups", []) or []:
        for p in group.get("params") or []:
            if getattr(p, "grad", None) is not None:
                p.grad = None
        for key, value in list(group.items()):
            if key == "params":
                continue
            if torch.is_tensor(value):
                group[key] = value.to(device)
    state = getattr(optimizer, "state", None)
    if state is not None:
        for bucket in state.values():
            if not isinstance(bucket, dict):
                continue
            for key, value in list(bucket.items()):
                if torch.is_tensor(value):
                    bucket[key] = value.to(device)
    for key, value in list(vars(optimizer).items()):
        if key in ("state", "param_groups"):
            continue
        if torch.is_tensor(value):
            setattr(optimizer, key, value.to(device))


def _move_module(module: Any, device) -> None:
    if module is None:
        return
    module.to(device)


def _named_modules(ctx: SwapContext) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = [("denoise", ctx.denoise), ("vae", ctx.vae)]
    for index, te in enumerate(ctx.text_encoders, start=1):
        items.append((f"text_encoder_{index}", te))
    return items


def _optimizer_train_mode(optimizer: Any) -> Optional[bool]:
    """Schedule-Free's `train_mode` flag, or None for an optimizer that has no such mode."""
    if optimizer is None or not getattr(optimizer, "param_groups", None):
        return None
    return bool(optimizer.param_groups[0].get("train_mode", False))


def _set_optimizer_train_mode(optimizer: Any, wanted: Optional[bool]) -> None:
    if optimizer is None or wanted is None:
        return
    current = _optimizer_train_mode(optimizer)
    if current is None:
        return
    if wanted and not current and hasattr(optimizer, "train"):
        optimizer.train()
    elif (not wanted) and current and hasattr(optimizer, "eval"):
        optimizer.eval()


def capture_modes(ctx: SwapContext) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for name, module in _named_modules(ctx):
        modes[name] = bool(module.training) if module is not None and hasattr(module, "training") else None
    for name, optimizer in (("denoise", ctx.denoise_optimizer), ("te", ctx.te_optimizer)):
        mode = _optimizer_train_mode(optimizer)
        if mode is not None:
            modes[f"{name}_opt_train_mode"] = mode
    return modes


def restore_modes(ctx: SwapContext, modes: dict[str, Any]) -> None:
    for name, module in _named_modules(ctx):
        flag = modes.get(name)
        if module is None or flag is None:
            continue
        module.train(flag)
    for name, optimizer in (("denoise", ctx.denoise_optimizer), ("te", ctx.te_optimizer)):
        _set_optimizer_train_mode(optimizer, modes.get(f"{name}_opt_train_mode"))


def _first_trainable_param(module: Any) -> Optional[tuple[str, torch.Tensor]]:
    if module is None or not hasattr(module, "named_parameters"):
        return None
    for name, param in module.named_parameters():
        if param.requires_grad:
            return name, param
    for name, param in module.named_parameters():
        return name, param
    return None


def _first_state_tensor(optimizer: Any, key: str) -> Optional[torch.Tensor]:
    state = getattr(optimizer, "state", None)
    if not state:
        return None
    for bucket in state.values():
        if isinstance(bucket, dict) and key in bucket and torch.is_tensor(bucket[key]):
            return bucket[key]
    return None


def log_snapshot(tag: str, ctx: SwapContext) -> None:
    bits: list[str] = [f"[swap {tag}]"]
    found = _first_trainable_param(ctx.denoise)
    if found is not None:
        name, param = found
        bits.append(f"denoise.{name} device={param.device} dtype={param.dtype}")
    if ctx.denoise_optimizer is not None:
        for key in ("z", "exp_avg_sq", "exp_avg"):
            tensor = _first_state_tensor(ctx.denoise_optimizer, key)
            if tensor is not None:
                bits.append(f"denoise_opt.{key} device={tensor.device} dtype={tensor.dtype}")
                break
    print(" ".join(bits), file=sys.stderr, flush=True)


def offload_to_cpu(ctx: SwapContext) -> None:
    stages = [
        ("offload_denoise", "Moving denoise model to CPU", lambda: _move_module(ctx.denoise, "cpu")),
        (
            "offload_text_encoders",
            "Moving text encoders to CPU",
            lambda: [_move_module(te, "cpu") for te in ctx.text_encoders],
        ),
        (
            "offload_optimizers",
            "Moving optimizer state to CPU",
            lambda: (
                optimizer_tensors_to(ctx.denoise_optimizer, "cpu"),
                optimizer_tensors_to(ctx.te_optimizer, "cpu"),
            ),
        ),
        ("offload_vae", "Moving VAE to CPU", lambda: _move_module(ctx.vae, "cpu")),
        ("flush_gpu", "Releasing GPU cache", lambda: flush_memory(ctx.device)),
    ]
    total = len(stages)
    for index, (stage, detail, action) in enumerate(stages, start=1):
        control.set_swap(stage, detail, index, total)
        action()


def reload_to_device(ctx: SwapContext, phase: str) -> None:
    stages: list[tuple[str, str, Any]] = []
    if phase == "encoding":
        stages.append(("reload_vae", "Moving VAE to GPU", lambda: _move_module(ctx.vae, ctx.device)))
    else:
        stages.extend(
            [
                ("reload_denoise", "Moving denoise model to GPU", lambda: _move_module(ctx.denoise, ctx.device)),
                (
                    "reload_text_encoders",
                    "Moving text encoders to GPU",
                    lambda: [_move_module(te, ctx.device) for te in ctx.text_encoders],
                ),
                (
                    "reload_optimizers",
                    "Moving optimizer state to GPU",
                    lambda: (
                        optimizer_tensors_to(ctx.denoise_optimizer, ctx.device),
                        optimizer_tensors_to(ctx.te_optimizer, ctx.device),
                    ),
                ),
            ]
        )
        if phase == "sampling":
            stages.append(("reload_vae", "VAE stays managed by sampler", lambda: None))
    total = max(1, len(stages))
    for index, (stage, detail, action) in enumerate(stages, start=1):
        control.set_swap(stage, detail, index, total)
        action()


def run_pause(ctx: SwapContext, phase: str) -> None:
    control.set_status(control.STATUS_PAUSING, paused_from=phase)
    ctx.saved_modes = capture_modes(ctx)
    log_snapshot("before-offload", ctx)
    offload_to_cpu(ctx)
    log_snapshot("after-offload", ctx)
    control.clear_swap()
    control.set_status(control.STATUS_PAUSED, paused_from=phase)


def run_resume(ctx: SwapContext, phase: str) -> None:
    control.set_status(control.STATUS_RESUMING, paused_from=phase)
    reload_to_device(ctx, phase)
    restore_modes(ctx, ctx.saved_modes or {})
    log_snapshot("after-reload", ctx)
    control.clear_swap()
    restore = {
        "encoding": control.STATUS_ENCODING,
        "training": control.STATUS_TRAINING,
        "sampling": control.STATUS_SAMPLING,
    }.get(phase, control.STATUS_TRAINING)
    control.set_status(restore, paused_from=None)


def at_safe_point(phase: str, ctx: Optional[SwapContext] = None) -> bool:
    """Handle pause/resume/stop at a swap-safe point. False means abort the phase."""
    cmd = control.poll_command()
    if cmd == "stop":
        control.set_status(control.STATUS_STOPPING, paused_from=None, swap=None)
        return False
    if cmd == "pause":
        if ctx is None:
            control.set_status(control.STATUS_PAUSED, paused_from=phase)
        else:
            run_pause(ctx, phase)
        cmd = None
    while control._ensure_state().get("status") == control.STATUS_PAUSED:
        cmd = control.poll_command()
        if cmd == "stop":
            control.set_status(control.STATUS_STOPPING, swap=None)
            return False
        if cmd == "resume":
            if ctx is None:
                control.set_status(phase if phase in ("encoding", "training", "sampling") else control.STATUS_TRAINING)
            else:
                run_resume(ctx, phase)
            break
        time.sleep(0.2)
    return control._ensure_state().get("status") != control.STATUS_STOPPING
