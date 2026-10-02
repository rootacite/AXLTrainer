#!/usr/bin/env python3
"""GPU probe for the unified Schedule-Free optimizers (UNet + text encoders).

Run from the repo root in the conda env named by `environment.yml`:

    conda activate axl
    python test/probe_optimizer_gpu.py

It runs the real `trainer/main.py` twice inside a throwaway mirror of the repo — symlinked sources
and a generated `config.toml`, through the same machinery `verify_mask_pipeline.py` uses (own
`AXL_RUNTIME_DIR`, ROCm-quieting env, the gfx1201 fault retry and its no-memory-cache escalation) —
on a small copy of the configured dataset:

    A  both optimizers live      the run under test
    B  te_learning_rate = 0.0    the control that says *whose* learning rate moves the TE weights

Both children run through the repo's `start_hook.sh`, i.e. under the `[environment].amdfq`
allocation patch (here `vmm`, `amdfq/amdfq-vmm-rs`) exactly as `start_train.sh` would preload it, so
the probe trains the path this machine actually trains on; `--no-hook` runs them bare, which is the
path the mask tiers use and the one that hits the intermittent gfx1201 Tensile fault here.

then reads both runs' TensorBoard scalars, `state.json` and exported `.safetensors`.

What the checks are for
    optimizers   every step's `UNet/LR/Effective_Actual_LR` and `TE/LR/Effective_Actual_LR` equal
                 `lr x min(1, step / warmup)`. Those scalars are `AdamWScheduleFree`'s own
                 `scheduled_lr`, written inside `step()` and nowhere else, so both optimizer objects
                 really stepped, each with its own lr and its own warmup.
    weights      every LoRA key group (UNet linear / UNet conv LoCon / te1 + te2 linear / te1 + te2
                 MLP) differs between the run's first step checkpoint and its `_final`, with no
                 NaN/Inf and the configured alpha scalars.
    ownership    in run B every `lora_te*` tensor is bit-identical between those two checkpoints
                 while the UNet's moved: with `te_learning_rate = 0` nothing moves the TE, so run
                 A's TE movement is that optimizer's own doing (a swapped parameter set or a
                 swapped lr would freeze the UNet in B instead).
    loss         finite, positive and below 1 at every step; `Train/Avg_Loss` reproduces
                 `synthesize_avg_loss(Train/Loss, steps_per_epoch)`; `state.json`'s loss agrees with
                 the last step. The trend is reported, not asserted: a step's loss is one random
                 timestep per sample, so a 30-step slope is noise-dominated by design.
    export       the file really holds trained UNet *and* TE weights: all three key families are
                 present, every group moved, and `SdxlFamily.load_lora` reloads it with zero
                 unmatched tensors into the right adapters. Its kohya metadata carries the values
                 this run was given, the two moved keys included (`ss_lr_warmup_steps`,
                 `ss_max_grad_norm`).
    hook         the children really ran under the patch: each log carries start_hook.sh's banner
                 naming `amdfq=vmm` and the .so, the resolved line comes from config.toml, and the
                 run did not need the fault retry. A patched run needs `amdfq` set in config.toml;
                 the probe refuses rather than silently running bare.

Nothing is written into the dataset: images are copied first, and every run gets its own
AXL_RUNTIME_DIR / output_dir / logging_dir under the report directory (kept for inspection; pass
`--clean` to delete the scratch dir).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# The tiers import the trainer as a package, and a script's sys.path[0] is this directory (which is
# also what makes `import verify_mask_pipeline` below work).
sys.path.insert(0, str(REPO_ROOT))

from verify_mask_pipeline import (  # noqa: E402 - the path above is what makes this import work
    Report,
    config_sections,
    fmt_values,
    guard_environment,
    launch_run,
    live_training_runs,
    tb_scalars,
)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TIERS = ("train", "hook", "weights", "loss", "reload")
RUN_TIER = "probe"
# "Not worse than predicting zeros" for an eps target: a real regression blows through this, a
# wobbly-but-healthy step does not.
LOSS_CEILING = 1.0
# A key group whose relative L2 movement is under this barely moved at all.
MOVE_FLOOR = 1e-6
# TensorBoard rounds every scalar to float32, so an identity between a logged and a recomputed
# float64 value cannot be tighter than that.
REL_TOL = 1e-6
GROUPS = ("unet-linear", "unet-conv", "te1-linear", "te1-mlp", "te2-linear", "te2-mlp")


# --------------------------------------------------------------------------------------
# dataset copy (read-only source)
# --------------------------------------------------------------------------------------


def configured_train_dirs() -> list[Path]:
    """The folders `[[environment.train_data]]` (or `train_data_dir`) names, that exist."""
    environment = config_sections().get("environment", {})
    raw = environment.get("train_data") or []
    dirs: list[Path] = []
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("path"):
                dirs.append(Path(str(block["path"])).expanduser())
    if not dirs and environment.get("train_data_dir"):
        dirs.append(Path(str(environment["train_data_dir"])).expanduser())
    return [path for path in dirs if path.is_dir()]


def copy_images(source: Path, dest: Path, count: int) -> int:
    """Copy up to `count` images and their sidecar captions; the source directory is never written."""
    names = sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXT
        and not path.name.lower().endswith(".mask.png")
    )
    copied = 0
    for path in names:
        if copied >= count:
            break
        target = dest / path.name
        if target.exists():  # two folders holding the same filename
            target = dest / f"{copied:02d}_{path.name}"
        shutil.copy2(path, target)
        sidecar = path.with_suffix(".txt")
        if sidecar.is_file():
            shutil.copy2(sidecar, target.with_suffix(".txt"))
        copied += 1
    return copied


def prepare_dataset(work: Path, images: int) -> tuple[Path, dict[str, Any]]:
    """One training folder built from the configured ones: `images` spread over them evenly."""
    sources = configured_train_dirs()
    if not sources:
        raise SystemExit(
            "config.toml names no existing training folder ([[environment.train_data]] / "
            "train_data_dir); the probe needs a dataset to copy"
        )
    dest = work / "dataset"
    dest.mkdir(parents=True, exist_ok=True)
    quota = [images // len(sources) + (1 if index < images % len(sources) else 0)
             for index in range(len(sources))]
    counts = {str(source): copy_images(source, dest, want) for source, want in zip(sources, quota)}
    copied = sum(counts.values())
    if copied < 2:
        raise SystemExit(f"only {copied} image(s) copied from {sources}; two are the minimum to train")
    sidecars = len(list(dest.glob("*.txt")))
    return dest, {"dir": str(dest), "images": copied, "captions": sidecars, "sources": counts}


# --------------------------------------------------------------------------------------
# the runs
# --------------------------------------------------------------------------------------


@dataclass
class ProbeRun:
    label: str
    te_lr: float
    result: Any
    output_name: str
    mirror_config: Path
    final: Optional[Path] = None
    first_step: Optional[Path] = None
    first_step_index: Optional[int] = None
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    series: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


def read_state(runtime_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((runtime_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def launch(rep: Report, args: argparse.Namespace, *, label: str, te_lr: float, data_dir: Path,
           work: Path) -> ProbeRun:
    name = f"opt_{label.lower()}"
    extra = {
        # The rank reduction the probe asks for; `[network]` is the one section the shared launcher
        # does not know about.
        "network": {
            "network_dim": args.network_dim,
            "network_alpha": args.network_alpha,
            "conv_dim": args.conv_dim,
            "conv_alpha": args.conv_alpha,
        },
        # config.toml has this off too; the launcher defaults it on because the mask tiers count
        # sample images.
        "training": {"sampling_enabled": False},
    }
    result = launch_run(
        name=name, data_dir=data_dir, seed=args.seed, steps=args.steps, work=work,
        masked=False, tier=RUN_TIER, batch_size=args.batch_size,
        save_every_override=max(1, args.steps // 3), lr=(args.unet_lr, te_lr), warmup=args.warmup,
        retries=args.retries, extra_sections=extra, hook=args.hook,
    )
    output_name = f"verify_{name}"
    run = ProbeRun(
        label=label, te_lr=te_lr, result=result, output_name=output_name,
        mirror_config=Path(result.claims.get("mirror_config") or (result.run_root / "mirror" / "config.toml")),
    )
    from trainer.checkpoints import discover_checkpoints

    run.checkpoints = discover_checkpoints(result.output_dir, output_name)
    finals = [item for item in run.checkpoints if item.get("final")]
    steps = sorted(
        (item for item in run.checkpoints if not item.get("final") and item.get("step") is not None),
        key=lambda item: item["step"],
    )
    run.final = Path(finals[0]["path"]) if finals else None
    if steps:
        run.first_step = Path(steps[0]["path"])
        run.first_step_index = int(steps[0]["step"])
    run.state = read_state(result.runtime_dir)
    for key, tag in (
        ("loss", "Train/Loss"),
        ("avg", "Train/Avg_Loss"),
        ("unet_lr", "UNet/LR/Effective_Actual_LR"),
        ("te_lr", "TE/LR/Effective_Actual_LR"),
    ):
        run.series[key] = tb_scalars(result.logging_dir, tag)
    return run


def tier_train(rep: Report, args: argparse.Namespace, work: Path) -> tuple[dict[str, Any], dict[str, ProbeRun]]:
    data_dir, dataset = prepare_dataset(work, args.images)
    rep.note(
        f"dataset: {dataset['images']} image(s) + {dataset['captions']} caption(s) copied from "
        + ", ".join(f"{path} x{count}" for path, count in dataset["sources"].items())
        + f" into {data_dir}"
    )
    runs = {"A": launch(rep, args, label="A", te_lr=args.te_lr, data_dir=data_dir, work=work)}
    if args.runs == "both":
        runs["B"] = launch(rep, args, label="B", te_lr=0.0, data_dir=data_dir, work=work)
    else:
        rep.note("--runs main: the frozen-TE control run (B) was skipped; the ownership checks cannot run")

    for label, run in runs.items():
        result = run.result
        tier = RUN_TIER
        rep.check(
            tier, f"{label}: the run finishes and writes both checkpoints",
            result.returncode == 0 and run.final is not None and run.first_step is not None,
            f"exit={result.returncode} in {result.seconds:.0f}s | final={run.final.name if run.final else None}"
            f" | first step checkpoint={run.first_step_index}"
            + (f" | attempts={result.attempts}" if result.attempts > 1 else ""),
            gpu_fault=bool(result.gpu_fault), attempts=result.attempts,
        )
        rep.check(
            tier, f"{label}: state.json says finished at the last logged step",
            run.state.get("status") == "finished"
            and int(run.state.get("training", {}).get("step", -1)) == len(run.series["loss"]),
            f"status={run.state.get('status')} step={run.state.get('training', {}).get('step')} "
            f"logged steps={len(run.series['loss'])}",
        )
        check_lr(rep, run, args)
    if args.runs == "both":
        check_control(rep, runs["A"], runs["B"])
    details = {"dataset": dataset, "runs": {label: describe_run(run) for label, run in runs.items()}}
    return details, runs


def describe_run(run: ProbeRun) -> dict[str, Any]:
    return {
        "te_lr": run.te_lr,
        "output_dir": str(run.result.output_dir),
        "logging_dir": str(run.result.logging_dir),
        "runtime_dir": str(run.result.runtime_dir),
        "run_dir": str(run.result.run_dir) if run.result.run_dir else None,
        "log": str(run.result.log_path),
        "mirror_config": str(run.mirror_config),
        "checkpoints": [item["dir"] for item in run.checkpoints],
        "seconds": round(run.result.seconds, 1),
        "steps": len(run.series["loss"]),
        "state": run.state.get("training", {}),
    }


def expected_lr(step: int, lr: float, warmup: int) -> float:
    """`AdamWScheduleFree`'s own ramp: `sched = (k+1)/warmup` while `k < warmup`, then 1.0."""
    if warmup <= 0:
        return lr
    return lr * min(1.0, step / warmup)


def check_lr(rep: Report, run: ProbeRun, args: argparse.Namespace) -> None:
    label = run.label
    loss_steps = [step for step, _ in run.series["loss"]]
    shape_ok = (
        [step for step, _ in run.series["unet_lr"]] == loss_steps
        and [step for step, _ in run.series["te_lr"]] == loss_steps
        and loss_steps == list(range(1, len(loss_steps) + 1))
    )
    rep.check(
        RUN_TIER, f"{label}: both LR tags are logged at every training step",
        shape_ok and bool(loss_steps),
        f"{len(loss_steps)} step(s), contiguous from 1, with Train/Loss + both LR tags",
    )
    for key, lr, name in (
        ("unet_lr", args.unet_lr, "UNet/LR/Effective_Actual_LR"),
        ("te_lr", run.te_lr, "TE/LR/Effective_Actual_LR"),
    ):
        series = run.series[key]
        worst = 0.0
        for step, value in series:
            worst = max(worst, abs(value - expected_lr(step, lr, args.warmup)))
        tol = max(1e-9, 1e-5 * lr)
        rep.check(
            RUN_TIER, f"{label}: {name} follows its own lr and warmup on every step",
            bool(series) and worst <= tol,
            f"max |logged - lr*min(1, step/{args.warmup})| = {worst:.3g} over {len(series)} step(s)",
            lr=lr, worst=worst, tolerance=tol,
        )
    if run.te_lr == 0.0:
        worst = max((abs(value) for _, value in run.series["te_lr"]), default=0.0)
        rep.check(
            RUN_TIER, f"{label}: the frozen TE lr logs as exactly 0",
            worst == 0.0,
            f"max TE/LR/Effective_Actual_LR = {worst:.3g}",
        )


def compare_configs(left: Path, right: Path, allowed: Iterable[str]) -> dict[str, Any]:
    """Flattened key/value diff of two generated configs (the repo's own flattening rule)."""
    from trainer.config import _load_toml_config

    a, b = _load_toml_config(str(left)), _load_toml_config(str(right))
    keys = sorted(set(a) | set(b))
    differing = [key for key in keys if a.get(key) != b.get(key)]
    return {
        "keys": len(keys),
        "differing": differing,
        "unexpected": [key for key in differing if key not in set(allowed)],
    }


def check_control(rep: Report, run_a: ProbeRun, run_b: ProbeRun) -> None:
    allowed = ("output_dir", "logging_dir", "output_name", "te_learning_rate")
    diff = compare_configs(run_a.mirror_config, run_b.mirror_config, allowed)
    rep.check(
        RUN_TIER, "the two runs differ only in the TE lr (a one-variable control)",
        not diff["unexpected"],
        f"{len(diff['differing'])} key(s) differ: {', '.join(diff['differing'])}",
        **{f"unexpected_{index}": key for index, key in enumerate(diff["unexpected"])},
    )
    rep.observe(RUN_TIER, "control config diff", "flattened keys that differ between run A and run B",
                differing=diff["differing"], compared_keys=diff["keys"])


# --------------------------------------------------------------------------------------
# tier: hook (was the [environment].amdfq patch really preloaded into the children?)
# --------------------------------------------------------------------------------------


def tier_hook(rep: Report, runs: dict[str, ProbeRun], args: argparse.Namespace) -> dict[str, Any]:
    tier = "hook"
    if not args.hook:
        rep.note("--no-hook: the children ran bare, without the [environment].amdfq patch")
        return {}
    from trainer.amdfq_patch import resolve_preload

    info = resolve_preload()
    rep.check(
        tier, "config.toml asks for an allocation patch to preload",
        info["choice"] in ("tail", "vmm") and bool(info["so"]),
        f"amdfq={info['choice']} LD_PRELOAD={info['so']}",
        choice=info["choice"], so=info["so"],
    )
    out: dict[str, Any] = {"resolved": dict(info)}
    for label, run in runs.items():
        text = run.result.log_path.read_text(encoding="utf-8", errors="replace")
        banner = next((line.strip() for line in text.splitlines() if "start_hook.sh: amdfq=" in line), None)
        mirror = f"workdir={run.result.run_root / 'mirror'}"
        rep.check(
            tier, f"{label}: start_hook.sh preloaded the patch into the trainer",
            banner is not None and f"amdfq={info['choice']}" in banner
            and info["so"] is not None and info["so"] in banner and mirror in banner,
            banner or "no start_hook.sh banner in the child's log",
        )
        rep.check(
            tier, f"{label}: the run stayed on the normal allocator path",
            not run.result.gpu_fault and run.result.attempts == 1,
            f"attempts={run.result.attempts} gfx1201_fault={run.result.gpu_fault}",
            attempts=run.result.attempts,
        )
        out[label] = {"banner": banner, "attempts": run.result.attempts,
                      "gpu_fault": run.result.gpu_fault}
    return out


# --------------------------------------------------------------------------------------
# tier: loss (TensorBoard + state.json)
# --------------------------------------------------------------------------------------


def ols_slope(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [float(step) for step, _ in points]
    ys = [float(value) for _, value in points]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mean_x) ** 2 for x in xs)
    if var <= 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var


def tier_loss(rep: Report, run: ProbeRun) -> dict[str, Any]:
    tier = "loss"
    losses = run.series["loss"]
    values = [value for _, value in losses]
    if not values:
        rep.check(tier, "the run logged a loss series", False, "Train/Loss is empty")
        return {}
    rep.check(
        tier, "every step's loss is finite and positive",
        all(math.isfinite(value) and value > 0 for value in values),
        f"{len(values)} step(s) ∈ [{min(values):.4g}, {max(values):.4g}]",
    )
    rep.check(
        tier, f"no step's loss exceeds {LOSS_CEILING} (the 'better than zeros' ceiling)",
        max(values) < LOSS_CEILING,
        f"max = {max(values):.4g}",
    )

    steps_per_epoch = 0
    training = run.state.get("training", {})
    epochs = int(training.get("epochs") or 0)
    total_steps = int(training.get("total_steps") or 0)
    if epochs > 0 and total_steps > 0:
        steps_per_epoch = total_steps // epochs
    from trainer.loss_log import synthesize_avg_loss

    rebuilt = synthesize_avg_loss(
        [{"step": step, "value": value} for step, value in losses], steps_per_epoch or None
    )
    old = [value for _, value in run.series["avg"]]
    new = [point["value"] for point in rebuilt]
    worst = 0.0
    if len(old) == len(new) == len(values):
        worst = max((abs(a - b) / max(abs(a), 1e-12) for a, b in zip(old, new)), default=0.0)
    rep.check(
        tier, "Train/Avg_Loss reproduces the trainer's own averaging rule",
        bool(old) and len(old) == len(new) == len(values) and worst <= REL_TOL,
        f"{len(old)} point(s) vs {len(new)} recomputed, max relative deviation {worst:.3g} "
        f"(steps_per_epoch={steps_per_epoch})",
        steps_per_epoch=steps_per_epoch, worst=worst,
    )

    last_state_loss = training.get("loss")
    last_logged = values[-1]
    got = last_state_loss is not None and abs(float(last_state_loss) - last_logged) <= REL_TOL * max(
        abs(last_logged), 1e-12
    )
    rep.check(
        tier, "state.json's loss agrees with the last logged step",
        got, f"state={last_state_loss} last Train/Loss={last_logged:.6g}",
    )

    third = max(1, len(values) // 3)
    head, tail = values[:third], values[-third:]
    rep.observe(
        tier, "loss series", "the per-step loss, its moving average and the trend the ceiling cannot see",
        points=len(values), minimum=min(values), maximum=max(values),
        mean=sum(values) / len(values), first_third=sum(head) / len(head),
        last_third=sum(tail) / len(tail), ols_slope=ols_slope(losses),
        moving_average_start=old[0] if old else None, moving_average_end=old[-1] if old else None,
    )
    return {"steps": len(values), "min": min(values), "max": max(values),
            "steps_per_epoch": steps_per_epoch, "slope": ols_slope(losses)}


# --------------------------------------------------------------------------------------
# tier: weights (the exported .safetensors)
# --------------------------------------------------------------------------------------


def is_mlp(key: str) -> bool:
    return "mlp_fc1" in key or "mlp_fc2" in key


def stem(key: str) -> str:
    return key[: key.rindex(".lora_")]


def classify(tensors: dict[str, Any]) -> dict[str, list[str]]:
    """Key groups: the UNet's two LoCon adapters (4-D vs 2-D weights) and each TE's Linear/MLP."""
    groups: dict[str, list[str]] = {}
    for key, tensor in tensors.items():
        if key.endswith(".alpha"):
            continue
        if key.startswith("lora_unet_"):
            name = "unet-conv" if tensor.dim() == 4 else "unet-linear"
        elif key.startswith("lora_te1_"):
            name = "te1-mlp" if is_mlp(key) else "te1-linear"
        elif key.startswith("lora_te2_"):
            name = "te2-mlp" if is_mlp(key) else "te2-linear"
        else:
            name = "other"
        groups.setdefault(name, []).append(key)
    return groups


def load_tensors(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path))


def compare_groups(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-group movement between two checkpoints (`right` is the later one)."""
    out: dict[str, dict[str, Any]] = {}
    for name, keys in classify(right).items():
        changed, max_delta, sq, ref = 0, 0.0, 0.0, 0.0
        alphas: set[float] = set()
        for key in keys:
            a, b = left.get(key), right[key]
            if a is None or a.shape != b.shape:
                changed += 1
                continue
            if not bool((a == b).all()):
                changed += 1
            delta = (a.float() - b.float()).abs().max()
            max_delta = max(max_delta, float(delta))
            sq += float((a.float() - b.float()).double().pow(2).sum())
            ref += float(b.float().double().pow(2).sum())
            alpha = right.get(stem(key) + ".alpha")
            if alpha is not None:
                alphas.add(round(float(alpha), 6))
        out[name] = {
            "keys": len(keys), "changed": changed, "max_abs_delta": max_delta,
            "relative_l2": (sq ** 0.5) / max(ref ** 0.5, 1e-12), "alphas": sorted(alphas),
        }
    return out


def tier_weights(rep: Report, runs: dict[str, ProbeRun], args: argparse.Namespace) -> dict[str, Any]:
    from trainer.checkpoints import read_lora_metadata
    from trainer.config import TrainConfig, _load_toml_config

    tier = "weights"
    out: dict[str, Any] = {}
    for label, run in runs.items():
        if run.final is None or run.first_step is None:
            rep.check(tier, f"{label}: the two checkpoints to compare exist", False,
                      "a run without both the first step checkpoint and _final cannot be compared")
            continue
        first, final = load_tensors(run.first_step), load_tensors(run.final)

        keys = set(final)
        paired = [key for key in keys if not key.endswith(".alpha")]
        unpaired = [
            key for key in paired
            if key.replace(".lora_down.weight", ".lora_up.weight") not in keys
            and key.replace(".lora_up.weight", ".lora_down.weight") not in keys
        ]
        families = sorted({key.split("_")[1] for key in keys if key.startswith("lora_")})
        nan_keys = [
            key for key, tensor in final.items()
            if tensor.numel() and not math.isfinite(float(tensor.float().abs().max()))
        ]
        dtypes = sorted({str(tensor.dtype) for tensor in final.values()})
        rep.check(
            tier, f"{label}: UNet + TE1 + TE2 tensors, each with both down/up halves",
            not unpaired and families == ["te1", "te2", "unet"],
            f"{len(keys)} tensors, families {', '.join(families)}, {len(unpaired)} unpaired key(s)",
            unpaired=unpaired[:5],
        )
        rep.check(
            tier, f"{label}: every exported tensor is finite bf16",
            not nan_keys and dtypes == ["torch.bfloat16"],
            f"dtypes {', '.join(dtypes)}; {len(nan_keys)} non-finite tensor(s)",
        )

        groups = compare_groups(first, final)
        out[label] = {"groups": groups, "checkpoints": {
            "first": f"{run.first_step.parent.name}/{run.first_step.name} (step {run.first_step_index})",
            "final": f"{run.final.parent.name}/{run.final.name}",
        }}
        rep.observe(
            tier, f"{label}: movement per key group",
            f"differences between step {run.first_step_index} and the final checkpoint",
            **groups,
        )

        expected_alphas = {
            "unet-conv": {float(args.conv_alpha)},
            "unet-linear": {float(args.network_alpha)},
            "te1-linear": {float(args.network_alpha)},
            "te1-mlp": {float(args.network_alpha)},
            "te2-linear": {float(args.network_alpha)},
            "te2-mlp": {float(args.network_alpha)},
        }
        alpha_ok = all(
            groups.get(name, {}).get("alphas") == sorted(values)
            for name, values in expected_alphas.items()
        )
        rep.check(
            tier, f"{label}: the alpha scalars are the configured network_alpha / conv_alpha",
            alpha_ok,
            "; ".join(f"{name}={groups.get(name, {}).get('alphas')}" for name in expected_alphas),
        )

        moved = {name: groups.get(name, {}) for name in GROUPS}
        if label == "A":
            ok = all(
                item.get("keys", 0) > 0 and item.get("changed", 0) > 0
                and item.get("relative_l2", 0.0) > MOVE_FLOOR
                for item in moved.values()
            )
            rep.check(
                tier, "A: every LoRA key group moved between the first checkpoint and _final",
                ok,
                "; ".join(
                    f"{name}: {item.get('changed', 0)}/{item.get('keys', 0)} keys, "
                    f"max|Δ|={item.get('max_abs_delta', 0):.3g}, relL2={item.get('relative_l2', 0):.3g}"
                    for name, item in moved.items()
                ),
                empty=[name for name, item in moved.items() if not item.get("keys")],
            )
        else:
            frozen = {name: groups.get(name, {}) for name in ("te1-linear", "te1-mlp", "te2-linear", "te2-mlp")}
            unet_moved = all(
                groups.get(name, {}).get("changed", 0) > 0 for name in ("unet-linear", "unet-conv")
            )
            te_frozen = all(
                item.get("changed", 0) == 0 and item.get("max_abs_delta", 0.0) == 0.0
                for item in frozen.values()
            )
            rep.check(
                tier, "B (te_lr = 0): the TE tensors are bit-identical and the UNet's are not",
                te_frozen and unet_moved,
                "; ".join(
                    f"{name}: {item.get('changed', 0)}/{item.get('keys', 0)} keys, "
                    f"max|Δ|={item.get('max_abs_delta', 0.0):.3g}"
                    for name, item in list(frozen.items())
                    + [(n, groups.get(n, {})) for n in ("unet-linear", "unet-conv")]
                ),
                te_frozen=te_frozen, unet_moved=unet_moved,
            )

        metadata = read_lora_metadata(run.final)
        cfg = TrainConfig.from_mapping(_load_toml_config(str(run.mirror_config)))
        expected = {
            "ss_network_dim": str(int(cfg.network_dim)),
            "ss_network_alpha": str(int(cfg.network_alpha)),
            "ss_network_type": "locon",
            "ss_network_args": f"conv_dim={int(cfg.conv_dim)} conv_alpha={int(cfg.conv_alpha)}",
            "ss_unet_lr": str(cfg.unet_learning_rate),
            "ss_text_encoder_lr": str(cfg.te_learning_rate),
            "ss_lr_warmup_steps": str(int(cfg.te_warmup_steps)),
            "ss_max_grad_norm": str(cfg.unet_max_grad_norm),
            "ss_min_snr_gamma": str(cfg.min_snr_gamma),
            "ss_steps": str(len(run.series["loss"])),
        }
        mismatched = {
            key: {"file": metadata.get(key), "expected": want}
            for key, want in expected.items()
            if metadata.get(key) != want
        }
        rep.check(
            tier, f"{label}: the kohya metadata carries the values this run was given",
            not mismatched,
            "; ".join(f"{key}={metadata.get(key)!r}" for key in expected),
            mismatched=mismatched,
        )
        out[label]["metadata"] = metadata
    return out


# --------------------------------------------------------------------------------------
# tier: reload (GPU, parent process, after the children have exited)
# --------------------------------------------------------------------------------------


def param_snapshot(module: Any) -> dict[str, Any]:
    """Adapter tensors as float32 on the CPU, so a comparison never depends on the device a module
    happens to sit on (the pipeline keeps its UNet on the GPU and its text encoders on the CPU)."""
    return {
        name: param.detach().float().cpu().clone()
        for name, param in module.named_parameters()
        if ".lora_" in name
    }


def quantize_like_file(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The adapter as a checkpoint would carry it: `save_lora` writes bf16, so a reloaded value can
    only differ from the init by what bf16 can express. Without this the bf16 rounding of a frozen
    adapter's own init (~2e-3 relative) drowns the movement it is supposed to show."""
    import torch

    return {name: tensor.to(torch.bfloat16).float().clone() for name, tensor in snapshot.items()}


def te_groups(snapshot: dict[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name in snapshot:
        group = "mlp" if (".mlp.fc1." in name or ".mlp.fc2." in name) else "linear"
        groups.setdefault(group, []).append(name)
    return groups


def unet_groups(snapshot: dict[str, Any]) -> dict[str, list[str]]:
    """The two LoCon adapters, told apart by the adapter weight's rank (`Conv2d` is 4-D)."""
    groups: dict[str, list[str]] = {}
    for name, tensor in snapshot.items():
        group = "conv" if tensor.dim() == 4 else "linear"
        groups.setdefault(group, []).append(name)
    return groups


def move_by_group(before: dict[str, Any], after: dict[str, Any], grouper: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, names in grouper(after).items():
        changed, max_delta, sq, ref = 0, 0.0, 0.0, 0.0
        for key in names:
            a, b = before.get(key), after[key]
            if a is None:
                continue
            if not bool((a == b).all()):
                changed += 1
            max_delta = max(max_delta, float((a.float() - b.float()).abs().max()))
            sq += float((a.float() - b.float()).double().pow(2).sum())
            ref += float(b.float().double().pow(2).sum())
        out[name] = {"keys": len(names), "changed": changed, "max_abs_delta": max_delta,
                     "relative_l2": (sq ** 0.5) / max(ref ** 0.5, 1e-12)}
    return out


def tier_reload(rep: Report, runs: dict[str, ProbeRun]) -> dict[str, Any]:
    tier = "reload"
    labelled = [(label, run) for label, run in runs.items() if run.final is not None]
    if not labelled:
        rep.check(tier, "a final checkpoint to reload", False, "no run produced one")
        return {}

    from trainer.config import TrainConfig, _load_toml_config
    from verify_mask_pipeline import Pipeline

    _, run_a = next((item for item in labelled if item[0] == "A"), labelled[0])
    cfg = TrainConfig.from_mapping(_load_toml_config(str(run_a.mirror_config)))
    print(f"      [{tier}] loading the base model once to reload the exported checkpoints", flush=True)
    pipeline = Pipeline(cfg)
    out: dict[str, Any] = {}
    try:
        # The adapter this pipeline built is the runs' own starting point (same seed, same call
        # order as trainer/main.py), captured before anything is loaded over it and quantized the
        # way the checkpoint file carries it, so the reload differs from the init by movement only.
        init_unet = quantize_like_file(dict(pipeline._init_lora))
        init_tes = [quantize_like_file(param_snapshot(te)) for te in pipeline.modules.text_encoders]
        for label, run in labelled:
            info = pipeline.load_checkpoint(run.final)
            rep.check(
                tier, f"{label}: the checkpoint reloads into this LoRA with nothing unmatched",
                int(info.get("loaded", 0)) > 0 and int(info.get("skipped", -1)) == 0,
                f"loaded={info.get('loaded')} skipped={info.get('skipped')} "
                f"step={info.get('step')} (file {run.final.parent.name})",
                loaded=info.get("loaded"), skipped=info.get("skipped"),
            )
            te_moved: dict[str, Any] = {}
            for index, (before, te) in enumerate(zip(init_tes, pipeline.modules.text_encoders), start=1):
                for group, item in move_by_group(before, param_snapshot(te), te_groups).items():
                    te_moved[f"te{index}-{group}"] = item
            unet_moved = move_by_group(init_unet, param_snapshot(pipeline.modules.denoise), unet_groups)
            if label == "A":
                ok = all(
                    item["keys"] > 0 and item["changed"] > 0 and item["relative_l2"] > MOVE_FLOOR
                    for item in list(te_moved.values()) + list(unet_moved.values())
                )
                rep.check(
                    tier, "A: every group of the exported file differs from the adapter's own init",
                    ok,
                    "; ".join(f"{name}: {item['changed']}/{item['keys']} tensors, "
                              f"max|Δ|={item['max_abs_delta']:.3g}, relL2={item['relative_l2']:.3g}"
                              for name, item in {**te_moved, **unet_moved}.items()),
                )
            else:
                rep.check(
                    tier, "B: the reloaded TE equals a freshly initialized adapter bit for bit",
                    all(item["changed"] == 0 and item["max_abs_delta"] == 0.0
                        for item in te_moved.values()),
                    "; ".join(f"{name}: {item['changed']}/{item['keys']} tensors differ, "
                              f"max|Δ|={item['max_abs_delta']:.3g}" for name, item in te_moved.items()),
                )
            rep.check(
                tier, f"{label}: reloading moves the UNet adapter into the file",
                all(item["changed"] > 0 for item in unet_moved.values()),
                "; ".join(f"{name}: {item['changed']}/{item['keys']} tensors changed, "
                          f"max|Δ|={item['max_abs_delta']:.3g}" for name, item in unet_moved.items()),
            )
            rep.observe(
                tier, f"{label}: distance from the freshly initialized adapter",
                "the file's values against the init this run started from, i.e. each group's whole "
                "training movement (not just its movement since the first step checkpoint)",
                te=te_moved, unet=unet_moved,
            )
            out[label] = {"loaded": info.get("loaded"), "skipped": info.get("skipped"),
                          "te": te_moved, "unet": unet_moved}
    finally:
        pipeline.release()
        del pipeline
    return out


# --------------------------------------------------------------------------------------
# cli / report
# --------------------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=24,
                        help="optimizer steps per run (the bucket sampler may round up)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="train_batch_size; config.toml uses 3")
    parser.add_argument("--images", type=int, default=9,
                        help="images copied out of the configured folders")
    parser.add_argument("--network-dim", type=int, default=8, help="LoCon rank (config.toml uses 32)")
    parser.add_argument("--network-alpha", type=int, default=4, help="config.toml uses 16")
    parser.add_argument("--conv-dim", type=int, default=4, help="LoCon conv rank (config.toml uses 16)")
    parser.add_argument("--conv-alpha", type=int, default=2, help="config.toml uses 8")
    parser.add_argument("--unet-lr", type=float, default=2e-4,
                        help="UNet LR (verify_mask_pipeline's measurement value; config.toml uses 5e-5)")
    parser.add_argument("--te-lr", type=float, default=2e-5,
                        help="TE LR for run A (config.toml uses 5e-6); run B always uses 0")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Schedule-Free warmup for both optimizers (config.toml uses 250)")
    parser.add_argument("--seed", type=int, default=None, help="defaults to config.toml's seed")
    parser.add_argument("--tiers", default="all", help="train,hook,weights,loss,reload | all")
    parser.add_argument("--no-hook", dest="hook", action="store_false",
                        help="run the children bare, without the [environment].amdfq patch")
    parser.add_argument("--runs", default="both", choices=("both", "main"),
                        help="'main' skips the frozen-TE control run (and its ownership checks)")
    parser.add_argument("--retries", type=int, default=2,
                        help="retries per run after the intermittent gfx1201 GPU memory fault")
    parser.add_argument("--report-dir", default=None, help="where the report is written")
    parser.add_argument("--work-dir", default=None, help="scratch dir (mirror, dataset copy, runs)")
    parser.add_argument("--clean", action="store_true", help="delete the scratch dir when done")
    parser.add_argument("--force", action="store_true", help="run even when a training run looks live")
    parser.add_argument("--allow-foreign-env", action="store_true",
                        help="skip the environment.yml conda env check")
    return parser.parse_args(argv)


def deviation_notes(args: argparse.Namespace) -> list[str]:
    """What the probe changes about config.toml, so a reader does not have to diff by hand."""
    sections = config_sections()
    training = sections.get("training", {})
    network = sections.get("network", {})
    unet = sections.get("unet_optimizer", {})
    te = sections.get("te_optimizer", {})
    pairs = [
        ("train_batch_size", training.get("train_batch_size"), args.batch_size),
        ("network_dim", network.get("network_dim"), args.network_dim),
        ("network_alpha", network.get("network_alpha"), args.network_alpha),
        ("conv_dim", network.get("conv_dim"), args.conv_dim),
        ("conv_alpha", network.get("conv_alpha"), args.conv_alpha),
        ("unet_learning_rate", unet.get("unet_learning_rate"), args.unet_lr),
        ("te_learning_rate", te.get("te_learning_rate"), args.te_lr),
        ("unet_warmup_steps", unet.get("unet_warmup_steps"), args.warmup),
        ("te_warmup_steps", te.get("te_warmup_steps"), args.warmup),
    ]
    notes = [
        f"{key}: config.toml {have} -> probe {want}"
        for key, have, want in pairs
        if have != want
    ]
    notes.append(
        "derived by the probe: one training folder (the configured ones copied into it), "
        "save_every_n_steps = steps/3, epoch = ceil(steps / ceil(images / batch)), "
        "max_data_loader_n_workers = 2, sampling_enabled = false (as in config.toml)"
    )
    return notes


def write_report(rep: Report, args: argparse.Namespace, details: dict[str, Any]) -> Path:
    out = rep.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": rep.env,
        "tiers": args.tiers,
        "options": {
            "runs": args.runs, "steps": args.steps, "batch_size": args.batch_size,
            "images": args.images, "seed": args.seed, "hook": args.hook,
            "network_dim": args.network_dim, "network_alpha": args.network_alpha,
            "conv_dim": args.conv_dim, "conv_alpha": args.conv_alpha,
            "unet_lr": args.unet_lr, "te_lr": args.te_lr, "warmup": args.warmup,
            "model": config_sections().get("environment", {}).get("pretrained_model_name_or_path"),
            "work_dir": str(args.work_dir),
        },
        "checks": [check.__dict__ for check in rep.checks],
        "observations": [obs.__dict__ for obs in rep.observations],
        "notes": rep.notes,
        "details": details,
        "failures": len(rep.failures),
        "seconds": round(time.time() - rep.started, 1),
    }
    (out / "probe_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    lines = [
        "# Optimizer probe (Schedule-Free UNet + TE)",
        "",
        f"- Interpreter `{rep.env.get('executable')}` (prefix `{rep.env.get('prefix')}`), "
        f"torch {rep.env.get('torch')} / HIP {rep.env.get('torch_hip')}, GPU {rep.env.get('gpu')}",
        f"- Tiers `{', '.join(args.tiers)}` | runs={args.runs}, steps={args.steps}, "
        f"batch={args.batch_size}, images={args.images}, seed={args.seed}",
        f"- LoCon rank {args.network_dim}/{args.network_alpha} (conv {args.conv_dim}/{args.conv_alpha}), "
        f"lrs {args.unet_lr}/{args.te_lr}, warmup {args.warmup}",
        f"- Wall clock {payload['seconds']:.0f}s | {len(rep.checks)} checks "
        f"({len(rep.failures)} failed) | {len(rep.observations)} observations",
        "",
        "## Checks",
        "",
        "| tier | check | result | measured |",
        "| --- | --- | --- | --- |",
    ]
    for check in rep.checks:
        measured = fmt_values(check.values).strip(" ()").replace("; ", " ")
        lines.append(f"| {check.tier} | {check.name} | {'PASS' if check.ok else '**FAIL**'} | {measured} |")
    lines += ["", "## Observations (measured, not asserted)", ""]
    for obs in rep.observations:
        lines += [f"### [{obs.tier}] {obs.name}", "", obs.detail, "", "```json",
                  json.dumps(obs.values, indent=2, ensure_ascii=False, default=str), "```", ""]
    if details:
        lines += ["## Detail per tier", "", "```json",
                  json.dumps(details, indent=2, ensure_ascii=False, default=str), "```", ""]
    if rep.notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in rep.notes] + [""]
    path = out / "probe_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: Iterable[str] | None = None) -> int:
    os.chdir(REPO_ROOT)  # config.toml is read relative to the repo root
    args = parse_args(argv)
    tiers = list(TIERS if args.tiers == "all" else (t.strip() for t in args.tiers.split(",") if t.strip()))
    for tier in tiers:
        if tier not in TIERS:
            raise SystemExit(f"unknown tier: {tier}")
    report_dir = (Path(args.report_dir) if args.report_dir
                  else Path("/tmp/axl-probe-optimizer") / time.strftime("%Y%m%d_%H%M%S"))
    work = Path(args.work_dir) if args.work_dir else report_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.report_dir, args.work_dir, args.tiers = report_dir, work, tiers
    rep = Report(report_dir)
    print(f"== optimizer probe | tiers={','.join(tiers)} | report={report_dir}", flush=True)

    guard_environment(args, rep)
    live = live_training_runs()
    if live and not args.force:
        raise SystemExit("refusing to run with a live training process (pass --force to override):\n  "
                         + "\n  ".join(live))
    if live:
        rep.note(f"--force used while these looked live: {live}")
    if args.seed is None:
        args.seed = int(config_sections().get("training", {}).get("seed", 1234))
    if args.hook:
        from trainer.amdfq_patch import launch_env_line

        try:
            line = launch_env_line()
        except FileNotFoundError as exc:
            raise SystemExit(f"refusing to run: {exc}")
        choice = line.split("|", 1)[0]
        if choice not in ("tail", "vmm"):
            raise SystemExit(
                f"refusing to run: config.toml selects amdfq={choice!r}, so the patch this probe "
                "preloads would be nothing; set [environment].amdfq to vmm (or pass --no-hook to "
                "run bare on purpose)"
            )
        rep.note(f"children run through start_hook.sh: {line}")

    rep.note("Read-only on the dataset: images are copied into the scratch dir, and every run gets "
             "its own AXL_RUNTIME_DIR / output_dir / logging_dir.")
    for note in deviation_notes(args):
        rep.note(note)
    if str(work.resolve()).startswith(("/tmp", "/dev/shm")):
        rep.note(f"the scratch dir {work} is on tmpfs (RAM-backed); pass --clean to drop it "
                 f"or --work-dir to put the mirrors and checkpoints on a disk")

    details: dict[str, Any] = {}
    runs: dict[str, ProbeRun] = {}
    try:
        if "train" in tiers:
            details["train"], runs = tier_train(rep, args, work)
        if "hook" in tiers:
            if runs:
                details["hook"] = tier_hook(rep, runs, args)
            else:
                rep.note("--tiers without `train`: the hook tier reads the children's logs")
        if "loss" in tiers:
            if "A" in runs:
                details["loss"] = tier_loss(rep, runs["A"])
            else:
                rep.note("--tiers without `train`: the loss tier needs run A's scalars")
        if "weights" in tiers:
            details["weights"] = tier_weights(rep, runs, args)
        if "reload" in tiers:
            if runs:
                details["reload"] = tier_reload(rep, runs)
            else:
                rep.note("--tiers without `train`: the reload tier needs a run's checkpoint")
    finally:
        report_path = write_report(rep, args, details)

    print("")
    print(f"== {len(rep.checks)} checks, {len(rep.failures)} failed | report: {report_path}", flush=True)
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)
    print("DONE" if not rep.failures else "FAILED", flush=True)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
