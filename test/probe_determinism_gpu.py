#!/usr/bin/env python3
"""GPU probe: does one fixed seed reproduce a training run bit for bit?

Run from the repo root in the conda env named by `environment.yml`:

    conda activate axl
    python test/probe_determinism_gpu.py

Proposition under test (written down before the runs; `PREDICTION` repeats it in the report): with
two configs that differ only in the three identity keys -- `output_dir`, `logging_dir`,
`output_name` -- two runs of the real `trainer/main.py`, on the same dataset copy, seeded from the
same `[training].seed`, do NOT log the same `Train/Loss` series: the first steps agree and the
series splits at some step. Deciding measurement: the `Train/Loss` scalar TensorBoard writes at each
step, compared as its exact float32 bytes (`struct.pack('<f', value)`). Two runs that log the same
bytes at every step refute the proposition.

Runs (each gets its own mirror, `AXL_RUNTIME_DIR`, `output_dir` and `logging_dir`; the dataset copy
is shared, so every run reads the same files):

    W    warms the dataset's `.latents_cache`, so the measured runs read it instead of encoding
    A1   first measured run
    A2   second measured run -- same config, same seed, same dataset, same cache
    --repeats 3 adds A3, a third sample of the same arm (two runs that agree cannot tell
    "reproducible" from "happened to agree")

Everything that has to hold before the proposition can be read is checked as an `instrument` check:

  * the mirror configs, flattened the way `trainer/config.py` flattens them, differ only in the
    three identity keys -- a one-variable comparison;
  * each measured run was launched by this invocation. The shared launcher reuses a completed run
    found in a work dir it has seen before (`done.json`), which would compare a run with itself, so
    the child log's mtime is checked against the probe's own start time;
  * the cache file count does not move between the measured runs (both read W's cache);
  * every logged scalar is exactly a float32, so the byte comparison is not hiding a wider value;
  * the runs log the same steps and the same LR series (`UNet/LR/Effective_Actual_LR` is a pure
    function of step, lr and warmup, so a difference there would mean the arms were not configured
    the same).

Scope -- what this probe does not decide:

  * which operation introduces the difference. It locates the first divergent step; it does not
    attribute the split to attention backend selection, reduction order, allocator addresses or
    anything else. That attribution is a separate experiment;
  * whether the weights diverge (the `weights` tier reports tensor payload hashes and per-tensor
    bit identity for the final checkpoints of every pair, as an observation);
  * anything about a `config.toml` other than the one this machine has: the scratch config is built
    from the repo's own sections plus the probe's overrides, and the dataset is a small copy of the
    configured folders.

Nothing is written into the dataset: images are copied into the scratch directory first (the latent
cache W writes into that copy belongs to the copy). Pass `--clean` to delete the scratch dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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
TIERS = ("runs", "hook", "instrument", "repro", "weights")
RUN_TIER = "determinism"
# The keys two arms of this probe are allowed to disagree on: they name where run state goes, and
# the same value in both runs would make the two runs write over each other.
IDENTITY_KEYS = ("output_dir", "logging_dir", "output_name")
SERIES = (
    ("loss", "Train/Loss"),
    ("avg", "Train/Avg_Loss"),
    ("unet_lr", "UNet/LR/Effective_Actual_LR"),
    ("te_lr", "TE/LR/Effective_Actual_LR"),
)
PREDICTION = (
    "The same seed does not reproduce the loss series: the runs agree at the start and split at "
    "some later step, so the `repro` check fails and the byte comparison names the first step where "
    "the two runs' Train/Loss values differ."
)


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


def copy_images(source: Path, dest: Path, count: int) -> tuple[int, int]:
    """Up to `count` images and their sidecars from `source` into `dest`: (placed, already there).

    A file the destination already holds is left as it is. Copying it under a prefixed name instead
    would silently enlarge the dataset — and so the epoch length and the step count — which is how a
    first version of this probe turned a 9-image dataset into an 18-image one and made its runs
    incomparable with the ones they were meant to repeat.
    """
    names = sorted(
        path
        for path in source.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXT
        and not path.name.lower().endswith(".mask.png")
    )
    placed = reused = 0
    for path in names:
        if placed + reused >= count:
            break
        if (dest / path.name).exists():
            reused += 1
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        sidecar = path.with_suffix(".txt")
        if sidecar.is_file():
            shutil.copy2(sidecar, target.with_suffix(".txt"))
        placed += 1
    return placed, reused


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
    pairs = [copy_images(source, dest, want) for source, want in zip(sources, quota)]
    counts = {str(source): placed for (placed, _), source in zip(pairs, sources)}
    copied = sum(placed for placed, _ in pairs)
    reused = sum(kept for _, kept in pairs)
    if copied + reused < 2:
        raise SystemExit(f"only {copied + reused} image(s) placed in {dest}; two are the minimum to train")
    sidecars = len(list(dest.glob("*.txt")))
    return dest, {"dir": str(dest), "images": copied + reused, "copied": copied, "reused": reused,
                  "captions": sidecars, "sources": counts}


def cache_files(data_dir: Path) -> int:
    """How many latents the dataset folder's cache holds (the instrument for "who encoded it")."""
    cache = data_dir / ".latents_cache"
    return len(list(cache.glob("*.pt"))) if cache.is_dir() else 0


# --------------------------------------------------------------------------------------
# the runs
# --------------------------------------------------------------------------------------


@dataclass
class ProbeRun:
    name: str
    result: Any
    output_name: str
    mirror_config: Path
    cache_before: int = 0
    cache_after: int = 0
    series: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    final: Path | None = None
    step_paths: dict[int, Path] = field(default_factory=dict)


def read_state(runtime_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((runtime_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_cache_plan(spec: str, count: int) -> list[str]:
    """`--cache-plan clear,keep` — what to do with the dataset's latent cache before each arm."""
    if not spec.strip():
        return ["keep"] * count
    parts = [part.strip().lower() for part in spec.split(",") if part.strip()]
    for part in parts:
        if part not in ("clear", "keep"):
            raise SystemExit(f"--cache-plan takes `clear` or `keep` per arm, not {part!r}")
    if len(parts) > count:
        raise SystemExit(f"--cache-plan names {len(parts)} arm(s) but this run has {count}")
    return parts + ["keep"] * (count - len(parts))


def clear_cache(data_dir: Path) -> int:
    """Drop the dataset folder's cached latents, so the next run has to encode them itself."""
    cache = data_dir / ".latents_cache"
    files = sorted(cache.glob("*.pt")) if cache.is_dir() else []
    for path in files:
        path.unlink()
    return len(files)


def launch(rep: Report, args: argparse.Namespace, *, name: str, data_dir: Path, work: Path,
           clear: bool = False) -> ProbeRun:
    extra = {
        # The rank reduction the probe asks for; `[network]` is the one section the shared launcher
        # does not know about.
        "network": {
            "network_dim": args.network_dim,
            "network_alpha": args.network_alpha,
            "conv_dim": args.conv_dim,
            "conv_alpha": args.conv_alpha,
        },
        # Sampling is off: this probe is about the training step's arithmetic, and a sample pass in
        # the middle of it would add its own RNG use and its own GPU work.
        "training": {"sampling_enabled": False},
    }
    if args.network_dropout is not None:
        extra["network"]["network_dropout"] = args.network_dropout
    if args.noise_offset is not None:
        extra["optimization"] = {"noise_offset": args.noise_offset}
    if clear:
        cleared = clear_cache(data_dir)
        rep.note(f"{name}: cleared {cleared} cached latent(s) before launching (this run encodes)")
    before = cache_files(data_dir)
    result = launch_run(
        name=name, data_dir=data_dir, seed=args.seed, steps=args.steps, work=work,
        masked=False, tier=RUN_TIER, batch_size=args.batch_size,
        save_every_override=max(1, args.steps // 3), lr=(args.unet_lr, args.te_lr),
        warmup=args.warmup, retries=args.retries, extra_sections=extra, hook=args.hook,
    )
    output_name = f"verify_{name}"
    run = ProbeRun(
        name=name, result=result, output_name=output_name,
        mirror_config=Path(result.claims.get("mirror_config") or (result.run_root / "mirror" / "config.toml")),
        cache_before=before, cache_after=cache_files(data_dir),
    )
    from trainer.checkpoints import discover_checkpoints

    run.checkpoints = discover_checkpoints(result.output_dir, output_name)
    finals = [item for item in run.checkpoints if item.get("final")]
    run.final = Path(finals[0]["path"]) if finals else None
    for item in run.checkpoints:
        if item.get("step") is not None:
            run.step_paths[int(item["step"])] = Path(item["path"])
    run.state = read_state(result.runtime_dir)
    if getattr(args, "snapshot_cache", False):
        # Keep what this run's warm pass wrote: the files are the only record of an encode's own
        # output, and the next arm (a `clear` plan) deletes them.
        snap = Path(work) / "cache_snapshots" / name
        snap.mkdir(parents=True, exist_ok=True)
        for path in (data_dir / ".latents_cache").glob("*.pt"):
            shutil.copy2(path, snap / path.name)
        rep.note(f"{name}: snapshotted {len(list(snap.glob('*.pt')))} cached latent(s) to {snap}")
    for key, tag in SERIES:
        run.series[key] = tb_scalars(result.logging_dir, tag)
    return run


def tier_runs(rep: Report, args: argparse.Namespace, work: Path) -> tuple[dict[str, Any], dict[str, ProbeRun]]:
    data_dir, dataset = prepare_dataset(work, args.images)
    rep.note(
        f"dataset: {dataset['images']} image(s) ({dataset['copied']} copied, {dataset['reused']} "
        f"already there) + {dataset['captions']} caption(s) from "
        + ", ".join(f"{path} x{count}" for path, count in dataset["sources"].items())
        + f" into {data_dir}"
    )
    suffix = getattr(args, "suffix", "")
    names = (["W"] if args.warm else []) + [f"A{index}{suffix}" for index in range(1, args.repeats + 1)]
    measured = [name for name in names if not name.startswith("W")]
    plan = parse_cache_plan(getattr(args, "cache_plan", ""), len(measured))
    runs: dict[str, ProbeRun] = {}
    for name in names:
        clear = name in measured and plan[measured.index(name)] == "clear"
        run = launch(rep, args, name=name, data_dir=data_dir, work=work, clear=clear)
        runs[name] = run
        rep.note(
            f"{name}: {run.result.seconds:.0f}s, cache {run.cache_before} -> {run.cache_after} "
            f"file(s), run dir {run.result.run_dir}"
        )
    expected_steps = max(1, int(runs[names[0]].result.claims.get("steps_requested", args.steps)))
    for name, run in runs.items():
        rep.check(
            RUN_TIER, f"{name}: the run finishes and writes its checkpoints",
            run.result.returncode == 0 and run.final is not None and bool(run.step_paths),
            f"exit={run.result.returncode} in {run.result.seconds:.0f}s | final="
            f"{run.final.name if run.final else None} | step checkpoints "
            f"{sorted(run.step_paths)} | logged {len(run.series['loss'])} step(s)",
            gpu_fault=bool(run.result.gpu_fault), attempts=run.result.attempts,
        )
        steps = [step for step, _ in run.series["loss"]]
        rep.check(
            RUN_TIER, f"{name}: every training step logs loss and both LRs, contiguously",
            steps == list(range(1, len(steps) + 1)) and bool(steps)
            and all([step for step, _ in run.series[key]] == steps for key, _ in SERIES[1:]),
            f"{len(steps)} step(s) 1..{len(steps)}, {len(run.series['avg'])} with Train/Avg_Loss",
        )
        rep.check(
            RUN_TIER, f"{name}: state.json agrees with the last logged step",
            run.state.get("status") == "finished"
            and int((run.state.get("training") or {}).get("step", -1)) == len(steps),
            f"status={run.state.get('status')} "
            f"step={(run.state.get('training') or {}).get('step')} loss="
            f"{(run.state.get('training') or {}).get('loss')}",
        )
    details = {"dataset": dataset, "runs": {name: describe_run(run) for name, run in runs.items()}}
    return details, runs


def describe_run(run: ProbeRun) -> dict[str, Any]:
    return {
        "output_dir": str(run.result.output_dir),
        "logging_dir": str(run.result.logging_dir),
        "runtime_dir": str(run.result.runtime_dir),
        "run_dir": str(run.result.run_dir) if run.result.run_dir else None,
        "log": str(run.result.log_path),
        "mirror_config": str(run.mirror_config),
        "checkpoints": [item["dir"] for item in run.checkpoints],
        "cache_before": run.cache_before,
        "cache_after": run.cache_after,
        "seconds": round(run.result.seconds, 1),
        "steps": len(run.series["loss"]),
        "state": run.state.get("training", {}),
    }


# --------------------------------------------------------------------------------------
# tier: hook (did the children really run under the [environment].amdfq patch?)
# --------------------------------------------------------------------------------------


def tier_hook(rep: Report, runs: dict[str, ProbeRun], args: argparse.Namespace) -> dict[str, Any]:
    tier = "hook"
    if not args.hook:
        rep.note("--no-hook: the children ran bare, without the [environment].amdfq patch")
        return {}
    from trainer.amdfq_patch import resolve_preload

    info = resolve_preload()
    out: dict[str, Any] = {"resolved": dict(info)}
    for name, run in runs.items():
        text = run.result.log_path.read_text(encoding="utf-8", errors="replace")
        banner = next((line.strip() for line in text.splitlines() if "start_hook.sh: amdfq=" in line), None)
        rep.check(
            tier, f"{name}: start_hook.sh preloaded the patch into the trainer",
            banner is not None and f"amdfq={info['choice']}" in banner
            and info["so"] is not None and info["so"] in banner,
            banner or "no start_hook.sh banner in the child's log",
        )
        rep.check(
            tier, f"{name}: the run stayed on the normal allocator path",
            not run.result.gpu_fault and run.result.attempts == 1,
            f"attempts={run.result.attempts} gfx1201_fault={run.result.gpu_fault}",
        )
        out[name] = {"banner": banner, "attempts": run.result.attempts,
                     "gpu_fault": run.result.gpu_fault}
    return out


# --------------------------------------------------------------------------------------
# tier: instrument (was the comparison one variable, between two runs that really ran?)
# --------------------------------------------------------------------------------------


def compare_configs(left: Path, right: Path) -> dict[str, Any]:
    """Flattened key/value diff of two generated configs (the repo's own flattening rule)."""
    from trainer.config import _load_toml_config

    a, b = _load_toml_config(str(left)), _load_toml_config(str(right))
    keys = sorted(set(a) | set(b))
    differing = [key for key in keys if a.get(key) != b.get(key)]
    return {
        "keys": len(keys),
        "differing": differing,
        "unexpected": [key for key in differing if key not in IDENTITY_KEYS],
    }


def float32_bytes(value: float) -> bytes:
    """The exact bytes of a logged scalar: TensorBoard stores `simple_value` as a float32."""
    return struct.pack("<f", value)


def is_exact_float32(value: float) -> bool:
    return struct.unpack("<f", struct.pack("<f", value))[0] == value


def compare_series(left: list[tuple[int, float]], right: list[tuple[int, float]]) -> dict[str, Any]:
    """How two per-step scalar series relate, bit for bit."""
    pairs = list(zip(left, right))
    equal = [(a, b) for a, b in pairs if a[0] == b[0] and float32_bytes(a[1]) == float32_bytes(b[1])]
    first = next((index for index, (a, b) in enumerate(pairs) if float32_bytes(a[1]) != float32_bytes(b[1])),
                 None)
    diffs = [abs(a[1] - b[1]) for a, b in pairs]
    rel = [(abs(a[1] - b[1]) / abs(a[1])) if a[1] else 0.0 for a, b in pairs]
    out: dict[str, Any] = {
        "steps_left": len(left),
        "steps_right": len(right),
        "steps_compared": len(pairs),
        "steps_bit_identical": len(equal),
        "first_differing_index": first,
        "first_differing_step": pairs[first][0][0] if first is not None else None,
        "max_abs_diff": max(diffs, default=0.0),
        "max_rel_diff": max(rel, default=0.0),
    }
    if first is not None:
        a, b = pairs[first]
        out["first_differing"] = {
            "step": a[0],
            "left": a[1], "left_hex": float32_bytes(a[1]).hex(),
            "right": b[1], "right_hex": float32_bytes(b[1]).hex(),
            "abs_diff": abs(a[1] - b[1]),
            "rel_diff": (abs(a[1] - b[1]) / abs(a[1])) if a[1] else 0.0,
        }
        out["abs_diff_at_first"] = abs(a[1] - b[1])
        out["abs_diff_at_last"] = abs(pairs[-1][0][1] - pairs[-1][1][1])
        out["identical_prefix"] = first == len(equal)
    return out


def tier_instrument(rep: Report, args: argparse.Namespace, runs: dict[str, ProbeRun],
                    started: float) -> dict[str, Any]:
    tier = "instrument"
    measured = [name for name in runs if name.startswith("A")]
    out: dict[str, Any] = {}

    # 1. One variable: the configs differ only in where each run writes.
    for index, left in enumerate(measured):
        for right in measured[index + 1:]:
            diff = compare_configs(runs[left].mirror_config, runs[right].mirror_config)
            rep.check(
                tier, f"{left} vs {right}: the mirror configs differ only in the identity keys",
                not diff["unexpected"],
                f"{len(diff['differing'])} of {diff['keys']} flattened key(s) differ: "
                f"{', '.join(diff['differing']) or 'none'}",
                **{f"unexpected_{n}": key for n, key in enumerate(diff["unexpected"])},
            )

    # 2. The runs really ran now. The shared launcher reuses a completed run it finds in a work
    #    dir, which would hand back a previous run's artifacts and compare a run with itself.
    for name in measured:
        log = runs[name].result.log_path
        mtime = log.stat().st_mtime if log.is_file() else 0.0
        rep.check(
            tier, f"{name}: this invocation launched the run (not a reused one)",
            mtime >= started - 1.0,
            f"log mtime {time.strftime('%H:%M:%S', time.localtime(mtime))} vs probe start "
            f"{time.strftime('%H:%M:%S', time.localtime(started))}"
            if mtime else "no child log",
        )

    # 3. Both measured runs read a warm cache: neither encoded the latents itself.
    if not args.warm:
        rep.note("--no-warm: no warming run, so the first measured run wrote the cache the second "
                 "read -- a known asymmetry of this arm")
    else:
        counts = [runs[name].cache_before for name in measured]
        rep.check(
            tier, "the measured runs started with the latents already cached",
            bool(counts) and all(count > 0 for count in counts),
            f"cache file count at launch: "
            + ", ".join(f"{name}={runs[name].cache_before}" for name in measured),
        )
        for name in measured:
            run = runs[name]
            rep.check(
                tier, f"{name}: the run read the cache instead of encoding",
                run.cache_after == run.cache_before,
                f"cache {run.cache_before} -> {run.cache_after} file(s): "
                f"{'unchanged' if run.cache_after == run.cache_before else 'the run wrote new latents'}",
            )

    # 4. Every compared value is exactly a float32, so byte equality is the value's own resolution.
    values = [(name, key, value) for name, run in runs.items() for key, _ in SERIES
              for _, value in run.series[key]]
    bad = [(name, key, value) for name, key, value in values if not is_exact_float32(value)]
    rep.check(
        tier, "every logged scalar is exactly representable as a float32",
        not bad,
        f"{len(values)} scalar(s) checked, {len(bad)} not a float32"
        + (f", e.g. {bad[0]}" if bad else ""),
    )

    # 5. The arms did the same work: same steps, same LR series.
    for index, left in enumerate(measured):
        for right in measured[index + 1:]:
            same_steps = [step for step, _ in runs[left].series["loss"]] == \
                [step for step, _ in runs[right].series["loss"]]
            lr_same = all(
                compare_series(runs[left].series[key], runs[right].series[key])["steps_bit_identical"]
                == len(runs[left].series[key])
                for key, _ in SERIES[2:]
            )
            rep.check(
                tier, f"{left} vs {right}: same step count and bit-identical LR series",
                same_steps and lr_same,
                f"steps {len(runs[left].series['loss'])} vs {len(runs[right].series['loss'])} | "
                f"LR series {'identical' if lr_same else 'DIFFER'}",
            )

    out["config_diffs"] = {
        f"{left} vs {right}": compare_configs(runs[left].mirror_config, runs[right].mirror_config)
        for index, left in enumerate(measured) for right in measured[index + 1:]
    }
    out["torch_state"] = torch_state()
    return out


def torch_state() -> dict[str, Any]:
    """The process-wide switches that decide which kernels the child's math runs on."""
    import torch

    cuda = torch.backends.cuda
    return {
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "bf16_reduced_precision_reduction": bool(
            getattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", None)
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "flash_sdp": bool(cuda.flash_sdp_enabled()),
        "mem_efficient_sdp": bool(cuda.mem_efficient_sdp_enabled()),
        "math_sdp": bool(cuda.math_sdp_enabled()),
    }


# --------------------------------------------------------------------------------------
# tier: repro (the proposition)
# --------------------------------------------------------------------------------------


def tier_repro(rep: Report, runs: dict[str, ProbeRun], args: argparse.Namespace) -> dict[str, Any]:
    tier = "repro"
    measured = [name for name in runs if name.startswith("A")]
    out: dict[str, Any] = {"pairs": {}}
    for index, left in enumerate(measured):
        for right in measured[index + 1:]:
            pair = f"{left} vs {right}"
            out["pairs"][pair] = {
                key: compare_series(runs[left].series[key], runs[right].series[key])
                for key, _ in SERIES
            }
    first_pair = f"{measured[0]} vs {measured[1]}"
    loss = out["pairs"][first_pair]["loss"]
    rep.check(
        tier, f"{first_pair}: the same seed reproduces Train/Loss byte for byte at every step",
        loss["steps_bit_identical"] == loss["steps_compared"] and loss["steps_compared"] > 0,
        f"{loss['steps_bit_identical']}/{loss['steps_compared']} step(s) bit-identical, first "
        f"difference at step {loss['first_differing_step']}"
        + (f" ({loss['first_differing']['left_hex']} vs {loss['first_differing']['right_hex']})"
           if "first_differing" in loss else "")
        + f" | max |Δ| {loss['max_abs_diff']:.3g}",
    )
    avg = out["pairs"][first_pair]["avg"]
    rep.observe(
        tier, f"{first_pair}: Train/Avg_Loss",
        "the same comparison for the windowed average the Dashboard draws",
        **avg,
    )
    for pair, series in out["pairs"].items():
        rep.observe(
            tier, f"{pair}: Train/Loss",
            "bit-for-bit comparison of the two runs' per-step loss scalars",
            **series["loss"],
        )
    # The state.json loss is the last step's value, written from the same tensor the logger saw.
    for pair in out["pairs"]:
        left, right = pair.split(" vs ")
        a = (runs[left].state.get("training") or {}).get("loss")
        b = (runs[right].state.get("training") or {}).get("loss")
        out.setdefault("state_loss", {})[pair] = {"left": a, "right": b, "equal": a == b}
        rep.observe(tier, f"{pair}: state.json loss", "the last step's loss as the trainer published it",
                    left=a, right=b, equal=a == b)
    return out


# --------------------------------------------------------------------------------------
# tier: weights (did the files diverge too?)
# --------------------------------------------------------------------------------------


def tensor_bytes(tensor: Any) -> bytes:
    import torch

    return tensor.detach().to("cpu").contiguous().flatten().view(torch.uint8).numpy().tobytes()


def payload_sha(path: Path) -> str:
    """sha256 of a `.safetensors` file's tensor payload: the bytes after the JSON header.

    Whole-file hashes cannot be compared: the header carries `modelspec.date`.
    """
    blob = path.read_bytes()
    header = struct.unpack("<Q", blob[:8])[0]
    return hashlib.sha256(blob[8 + header:]).hexdigest()


def compare_files(left: Path, right: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    with safe_open(str(left), framework="pt") as a, safe_open(str(right), framework="pt") as b:
        keys_a, keys_b = list(a.keys()), list(b.keys())
        identical, differing, worst = 0, [], {"key": None, "max_abs_diff": 0.0}
        for key in keys_a:
            if key not in keys_b:
                continue
            ta, tb = a.get_tensor(key), b.get_tensor(key)
            if ta.shape != tb.shape or ta.dtype != tb.dtype:
                differing.append(key)
                continue
            if tensor_bytes(ta) == tensor_bytes(tb):
                identical += 1
                continue
            differing.append(key)
            if ta.is_floating_point():
                diff = float((ta.to(torch.float32) - tb.to(torch.float32)).abs().max())
                if diff > worst["max_abs_diff"]:
                    worst = {"key": key, "max_abs_diff": diff}
    return {
        "left": str(left), "right": str(right),
        "keys_left": len(keys_a), "keys_right": len(keys_b),
        "tensors_bit_identical": identical,
        "tensors_differing": len(differing),
        "first_differing_keys": differing[:5],
        "largest_difference": worst,
        "payload_sha_left": payload_sha(left),
        "payload_sha_right": payload_sha(right),
        "payload_identical": payload_sha(left) == payload_sha(right),
    }


def tier_weights(rep: Report, runs: dict[str, ProbeRun], args: argparse.Namespace) -> dict[str, Any]:
    tier = "weights"
    measured = [name for name in runs if name.startswith("A")]
    out: dict[str, Any] = {}
    for index, left in enumerate(measured):
        for right in measured[index + 1:]:
            pair = f"{left} vs {right}"
            entry: dict[str, Any] = {}
            if runs[left].final and runs[right].final:
                entry["final"] = compare_files(runs[left].final, runs[right].final)
                rep.observe(
                    tier, f"{pair}: final checkpoint tensors",
                    "per-tensor bit identity and the payload hash of the two exports",
                    **entry["final"],
                )
            common_steps = sorted(set(runs[left].step_paths) & set(runs[right].step_paths))
            if common_steps:
                step = common_steps[0]
                entry[f"step_{step}"] = compare_files(runs[left].step_paths[step],
                                                      runs[right].step_paths[step])
                rep.observe(
                    tier, f"{pair}: checkpoint at step {step} tensors",
                    "the earliest checkpoint both runs wrote: how far apart the weights already are",
                    **entry[f"step_{step}"],
                )
            out[pair] = entry
    return out


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=24,
                        help="optimizer steps per run (the dataset rounds up to whole epochs)")
    parser.add_argument("--batch-size", type=int, default=2, help="train_batch_size; config.toml uses 3")
    parser.add_argument("--images", type=int, default=9,
                        help="images copied out of the configured folders")
    parser.add_argument("--repeats", type=int, default=2,
                        help="measured runs of the same arm (the user asked for 2; 3 tells "
                             "'reproducible' from 'happened to agree')")
    parser.add_argument("--suffix", default="",
                        help="appended to every run name, so a later invocation can add runs to a "
                             "work dir the shared launcher has already seen (it reuses a completed "
                             "run there, which compares a run with itself)")
    parser.add_argument("--network-dropout", type=float, default=None,
                        help="override [network].network_dropout for every run (0.0 disables it)")
    parser.add_argument("--noise-offset", type=float, default=None,
                        help="override [optimization].noise_offset for every run (0.0 disables it)")
    parser.add_argument("--cache-plan", default="",
                        help="comma-separated `clear`/`keep` per measured arm: what to do with the "
                             "dataset's latent cache before launching it. `clear,keep` makes the "
                             "first arm encode its own latents and the rest read them")
    parser.add_argument("--snapshot-cache", action="store_true",
                        help="copy the dataset's latent cache after every run, so what each encode "
                             "wrote can be compared byte for byte")
    parser.add_argument("--no-warm", dest="warm", action="store_false",
                        help="skip the cache-warming run (the first measured run then encodes)")
    parser.add_argument("--network-dim", type=int, default=8, help="LoCon rank (config.toml uses 32)")
    parser.add_argument("--network-alpha", type=int, default=4, help="config.toml uses 16")
    parser.add_argument("--conv-dim", type=int, default=4, help="LoCon conv rank (config.toml uses 16)")
    parser.add_argument("--conv-alpha", type=int, default=2, help="config.toml uses 8")
    parser.add_argument("--unet-lr", type=float, default=2e-4,
                        help="UNet LR (config.toml uses 1e-4)")
    parser.add_argument("--te-lr", type=float, default=2e-5,
                        help="TE LR (config.toml uses 8e-6)")
    parser.add_argument("--warmup", type=int, default=10, help="Schedule-Free warmup (config.toml uses 250)")
    parser.add_argument("--seed", type=int, default=None, help="defaults to config.toml's seed")
    parser.add_argument("--tiers", default="all", help="runs,hook,instrument,repro,weights | all")
    parser.add_argument("--no-hook", dest="hook", action="store_false",
                        help="run the children bare, without the [environment].amdfq patch")
    parser.add_argument("--retries", type=int, default=2,
                        help="retries per run after the intermittent gfx1201 GPU memory fault")
    parser.add_argument("--report-dir", default=None, help="where the report is written")
    parser.add_argument("--work-dir", default=None, help="scratch dir (mirrors, dataset copy, runs)")
    parser.add_argument("--clean", action="store_true", help="delete the scratch dir when done")
    parser.add_argument("--force", action="store_true", help="run even when a training run looks live")
    parser.add_argument("--allow-foreign-env", action="store_true",
                        help="skip the environment.yml conda env check")
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2: one run cannot be compared with anything")
    return args


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
    notes = [f"{key}: config.toml {have} -> probe {want}" for key, have, want in pairs if have != want]
    if args.seed is not None:
        notes.append(f"seed: config.toml {training.get('seed')} -> probe {args.seed}")
    notes.append(
        "derived by the probe: one training folder (the configured ones copied into it), "
        "sampling_enabled = false, save_every_n_steps = steps/3, "
        "epoch = ceil(steps / ceil(images / batch)), max_data_loader_n_workers = 2"
    )
    return notes


def write_report(rep: Report, args: argparse.Namespace, details: dict[str, Any], verdict: str) -> Path:
    out = rep.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": rep.env,
        "prediction": PREDICTION,
        "verdict": verdict,
        "tiers": args.tiers,
        "options": {
            "repeats": args.repeats, "warm": args.warm, "steps": args.steps,
            "batch_size": args.batch_size, "images": args.images, "seed": args.seed,
            "hook": args.hook, "network_dim": args.network_dim, "network_alpha": args.network_alpha,
            "conv_dim": args.conv_dim, "conv_alpha": args.conv_alpha, "unet_lr": args.unet_lr,
            "te_lr": args.te_lr, "warmup": args.warmup,
            "network_dropout": args.network_dropout, "cache_plan": args.cache_plan,
            "noise_offset": args.noise_offset,
            "suffix": getattr(args, "suffix", ""),
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
        "# Training-determinism probe (same seed twice, isolated real path)",
        "",
        f"- **Verdict**: {verdict}",
        f"- **Prediction (written before the runs)**: {PREDICTION}",
        f"- Interpreter `{rep.env.get('executable')}` (prefix `{rep.env.get('prefix')}`), "
        f"torch {rep.env.get('torch')} / HIP {rep.env.get('torch_hip')}, GPU {rep.env.get('gpu')}",
        f"- Tiers `{', '.join(args.tiers)}` | repeats={args.repeats}, warm={args.warm}, "
        f"steps={args.steps}, batch={args.batch_size}, images={args.images}, seed={args.seed}",
        f"- Wall clock {payload['seconds']:.0f}s | {len(rep.checks)} checks "
        f"({len(rep.failures)} failed) | {len(rep.observations)} observations",
        "",
        "The `repro` check is the proposition: it FAILS when the two runs' loss bytes differ, which "
        "is the prediction. `instrument`, `runs`, `hook` and `weights` report on the setup instead; "
        "a failure there means the comparison itself is not trustworthy.",
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
                  else Path("/tmp/axl-probe-determinism") / time.strftime("%Y%m%d_%H%M%S"))
    work = Path(args.work_dir) if args.work_dir else report_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.report_dir, args.work_dir, args.tiers = report_dir, work, tiers
    rep = Report(report_dir)
    started = time.time()
    print(f"== determinism probe | tiers={','.join(tiers)} | report={report_dir}", flush=True)
    print(f"   prediction: {PREDICTION}", flush=True)

    guard_environment(args, rep)
    live = live_training_runs()
    if live and not args.force:
        raise SystemExit("refusing to run with a live training process (pass --force to override):\n  "
                         + "\n  ".join(live))
    if live:
        rep.note(f"--force used while these looked live: {live}")
    if args.seed is None:
        args.seed = int(config_sections().get("training", {}).get("seed", 1234))
    rep.note(f"seed {args.seed} for every measured run (config.toml's value unless --seed)")
    if args.hook:
        from trainer.amdfq_patch import launch_env_line

        try:
            line = launch_env_line()
        except FileNotFoundError as exc:
            raise SystemExit(f"refusing to run: {exc}")
        if line.split("|", 1)[0] not in ("tail", "vmm"):
            raise SystemExit(
                "refusing to run: config.toml selects no amdfq patch, so the patch this probe "
                "preloads would be nothing; set [environment].amdfq to vmm (or pass --no-hook)"
            )
        rep.note(f"children run through start_hook.sh: {line}")
    rep.note("Read-only on the dataset: images are copied into the scratch dir, and every run gets "
             "its own AXL_RUNTIME_DIR / output_dir / logging_dir.")
    for note in deviation_notes(args):
        rep.note(note)

    details: dict[str, Any] = {}
    runs: dict[str, ProbeRun] = {}
    verdict = "not run"
    try:
        if "runs" in tiers:
            details["runs"], runs = tier_runs(rep, args, work)
        if "hook" in tiers:
            if runs:
                details["hook"] = tier_hook(rep, runs, args)
            else:
                rep.note("--tiers without `runs`: the hook tier reads the children's logs")
        if "instrument" in tiers:
            if runs:
                details["instrument"] = tier_instrument(rep, args, runs, started)
            else:
                rep.note("--tiers without `runs`: the instrument tier reads the children's output")
        if "repro" in tiers:
            if len(runs) >= 2:
                details["repro"] = tier_repro(rep, runs, args)
                measured = [name for name in runs if name.startswith("A")]
                # The pair key carries the run names, which `--suffix` can rename: naming the pair
                # literally here is how the verdict line once reported success over a failed check.
                first = f"{measured[0]} vs {measured[1]}" if len(measured) >= 2 else ""
                loss = details["repro"]["pairs"].get(first, {}).get("loss", {})
                split = loss.get("steps_bit_identical") != loss.get("steps_compared")
                verdict = (
                    f"the same seed is NOT reproducible: {loss.get('steps_bit_identical')}/"
                    f"{loss.get('steps_compared')} steps bit-identical, first difference at step "
                    f"{loss.get('first_differing_step')}" if split else
                    "the same seed reproduced every compared value byte for byte in this sample"
                )
            else:
                rep.note("--tiers without `runs`: the repro tier needs two runs")
        if "weights" in tiers:
            if len(runs) >= 2:
                details["weights"] = tier_weights(rep, runs, args)
            else:
                rep.note("--tiers without `runs`: the weights tier needs two runs")
    finally:
        report_path = write_report(rep, args, details, verdict)

    print("")
    print(f"== {len(rep.checks)} checks, {len(rep.failures)} failed | report: {report_path}", flush=True)
    print(f"== verdict: {verdict}", flush=True)
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)
    print("DONE" if not rep.failures else "FAILED", flush=True)
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
