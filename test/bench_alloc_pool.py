#!/usr/bin/env python3
"""Does the VMM hook's allocation pool make a training step faster?

Run from the repo root in the conda env named by `environment.yml`:

    python test/bench_alloc_pool.py --probe        # one short run: shakedown, warms the subset cache
    python test/bench_alloc_pool.py                # warm-up + baseline + 16/32/64/128/256 MiB + baseline repeat
    python test/bench_alloc_pool.py --plot-only    # re-render the report from the stored JSON

Each condition is the *same* workload, taken from the repo's `config.toml`: the same batch size,
resolution, bucketing, optimizers, precision, gradient checkpointing and save cadence, against one
shared subset of the configured dataset. Only the run-scoped keys differ (the subset directory, the
per-run output/log/runtime directories, the run name, the epoch count) plus the pool size under test.

Every run is a real `trainer/main.py` child under the hook: a mirror repo (symlinked trainer sources)
with a generated `config.toml`, `LD_PRELOAD` and `AMDFQ_*` resolved from that mirror's config through
`trainer/amdfq_patch.resolve_preload` — the same path `start_train.sh` takes. Step timing comes from
the run's own TensorBoard events (one scalar event per optimizer step, each with a wall time), which
is also where `api.py`'s dashboard reads them.

What one run reports
  step time   adjacent `Train/Avg_Loss` events' wall-time deltas; the first `--skip-first` steps are
              dropped as warm-up and any delta that carries a save+sample (the step after every
              `save_every_n_steps`) is dropped too, because that delta is a sampling cost, not a step
  vram        `/sys/class/drm/cardN/device/mem_info_vram_{used,total}` sampled every second while the
              child runs — the peak is what a pool costs the card
  hook        line counts from the child's stderr: allocations, frees, served/forwarded, the walk
              over `amdfq_vmm_rs::pool` lines (pools created/released, allocations served from one),
              `hipMemCreate` failures, and the allocator's own OOM warnings

Deviation from the literal `config.toml`, on purpose and identical in every condition: `epoch` is
capped (time), `train_data_dir` is a symlink subset (controlled shape mix, one shared latent cache),
the three output directories are per-run, `[[validation.samples]]` is dropped (the TOML writer cannot
express an array of tables, so the child renders its one flat `sample_*` set at the final save), and
this machine's `config.toml` switches are kept as they are.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# `test/` is sys.path[0] for this script, and the trainer imports need the repo root.
sys.path.insert(0, str(REPO_ROOT))

from verify_mask_pipeline import (  # noqa: E402
    child_env,
    config_sections,
    gpu_memory_fault,
    make_mirror,
    reap_run_processes,
    write_toml,
)
from trainer.amdfq_patch import resolve_preload  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_WORK_DIR = Path("/var/tmp/axl-pool-bench")
DEFAULT_POOL_SIZES = (16, 32, 64, 128, 256)
DEFAULT_SUBSET = 120
DEFAULT_EPOCHS = 3
DEFAULT_SKIP_FIRST = 20
# Step deltas a run needs before its median counts as a measurement. The configured workload yields
# ~180 of them, and a run that dies late (the allocator can abort the child near the VRAM limit) then
# still has one.
MIN_DELTAS = 120
ATTEMPTS = 2


# --------------------------------------------------------------------------------------
# dataset subset
# --------------------------------------------------------------------------------------


def image_files(data_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXT and not p.name.endswith(".mask.png")
    )


def build_subset(work: Path, data_dir: Path, count: int) -> tuple[Path, int]:
    """One symlink subset shared by every condition.

    The latent cache key is `sha1(abs_path::bucket::fit)`, so the subset has to have *one* path for
    all runs: then the first run encodes it and every later run reads the same `.latents_cache`.
    """
    subset = work / "dataset"
    images = image_files(data_dir)
    if not images:
        raise SystemExit(f"no images under {data_dir}")
    stride = max(1, len(images) // count)
    picked = images[::stride]
    if not subset.exists():
        subset.mkdir(parents=True)
        for img in picked:
            os.symlink(img.resolve(), subset / img.name)
            sidecar = img.with_suffix(".txt")
            if sidecar.is_file():
                os.symlink(sidecar.resolve(), subset / sidecar.name)
            mask = img.with_name(img.stem + ".mask.png")
            if mask.is_file():
                os.symlink(mask.resolve(), subset / mask.name)
    else:
        picked = image_files(subset)
    return subset, len(picked)


# --------------------------------------------------------------------------------------
# vram sampling (same card order as the hook: amdgpu cards with the counters, by cardN)
# --------------------------------------------------------------------------------------


@dataclass
class VramSampler:
    card: Optional[Path] = None
    interval: float = 1.0
    started: float = 0.0
    samples: list[tuple[float, int, int]] = field(default_factory=list)
    _last: float = 0.0

    def __post_init__(self) -> None:
        cards: list[tuple[int, Path]] = []
        for entry in sorted(Path("/sys/class/drm").glob("card*")):
            name = entry.name[len("card") :]
            if not name.isdigit():
                continue
            device = entry / "device"
            if (device / "mem_info_vram_total").is_file():
                cards.append((int(name), device))
        cards.sort(key=lambda pair: pair[0])
        if cards:
            self.card = cards[0][1]

    def poll(self, now: Optional[float] = None) -> None:
        if self.card is None:
            return
        now = time.time() if now is None else now
        if now - self._last < self.interval:
            return
        self._last = now
        try:
            used = int((self.card / "mem_info_vram_used").read_text().strip())
            total = int((self.card / "mem_info_vram_total").read_text().strip())
        except (OSError, ValueError):
            return
        self.samples.append((now - self.started, used, total))

    @property
    def peak(self) -> Optional[int]:
        return max((used for _, used, _ in self.samples), default=None)

    @property
    def mean(self) -> Optional[int]:
        if not self.samples:
            return None
        return int(sum(used for _, used, _ in self.samples) / len(self.samples))

    @property
    def total(self) -> Optional[int]:
        return self.samples[-1][2] if self.samples else None


# --------------------------------------------------------------------------------------
# what a run produced
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    name: str
    pool_mib: int
    measured: bool
    mirror: Path
    run_root: Path
    runtime_dir: Path
    log_path: Path
    logging_dir: Path
    output_dir: Path
    seconds: float = 0.0
    returncode: int = 0
    attempts: int = 1
    gpu_fault: bool = False
    run_id: Optional[str] = None
    steps_seen: int = 0
    step_times: list[tuple[int, float]] = field(default_factory=list)
    loss_series: list[tuple[int, float]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    hook: dict[str, Any] = field(default_factory=dict)
    vram: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "pool_mib": self.pool_mib,
            "measured": self.measured,
            "seconds": round(self.seconds, 1),
            "returncode": self.returncode,
            "attempts": self.attempts,
            "gpu_fault": self.gpu_fault,
            "run_id": self.run_id,
            "steps_seen": self.steps_seen,
            "step_times": [[step, round(delta, 4)] for step, delta in self.step_times],
            "loss_series": [[step, value] for step, value in self.loss_series],
            "stats": self.stats,
            "hook": self.hook,
            "vram": self.vram,
            "errors": self.errors,
            "mirror": str(self.mirror),
            "log": str(self.log_path),
        }
        return data

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "RunResult":
        """Rebuild enough of a run to render the report from the stored JSON (`--plot-only`)."""
        return cls(
            name=data["name"],
            pool_mib=int(data["pool_mib"]),
            measured=bool(data["measured"]),
            mirror=Path(data["mirror"]),
            run_root=Path(data["log"]).parent,
            runtime_dir=Path(data["log"]).parent / "runtime",
            log_path=Path(data["log"]),
            logging_dir=Path(data["log"]).parent / "logs",
            output_dir=Path(data["log"]).parent / "outputs",
            seconds=float(data.get("seconds", 0.0)),
            returncode=int(data.get("returncode", 0)),
            attempts=int(data.get("attempts", 1)),
            gpu_fault=bool(data.get("gpu_fault")),
            run_id=data.get("run_id"),
            steps_seen=int(data.get("steps_seen", 0)),
            step_times=[(int(step), float(delta)) for step, delta in data.get("step_times", [])],
            loss_series=[(int(step), float(value)) for step, value in data.get("loss_series", [])],
            stats=data.get("stats", {}),
            hook=data.get("hook", {}),
            vram=data.get("vram", {}),
            errors=list(data.get("errors", [])),
        )


def hook_counts(log_path: Path) -> dict[str, Any]:
    """Line counts out of the child's stderr. Only the pool's own lines move with the knob; the two
    allocation gates log one line per call either way."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    pool_lines = [line for line in text.splitlines() if "amdfq_vmm_rs::pool" in line]

    def count(needle: str) -> int:
        return text.count(needle)

    return {
        "hipMalloc": count("amdfq_vmm_rs::hooks hipMalloc("),
        "hipFree": count("amdfq_vmm_rs::hooks hipFree("),
        "served": count("ret=0 served "),
        "pooled": count("ret=0 pooled "),
        "forwarded": count("ret=0 forwarded "),
        "released": count("-> released"),
        "forwarded_free": count("forwarded size="),
        "untracked": count("untracked"),
        "duplicate": count("duplicate "),
        "create_failure": count("hipMemCreate("),
        "oom_warning": count("memory allocation failed with OOM"),
        "pool_lines": len(pool_lines),
        "pools_created": sum(1 for line in pool_lines if "created" in line),
        "pools_released": sum(1 for line in pool_lines if "released" in line),
        "pool_refused": sum(1 for line in pool_lines if "refused" in line),
    }


def read_step_events(logging_dir: Path) -> list[tuple[int, float, float]]:
    """`Train/Avg_Loss` events, ascending by step: one per optimizer step, each with a wall time and
    the loss the step produced (the loss is the pool's correctness probe — the training math must be
    bit-for-bit the same trajectory whatever the allocator did)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not logging_dir.is_dir():
        return []
    runs = sorted(
        (p for p in logging_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        return []
    accumulator = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0})
    accumulator.Reload()
    if "scalars" not in accumulator.Tags() or "Train/Avg_Loss" not in accumulator.Tags()["scalars"]:
        return []
    events = sorted(accumulator.Scalars("Train/Avg_Loss"), key=lambda e: e.step)
    return [(int(e.step), float(e.wall_time), float(e.value)) for e in events]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def step_statistics(
    events: list[tuple[int, float, float]], *, skip_first: int, cadence: int
) -> tuple[dict[str, Any], list[tuple[int, float]], list[tuple[int, float]]]:
    """Median/worst-case step time, with the warm-up steps and the sampling steps left out, plus the
    loss trajectory each step produced."""
    deltas: list[tuple[int, float]] = []
    losses: list[tuple[int, float]] = []
    for step, _, value in events:
        if step > skip_first:
            losses.append((step, value))
    for (prev_step, prev_time, _), (step, wall_time, _) in zip(events, events[1:]):
        if step <= skip_first:
            continue
        if cadence > 0 and prev_step % cadence == 0:
            # This delta spans the step checkpoint and its sample.
            continue
        delta = wall_time - prev_time
        if delta > 0:
            deltas.append((step, delta))
    times = [delta for _, delta in deltas]
    if not times:
        return {"n": 0}, deltas, losses
    median = statistics.median(times)
    return (
        {
            "n": len(times),
            "median_s": median,
            "p25_s": quantile(times, 0.25),
            "p75_s": quantile(times, 0.75),
            "min_s": min(times),
            "max_s": max(times),
            "mean_s": sum(times) / len(times),
            "steps_per_s": 1.0 / median,
            "loss_first": losses[0][1] if losses else None,
            "loss_last": losses[-1][1] if losses else None,
        },
        deltas,
        losses,
    )


# --------------------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------------------


def make_run_mirror(mirror: Path) -> None:
    if (mirror / "trainer").is_dir():
        return
    make_mirror(mirror)
    # `resolve_preload(root)` looks for the .so under `root/amdfq/...`, so the mirror needs the tree.
    os.symlink(REPO_ROOT / "amdfq", mirror / "amdfq")


def build_run_config(
    *, mirror: Path, subset: Path, run_root: Path, name: str, pool_mib: int, epochs: int
) -> dict[str, dict[str, Any]]:
    sections = config_sections()
    sections["environment"].update(
        {
            "train_data_dir": str(subset),
            "output_dir": str(run_root / "outputs"),
            "logging_dir": str(run_root / "logs"),
            "output_name": f"bench_{name}",
            "amdfq": "vmm",
            "amdfq_pool_mib": int(pool_mib),
        }
    )
    sections["training"]["epoch"] = epochs
    write_toml(mirror / "config.toml", sections)
    return sections


def run_once(
    *, name: str, pool_mib: int, measured: bool, work: Path, subset: Path,
    epochs: int, skip_first: int, quiet: bool, attempt: int,
) -> RunResult:
    run_root = work / "runs" / name
    mirror = run_root / "mirror"
    runtime_dir = run_root / "runtime"
    log_path = run_root / "train.out"
    for path in (runtime_dir, run_root / "outputs", run_root / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    make_run_mirror(mirror)
    sections = build_run_config(
        mirror=mirror, subset=subset, run_root=run_root, name=name, pool_mib=pool_mib, epochs=epochs
    )
    cadence = int(sections.get("training", {}).get("save_every_n_steps", 0) or 0)

    preload = resolve_preload(mirror)
    env = child_env(runtime_dir)
    env["AMDFQ_LOG_LEVEL"] = os.environ.get("AMDFQ_LOG_LEVEL", "info")
    if preload["choice"] in ("tail", "vmm"):
        env["LD_PRELOAD"] = str(preload["so"])
        if preload["choice"] == "vmm":
            if preload.get("va_status"):
                env["AMDFQ_VA_STATUS"] = str(preload["va_status"])
            if preload.get("vram_reserve") is not None:
                env["AMDFQ_VRAM_RESERVE"] = str(preload["vram_reserve"])
            if preload.get("va_never_reuse") is not None:
                env["AMDFQ_VA_NEVER_REUSE"] = str(preload["va_never_reuse"])
            if preload.get("pool"):
                env["AMDFQ_POOL_SIZE"] = str(preload["pool"])

    result = RunResult(
        name=name,
        pool_mib=pool_mib,
        measured=measured,
        mirror=mirror,
        run_root=run_root,
        runtime_dir=runtime_dir,
        log_path=log_path,
        logging_dir=run_root / "logs",
        output_dir=run_root / "outputs",
        attempts=attempt,
    )

    sampler = VramSampler()
    sampler.started = time.time()
    started = time.time()
    if not quiet:
        print(
            f"      [{name}] pool={pool_mib} MiB epochs={epochs} "
            f"LD_PRELOAD={Path(str(preload['so'])).name} AMDFQ_POOL_SIZE={env.get('AMDFQ_POOL_SIZE', '-')}",
            flush=True,
        )
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", "trainer/main.py"],
            cwd=mirror,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        state_path = runtime_dir / "state.json"
        last_step = None
        last_print = time.time()
        while proc.poll() is None:
            time.sleep(0.5)
            sampler.poll()
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            result.run_id = state.get("run_id") or result.run_id
            step = state.get("training", {}).get("step")
            if step is not None:
                last_step = step
            if not quiet and time.time() - last_print > 60:
                last_print = time.time()
                print(
                    f"      [{name}] step {last_step} loss={state.get('training', {}).get('loss')} "
                    f"({time.time() - started:.0f}s, vram_peak={(sampler.peak or 0) / 2**30:.2f} GiB)",
                    flush=True,
                )
        proc.wait()
    result.seconds = time.time() - started
    result.returncode = int(proc.returncode or 0)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    if result.run_id is None:
        marker = "bench_" + name
        import re

        found = re.search(rf"{re.escape(marker)}_(\d{{8}}_\d{{6}})", text)
        if found:
            result.run_id = f"{marker}_{found.group(1)}"

    events = read_step_events(result.logging_dir)
    result.steps_seen = events[-1][0] if events else 0
    result.stats, result.step_times, result.loss_series = step_statistics(
        events, skip_first=skip_first, cadence=cadence
    )
    result.hook = hook_counts(log_path)
    result.vram = {
        "peak_bytes": sampler.peak,
        "mean_bytes": sampler.mean,
        "total_bytes": sampler.total,
        "samples": len(sampler.samples),
    }
    result.gpu_fault = result.returncode != 0 and gpu_memory_fault(log_path)
    if result.returncode != 0:
        result.errors.append(f"exit {result.returncode}")
    if result.stats.get("n", 0) < MIN_DELTAS:
        result.errors.append(
            f"only {result.stats.get('n', 0)} step deltas ({result.steps_seen} steps reached), "
            f"need {MIN_DELTAS} for a median"
        )
    if not quiet:
        print(
            f"      [{name}] exit={result.returncode} in {result.seconds:.0f}s | "
            f"median {result.stats.get('median_s', float('nan')):.3f}s/step "
            f"({result.stats.get('steps_per_s', float('nan')):.3f} step/s, n={result.stats.get('n')}) | "
            f"vram_peak={(result.vram['peak_bytes'] or 0) / 2**30:.2f} GiB | "
            f"malloc {result.hook.get('hipMalloc')} pooled {result.hook.get('pooled')} "
            f"pools {result.hook.get('pools_created')}/{result.hook.get('pools_released')}"
            + ("".join(f" | {error}" for error in result.errors)),
            flush=True,
        )
    if result.returncode != 0:
        reaped = reap_run_processes(str(mirror))
        if reaped:
            result.errors.append(f"reaped {len(reaped)} orphaned worker processes")
    return result


def run_condition(**kwargs: Any) -> RunResult:
    """Up to `ATTEMPTS` identical attempts, so a run that cannot be measured does not delete its
    condition from the curve.

    Two ways a run gets here: the known gfx1201 Tensile fault kills a child now and then, and — more
    often on a 16 GB card — the workload's own allocator runs out of room and aborts the child. Both
    are properties of the workload, not of the pool, so the retry keeps the same environment: dodging
    the fault with `PYTORCH_NO_HIP_MEMORY_CACHING` or a smaller batch would change what is measured.
    A run that did reach enough steps to have a median is not retried, even if it then died — the
    median over those steps is a valid measurement, and the fact is recorded."""
    result = run_once(attempt=1, **kwargs)
    attempt = 1
    while result.stats.get("n", 0) < MIN_DELTAS and attempt < ATTEMPTS:
        attempt += 1
        why = "gfx1201 GPU memory fault" if result.gpu_fault else f"exit {result.returncode}"
        print(f"      [{result.name}] attempt {attempt - 1}: {why} before {MIN_DELTAS} deltas, "
              f"retrying", flush=True)
        result = run_once(attempt=attempt, **kwargs)
    return result


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def baseline_medians(runs: list[RunResult], *, session_only: bool = True) -> list[float]:
    """Baseline runs of the session that is being reported. A run read back from an earlier report
    (`--include-json`) has its name prefixed `pre_`: it is the "before the change" number, and it is
    compared against rather than folded into this session's noise band."""
    return [
        run.stats["median_s"]
        for run in runs
        if run.measured
        and run.pool_mib == 0
        and run.stats.get("median_s")
        and run.stats.get("n", 0) >= MIN_DELTAS
        and not (session_only and run.name.startswith("pre_"))
    ]


def pre_change_medians(runs: list[RunResult]) -> list[float]:
    return [
        run.stats["median_s"]
        for run in runs
        if run.measured
        and run.pool_mib == 0
        and run.stats.get("median_s")
        and run.stats.get("n", 0) >= MIN_DELTAS
        and run.name.startswith("pre_")
    ]


def baseline_band(runs: list[RunResult]) -> Optional[tuple[float, float]]:
    """Baseline step time interval between the runs' own 25th and 75th percentiles: the width the
    curve has to beat before an improvement means anything."""
    times = [
        t
        for run in runs
        if run.measured
        and run.pool_mib == 0
        and run.stats.get("n", 0) >= MIN_DELTAS
        and not run.name.startswith("pre_")
        for _, t in run.step_times
    ]
    if not times:
        medians = baseline_medians(runs)
        return (min(medians), max(medians)) if medians else None
    return quantile(times, 0.25), quantile(times, 0.75)


def render_plot(runs: list[RunResult], out_png: Path) -> Optional[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A run that reached fewer than MIN_DELTAS step deltas is not a measurement: it is a child the
    # workload killed on its own (the allocator at the VRAM limit does that here and then). It stays
    # in the report's condition table, with its error, and stays out of the curve.
    measured = [
        run for run in runs
        if run.measured and run.stats.get("n", 0) >= MIN_DELTAS
    ]
    if not measured:
        return "no measured run to plot"
    baselines = baseline_medians(runs)
    if not baselines:
        return "no baseline run to compare against"
    base_median = statistics.mean(baselines)
    band = baseline_band(runs) or (base_median, base_median)

    pooled = sorted((run for run in measured if run.pool_mib > 0), key=lambda run: run.pool_mib)
    sizes = [run.pool_mib for run in pooled]
    steps_per_s = [run.stats["steps_per_s"] for run in pooled]
    low = [1.0 / run.stats["p75_s"] for run in pooled]
    high = [1.0 / run.stats["p25_s"] for run in pooled]
    speedup = [base_median / run.stats["median_s"] for run in pooled]
    vram = [(run.vram["peak_bytes"] or 0) / 2**30 for run in pooled]
    pooled_share = [
        100.0 * run.hook.get("pooled", 0) / max(1, run.hook.get("hipMalloc", 1)) for run in pooled
    ]
    per_step_malloc = [
        run.hook.get("hipMalloc", 0) / max(1, run.stats["n"]) for run in pooled
    ]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    axis = axes[0]
    axis.axhspan(1.0 / band[1], 1.0 / band[0], color="tab:gray", alpha=0.25)
    axis.axhline(1.0 / base_median, color="tab:gray", linestyle="--", linewidth=1)
    before = pre_change_medians(runs)
    if before:
        axis.axhline(
            1.0 / statistics.mean(before),
            color="tab:red",
            linestyle=":",
            linewidth=1.2,
            label="baseline before the pool was added",
        )
        axis.legend(fontsize=8, loc="lower right")
    axis.errorbar(
        sizes, steps_per_s, yerr=[[s - l for s, l in zip(steps_per_s, low)],
                                 [h - s for s, h in zip(steps_per_s, high)]],
        marker="o", capsize=4, color="tab:blue",
    )
    axis.set_xlabel("PoolSize (MiB)")
    axis.set_ylabel("steps/s (median step time)")
    axis.set_title("step rate vs pool size\n(grey: baseline, dash: its median, band: its step spread)")
    axis.grid(alpha=0.3)

    axis = axes[1]
    axis.bar([str(size) for size in sizes], speedup, color="tab:green", alpha=0.8)
    axis.axhline(1.0, color="tab:gray", linestyle="--", linewidth=1)
    axis.set_xlabel("PoolSize (MiB)")
    axis.set_ylabel("speedup vs baseline (median)")
    axis.set_title("speedup and what the pool costs the card")
    axis.grid(alpha=0.3, axis="y")
    twin = axis.twinx()
    twin.plot([str(size) for size in sizes], vram, marker="s", color="tab:red", linewidth=1)
    twin.set_ylabel("peak VRAM (GiB)")
    # The card's own total, not the range of the points: a pool's cost is a few tens of MiB against
    # a 16 GiB card, and an axis that zooms into that makes a flat result look dramatic.
    twin.set_ylim(15.5, 16.05)

    axis = axes[2]
    axis.plot([str(size) for size in sizes], pooled_share, marker="o", color="tab:purple")
    axis.set_xlabel("PoolSize (MiB)")
    axis.set_ylabel("% of hipMalloc calls served from a pool")
    axis.set_ylim(0, 105)
    axis.set_title("what the pool absorbed")
    axis.grid(alpha=0.3)
    twin = axis.twinx()
    twin.plot([str(size) for size in sizes], per_step_malloc, marker="^", color="tab:orange")
    twin.set_ylabel("hipMalloc calls per step")

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=0)
    figure.suptitle(
        "amdfq-vmm-rs allocation pool: does PoolSize buy step rate?", fontsize=13
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=300)
    plt.close(figure)
    return None


def trajectory_delta(run: RunResult, reference: Optional[RunResult]) -> str:
    """Largest absolute loss difference against the baseline over the steps both reached."""
    if reference is None or not run.loss_series or not reference.loss_series:
        return "no baseline"
    theirs = dict(reference.loss_series)
    shared = [(step, value) for step, value in run.loss_series if step in theirs]
    if not shared:
        return "no shared step"
    worst = max(abs(value - theirs[step]) for step, value in shared)
    return "identical" if worst == 0.0 else f"{worst:.2e} over {len(shared)} steps"


def render_markdown(runs: list[RunResult], *, subset: Path, n_images: int, epochs: int,
                    batch_size: int, skip_first: int, work: Path, config_diff: dict[str, Any]) -> str:
    measured = [run for run in runs if run.measured and run.stats.get("n", 0) >= MIN_DELTAS]
    # Same split as the plot: a `pre_*` run is the earlier session's number, not part of this
    # session's noise band (see baseline_medians).
    baselines = [run for run in measured if run.pool_mib == 0 and run.stats.get("median_s")
                 and not run.name.startswith("pre_")]
    pool_runs = [run for run in measured if run.pool_mib > 0]

    lines = [
        "# amdfq-vmm-rs allocation pool: PoolSize benchmark",
        "",
        f"- work dir: `{work}`",
        f"- dataset subset: `{subset}` ({n_images} images)",
        f"- per run: epochs={epochs}, batch={batch_size}, "
        f"steps used after dropping the first {skip_first} and the sampling deltas",
        "- subset cache: built once by the first run, reused by every later one "
        "(`<data>/.latents_cache`, same path for all conditions)",
        "",
        "## Conditions",
        "",
        "| run | PoolSize (MiB) | exit | median s/step | p25 | p75 | steps/s | n | peak VRAM (GiB) | "
        "hipMalloc | pooled | pools created/released |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        stats = run.stats
        # n marks a run that did not reach MIN_DELTAS deltas: reported, not measured.
        flag = "" if stats.get("n", 0) >= MIN_DELTAS else " (short)"
        lines.append(
            f"| {run.name}{flag} | {run.pool_mib if run.pool_mib else 'off'} | {run.returncode} | "
            f"{stats.get('median_s', float('nan')):.3f} | {stats.get('p25_s', float('nan')):.3f} | "
            f"{stats.get('p75_s', float('nan')):.3f} | {stats.get('steps_per_s', float('nan')):.3f} | "
            f"{stats.get('n', 0)} | {(run.vram.get('peak_bytes') or 0) / 2**30:.2f} | "
            f"{run.hook.get('hipMalloc', 0)} | {run.hook.get('pooled', 0)} | "
            f"{run.hook.get('pools_created', 0)}/{run.hook.get('pools_released', 0)} |"
        )

    lines += ["", "## Verdict", ""]
    before = pre_change_medians(runs)
    before_median = statistics.mean(before) if before else None
    if not baselines:
        lines.append("No baseline run completed in this session; nothing to compare.")
    else:
        base_median = statistics.mean(baseline_medians(runs))
        band = baseline_band(baselines) or (base_median, base_median)
        base_spread_pct = (band[1] - band[0]) / base_median * 100
        base_repeat = ""
        if len(baselines) > 1:
            low, high = sorted(run.stats["median_s"] for run in baselines)
            drift = (high - low) / low * 100
            base_repeat = (
                f"The two baseline runs of this session differ by {drift:.1f}% "
                f"({low:.3f} s vs {high:.3f} s median); the baseline's own step spread is "
                f"{base_spread_pct:.0f}% of its median. "
            )
            if drift > base_spread_pct:
                base_repeat += (
                    "The session drifted more than the within-run spread, so treat the curve as "
                    "indicative. "
                )
        if before_median is not None:
            delta = (before_median - base_median) / base_median * 100
            base_repeat += (
                f"Before the pool was added the same workload measured {before_median:.3f} s/step "
                f"({1.0 / before_median:.3f} steps/s, {len(before)} run(s)); this session's "
                f"pool-off baseline is {delta:+.1f}% against it"
                + (
                    ", i.e. within the spread, so the two sessions are comparable."
                    if abs(delta) <= max(base_spread_pct, 2.0)
                    else ", i.e. the two sessions are NOT comparable — read the pool sizes against "
                    "this session's own baseline."
                )
            )
        lines.append(
            f"{base_repeat}{' ' if base_repeat else ''}Baseline median {base_median:.3f} s/step "
            f"({1.0 / base_median:.3f} steps/s, {len(baselines)} run(s) this session)."
        )
        lines.append("")
        header = "| PoolSize (MiB) | median s/step | vs session baseline | peak VRAM vs baseline | pooled share | loss trajectory |"
        if before_median is not None:
            header = (
                "| PoolSize (MiB) | median s/step | vs session baseline | vs pre-change baseline | "
                "peak VRAM vs baseline | pooled share | loss trajectory |"
            )
        lines.append(header)
        separator = "| --- | --- | --- | --- | --- | --- |"
        if before_median is not None:
            separator = "| --- | --- | --- | --- | --- | --- | --- |"
        lines.append(separator)
        base_vram = statistics.mean(
            (run.vram.get("peak_bytes") or 0) for run in baselines
        )
        reference = max(baselines, key=lambda run: len(run.loss_series)) if baselines else None
        for run in sorted(pool_runs, key=lambda run: run.pool_mib):
            change = (base_median - run.stats["median_s"]) / base_median * 100
            vram_change = (
                ((run.vram.get("peak_bytes") or 0) - base_vram) / base_vram * 100
                if base_vram
                else 0.0
            )
            share = 100.0 * run.hook.get("pooled", 0) / max(1, run.hook.get("hipMalloc", 1))
            row = (
                f"| {run.pool_mib} | {run.stats['median_s']:.3f} | {change:+.1f}% | "
            )
            if before_median is not None:
                row += f"{(before_median - run.stats['median_s']) / before_median * 100:+.1f}% | "
            row += f"{vram_change:+.1f}% | {share:.1f}% | {trajectory_delta(run, reference)} |"
            lines.append(row)
        lines.append("")
        lines.append(
            "The loss column is the largest absolute difference against the session baseline's "
            "`Train/Avg_Loss` trajectory over the steps both runs reached: the pool changes where "
            "memory comes from, and the training math has to come out identical anyway."
        )
        short = [
            run.name
            for run in runs
            if run.measured and run.stats.get("n", 0) < MIN_DELTAS
        ]
        if short:
            lines.append("")
            lines.append(
                "Out of the curve (they never reached "
                f"{MIN_DELTAS} step deltas): "
                + ", ".join(
                    f"`{name}` — {next(r for r in runs if r.name == name).steps_seen} steps, "
                    f"exit {next(r for r in runs if r.name == name).returncode}"
                    for name in short
                )
                + ". The workload aborts its own child when the allocator runs out of room, and on "
                "this card that happens more often as the pool grows: a bigger pool parks more "
                "committed VRAM between its blocks' frees and the allocator's next request."
            )
        lines.append("")

    lines += [
        "## What was changed to run this (identical in every condition)",
        "",
        "```json",
        json.dumps(config_diff, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Caveats",
        "",
        "- One run per condition: the curve is a difference of medians over "
        f"{measured[0].stats.get('n', 0) if measured else 0} step deltas, not a distribution of runs.",
        "- The workload sits at the VRAM limit on a 16 GB card (the baseline log carries the "
        "allocator's own OOM warnings), so an allocation pool changes that equilibrium and a larger "
        "pool is not automatically faster.",
        "- A pooled block's neighbours inside its pool are live allocations: an over-read still lands "
        "in mapped memory, but an over-*write* past a block end can now reach another block instead "
        "of the throwaway pad behind a solo allocation.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def ordered_runs(runs: dict[str, RunResult]) -> list[RunResult]:
    """Insertion order: the run names are built from the condition list, which is the order the
    report should read in."""
    return list(runs.values())


def include_earlier(paths: list[Path], *, as_baseline: bool = False) -> list[RunResult]:
    """Runs read back from an earlier report, kept in this one as `pre_*`: the measurement taken
    before the pool existed, which the pool sizes are also compared against.

    `as_baseline` is for a report collected *before* the hook knew the knob at all: those conditions
    were named after pool sizes that the hook ignored, so their `pool_mib` is meaningless and they
    are baselines."""
    earlier: list[RunResult] = []
    for path in paths:
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"bench: cannot read {path}: {exc}", file=sys.stderr, flush=True)
            continue
        for data in stored.get("runs", []):
            run = RunResult.from_payload(data)
            run.name = f"pre_{run.name}"
            if as_baseline:
                run.pool_mib = 0
            earlier.append(run)
        print(f"bench: kept {len(stored.get('runs', []))} run(s) from {path} as pre_*"
              + (" (baselines)" if as_baseline else ""), flush=True)
    return earlier


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="default: <work-dir>/report")
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET,
                        help="approximate number of images in the shared subset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--skip-first", type=int, default=DEFAULT_SKIP_FIRST)
    parser.add_argument("--pool-sizes", type=int, nargs="*", default=list(DEFAULT_POOL_SIZES))
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the discarded warm-up run (its info-level log is the "
                             "allocation-count evidence)")
    parser.add_argument("--no-baseline-repeat", action="store_true")
    parser.add_argument("--extra-conditions", nargs="*", default=[],
                        help="re-run only these names, e.g. pool_64 baseline_1")
    parser.add_argument("--probe", action="store_true",
                        help="one short baseline run: shakedown and latent-cache warm-up")
    parser.add_argument("--include-json", type=Path, action="append", default=[],
                        help="runs from an earlier report to keep in this one, named `pre_*` "
                             "(the 'before the change' baseline)")
    parser.add_argument("--include-json-baseline", type=Path, action="append", default=[],
                        help="runs from a report collected before the hook knew the pool knob: "
                             "their pool sizes were ignored, so they count as baselines")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    work: Path = args.work_dir
    report_dir: Path = args.report_dir or (work / "report")
    work.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "bench.json"
    png_path = report_dir / "pool_bench.png"
    md_path = report_dir / "pool_bench_report.md"

    if args.plot_only:
        if not json_path.is_file():
            print(f"plot-only: {json_path} does not exist yet", file=sys.stderr, flush=True)
            return 2
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        stored_runs = (
            include_earlier(args.include_json)
            + include_earlier(args.include_json_baseline, as_baseline=True)
            + [RunResult.from_payload(data) for data in stored.get("runs", [])]
        )
        error = render_plot(stored_runs, png_path)
        if error:
            print(f"plot-only: {error}", file=sys.stderr, flush=True)
            return 1
        md_path.write_text(
            render_markdown(
                stored_runs,
                subset=Path(stored.get("subset", "")),
                n_images=int(stored.get("n_images", 0)),
                epochs=int(stored.get("epochs", 0)),
                batch_size=int(stored.get("batch_size", 1)),
                skip_first=int(stored.get("skip_first", 0)),
                work=work,
                config_diff=stored.get("config_diff", {}),
            ),
            encoding="utf-8",
        )
        print(f"plot-only: rewrote {png_path} and {md_path} from {json_path}", flush=True)
        return 0

    repo_config = config_sections()
    data_dir = Path(str(repo_config["environment"]["train_data_dir"]))
    batch_size = int(repo_config["training"]["train_batch_size"])
    subset, n_images = build_subset(work, data_dir, args.subset)
    print(
        f"bench: subset {subset} ({n_images} images), batch {batch_size}, {args.epochs} epoch(s), "
        f"work {work} (the sampler's batches per epoch, and so the step count, is the run's own)",
        flush=True,
    )

    conditions: list[tuple[str, int, bool]]
    if args.probe:
        conditions = [("probe", 0, True)]
    else:
        conditions = []
        if not args.no_warmup:
            conditions.append(("warmup", 0, False))
        conditions.append(("baseline_1", 0, True))
        for size in args.pool_sizes:
            conditions.append((f"pool_{size}", int(size), True))
        if not args.no_baseline_repeat:
            conditions.append(("baseline_2", 0, True))
    if args.extra_conditions:
        wanted = set(args.extra_conditions)
        known = {name: (name, mib, measured) for name, mib, measured in conditions}
        # A re-run may name a condition outside the default list, e.g. pool_512.
        for name in args.extra_conditions:
            if name not in known:
                size = name.split("_")[-1]
                known[name] = (name, int(size) if size.isdigit() else 0, True)
        conditions = [known[name] for name in args.extra_conditions if name in known]
        print(f"bench: re-running only {sorted(wanted)}", flush=True)

    config_diff = {
        "kept from config.toml": [
            "train_batch_size", "train_resolution", "bucket_reso_steps", "min/max_bucket_reso",
            "mixed_precision", "gradient_checkpointing_*", "learning rates", "seed",
            "save_every_n_steps", "cache_latents*", "max_data_loader_n_workers",
            "amdfq", "amdfq_vram_reserve_gib", "amdfq_va_never_reuse",
        ],
        "changed for the benchmark": {
            "train_data_dir": f"{subset} (symlink subset of the configured dataset)",
            "output_dir / logging_dir": "per-run under the work dir",
            "output_name": "bench_<condition>",
            "epoch": args.epochs,
            "amdfq_pool_mib": "the condition under test (0 = off)",
            "[[validation.samples]]": "dropped by the TOML writer; the flat sample_* set renders "
                                      "one image at the final save",
        },
    }

    # Runs read back from an earlier report are carried into this one's JSON as well, so the stored
    # file is the whole comparison rather than only this session's half of it.
    earlier = include_earlier(args.include_json) + include_earlier(
        args.include_json_baseline, as_baseline=True
    )
    runs: dict[str, RunResult] = {}
    for name, pool_mib, measured in conditions:
        print(f"bench: {name} (pool {pool_mib or 'off'} MiB)", flush=True)
        run = run_condition(
            name=name,
            pool_mib=pool_mib,
            measured=measured,
            work=work,
            subset=subset,
            epochs=args.epochs,
            skip_first=args.skip_first,
            quiet=args.quiet,
        )
        runs[name] = run
        json_path.write_text(
            json.dumps(
                {
                    "subset": str(subset),
                    "n_images": n_images,
                    "epochs": args.epochs,
                    "batch_size": batch_size,
                    "skip_first": args.skip_first,
                    "config_diff": config_diff,
                    "runs": [run.payload() for run in earlier + ordered_runs(runs)],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if args.probe:
            print(
                f"probe ok: steps {run.steps_seen}, median "
                f"{run.stats.get('median_s', float('nan')):.3f}s, hook malloc "
                f"{run.hook.get('hipMalloc')}, vram samples {run.vram['samples']}",
                flush=True,
            )
            return 0 if not run.errors else 1
        if run.errors:
            print(f"bench: {name} reported {run.errors}", flush=True)

    ordered = earlier + [runs[name] for name, _, _ in conditions]
    error = render_plot(ordered, png_path)
    if error:
        print(f"bench: plot not written: {error}", flush=True)
    md_path.write_text(
        render_markdown(
            ordered,
            subset=subset,
            n_images=n_images,
            epochs=args.epochs,
            batch_size=batch_size,
            skip_first=args.skip_first,
            work=work,
            config_diff=config_diff,
        ),
        encoding="utf-8",
    )
    print(f"bench: wrote {json_path}, {md_path}" + (f", {png_path}" if not error else ""), flush=True)
    return 0 if not any(run.errors for run in ordered if run.measured) else 1


if __name__ == "__main__":
    raise SystemExit(main())
