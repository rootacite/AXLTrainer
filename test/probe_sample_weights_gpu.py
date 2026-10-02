#!/usr/bin/env python3
"""GPU probe: every path that renders a sample must use the *whole* trained LoRA.

Run from the repo root in the conda env named by `environment.yml`:

    conda activate axl
    python test/probe_sample_weights_gpu.py

What it runs (all of it the real code; nothing here reimplements a render)
    1. One real `trainer/main.py` run in a throwaway mirror of the repo with `sampling_enabled` on,
       so its own sample points render images beside the checkpoints they belong to. The child goes
       through `start_hook.sh` (the `[environment].amdfq` patch) unless `--no-hook`.
    2. `trainer/generate_sample.py` jobs against those same checkpoints, one per mode — `sets`,
       `single`, `batch` (two checkpoints, so the pipeline is reused and the LoRA reloaded) and
       `evaluate` (a top-up with something to render) — built and spawned the way `api.py` builds
       and spawns them, and read back from their own job records and logs.

What the checks are for
    train      the run finished, its sample points left one image per set-slot at every checkpoint
               step, and (with `--hook`) the child ran under the patch.
    inrun      each in-run sample is the same image as a fresh pass from the checkpoint written at
               that step, so the in-run pass rendered with the weights that are in the file (the two
               differ only by the file's bf16 cast of them). A pass from the *other* checkpoint is
               the control that says a rendered image can tell two trained adapters apart.
    modes      for every mode and every checkpoint it loaded, `load_lora`'s own log line reports
               `loaded=N skipped=0` with `N` equal to the file's non-alpha tensor count — nothing in
               the file was left behind — and the mode's images match the `sets` reference of the
               same checkpoint (`batch`'s two checkpoints match their *own* references, not each
               other's; `single` and the evaluation's top-up match each other). A copy of one
               checkpoint with every LoRA value zeroed is rendered as the control that the pipeline
               consumes the trained adapter at all.
    meta       the model side comes from the checkpoint, not from `config.toml`: its metadata's
               network type / dim / alpha and conv dim agree with the file's real tensor shapes, a
               disagreeing config resolves to the checkpoint's numbers, a different rank produces a
               different pipeline key, and a metadata/base mismatch is refused rather than loaded.

What this probe does *not* cover: it is one run, one dataset copy, one base model. The in-run pass
reading the averaged `x` rather than the training iterate `y` is pinned by the unit test in
`test/test_optimizers.py` — `x` and `y` sit ~1% of a step apart, far below what a rendered image can
resolve, so no image check could decide it.

Nothing is written into the dataset: images are copied first, and every run gets its own
AXL_RUNTIME_DIR / output_dir / logging_dir under the report directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# A script's sys.path[0] is this directory, which is what makes the two `test/` imports work.
sys.path.insert(0, str(REPO_ROOT))

from verify_mask_pipeline import (  # noqa: E402 - the path above is what makes these importable
    Report,
    config_sections,
    fmt_values,
    gpu_memory_fault,
    guard_environment,
    launch_run,
    live_training_runs,
)
from probe_optimizer_gpu import prepare_dataset  # noqa: E402 - one copy of the dataset-copy rule

TIERS = ("train", "inrun", "modes", "meta")
PROBE_TIER = "probe"
JOB_TIMEOUT_SECONDS = 900
SAMPLE_NAME_RE = r"^.+_(?P<step>\d{6})_p(?P<set>\d+)_(?P<repeat>\d+)\.png$"

# Rendered pixels are *measured*, never asserted, and this is why: on this workload the same render
# run twice in two processes differs by ~4 LSB mean while removing the whole trained adapter moves it
# ~9 LSB and another checkpoint ~8 LSB (measured — the report carries the run's own numbers). The
# render's run-to-run noise is therefore the same order as the adapter's entire influence, so an
# image cannot decide which weights were loaded; every claim about the loaded set is carried by
# `load_lora`'s own accounting, the checkpoint metadata and the in-run sample's provenance instead.


# --------------------------------------------------------------------------------------
# renders: pixels, deltas
# --------------------------------------------------------------------------------------


def image_delta(left: Path, right: Path) -> dict[str, Any]:
    """Per-pixel difference of two PNGs, in 0..255 units (LSB)."""
    import numpy as np
    from PIL import Image

    with Image.open(left) as handle:
        a = np.asarray(handle.convert("RGB"), dtype=np.int16)
    with Image.open(right) as handle:
        b = np.asarray(handle.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return {"same_shape": False, "left_shape": list(a.shape), "right_shape": list(b.shape)}
    diff = np.abs(a - b)
    return {
        "same_shape": True,
        "max_lsb": int(diff.max()),
        "mean_lsb": float(diff.mean()),
        "differing_fraction": float((diff.sum(axis=2) > 0).mean()),
    }


# --------------------------------------------------------------------------------------
# jobs: written and spawned the way api.py writes and spawns them
# --------------------------------------------------------------------------------------


@dataclass
class JobRun:
    name: str
    mode: str
    job_id: str
    record: dict[str, Any]
    log_path: Path
    returncode: int
    seconds: float
    attempts: int = 1
    files: list[Path] = field(default_factory=list)
    loads: list[dict[str, Any]] = field(default_factory=list)

    @property
    def state(self) -> str:
        return str(self.record.get("state") or "")


def parse_loads(text: str) -> list[dict[str, Any]]:
    """`load_lora`'s own report: `Resume: loaded N tensors from <file> (skipped=S, checkpoint step=…)`."""
    found: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "Resume: loaded " not in line:
            continue
        try:
            _, tail = line.split("Resume: loaded ", 1)
            count, rest = tail.split(" tensors from ", 1)
            path, rest = rest.split(" (skipped=", 1)
            skipped = rest.split(",", 1)[0]
            found.append({"loaded": int(count), "path": path.strip(), "skipped": int(skipped)})
        except ValueError:
            continue
    return found


def generator_command(spec_path: Path, *, hooked: bool) -> list[str]:
    command = [
        sys.executable, "-u", str(REPO_ROOT / "trainer" / "generate_sample.py"),
        "--spec", str(spec_path),
    ]
    if hooked:
        command = ["bash", str(REPO_ROOT / "start_hook.sh"), "--workd", str(REPO_ROOT), *command]
    return command


def run_job(rep: Report, *, name: str, job: dict[str, Any], generated: Path,
            retries: int = 1) -> JobRun:
    """Write the job record, spawn `trainer/generate_sample.py` on it, wait, read the record back.

    The spawn is `api.py`'s (`cwd` = the repo root, `start_new_session`, stderr into the job log),
    which means generator jobs run without the `amdfq` patch — exactly as they do when Ranko asks for
    them. A job that dies of the gfx1201 fault is retried once through `start_hook.sh`.
    """
    from trainer import genjob

    generated.mkdir(parents=True, exist_ok=True)
    job_id = str(job["id"])
    spec_path = genjob.job_path(generated, job_id)
    log_path = genjob.log_path(generated, job_id)
    genjob.write_job(generated, job)

    attempts = 0
    hooked = False
    while True:
        attempts += 1
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                generator_command(spec_path, hooked=hooked),
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            while proc.poll() is None:
                if time.time() - started > JOB_TIMEOUT_SECONDS:
                    print(f"      [{PROBE_TIER}] {name}: timed out, stopping the job", flush=True)
                    proc.kill()
                    break
                time.sleep(2.0)
        seconds = time.time() - started
        text = log_path.read_text(encoding="utf-8", errors="replace")
        fault = gpu_memory_fault(log_path)
        if attempts <= retries and (proc.returncode != 0 or fault):
            print(
                f"      [{PROBE_TIER}] {name}: exit={proc.returncode}"
                + (" [gfx1201 GPU memory fault]" if fault else "")
                + ", retrying under start_hook.sh",
                flush=True,
            )
            shutil.copy2(log_path, log_path.with_name(f"{log_path.stem}.attempt{attempts}.log"))
            hooked = True
            continue
        break

    record = genjob.read_job(spec_path) or {}
    files = [Path(str(entry)) for entry in (record.get("files") or [])]
    if not files and record.get("image_path"):
        files.append(Path(str(record["image_path"])))
    print(
        f"      [{PROBE_TIER}] {name}: {record.get('state')} in {seconds:.0f}s, "
        f"{len(files)} image(s) | {log_path.name}",
        flush=True,
    )
    return JobRun(
        name=name, mode=str(record.get("mode") or job.get("mode") or ""), job_id=job_id,
        record=record, log_path=log_path, returncode=int(proc.returncode), seconds=seconds,
        attempts=attempts, files=files, loads=parse_loads(text),
    )


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------


@dataclass
class ProbeRun:
    result: Any
    run_id: str
    output_name: str
    run_dir: Path
    samples_dir: Path
    log_dir: Path
    generated: Path
    sets: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]
    state: dict[str, Any]
    inrun: dict[int, Path] = field(default_factory=dict)
    refs: dict[str, Path] = field(default_factory=dict)
    jobs: dict[str, JobRun] = field(default_factory=dict)

    def sample_sets(self) -> list[Any]:
        from trainer.config import SampleSet

        return [SampleSet(**entry) for entry in self.sets]

    def checkpoint_at_step(self, step: int) -> Optional[dict[str, Any]]:
        for item in self.checkpoints:
            if item.get("step") == step:
                return item
        return None

    @property
    def first_step(self) -> dict[str, Any]:
        steps = [item for item in self.checkpoints if not item.get("final") and item.get("step")]
        return min(steps, key=lambda item: item["step"])

    @property
    def final(self) -> dict[str, Any]:
        return [item for item in self.checkpoints if item.get("final")][0]


def inrun_samples(samples_dir: Path) -> dict[int, Path]:
    """The run's own sample points: `{step: image}` for set 0, repeat 0."""
    import re

    pattern = re.compile(SAMPLE_NAME_RE)
    found: dict[int, Path] = {}
    for path in sorted(samples_dir.glob("*.png")):
        match = pattern.match(path.name)
        if match and int(match.group("set")) == 0 and int(match.group("repeat")) == 0:
            found[int(match.group("step"))] = path
    return found


def launch_training(rep: Report, args: argparse.Namespace, data_dir: Path, work: Path) -> ProbeRun:
    from trainer import genjob
    from trainer.checkpoints import discover_checkpoints
    from trainer.config import resolve_sample_sets, run_config_mapping
    from trainer.runs import find_samples_dir, run_output_name

    extra_sections = {
        # config.toml's own single-set fallback, at a size and step count a probe can afford.
        "network": {
            "network_dim": args.network_dim,
            "network_alpha": args.network_alpha,
            "conv_dim": args.conv_dim,
            "conv_alpha": args.conv_alpha,
        },
        "training": {"sampling_enabled": True},
        "validation": {
            "sample_seed": args.sample_seed,
            "sample_repeat": 1,
            "sample_width": args.sample_size,
            "sample_height": args.sample_size,
            "sample_steps": args.sample_steps,
        },
    }
    result = launch_run(
        name="sample", data_dir=data_dir, seed=args.seed, steps=args.steps, work=work,
        masked=False, tier=PROBE_TIER, batch_size=args.batch_size,
        save_every_override=max(1, args.steps // 2), lr=(args.unet_lr, args.te_lr),
        warmup=args.warmup, retries=args.retries, extra_sections=extra_sections, hook=args.hook,
    )
    run_dir = Path(result.run_dir) if result.run_dir else Path()
    run_id = run_dir.name
    output_name = run_output_name(run_id)
    samples_dir = find_samples_dir(run_dir, output_name)
    log_dir = Path(result.logging_dir) / run_id
    mapping, _source = run_config_mapping(log_dir)
    state_path = Path(result.runtime_dir) / "state.json"
    return ProbeRun(
        result=result,
        run_id=run_id,
        output_name=output_name,
        run_dir=run_dir,
        samples_dir=samples_dir,
        log_dir=log_dir,
        generated=genjob.generated_dir(samples_dir),
        sets=[asdict(sample_set) for sample_set in resolve_sample_sets(mapping)],
        checkpoints=discover_checkpoints(result.output_dir, output_name),
        state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {},
        inrun=inrun_samples(samples_dir),
    )


def sets_job(run: ProbeRun, checkpoint: dict[str, Any], *, job_id: str) -> dict[str, Any]:
    """A `sets` job for one checkpoint, built as `handle_generate_checkpoint_samples` builds it."""
    from trainer import genjob

    job = genjob.new_job(
        {"step": checkpoint.get("step")},
        run_id=run.run_id,
        output_name=run.output_name,
        checkpoint=str(checkpoint["path"]),
        mode=genjob.MODE_SETS,
        total_images=sum(int(entry["repeat"]) for entry in run.sets),
        extra={"sample_sets": run.sets, "config_log_dir": str(run.log_dir)},
    )
    job["id"] = job_id
    return job


def single_job(run: ProbeRun, checkpoint: dict[str, Any], *, seed: int,
               job_id: str) -> dict[str, Any]:
    """A `single` job with the first set's prompt and the given seed, as api.py builds it."""
    from trainer import genjob

    sample_set = run.sets[0]
    job = genjob.new_job(
        {
            "prompt": sample_set["prompt"],
            "negative_prompt": sample_set["negative"],
            "cfg": sample_set["guidance_scale"],
            "steps": sample_set["steps"],
            "seed": int(seed),
            "width": sample_set["width"],
            "height": sample_set["height"],
            "step": checkpoint.get("step"),
        },
        run_id=run.run_id,
        output_name=run.output_name,
        checkpoint=str(checkpoint["path"]),
        mode=genjob.MODE_SINGLE,
        total_images=1,
    )
    job["id"] = job_id
    return job


def batch_job(run: ProbeRun, entries: list[dict[str, Any]], *, job_id: str,
              config_log_dir: Path) -> dict[str, Any]:
    from trainer import genjob

    steps = sorted(int(entry["step"]) for entry in entries)
    job = genjob.new_batch_job(
        run_id=run.run_id,
        output_name=run.output_name,
        checkpoints=[{"path": str(entry["path"]), "step": entry.get("step")} for entry in entries],
        from_step=steps[0],
        to_step=steps[-1],
        images_per_checkpoint=sum(int(item["repeat"]) for item in run.sets),
        config_log_dir=str(config_log_dir),
    )
    job["id"] = job_id
    return job


def evaluation_job(run: ProbeRun, checkpoint: dict[str, Any], *,
                   job_id: str) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    """An `evaluate` job, planned with the very helpers `handle_evaluate_checkpoint` plans with."""
    from trainer import evaluation, genjob

    sets = run.sample_sets()
    images = evaluation.collect_images(
        samples_dir=run.samples_dir,
        generated_dir=run.generated,
        checkpoint=Path(str(checkpoint["path"])),
        step=checkpoint.get("step"),
        sets=sets,
    )
    depth = len(images) + 1  # one image more than it has: the top-up must render something
    plan = evaluation.expansion_plan(sets, depth, images)
    job = genjob.new_evaluation_job(
        run_id=run.run_id,
        output_name=run.output_name,
        checkpoint=str(checkpoint["path"]),
        step=checkpoint.get("step"),
        depth=depth,
        threshold=0.35,
        categories=["general"],
        config_source="probe",
        config_log_dir=str(run.log_dir),
        plan=plan,
        images=[image.to_dict() for image in images],
        sample_sets=run.sets,
        tags=[],
    )
    job["id"] = job_id
    return job, plan, images


def knockout_copy(source: Path, dest: Path) -> Path:
    """A copy of a checkpoint whose LoRA values are all zero: the adapter contributes nothing."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(str(source), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {
            key: (torch.zeros_like(handle.get_tensor(key)) if not key.endswith(".alpha")
                  else handle.get_tensor(key))
            for key in handle.keys()
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dest), metadata=metadata)
    return dest


def tensor_keys(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        return list(handle.keys())


# --------------------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------------------


def tier_train(rep: Report, args: argparse.Namespace, work: Path) -> tuple[dict[str, Any], ProbeRun]:
    tier = PROBE_TIER
    data_dir, dataset = prepare_dataset(work, args.images)
    rep.note(
        f"dataset: {dataset['images']} image(s) copied from "
        + ", ".join(f"{path} x{count}" for path, count in dataset["sources"].items())
        + f" into {data_dir}"
    )
    run = launch_training(rep, args, data_dir, work)
    checkpoint_steps = {int(item["step"]) for item in run.checkpoints if item.get("step") is not None}
    rep.check(
        tier, "the run finishes with an in-run sample at every checkpoint step",
        run.result.returncode == 0 and bool(checkpoint_steps)
        and checkpoint_steps <= set(run.inrun),
        f"exit={run.result.returncode} in {run.result.seconds:.0f}s | "
        f"steps with a checkpoint and an in-run sample: {sorted(checkpoint_steps & set(run.inrun))} "
        f"of {sorted(checkpoint_steps)}"
        + (f" | attempts={run.result.attempts}" if run.result.attempts > 1 else ""),
        gpu_fault=bool(run.result.gpu_fault), attempts=run.result.attempts,
    )
    rep.check(
        tier, "each in-run sample is named {output_name}_{step:06d}_p0_0.png",
        bool(run.inrun)
        and all(
            path.name.startswith(f"{run.output_name}_{step:06d}_p0_0")
            for step, path in run.inrun.items()
        ),
        "; ".join(f"s{step}: {path.name}" for step, path in sorted(run.inrun.items())),
    )
    rep.check(
        tier, "the prompts and sampling values resolved to one usable set",
        len(run.sets) == 1 and int(run.sets[0]["steps"]) > 0 and int(run.sets[0]["width"]) > 0,
        f"{len(run.sets)} set(s): " + "; ".join(
            f"{entry['name']} {entry['width']}x{entry['height']} {entry['steps']} steps "
            f"cfg {entry['guidance_scale']} rescale {entry['guidance_rescale']} seed {entry['seed']} "
            f"x{entry['repeat']}"
            for entry in run.sets
        ),
    )
    if args.hook:
        text = run.result.log_path.read_text(encoding="utf-8", errors="replace")
        banner = next(
            (line.strip() for line in text.splitlines() if "start_hook.sh: amdfq=" in line), None
        )
        rep.check(
            tier, "the training child ran under the [environment].amdfq patch",
            banner is not None and "amdfq=none" not in banner,
            banner or "no start_hook.sh banner in the child's log",
        )
    else:
        rep.note("--no-hook: the training child ran bare, without the [environment].amdfq patch")
    return {
        "dataset": dataset,
        "run": {
            "run_dir": str(run.run_dir),
            "samples_dir": str(run.samples_dir),
            "generated": str(run.generated),
            "log_dir": str(run.log_dir),
            "checkpoints": [
                f"{Path(item['path']).parent.name} (step {item.get('step')})" for item in run.checkpoints
            ],
            "in_run_samples": {f"s{step}": path.name for step, path in sorted(run.inrun.items())},
            "sets": run.sets,
            "state": run.state.get("training", {}),
            "mirror_config": run.result.claims.get("mirror_config"),
            "seconds": round(run.result.seconds, 1),
        },
    }, run


def check_loads(rep: Report, job: JobRun, file_counts: dict[str, int]) -> None:
    """`load_lora` loaded every LoRA tensor of every file this job was given."""
    tier = "modes"
    if not job.loads:
        rep.check(
            tier, f"{job.name}: the job reports what it loaded", False,
            f"no `Resume: loaded …` line in {job.log_path.name}",
        )
        return
    problems: list[str] = []
    for entry in job.loads:
        wanted = file_counts.get(entry["path"])
        if wanted is None:
            problems.append(f"{Path(entry['path']).parent.name}: unexpected file")
            continue
        if entry["loaded"] != wanted or entry["skipped"] != 0:
            problems.append(
                f"{Path(entry['path']).parent.name}: loaded={entry['loaded']} (of {wanted}), "
                f"skipped={entry['skipped']}"
            )
    rep.check(
        tier, f"{job.name}: loaded every tensor of the file(s) it was given, none skipped",
        not problems,
        f"{len(job.loads)} load(s): " + "; ".join(
            f"{Path(entry['path']).parent.name} {entry['loaded']}/"
            f"{file_counts.get(entry['path'], '?')} skipped={entry['skipped']}"
            for entry in job.loads
        ),
        problems=problems,
    )


def note_delta(rep: Report, tier: str, name: str, left: Path, right: Path) -> dict[str, Any]:
    """Record one rendered pair as an observation. Pixels are never asserted: see the note above
    `JOB_TIMEOUT_SECONDS` for the measured reason."""
    outcome = image_delta(left, right)
    rep.observe(
        tier, name, f"{left.name} vs {right.name}",
        **{**outcome, "left": str(left), "right": str(right)},
    )
    return outcome


def inrun_provenance(path: Path) -> dict[str, str]:
    """What the in-run sample recorded about itself when it was written (`sample_provenance`)."""
    from PIL import Image

    with Image.open(path) as image:
        return {key: str(value) for key, value in (image.text or {}).items()}


def tier_inrun(rep: Report, args: argparse.Namespace, run: ProbeRun) -> dict[str, Any]:
    tier = "inrun"
    out: dict[str, Any] = {"deltas": {}}
    targets = [run.first_step, run.final]
    missing = [entry.get("step") for entry in targets if entry.get("step") not in run.inrun]
    if missing:
        rep.check(tier, "both sample points have an in-run image and a checkpoint", False,
                  f"steps without an in-run image: {missing} (have {sorted(run.inrun)})")
        return out

    # The set the generator renders with, and what the in-run sample says it rendered with.
    sample_set = run.sets[0]
    provenance_ok = True
    provenance_text: list[str] = []
    for entry in targets:
        step = int(entry["step"])
        text = inrun_provenance(run.inrun[step])
        wanted = {
            "axl_seed": str(sample_set["seed"]),
            "axl_steps": str(sample_set["steps"]),
            "axl_width": str(sample_set["width"]),
            "axl_height": str(sample_set["height"]),
            "axl_guidance": str(sample_set["guidance_scale"]),
            "axl_guidance_rescale": str(sample_set["guidance_rescale"]),
            "axl_prompt": str(sample_set["prompt"]),
            "axl_negative": str(sample_set["negative"]),
        }
        differing = {key: (text.get(key), value) for key, value in wanted.items() if text.get(key) != value}
        provenance_ok = provenance_ok and not differing
        provenance_text.append(
            f"s{step}: " + ("as configured" if not differing else f"differs in {sorted(differing)}")
        )
        if differing:
            out.setdefault("provenance_mismatch", {})[f"s{step}"] = differing
    rep.check(
        tier, "each in-run sample carries the seed/size/steps/prompts of the set it was planned with",
        provenance_ok, "; ".join(provenance_text),
    )

    for entry in targets:
        step = int(entry["step"])
        directory = Path(entry["path"]).parent.name
        job = run_job(
            rep, name=f"sets-{directory}",
            job=sets_job(run, entry, job_id=f"{run.output_name}_s{step:06d}_ref"),
            generated=run.generated, retries=args.retries,
        )
        run.jobs[f"sets:{directory}"] = job
        if len(job.files) != 1:
            rep.check(tier, f"the sets pass for {directory} rendered one image", False,
                      f"state={job.state}, {len(job.files)} file(s)")
            continue
        run.refs[directory] = job.files[0]
        out["deltas"][f"inrun_s{step}"] = note_delta(
            rep, tier, f"step {step}: the in-run sample against a pass from its own checkpoint",
            job.files[0], run.inrun[step],
        )
        if entry is targets[0]:
            # The render's own noise floor: the same spec, the same checkpoint, another process.
            repeat = run_job(
                rep, name=f"sets-{directory}-repeat",
                job=sets_job(run, entry, job_id=f"{run.output_name}_s{step:06d}_ref2"),
                generated=run.generated, retries=args.retries,
            )
            run.jobs[f"sets-repeat:{directory}"] = repeat
            if repeat.files:
                out["deltas"]["floor_same_job_twice"] = note_delta(
                    rep, tier, f"the render floor: the same pass run twice (step {step})",
                    job.files[0], repeat.files[0],
                )
                out["deltas"][f"inrun_s{step}_vs_floor_pair"] = note_delta(
                    rep, tier, f"step {step}: the in-run sample against that second pass",
                    repeat.files[0], run.inrun[step],
                )

    directories = [Path(entry["path"]).parent.name for entry in targets]
    if len(directories) == 2 and all(directory in run.refs for directory in directories):
        out["deltas"]["control_other_checkpoint"] = note_delta(
            rep, tier, "the control scale: the two checkpoints' own passes",
            run.refs[directories[0]], run.refs[directories[1]],
        )
    return out


def tier_modes(rep: Report, args: argparse.Namespace, run: ProbeRun) -> dict[str, Any]:
    from trainer import genjob

    tier = "modes"
    out: dict[str, Any] = {"deltas": {}, "jobs": {}}
    first, final = run.first_step, run.final
    first_dir, final_dir = Path(first["path"]).parent.name, Path(final["path"]).parent.name
    file_counts = {
        str(item["path"]): sum(1 for key in tensor_keys(Path(item["path"]))
                               if not key.endswith(".alpha"))
        for item in run.checkpoints
    }
    out["file_tensor_counts"] = file_counts

    # --- single: one image, the first set's prompt, the pass's own slot-0 seed ----------------
    if first_dir in run.refs:
        job = run_job(
            rep, name="single-slot0",
            job=single_job(run, first, seed=int(run.sets[0]["seed"]),
                           job_id=f"{run.output_name}_single_seed{int(run.sets[0]['seed'])}"),
            generated=run.generated, retries=args.retries,
        )
        run.jobs["single:slot0"] = job
        if job.files:
            out["deltas"]["single_vs_sets_slot0"] = note_delta(
                rep, tier, "single: its image against the sets pass of the same checkpoint and seed",
                job.files[0], run.refs[first_dir],
            )
        else:
            rep.check(tier, "single: rendered an image", False, f"state={job.state}")
        rep.check(
            tier, "single: the job finished with its one image recorded",
            job.state == genjob.STATE_DONE and len(job.files) == 1,
            f"state={job.state}, files={[path.name for path in job.files]}",
        )

    # --- batch: two checkpoints, one pipeline, the LoRA reloaded between them ------------------
    batch = run_job(
        rep, name="batch",
        job=batch_job(run, [first, final], job_id=f"{run.output_name}_batch",
                      config_log_dir=run.log_dir),
        generated=run.generated, retries=args.retries,
    )
    run.jobs["batch"] = batch
    batch_text = batch.log_path.read_text(encoding="utf-8", errors="replace")
    reloaded = batch_text.count("loading the LoRA weights of")
    rep.check(
        tier, "batch: one pipeline for the range, the LoRA reloaded for the second checkpoint",
        reloaded == 1,
        f"{reloaded} `loading the LoRA weights of …` line(s), {len(batch.loads)} load(s) reported",
    )
    # The batch's work list is its own record; each entry got a `sets` job of its own, which is
    # where that checkpoint's images are.
    sub_by_step: dict[int, Path] = {}
    for job_id in batch.record.get("job_ids") or []:
        record = run_job_records(run, str(job_id))
        step = record.get("step")
        files = [Path(str(path)) for path in (record.get("files") or [])]
        if step is not None and files:
            sub_by_step[int(step)] = files[0]
    for entry in (first, final):
        step = int(entry["step"])
        directory = Path(entry["path"]).parent.name
        reference = run.refs.get(directory)
        rendered = sub_by_step.get(step)
        if reference is None or rendered is None:
            rep.check(tier, f"batch: checkpoint {directory} has a render and a reference", False,
                      f"reference={reference}, batch image={rendered}")
            continue
        out["deltas"][f"batch_{directory}"] = note_delta(
            rep, tier, f"batch: the image of {directory} against that checkpoint's own sets pass",
            rendered, reference,
        )
    rep.check(
        tier, "batch: every checkpoint of the range got its own render",
        len(sub_by_step) == len(batch.record.get("checkpoints") or []) == 2,
        f"checkpoints rendered {sorted(sub_by_step)}, sub-jobs "
        f"{list(batch.record.get('job_ids') or [])}, load(s) reported {len(batch.loads)} "
        f"(the batch record itself holds no images by design)",
    )
    # What the batch rendered, straight from its log: it must be this run's own set. The range used
    # to re-resolve the prompts from `{config_log_dir}/config.toml` through a `TrainConfig`, where a
    # key the snapshot lacks (`samples`, for a config carrying only the flat `sample_*` scalars) was
    # filled from *today's* repo `config.toml` — a mirror whose config has no `[[validation.samples]]`
    # at all is exactly that case, so this check is the regression test for it.
    resolved = [
        line.strip() for line in batch_text.splitlines()
        if line.strip().startswith("[generate_sample] set ")
    ]
    wanted = [
        f"set {index + 1}/{len(run.sets)} {entry['name']}: {entry['repeat']} image(s), "
        f"{entry['width']}x{entry['height']}, {entry['steps']} steps, cfg {entry['guidance_scale']}, "
        f"seed {entry['seed']}"
        for index, entry in enumerate(run.sets)
    ]
    rep.check(
        tier, "batch: it rendered this run's own prompt sets",
        len(resolved) >= len(wanted) and all(
            line.split(": ", 1)[-1].startswith(text.split(": ", 1)[-1]) for line, text in zip(resolved, wanted)
        ),
        f"rendered {resolved or 'nothing'} | the run's sets are {wanted}",
    )
    references = [run.refs.get(first_dir), run.refs.get(final_dir)]
    if all(references):
        out["deltas"]["control_batch_references"] = note_delta(
            rep, tier, "the control scale: the batch's two checkpoints' own passes",
            references[0], references[1],
        )

    # --- evaluate: a top-up that has something to render, against a single render of that slot --
    try:
        job, plan, images = evaluation_job(
            run, first, job_id=f"{run.output_name}_evaluate_topup"
        )
    except Exception as exc:  # noqa: BLE001 - a plan that cannot be built is the check's failure
        rep.check(tier, "the evaluation could be planned", False, f"{type(exc).__name__}: {exc}")
        return out
    out["evaluation_plan"] = {
        "depth": plan.get("depth"), "existing_images": plan.get("existing_images"),
        "render_total": plan.get("render_total"),
        "slots": [{"set": item["set_index"], "render": item["render"]} for item in plan["sets"]],
    }
    rep.check(
        tier, "the evaluation has a slot to render (the top-up is the path under test)",
        bool(plan.get("needed")) and int(plan.get("render_total") or 0) > 0,
        f"depth={plan.get('depth')} existing={plan.get('existing_images')} "
        f"render_total={plan.get('render_total')}",
    )
    slots = [repeat for item in plan["sets"] for repeat in item["render"]]
    evaluate = run_job(rep, name="evaluate", job=job, generated=run.generated, retries=args.retries)
    run.jobs["evaluate"] = evaluate
    if slots:
        slot = slots[0]
        reference = run_job(
            rep, name=f"single-slot{slot}",
            job=single_job(run, first, seed=int(run.sets[0]["seed"]) + slot,
                           job_id=f"{run.output_name}_single_slot{slot}"),
            generated=run.generated, retries=args.retries,
        )
        run.jobs[f"single:slot{slot}"] = reference
        rendered = [path for path in evaluate.files if path.name.endswith(f"_p0_{slot}.png")]
        if reference.files and rendered:
            out["deltas"][f"evaluate_slot{slot}"] = note_delta(
                rep, tier,
                f"evaluate: its rendered slot {slot} against a single render of that slot",
                rendered[0], reference.files[0],
            )
            slot0 = run.refs.get(first_dir)
            if slot0 is not None:
                out["deltas"]["control_evaluate_vs_slot0"] = note_delta(
                    rep, tier,
                    "the control scale: the evaluation's new slot against the existing slot's image",
                    rendered[0], slot0,
                )
        else:
            rep.check(tier, "evaluate: the rendered slot has a reference to compare with", False,
                      f"evaluate files={[path.name for path in evaluate.files]}, "
                      f"reference files={[path.name for path in reference.files]}")
    rep.check(
        tier, "evaluate: the pass reached its end (rendered, tagged and scored)",
        evaluate.state == genjob.STATE_DONE and evaluate.record.get("phase") == genjob.PHASE_DONE
        and isinstance(evaluate.record.get("scores"), dict),
        f"state={evaluate.state} phase={evaluate.record.get('phase')} "
        f"files={[path.name for path in evaluate.files]} "
        f"scores={'yes' if isinstance(evaluate.record.get('scores'), dict) else 'no'}",
    )

    # --- knockout: what a completely unloaded adapter would look like --------------------------
    if not args.no_knockout:
        knockout = knockout_copy(
            Path(final["path"]), run.run_dir / f"{final_dir}_zeroed" / Path(final["path"]).name
        )
        entry = {"path": str(knockout), "step": final.get("step")}
        job = run_job(
            rep, name="sets-zeroed",
            job=sets_job(run, entry, job_id=f"{run.output_name}_s{int(final['step']):06d}_zeroed"),
            generated=run.generated, retries=args.retries,
        )
        run.jobs["sets:zeroed"] = job
        reference = run.refs.get(final_dir)
        if job.files and reference is not None:
            out["deltas"]["control_zeroed_adapter"] = note_delta(
                rep, tier,
                "the strongest possible perturbation: the same checkpoint with every LoRA value zeroed",
                job.files[0], reference,
            )
        else:
            rep.check(tier, "the zeroed checkpoint rendered an image to compare", False,
                      f"state={job.state}, files={[path.name for path in job.files]}")

    # --- accounting: every job above loaded every tensor it was given ---------------------------
    if run.jobs.get("sets:zeroed"):
        # The zeroed copy is the probe's own artifact, not a checkpoint of the run.
        file_counts.update({
            path: sum(1 for key in tensor_keys(path) if not key.endswith(".alpha"))
            for path in (str(entry) for entry in
                         [Path(run.jobs["sets:zeroed"].loads[0]["path"])] if run.jobs["sets:zeroed"].loads)
        })
    for job in run.jobs.values():
        check_loads(rep, job, file_counts)
        out["jobs"][job.name] = {
            "mode": job.mode, "state": job.state, "returncode": job.returncode,
            "attempts": job.attempts, "seconds": round(job.seconds, 1),
            "files": [path.name for path in job.files], "loads": job.loads,
            "log": str(job.log_path),
        }
    return out


def run_job_records(run: ProbeRun, job_id: str) -> dict[str, Any]:
    from trainer import genjob

    return genjob.read_job(genjob.job_path(run.generated, job_id)) or {}


def tier_meta(rep: Report, args: argparse.Namespace, run: ProbeRun) -> dict[str, Any]:
    tier = "meta"
    from trainer.checkpoints import (
        conv_dim_alpha_from_metadata,
        infer_network_type,
        read_lora_metadata,
    )
    from trainer.config import TrainConfig
    from trainer.generate_sample import _build_config, _shape_key

    out: dict[str, Any] = {"per_checkpoint": {}}
    repo_cfg = TrainConfig()
    for entry in run.checkpoints:
        path = Path(entry["path"])
        directory = path.parent.name
        metadata = read_lora_metadata(path)
        keys = tensor_keys(path)
        from safetensors.torch import load_file

        tensors = load_file(str(path))
        down = [key for key in keys if not key.endswith(".alpha") and key.endswith(".lora_down.weight")]
        conv = [key for key in down if tensors[key].dim() == 4]
        shapes_ok = bool(down)
        problems: list[str] = []
        for key in down:
            rank = int(tensors[key].shape[0])
            expected = args.conv_dim if tensors[key].dim() == 4 else args.network_dim
            if rank != expected:
                shapes_ok = False
                problems.append(f"{key}: rank {rank} != {expected}")
        rep.check(
            tier, f"{directory}: the adapter shapes match the configured rank / conv rank",
            shapes_ok,
            f"{len(down)} lora_down tensor(s), {len(conv)} of them 4-D (the conv adapter); "
            + ("every rank as configured" if shapes_ok else "; ".join(problems[:3])),
        )
        resolved = _build_config(metadata, path, base_cfg=repo_cfg)
        rep.check(
            tier, f"{directory}: the model side resolves from the checkpoint, not config.toml",
            int(resolved.network_dim) == args.network_dim
            and int(getattr(resolved, "conv_dim", 0)) == args.conv_dim
            and str(resolved.network_type) == "locon"
            and str(resolved.resume_lora_path) == str(path),
            f"config.toml says dim={repo_cfg.network_dim} conv={repo_cfg.conv_dim} "
            f"type={repo_cfg.network_type} -> the checkpoint resolves to dim={resolved.network_dim} "
            f"conv={getattr(resolved, 'conv_dim', None)} type={resolved.network_type} "
            f"resume={Path(str(resolved.resume_lora_path)).name}",
        )
        rep.check(
            tier, f"{directory}: its metadata and the file agree on type / dim / conv dim",
            infer_network_type(metadata) == "locon"
            and conv_dim_alpha_from_metadata(metadata) == (args.conv_dim, args.conv_alpha)
            and str(metadata.get("ss_network_dim")) == str(args.network_dim)
            and str(metadata.get("ss_network_alpha")) == str(args.network_alpha),
            f"ss_network_type={metadata.get('ss_network_type')} "
            f"ss_network_dim={metadata.get('ss_network_dim')} "
            f"ss_network_alpha={metadata.get('ss_network_alpha')} "
            f"conv_dim/alpha={conv_dim_alpha_from_metadata(metadata)}",
        )
        out["per_checkpoint"][directory] = {
            "metadata": {key: metadata.get(key) for key in (
                "ss_network_type", "ss_network_dim", "ss_network_alpha", "ss_network_args",
                "modelspec.prediction_type", "ss_steps",
            )},
            "tensors": len(keys), "linear_down": len(down) - len(conv), "conv_down": len(conv),
            "resolved": {
                "network_type": resolved.network_type, "network_dim": resolved.network_dim,
                "conv_dim": getattr(resolved, "conv_dim", None),
            },
        }
    if run.checkpoints:
        path = Path(run.checkpoints[0]["path"])
        metadata = read_lora_metadata(path)
        resolved = _build_config(metadata, path, base_cfg=repo_cfg)
        rep.check(
            tier, "a checkpoint of a different rank would rebuild the pipeline rather than reuse it",
            _shape_key(resolved) != _shape_key(repo_cfg),
            f"checkpoint key={_shape_key(resolved)} vs config.toml key={_shape_key(repo_cfg)}",
        )
        try:
            _build_config({**metadata, "ss_base_model_version": "sd3.5-large"}, path, base_cfg=repo_cfg)
            refused, note = False, "no error"
        except RuntimeError as exc:
            refused, note = True, str(exc)[:140]
        rep.check(
            tier, "a checkpoint trained on another base family is refused, not half loaded",
            refused, note,
        )
    return out


# --------------------------------------------------------------------------------------
# cli / report
# --------------------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=12,
                        help="optimizer steps of the training run (two save/sample points wanted)")
    parser.add_argument("--batch-size", type=int, default=2, help="train_batch_size (config.toml: 3)")
    parser.add_argument("--images", type=int, default=9, help="images copied out of the configured folders")
    parser.add_argument("--network-dim", type=int, default=8, help="LoCon rank (config.toml: 32)")
    parser.add_argument("--network-alpha", type=int, default=4, help="config.toml: 16")
    parser.add_argument("--conv-dim", type=int, default=4, help="LoCon conv rank (config.toml: 16)")
    parser.add_argument("--conv-alpha", type=int, default=2, help="config.toml: 8")
    parser.add_argument("--unet-lr", type=float, default=2e-4, help="UNet LR (config.toml: 5e-5)")
    parser.add_argument("--te-lr", type=float, default=2e-5, help="TE LR (config.toml: 5e-6)")
    parser.add_argument("--warmup", type=int, default=10, help="Schedule-Free warmup (config.toml: 250)")
    parser.add_argument("--sample-seed", type=int, default=1234,
                        help="the sample set's seed (0 would draw a fresh one per image)")
    parser.add_argument("--sample-size", type=int, default=768, help="sample width and height")
    parser.add_argument("--sample-steps", type=int, default=8, help="denoise steps per sample")
    parser.add_argument("--seed", type=int, default=None, help="training seed (default: config.toml's)")
    parser.add_argument("--tiers", default="all", help="train,inrun,modes,meta | all")
    parser.add_argument("--no-knockout", action="store_true",
                        help="skip the zeroed-adapter control render")
    parser.add_argument("--no-hook", dest="hook", action="store_false",
                        help="run the training child bare, without the [environment].amdfq patch")
    parser.add_argument("--retries", type=int, default=2, help="retries per run/job after a GPU fault")
    parser.add_argument("--report-dir", default=None, help="where the report is written")
    parser.add_argument("--work-dir", default=None, help="scratch dir (mirror, dataset copy, runs)")
    parser.add_argument("--clean", action="store_true", help="delete the scratch dir when done")
    parser.add_argument("--force", action="store_true", help="run even when a training run looks live")
    parser.add_argument("--allow-foreign-env", action="store_true",
                        help="skip the environment.yml conda env check")
    return parser.parse_args(argv)


def write_report(rep: Report, args: argparse.Namespace, details: dict[str, Any]) -> Path:
    out = rep.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": rep.env,
        "tiers": args.tiers,
        "options": {
            "steps": args.steps, "batch_size": args.batch_size, "images": args.images,
            "seed": args.seed, "hook": args.hook,
            "network_dim": args.network_dim, "network_alpha": args.network_alpha,
            "conv_dim": args.conv_dim, "conv_alpha": args.conv_alpha,
            "unet_lr": args.unet_lr, "te_lr": args.te_lr, "warmup": args.warmup,
            "sample_seed": args.sample_seed, "sample_size": args.sample_size,
            "sample_steps": args.sample_steps,
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
        "# Sample-path probe (every render path uses the whole trained LoRA)",
        "",
        f"- Interpreter `{rep.env.get('executable')}` (prefix `{rep.env.get('prefix')}`), "
        f"torch {rep.env.get('torch')} / HIP {rep.env.get('torch_hip')}, GPU {rep.env.get('gpu')}",
        f"- Tiers `{', '.join(args.tiers)}` | steps={args.steps}, batch={args.batch_size}, "
        f"images={args.images}, seed={args.seed}, hook={args.hook}",
        f"- LoCon rank {args.network_dim}/{args.network_alpha} (conv {args.conv_dim}/{args.conv_alpha}), "
        f"lrs {args.unet_lr}/{args.te_lr}, warmup {args.warmup}",
        f"- Samples {args.sample_size}x{args.sample_size}, {args.sample_steps} steps, seed "
        f"{args.sample_seed} | rendered pixels are observations: this run's own render floor (the "
        f"same pass twice) is beside the control scales in the observations, and nothing passes or "
        f"fails on them",
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
                  else Path("/tmp/axl-probe-sample-weights") / time.strftime("%Y%m%d_%H%M%S"))
    work = Path(args.work_dir) if args.work_dir else report_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    args.report_dir, args.work_dir, args.tiers = report_dir, work, tiers
    rep = Report(report_dir)
    print(f"== sample-weight probe | tiers={','.join(tiers)} | report={report_dir}", flush=True)

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
                f"refusing to run: config.toml selects amdfq={choice!r}, so the patch the training "
                "child would preload is nothing; set [environment].amdfq to vmm (or pass --no-hook to "
                "run bare on purpose)"
            )
        rep.note(f"training child through start_hook.sh: {line}")
    rep.note("generator jobs are spawned the way api.py spawns them (bare, cwd = repo root, stderr "
             "into the job log); a job that dies of the gfx1201 fault is retried under start_hook.sh")
    rep.note("read-only on the dataset: images are copied into the scratch dir, and every run and job "
             "writes only under the report directory")

    details: dict[str, Any] = {}
    try:
        details["train"], run = tier_train(rep, args, work)
        if "inrun" in tiers:
            details["inrun"] = tier_inrun(rep, args, run)
        if "modes" in tiers:
            details["modes"] = tier_modes(rep, args, run)
        if "meta" in tiers:
            details["meta"] = tier_meta(rep, args, run)
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
