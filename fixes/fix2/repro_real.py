#!/usr/bin/env python3
"""The path that reproduces the gfx1201 page fault — the real trainer, on the packaged config.

`repro.py`, `repro_unet.py` and `repro_joint.py` in this directory show that no standalone GEMM
shape is enough: the abort needs the real training process. This script is the other half of the
table — it drives the unmodified `trainer/main.py` with the exact configuration that faulted
(`resources/original-config.toml`, SDXL, 6 packaged images, `train_batch_size=3`,
`network_dim=36`, seed 1145141920) and dies at step 101 of 120, every time.

Because the fault is driven by the caption-shuffle RNG (seed -> tokenized prompt -> number of CLIP
chunks -> encoder sequence length), the same script can also run the *surviving* variants, so one
command produces both sides of the table:

    python repro_real.py                      # original (faults) + the surviving variants
    python repro_real.py --only original      # just the 100% path
    python repro_real.py --list               # show the grid without running it
    python repro_real.py --only original --repeats 3

Weights are referenced from `trainer/config.toml` (`pretrained_model_name_or_path`), never copied;
override with `--model`. Everything else travels with this directory: the config and the six
images+captions live in `resources/`, and the run is assembled in a throwaway directory
(`/tmp/axl-fix2-repro` by default) so the repository and the datasets are never written to.

Needs the `axl` conda env and a GPU. Run it from anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
FAULT_MARKERS = ("HSA_STATUS_ERROR_MEMORY_FAULT", "Memory access fault by GPU node",
                 "GCVM_L2_PROTECTION_FAULT", "Page not present or supervisor privilege")
STEP_RE = re.compile(r"step=(\d+)\s+loss=")

KANAE_DIR = Path("/home/acite/LLM/Character/kanae")

# The packaged run samples at 768x768, 8 denoise steps, 1 repeat (`[validation]`). The `fast` and
# `mech` grids shrink that to 512x512 to keep a short run short; the sample *event* — the model
# offload/reload and the VAE pass — is what the fault needs, the number of denoise steps is not
# (see the mech table in README.md).
SAMPLE_SEED_1 = {"validation.sample_seed": 1}

CHEAP_SAMPLES = {
    "validation.sample_steps": 8,
    "validation.sample_repeat": 1,
    "validation.sample_width": 512,
    "validation.sample_height": 512,
}


# The packed config is the faulting run's; a variant only shifts one of the inputs the fault
# depends on, so each row of the table differs from the original in exactly one way.
#
# `packaged` rows are the second sighting (fixes/fix2-ex.md): SDXL, `train_batch_size` 3,
# `network_dim` 36, the six packaged kanae images, and they die at step 101 of 120. The `kanae`
# rows are the first sighting (fixes/fix2.txt): the same repo on the full kanae dataset with
# `train_batch_size = 2` + `network_dim = 36`, which aborted on the first backward.
GRID: list[tuple[str, dict]] = [
    ("original (packaged 6 images)", {}),
    ("original, run again", {}),
    ("seed 1145141919 (the surviving seed)", {"seed": 1145141919}),
    ("no HIP memory caching", {"no_hip_memory_caching": True}),
    ("sampling every 30 steps disabled", {"save_every_n_steps": 0}),
    ("network_dim 32", {"network_dim": 32}),
    ("fix2.txt: batch 2, dim 36, kanae", {"dataset": "kanae", "batch_size": 2, "steps": 3,
                                          "save_every_n_steps": 0}),
    ("fix2.txt control: batch 3, dim 36, kanae", {"dataset": "kanae", "steps": 3,
                                                  "save_every_n_steps": 0}),
    ("fix2.txt: batch 2, dim 36, 3-chunk captions", {"dataset": "kanae", "batch_size": 2,
                                                     "steps": 3, "save_every_n_steps": 0,
                                                     "caption_chunks": 3}),
    # The short one: same packaged config, but the 30-step cadence packed to every 2 steps, so the
    # sample path and the transition it needs land inside the first ten steps instead of step 101.
    ("fast: cadence 2, cheap samples", {"steps": 10, "save_every_n_steps": 2,
                                        "set": CHEAP_SAMPLES}),
]

# Which meta-operation actually sets the fault up? Each row removes or moves one thing the 100% row
# does, and the interesting signal is not "does it die" but *where*: the fault sits on a 462 -> 231
# shape transition (see seed_step_shapes.py), so a row that moves the death step or survives 120
# steps tells us which part of the step history has to be in place for the transition to be fatal.
#
# `validation.*` is shrunk in some rows: 35 denoise steps x 3 repeats at 1280x720 costs ~20 s per
# sample, which would dominate a short-cadence run. Sample *size* is itself one of the variables, so
# the first row re-runs the packaged cadence with the cheap samples as the control.
# The mech grid shows the death follows the *cadence events*, not the step count: with a 30-step
# cadence the packaged run dies at the 3rd sample (step 90) or 11 steps later in training, a 10-step
# cadence dies one step after its 3rd sample, a 5-step cadence dies at its 4th. So the shortest
# faithful repro packs the cadence instead of shortening the run.
FAST_GRID: list[tuple[str, dict]] = [
    ("fast: cadence 1, packaged samples", {"steps": 8, "save_every_n_steps": 1}),
    ("fast: cadence 1, cheap samples", {"steps": 8, "save_every_n_steps": 1, "set": CHEAP_SAMPLES}),
    ("fast: cadence 2, cheap samples", {"steps": 10, "save_every_n_steps": 2,
                                        "set": CHEAP_SAMPLES}),
    ("fast: cadence 3, cheap samples", {"steps": 12, "save_every_n_steps": 3,
                                        "set": CHEAP_SAMPLES}),
    # Cadence 1 survived 8 steps with 8 samples; does the densest cadence ever reach the bad layout?
    ("fast: cadence 1, cheap samples, 24 steps", {"steps": 24, "save_every_n_steps": 1,
                                                  "set": CHEAP_SAMPLES}),
]

# `network_dim = 32` aborts at step 4 - before the first sample - so the training path alone can
# reach the bad layout for some K. This scan looks for an even faster one: twelve steps is enough to
# catch a step-1..step-12 death, and the cadence stays packaged (30) so no sample happens inside the
# window and nothing but the rank differs from the canonical config.
# Silent damage: every captured fault is `RW: 0x0`, i.e. the Tensile kernel *reads* one page past a
# buffer. That cannot scribble on a neighbour; what it can do is read garbage into its own GEMM
# result when the page happens to be mapped, and then nothing crashes and no dmesg line appears.
# These rows attack that from two sides:
#   * `sample_seed = 1` gives the sample pass its own generator (`sampling.py` otherwise draws its
#     seed from the global RNG, which would make the trajectories diverge for innocent reasons), so
#     a cadence-on run and a cadence-off run must be bitwise identical for all 120 steps if the
#     sample pass leaves no trace on training state;
#   * different allocator layouts (cached / hipMalloc-per-tensor / expandable segments) put the same
#     computation at different addresses, so if an overread ever feeds garbage into a result, the
#     numbers in those runs cannot stay byte-identical.
INTEGRITY_GRID: list[tuple[str, dict]] = [
    ("integrity: seed 1145141920, cached allocator", {"steps": 120,
                                                      "set": SAMPLE_SEED_1}),
    ("integrity: seed 1145141920, no-hip caching",
     {"steps": 120, "no_hip_memory_caching": True, "set": SAMPLE_SEED_1}),
    ("integrity: seed 1145141919, cached allocator", {"seed": 1145141919, "steps": 120,
                                                      "set": SAMPLE_SEED_1}),
    ("integrity: seed 1145141919, no-hip caching", {"seed": 1145141919, "steps": 120, "no_hip_memory_caching": True, "set": SAMPLE_SEED_1}),
    ("integrity: seed 1145141919, expandable segments", {"seed": 1145141919, "steps": 120, "set": SAMPLE_SEED_1,
      "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}),
    ("integrity: seed 1145141919, sampling off", {"seed": 1145141919, "steps": 120, "save_every_n_steps": 0,
                                                  "set": SAMPLE_SEED_1}),
    # `sample_seed` turned out to flip the outcome too: seed 1145141919 dies at step 101 with
    # sample_seed 1, and survived twice with the packaged sample_seed 0. These two rows separate
    # "the sample seed decides it" from "run-to-run variance".
    ("integrity: seed 1145141919, packaged sample seed", {"seed": 1145141919, "steps": 120}),
    ("integrity: seed 1145141919, sample_seed 1 again", {"seed": 1145141919, "steps": 120, "set": SAMPLE_SEED_1}),
]

# The first integrity pass was confounded: two runs of the same config do not even agree on their
# cached latents (MIOpen picks different conv algorithms per process), so every trajectory differs
# from step 1 for a reason that has nothing to do with the fault. These rows run with `--data-dir`,
# i.e. one shared latent cache, and then add the comparison that matters: the same configuration
# twice, to see whether the training path is bitwise reproducible at all once its inputs are fixed.
INTEGRITY2_GRID: list[tuple[str, dict]] = [
    (f"integrity2 seed {seed}: {name}", {"seed": seed, "steps": 120, **overrides})
    for seed in (1145141920, 1145141919)
    for name, overrides in (
        ("cached allocator (a)", {"set": SAMPLE_SEED_1}),
        ("cached allocator (b, repeat)", {"set": SAMPLE_SEED_1}),
        ("no-hip caching", {"no_hip_memory_caching": True, "set": SAMPLE_SEED_1}),
        ("sampling off", {"save_every_n_steps": 0, "set": SAMPLE_SEED_1}),
    )
]

DIM_GRID: list[tuple[str, dict]] = [
    (f"dim scan: network_dim {dim}", {"steps": 12, "network_dim": dim})
    for dim in (16, 24, 32, 40, 48, 64)
]

# The first scan put the fastest abort at step 4 for `network_dim` 24 and 32. This second pass fills
# in the ranks between them, looking for one that dies even earlier.
DIM2_GRID: list[tuple[str, dict]] = [
    (f"dim scan 2: network_dim {dim}", {"steps": 12, "network_dim": dim})
    for dim in (8, 12, 20, 28, 36, 44, 56)
]
# The step-4 deaths happen without any sample, so the cadence is not what makes them fatal. Does the
# allocator layout still decide them? Same two ranks with hipMalloc-per-tensor.
DIM2_GRID += [
    (f"dim scan 2: network_dim {dim}, no-hip caching",
     {"steps": 12, "network_dim": dim, "no_hip_memory_caching": True})
    for dim in (24, 32)
]

MECH_GRID: list[tuple[str, dict]] = [
    ("mech: cadence 30 (control), cheap samples", {"steps": 120, "set": CHEAP_SAMPLES}),
    ("mech: cadence 5, cheap samples", {"steps": 40, "save_every_n_steps": 5,
                                        "set": CHEAP_SAMPLES}),
    ("mech: cadence 10, cheap samples", {"steps": 40, "save_every_n_steps": 10,
                                         "set": CHEAP_SAMPLES}),
    # Still swaps the models and decodes through the VAE, but does almost no denoising work: this
    # separates "a sample happened" from "the sample was expensive".
    ("mech: cadence 30, one-step samples", {"steps": 120,
                                            "set": {**CHEAP_SAMPLES,
                                                    "validation.sample_steps": 1}}),
    ("mech: sampling off", {"steps": 120, "save_every_n_steps": 0}),
    ("mech: gradient checkpointing off",
     {"steps": 120, "set": {"optimization.gradient_checkpointing_unet": False,
                            "optimization.gradient_checkpointing_te": False}}),
    ("mech: no latent disk cache",
     {"steps": 120, "set": {"optimization.cache_latents": False,
                            "optimization.cache_latents_to_disk": False}}),
    ("mech: no dataloader workers",
     {"steps": 120, "set": {"infrastructure.max_data_loader_n_workers": 0,
                            "infrastructure.persistent_workers": False}}),
    # If the overrun is fatal only because of where the caching allocator put things, a run that
    # returns the pool to a known state every step should never reach the bad layout.
    # The sample path is also the only place a large VAE decode allocation happens; a 256x256,
    # one-step sample keeps every offload/reload but shrinks that allocation to almost nothing.
    ("mech: cadence 30, tiny samples (256px, 1 step)",
     {"steps": 120, "set": {"validation.sample_steps": 1, "validation.sample_repeat": 1,
                            "validation.sample_width": 256, "validation.sample_height": 256}}),
    ("mech: flush memory every step",
     {"steps": 120, "set": {"optimization.flush_memory_every_step": True}}),
]


# ----------------------------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------------------------


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "api.py").is_file() and (candidate / "trainer" / "config.toml").is_file():
            return candidate
    raise SystemExit(f"could not find the repository root above {start} (needs api.py, trainer/)")


REPO_ROOT = find_repo_root(HERE)


def model_path(override: str | None) -> str:
    """The pipeline path the trainer would use: `trainer/config.toml` unless overridden."""
    if override:
        return override
    import tomllib

    with open(REPO_ROOT / "trainer" / "config.toml", "rb") as handle:
        return str(tomllib.load(handle)["environment"]["pretrained_model_name_or_path"])


def stage_dataset(dest: Path, source: Path) -> int:
    """Copy images+captions next to the run, so the run never writes into `resources/` or into
    the user's dataset directory (the trainer puts its latent cache inside `train_data_dir`)."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    count = 0
    for src in sorted(source.rglob("*")):
        if src.is_file() and src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".txt"):
            shutil.copy2(src, dest / src.name)
            count += src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    if not count:
        raise SystemExit(f"no images in {source}")
    return count


def pad_captions_to_chunks(data_dir: Path, chunks: int, tokenizer: object) -> int:
    """Grow every caption until its CLIP token stream needs `chunks` encoder chunks (77 tokens
    each), i.e. until the LoRA GEMM's `M` is `batch * 77 * chunks` on every step.

    `fixes/fix2.txt` derives its faulting size from `seq 231` (3 chunks), which no natural kanae
    caption reaches (the longest is 145 tokens -> 2 chunks), so the derivation is only testable
    with captions that long.
    """
    filler = "detailed background, soft lighting, highres, absurdres, illustration, " \
             "beautiful eyes, hair ornament, scenery, depth of field"
    padded = 0
    for caption in sorted(data_dir.glob("*.txt")):
        text = caption.read_text(encoding="utf-8").strip()
        while len(tokenizer(text, add_special_tokens=False, truncation=False).input_ids) \
                <= 75 * (chunks - 1) and len(text) < 4000:
            text = f"{text}, {filler}"
        if text != caption.read_text(encoding="utf-8").strip():
            caption.write_text(text, encoding="utf-8")
            padded += 1
    return padded


def make_mirror(dest: Path) -> None:
    """A throwaway repo root: symlinked trainer sources plus the generated config.toml, so the
    unmodified `trainer/main.py` runs against this script's config instead of the repo's."""
    (dest / "trainer").mkdir(parents=True, exist_ok=True)
    links = [(src, dest / "trainer" / src.name) for src in (REPO_ROOT / "trainer").glob("*.py")]
    links.append((REPO_ROOT / "text_processing.py", dest / "text_processing.py"))
    for src, target in links:
        if not target.is_symlink():  # re-running into a kept work dir must not fail
            os.symlink(src, target)


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def write_config(dest: Path, sections: dict, variant: dict) -> None:
    environment = sections["environment"]
    environment["train_data_dir"] = variant["data_dir"]
    environment["output_dir"] = variant["output_dir"]
    environment["logging_dir"] = variant["logging_dir"]
    environment["output_name"] = variant["output_name"]
    environment["pretrained_model_name_or_path"] = variant["model"]

    training = sections["training"]
    training["seed"] = variant["seed"]
    training["epoch"] = variant["epochs"]
    training["save_every_n_epochs"] = 0  # epoch checkpoints are pure disk cost for this repro
    if variant["save_every_n_steps"] is not None:
        training["save_every_n_steps"] = variant["save_every_n_steps"]
    training["train_batch_size"] = variant["batch_size"]
    if variant["network_dim"] is not None:
        sections["network"]["network_dim"] = variant["network_dim"]
    for dotted, value in (variant.get("set") or {}).items():
        section, _, key = dotted.partition(".")
        if section not in sections:
            raise SystemExit(f"no [{section}] table in the packaged config ({dotted})")
        sections[section][key] = value

    lines: list[str] = []
    for name, table in sections.items():
        lines.append(f"[{name}]")
        for key, value in table.items():
            if value is None:
                continue
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# running one child
# ----------------------------------------------------------------------------------------------


def child_env(runtime_dir: Path, *, no_hip_memory_caching: bool,
              extra: dict | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "AXL_RUNTIME_DIR": str(runtime_dir),
        "PYTHONUNBUFFERED": "1",
        # Same ROCm quieting/caching as start_train.sh.
        "AMD_LOG_LEVEL": "0",
        "CK_LOG_LEVEL": "0",
        "MIOPEN_ENABLE_LOGGING": "0",
        "MIOPEN_ENABLE_LOGGING_CMD": "0",
        "MIOPEN_LOG_LEVEL": "1",
        "MIOPEN_LOG_BUFFER_SIZE": "0",
        "MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS": "0",
        "MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_BWD_XDLOPS_AI_HEUR": "0",
        "MIOPEN_DEBUG_ENABLE_AI_IMMED_MODE_FALLBACK": "0",
        "MIOPEN_CUSTOM_CACHE_DIR": str(Path.home() / ".cache" / "miopen"),
        "MIOPEN_USER_DB_PATH": str(Path.home() / ".config" / "miopen"),
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
    })
    if no_hip_memory_caching:
        # fixes/fix2.txt: hipMalloc per allocation never reuses the blocks the overrunning Tensile
        # kernel walks past, at ~2.5x the step time.
        env["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
    for key, value in (extra or {}).items():
        env[key] = str(value)
    return env


def reap_run_processes(marker: str) -> list[int]:
    """Kill leftovers of a dead child (DataLoader forkserver workers survive its SIGABRT).

    `marker` is that run's own mirror path, so only its processes can match.
    """
    import signal

    me, parent = os.getpid(), os.getppid()
    reaped: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if marker in cmdline and "forkserver" in cmdline:
            try:
                os.kill(pid, signal.SIGTERM)
                reaped.append(pid)
            except OSError:
                pass
    return reaped


def log_has_fault(log_path: Path) -> bool:
    """True once the child has printed the KFD abort, i.e. before it spends time on teardown."""
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 4096))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return False
    return any(marker in tail for marker in FAULT_MARKERS)


def prune_weight_dirs(output_dir: Path, output_name: str) -> int:
    """Delete step/epoch checkpoint dirs of a finished run; samples and logs stay.

    A 120-step run writes ~60 epoch checkpoints at ~15 MB, which is a lot of tmpfs for a crash
    repro that only needs the log.
    """
    removed = 0
    for child in output_dir.glob(f"{output_name}_*"):
        if not child.is_dir() or child.name.endswith("_samples"):
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    return removed


def run_variant(label: str, overrides: dict, *, args: argparse.Namespace, work: Path,
                index: int) -> dict:
    import tomllib

    with open(RESOURCES / "original-config.toml", "rb") as handle:
        sections = tomllib.load(handle)

    claimed_seed = re.search(r"seed (\d+)", label)
    if claimed_seed and int(claimed_seed.group(1)) != int(overrides.get("seed", 1145141920)):
        # Labels are documentation; without this guard this grid ran copies of seed 1145141920 for
        # hours while claiming 1145141919.
        raise SystemExit(f"row {label!r} claims seed {claimed_seed.group(1)} but would run "
                         f"{overrides.get('seed', 1145141920)}; set 'seed' in the row")

    cli_env = {}
    for item in args.env:
        key, _, value = item.partition("=")
        cli_env[key] = value

    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    run_root = work / f"{index:02d}_{slug}"
    mirror, runtime_dir = run_root / "mirror", run_root / "runtime"
    output_dir, logging_dir = run_root / "outputs", run_root / "logs"
    for path in (run_root, runtime_dir, output_dir, logging_dir):
        path.mkdir(parents=True, exist_ok=True)
    make_mirror(mirror)

    source = KANAE_DIR if overrides.get("dataset") == "kanae" else Path(args.dataset)
    if not source.is_dir():
        raise SystemExit(f"dataset not found: {source}")
    if args.data_dir:
        # One shared directory means one latent cache: the VAE encode is *not* reproducible run to
        # run on this box (MIOpen picks different conv algorithms per process), so runs that must be
        # comparable have to read identical latents instead of each encoding its own copy.
        data_dir = Path(args.data_dir)
        staged = [p for p in data_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS] \
            if data_dir.is_dir() else []
        images = len(staged) if staged else stage_dataset(data_dir, source)
    else:
        data_dir = run_root / "data"
        images = stage_dataset(data_dir, source)
    padded = 0
    if overrides.get("caption_chunks"):
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
        padded = pad_captions_to_chunks(data_dir, int(overrides["caption_chunks"]),
                                        tokenizer)

    steps = int(overrides.get("steps", args.steps))
    batch_size = int(overrides.get("batch_size", sections["training"]["train_batch_size"]))
    steps_per_epoch = max(1, -(-images // batch_size))
    epochs = max(1, -(-steps // steps_per_epoch))
    # The trainer only knows whole epochs, so a row always runs at least one epoch.
    steps = epochs * steps_per_epoch
    output_name = "repro"
    write_config(mirror / "trainer" / "config.toml", sections, {
        "data_dir": str(data_dir), "output_dir": str(output_dir),
        "logging_dir": str(logging_dir), "output_name": output_name,
        "model": args.model_path, "seed": int(overrides.get("seed", 1145141920)),
        "epochs": epochs, "batch_size": batch_size,
        "network_dim": overrides.get("network_dim"),
        "save_every_n_steps": overrides.get("save_every_n_steps", 30),
        "set": overrides.get("set"),
    })

    if args.dry_run:
        print(f"      {label}: staged {images} images, wrote "
              f"{mirror / 'trainer' / 'config.toml'} (dry run)", flush=True)
        return {"label": label, "seed": int(overrides.get("seed", 1145141920)),
                "batch_size": batch_size, "network_dim": sections["network"]["network_dim"],
                "steps_requested": steps, "last_step": None, "verdict": "dry-run",
                "exit_code": None, "seconds": 0.0, "dataset": source.name, "images": images,
                "no_hip_memory_caching": bool(overrides.get("no_hip_memory_caching")),
                "save_every_n_steps": overrides.get("save_every_n_steps", 30),
                "config": str(mirror / "trainer" / "config.toml")}

    log_path = run_root / "train.out"
    started = time.time()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([sys.executable, "-u", "trainer/main.py"], cwd=mirror,
                                env=child_env(runtime_dir,
                                              no_hip_memory_caching=bool(overrides.get("no_hip_memory_caching")),
                                              extra={**cli_env, **(overrides.get("env") or {})}),
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        state_path = runtime_dir / "state.json"
        last_print = 0.0
        fault_seen_at: float | None = None
        while proc.poll() is None:
            time.sleep(2.0)
            if fault_seen_at is None and log_has_fault(log_path):
                # The KFD abort is printed well before the coredump handler gives up (~25 s), so a
                # caller in a hurry can stop the process here instead of paying for the teardown.
                # The marker, the kernel name and the dmesg entry are already on disk.
                fault_seen_at = time.time()
            if fault_seen_at and args.kill_after_fault and \
                    time.time() - fault_seen_at > args.kill_after_fault:
                proc.kill()
                break
            if not args.quiet and time.time() - last_print > 20:
                last_print = time.time()
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    step = state.get("training", {}).get("step")
                    print(f"      {label}: step {step}/{steps} ({time.time() - started:.0f}s)",
                          flush=True)
                except (OSError, json.JSONDecodeError):
                    pass
        proc.wait()
    seconds = time.time() - started

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    fault = proc.returncode != 0 and any(marker in log_text for marker in FAULT_MARKERS)
    if fault:
        verdict = "FAULT"
    elif proc.returncode == 0:
        verdict = "ok"
    else:
        verdict = f"error({proc.returncode})"
    reached = max((int(m.group(1)) for m in STEP_RE.finditer(log_text)), default=None)
    reaped = reap_run_processes(str(mirror)) if proc.returncode != 0 else []
    pruned = prune_weight_dirs(output_dir, output_name) if args.prune_checkpoints else 0

    entry = {
        "label": label, "seed": int(overrides.get("seed", 1145141920)),
        "batch_size": batch_size, "network_dim": sections["network"]["network_dim"],
        "no_hip_memory_caching": bool(overrides.get("no_hip_memory_caching")),
        "save_every_n_steps": overrides.get("save_every_n_steps", 30),
        "dataset": source.name, "images": images, "caption_chunks": overrides.get("caption_chunks"),
        "padded_captions": padded, "steps_requested": steps, "last_step": reached,
        "verdict": verdict,
        "exit_code": proc.returncode, "seconds": round(seconds, 1),
        "log": str(log_path), "reaped_workers": reaped, "pruned_dirs": pruned,
        "killed_after_fault": bool(fault_seen_at and args.kill_after_fault and
                                   proc.returncode == -9),
        "tensorboard": str(logging_dir),
    }
    print(f"      {label}: {verdict}, last step "
          f"{reached if reached is not None else '-'}, exit={proc.returncode}, "
          f"{seconds:.0f}s" + (f", reaped {len(reaped)} orphaned workers" if reaped else ""),
          flush=True)
    return entry


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------


def pick_grid(args: argparse.Namespace) -> list[tuple[str, dict]]:
    grid = {"mech": MECH_GRID, "fast": FAST_GRID, "dim": DIM_GRID,
            "integrity": INTEGRITY_GRID, "dim2": DIM2_GRID,
            "integrity2": INTEGRITY2_GRID}.get(args.grid, GRID)
    if args.row:
        wanted = args.row.split("|")
        selected = [item for item in grid if item[0] in wanted]
        if not selected:
            raise SystemExit(f"--row matched no row in --grid {args.grid}; labels: "
                             + " | ".join(label for label, _ in grid))
        return selected
    if args.only:
        wanted = [name.strip() for name in args.only.split(",")]
        grid = [item for item in grid if any(w in item[0] for w in wanted)]
        if not grid:
            raise SystemExit(f"--only {args.only} matched no row; rows: "
                             + ", ".join(label for label, _ in grid))
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=120,
                        help="training steps to request; the original config reached 120 and died "
                             "at 101")
    parser.add_argument("--only", default="",
                        help="comma-separated substrings; a row runs if it contains any of them "
                             "(so `--only fast` runs every fast row)")
    parser.add_argument("--row", default="",
                        help="exact row labels separated by `|`, for when --only is too loose")
    parser.add_argument("--grid", default="table",
                        choices=("table", "mech", "fast", "dim", "dim2", "integrity", "integrity2"),
                        help="`table` is the failing-vs-surviving grid, `mech` the meta-operation "
                             "isolation grid (same harness, one knob moved per row)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="how often to repeat every selected row")
    parser.add_argument("--list", action="store_true", help="print the grid and exit")
    parser.add_argument("--model", default=None, help="override pretrained_model_name_or_path")
    parser.add_argument("--env", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra child environment for every row, e.g. "
                             "--env HSA_SVM_GUARD_PAGES=0 (ROCm 10.0.0 environment variables, see "
                             "the ROCm environment variables reference)")
    parser.add_argument("--data-dir", default="",
                        help="use this directory as train_data_dir for every row (staged "
                             "once), so the runs share one latent cache")
    parser.add_argument("--dataset", default=str(RESOURCES / "dataset"),
                        help="images+captions copied into each run unless the row names its "
                             "own dataset (the `kanae` rows use " + str(KANAE_DIR) + ")")
    parser.add_argument("--work-dir", default="/tmp/axl-fix2-repro")
    parser.add_argument("--out", default=str(RESOURCES / "real-results.json"))
    parser.add_argument("--prune-checkpoints", type=int, default=1)
    parser.add_argument("--kill-after-fault", type=int, default=0,
                        help="seconds to wait after the fault appears in the log before killing "
                             "the child, instead of waiting ~25 s for its coredump teardown "
                             "(0 = let it exit on its own, which is the default and keeps the "
                             "authentic -6 exit code)")
    parser.add_argument("--dry-run", action="store_true",
                        help="stage the run and write its config, but do not start training "
                             "(catches bad --grid mech override keys without burning GPU time)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    grid = pick_grid(args)
    if args.list:
        for label, overrides in grid:
            print(f"{label:<44} {overrides}")
        return 0

    model = model_path(args.model)
    if not Path(model).is_dir():
        print(f"model path not found: {model}\n"
              f"point --model at a diffusers SDXL directory (trainer/config.toml holds the "
              f"author's path)", file=sys.stderr)
        return 2
    if os.environ.get("CONDA_DEFAULT_ENV") != "axl" and not args.quiet:
        print(f"note: CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}, training is "
              f"verified with the `axl` env", file=sys.stderr)

    args.model_path = model
    work = Path(args.work_dir)
    print(f"== {len(grid) * args.repeats} training runs through `trainer/main.py`")
    print(f"== model: {model}")
    print(f"== packaged dataset: {args.dataset} ({KANAE_DIR} for the fix2.txt rows)")
    print(f"== work dir: {work}\n")

    results: list[dict] = []
    index = 0
    for label, overrides in grid:
        for repeat in range(1, args.repeats + 1):
            index += 1
            name = label if args.repeats == 1 else f"{label} #{repeat}"
            results.append(run_variant(name, overrides, args=args, work=work, index=index))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n== table: the faulting paths and what switches them off")
    print(f"| {'row':<42} | {'dataset':>8} | {'steps':>5} | {'batch':>5} | {'dim':>3} | "
          f"{'seed':>10} | {'HIP cache':>9} | {'samples':>7} | {'last step':>9} | result |")
    print("|" + "---|" * 10)
    for entry in results:
        samples = "off" if entry["save_every_n_steps"] == 0 else str(entry["save_every_n_steps"])
        print(f"| {entry['label']:<42} | {entry['dataset']:>8} | {entry['steps_requested']:>5} | "
              f"{entry['batch_size']:>5} | {entry['network_dim']:>3} | {entry['seed']:>10} | "
              f"{'off' if entry['no_hip_memory_caching'] else 'on':>9} | {samples:>7} | "
              f"{entry['last_step'] if entry['last_step'] is not None else '-':>9} | "
              f"{entry['verdict']} |")

    faults = [entry for entry in results if entry["verdict"] == "FAULT"]
    print(f"\n== {len(faults)}/{len(results)} runs hit the fault; json: {args.out}")
    for entry in results:
        print(f"   {entry['verdict']:<7} {entry['label']}: "
              f"last step {entry['last_step'] if entry['last_step'] is not None else '-'}"
              f"/{entry['steps_requested']}, {entry['seconds']:.0f}s"
              + (f", {entry['pruned_dirs']} checkpoints pruned"
                 if entry.get("pruned_dirs") else ""))
    if all(entry["verdict"] == "dry-run" for entry in results):
        print("DRY-RUN (nothing started)")
        return 0
    print("REPRODUCED" if faults else "NO-FAULT-REPRODUCED")
    return 0 if faults else 1


if __name__ == "__main__":
    raise SystemExit(main())
