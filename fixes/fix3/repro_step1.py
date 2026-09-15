#!/usr/bin/env python3
"""The step-1 abort of the current config, packaged: the real trainer, a throwaway mirror, no edits.

With `config.toml` as committed (`network_dim = 48`, `train_batch_size = 2`) the run dies inside the
**first** optimizer step, every time, with the gfx1201 Tensile page fault. This script is the
reproducer for `fixes/fix3/README.md`: it assembles a throwaway repo root under `/tmp`, stages a
dataset next to the run, starts the unmodified `trainer/main.py` there, and reports whether the child
faulted at which step.

    python repro_step1.py --list                     # the packaged rows
    python repro_step1.py --only "step-1 abort"      # the 100% row (~30 s)
    python repro_step1.py --grid dim                 # rank scan: which network_dim die at step 1
    python repro_step1.py --row "dim 64" --steps 3

Rows differ from the packaged config in exactly one way, so a row that survives and a row that dies
differ in that one knob. Every row is capped by `--steps` (default 1): the fault is on the first
backward, and a surviving row would otherwise spend six minutes on a 320-step epoch. `epoch = 1` and
`save_every_n_steps = 0` keep the run short — the step-1 death happens long before either matters.

Weights are referenced from `config.toml`'s `pretrained_model_name_or_path`, never copied. The dataset
is staged (copied) out of the user's data directory, because the trainer writes its latent cache into
`train_data_dir`. Needs the `axl_rocm_7_14` conda env and a GPU. Run it from anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
FAULT_MARKERS = ("HSA_STATUS_ERROR_MEMORY_FAULT", "Memory access fault by GPU node",
                 "GCVM_L2_PROTECTION_FAULT", "Page not present or supervisor privilege",
                 "Memory Fault Error")
STEP_RE = re.compile(r"step=(\d+)\s")


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "api.py").is_file() and (candidate / "config.toml").is_file():
            return candidate
    raise SystemExit(f"could not find the repository root above {start} (needs api.py, config.toml)")


REPO_ROOT = find_repo_root(HERE)


# ------------------------------------------------------------------------------------------------
# the rows
# ------------------------------------------------------------------------------------------------

# The packaged row set. `packaged` is the committed `config.toml` with the author's paths swapped
# for this run's; everything else is a single-knob change from it.
GRID: list[tuple[str, dict]] = [
    ("step-1 abort: repo config (dim 48, batch 2)", {}),
    ("step-1 abort, repeat", {}),
    ("dim 64 (the rank that trained 3230 steps on 2026-09-15)", {"network_dim": 64}),
    ("dim 48, batch 1", {"train_batch_size": 1}),
    ("dim 48, batch 4", {"train_batch_size": 4}),
    ("dim 48, hipMalloc per tensor", {"no_hip_memory_caching": True}),
    ("dim 48, guard pages off", {"env": {"HSA_SVM_GUARD_PAGES": "0"}}),
    ("dim 48, no dataloader workers",
     {"set": {"infrastructure.max_data_loader_n_workers": 0,
              "infrastructure.persistent_workers": False}}),
    ("dim 48, no sampling cadence", {"set": {"training.save_every_n_steps": 0}}),
    # Not runnable here: without UNet gradient checkpointing the step does not fit on a 16 GiB card.
    # Measured 2026-09-15 at 1280x768 batch 2: `torch.OutOfMemoryError: Tried to allocate 20.00 MiB
    # ... 14.99 GiB is allocated by PyTorch` inside `gelu`, and that failed allocation is what took
    # the desktop down with it. `fixes/fix2/README.md` recorded the same negative at batch 3.
    ("dim 48, gradient checkpointing off",
     {"set": {"optimization.gradient_checkpointing_unet": False},
      "skip": "OOMs by construction on a 16 GiB card (see README); --force runs it behind the "
              "VRAM watchdog, which fires long before the allocator reaches the card's limit"}),
    ("dim 48, te checkpointing on", {"set": {"optimization.gradient_checkpointing_te": True}}),
    ("dim 48, dropout 0", {"set": {"network.network_dropout": 0.0}}),
    ("dim 48, seed 1145141920", {"seed": 1145141920}),
    ("dim 48, 6 packaged images", {"dataset_subset": 6}),
    ("dim 48, 2 packaged images", {"dataset_subset": 2}),
    ("dim 48, 12 packaged images", {"dataset_subset": 12}),
    ("dim 48, no loss mask (alpha stripped)", {"dataset_alpha": False, "dataset_subset": 6}),
]

# Which `network_dim` values die on the first step? The rank is the LoRA GEMM's K, and `fixes/fix2`
# found the same question answered differently per rank on the ROCm 10 stack.
#
# Twelve steps, not one: the sampler walks through buckets as the epoch goes on, so the step a row dies
# at names the bucket that killed it. `probe_batches.py --batches 12` prints that map
# (0:1280x768, 1:1408x768, 2:1408x768, 3:1280x768, 4:512x1920, 5:1408x768, 6:1408x768, 7:1280x768,
# 8:1408x768, 9:512x2176, 10:1408x768, 11:512x2176), so "died at step 1" means the 1408x768 bucket.
DIM_GRID: list[tuple[str, dict]] = [
    (f"dim scan: network_dim {dim}", {"network_dim": dim, "steps": 12})
    for dim in (8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64)
]

# The batch size scales the LoRA GEMM's M (`M = batch * 77 * chunks`), so it is the other half of the
# shape. One row per (batch, dim) pair that the fault could depend on.
BATCH_GRID: list[tuple[str, dict]] = [
    (f"batch {batch}: dim {dim}", {"train_batch_size": batch, "network_dim": dim, "steps": 2})
    for batch in (1, 2, 3, 4)
    for dim in (24, 32, 48, 64)
]

# Which shapes are needed at all? These rows change the geometry the first batch is built from.
SHAPE_GRID: list[tuple[str, dict]] = [
    ("shape: no bucketing (train_resolution 1024)", {"set": {"bucketing.enable_bucket": False},
                                                     "steps": 2}),
    ("shape: bucket_reso_steps 64 (fixes/fix1 territory)", {"set": {"bucketing.bucket_reso_steps": 64},
                                                            "steps": 2}),
    ("shape: max_token_length 75 (one CLIP chunk)", {"set": {"network.max_token_length": 75},
                                                     "steps": 2}),
    ("shape: max_token_length 150 (two CLIP chunks)", {"set": {"network.max_token_length": 150},
                                                       "steps": 2}),
]

GRIDS = {"default": GRID, "dim": DIM_GRID, "batch": BATCH_GRID, "shape": SHAPE_GRID}

# A single-knob row that survives may be surviving on the allocator layout rather than on the knob,
# so "survives" is only a result once it repeats. Run with `--repeats 3`.
STABILITY_GRID: list[tuple[str, dict]] = [
    ("stability: dim 48, batch 2 (the abort)", {}),
    ("stability: dim 64, batch 2", {"network_dim": 64}),
    ("stability: dim 48, batch 1", {"train_batch_size": 1}),
    ("stability: dim 48, batch 4", {"train_batch_size": 4}),
    ("stability: dim 48, dropout 0", {"set": {"network.network_dropout": 0.0}}),
    ("stability: dim 48, te checkpointing on",
     {"set": {"optimization.gradient_checkpointing_te": True}}),
    ("stability: dim 48, hipMalloc per tensor", {"no_hip_memory_caching": True}),
    ("stability: dim 48, guard pages off", {"env": {"HSA_SVM_GUARD_PAGES": "0"}}),
]

GRIDS["stability"] = STABILITY_GRID

# The seed decides which images land in the first batch (the sampler shuffles per bucket with
# `seed + epoch`) and how their captions shuffle (hence the CLIP chunk count, hence `M`), so it is
# the cheapest way to move the first step's shapes without touching the config.
SEED_GRID: list[tuple[str, dict]] = [
    (f"seed scan: {seed}", {"seed": seed, "steps": 1})
    for seed in (1145141919, 1145141920, 1145141921, 1145141922, 1145141923, 1145141924)
]

GRIDS["seed"] = SEED_GRID

# The correlation the pack exists for: one bucket per row, one image set per bucket, so every step of
# that row draws the same spatial shape and the only thing changing between rows is the bucket and the
# rank. The image names come from `probe_batches.py --names-per-bucket`; `probe_batches.py --data-dir`
# on the staged directory (which the harness keeps) then reports the chunk counts that row actually
# ran with, because the caption shuffle is keyed on the image path.
#
# Rows stage their images, so they run under `--prewarm`: the encode pass is part of the allocation
# history the abort depends on, and the real run never performs it.
BUCKET_SETS: dict[str, list[str]] = {
    "1280x768": ["0389.png", "0103.png", "0023.png", "0414.png"],
    "1408x768": ["0002.png", "0013.png", "0014.png", "0018.png"],
    "512x1920": ["0005.png", "0015.png", "0019.png", "0025.png"],
    "512x2176": ["0016.png", "0022.png", "0026.png", "0029.png"],
}
BUCKET_GRID: list[tuple[str, dict]] = [
    (f"bucket {bucket}: dim {dim}",
     {"dataset_names": names, "network_dim": dim, "steps": 1})
    for bucket, names in BUCKET_SETS.items()
    for dim in (24, 48, 64)
]

GRIDS["bucket"] = BUCKET_GRID

# The 12-step rank scan runs into the VRAM watchdog on the high ranks: twelve steps of the largest
# buckets accumulate to ~14 GiB, which is the watchdog's neighbourhood, so those rows lose their
# verdict to a memory stop. Three steps cover buckets 1280x768 and 1408x768 — the two the abort lives
# on — and stay well inside the card, so every rank gets a verdict.
DIM3_GRID: list[tuple[str, dict]] = [
    (f"dim short scan: network_dim {dim}", {"network_dim": dim, "steps": 3})
    for dim in (8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64)
]

GRIDS["dim3"] = DIM3_GRID

# The packaged mini-dataset: six 2120x1280 images from the bucket the real run's first batch came
# from, each with a caption that needs exactly two CLIP chunks. Chosen by measurement, not by name:
# `probe_batches.py` shows every batch of this set is bucket 1280x768 / seq 154 / M=308, i.e. the
# shape the real run dies on — whereas the first six images of the real directory (tall 3.6k-pixel
# crops) put a single image in bucket 640x1664 and survive.
PACKAGED_NAMES = ["0389.png", "0103.png", "0023.png", "0414.png", "0067.png", "0031.png"]
PACKAGED_GRID: list[tuple[str, dict]] = [
    ("packaged mini-dataset (6 images, bucket 1280x768, 2 chunks)",
     {"dataset_names": PACKAGED_NAMES, "steps": 1}),
]

GRIDS["packaged"] = PACKAGED_GRID


# ------------------------------------------------------------------------------------------------
# assembling and running one child
# ------------------------------------------------------------------------------------------------


def model_path(override: str | None) -> str:
    if override:
        return override
    import tomllib

    with open(REPO_ROOT / "config.toml", "rb") as handle:
        return str(tomllib.load(handle)["environment"]["pretrained_model_name_or_path"])


def make_mirror(dest: Path) -> None:
    """Symlinked trainer sources: the unmodified `trainer/main.py`, this run's generated config."""
    (dest / "trainer").mkdir(parents=True, exist_ok=True)
    for src in (REPO_ROOT / "trainer").glob("*.py"):
        target = dest / "trainer" / src.name
        if not target.is_symlink():
            os.symlink(src, target)
    target = dest / "text_processing.py"
    if not target.is_symlink():
        os.symlink(REPO_ROOT / "text_processing.py", target)


def stage_dataset(dest: Path, source: Path, subset: int | None, keep_alpha: bool,
                  names: list[str] | None = None, keep: bool = False) -> int:
    """Copy images+captions next to the run: the trainer writes its latent cache inside the data dir.

    `subset` takes the first N images in `list_images` order, which is the order the sampler buckets
    them in — the first batch is then drawn from the same bucket the full dataset's first batch is.
    `names` stages an explicit list instead, which is how the packaged mini-dataset is built.
    `keep` reuses an already-staged directory, cache included: a warm `.latents_cache/` is what the
    real run has, and building one changes the allocations the first step starts from.
    """
    if keep and dest.is_dir() and any(dest.iterdir()):
        return len([p for p in dest.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS
                    and not p.name.lower().endswith(".mask.png")])
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    if names is None:
        names = sorted(p.name for p in source.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                       and not p.name.lower().endswith(".mask.png"))
    if subset:
        names = names[:subset]
    for name in names:
        shutil.copy2(source / name, dest / name)
        caption = source / f"{Path(name).stem}.txt"
        if caption.is_file():
            shutil.copy2(caption, dest / caption.name)
        mask = source / f"{Path(name).stem}.mask.png"
        if keep_alpha and mask.is_file():
            shutil.copy2(mask, dest / mask.name)
    if not keep_alpha:
        strip_alpha(dest)
    return len(names)


def strip_alpha(data_dir: Path) -> None:
    """Flatten alpha onto white so the loss mask has to come from a sidecar, not the channel.

    The real dataset is 518/640 alpha-bearing PNGs, so `load_loss_mask` builds a mask tensor per
    sample. This variant removes that ingredient without changing the pixels' RGB content.
    """
    from PIL import Image

    for path in sorted(data_dir.glob("*.png")):
        if path.name.endswith(".mask.png"):
            continue
        with Image.open(path) as img:
            if img.mode not in ("RGBA", "LA"):
                continue
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img.convert("RGB"), mask=img.convert("RGBA").getchannel("A"))
            background.save(path, format="PNG")


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
    training["epoch"] = 1
    training["save_every_n_epochs"] = 0
    training["save_every_n_steps"] = 0
    training["train_batch_size"] = variant["batch_size"]
    if variant["network_dim"] is not None:
        sections["network"]["network_dim"] = variant["network_dim"]
        sections["network"]["network_alpha"] = max(1, variant["network_dim"] // 2)
    for dotted, value in (variant.get("set") or {}).items():
        section, _, key = dotted.partition(".")
        if section not in sections:
            raise SystemExit(f"no [{section}] table in the packaged config ({dotted})")
        sections[section][key] = value

    lines: list[str] = []
    for name, table in sections.items():
        lines.append(f"[{name}]")
        for key, value in table.items():
            if value is None or isinstance(value, list):
                continue
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")


def child_env(runtime_dir: Path, *, no_hip: bool, extra: dict | None) -> dict[str, str]:
    """`start_train.sh`'s environment, so the child is the same process the real launcher starts."""
    env = dict(os.environ)
    env.update({
        "AXL_RUNTIME_DIR": str(runtime_dir),
        "PYTHONUNBUFFERED": "1",
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
    if no_hip:
        # Every tensor its own hipMalloc: the allocator cannot hand back a block whose neighbour is
        # unmapped, at ~2.5x the step time.
        env["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
    for key, value in (extra or {}).items():
        env[key] = str(value)
    return env


def reap_run_processes(marker: str) -> list[int]:
    """Kill the leftovers of a dead child: DataLoader forkserver workers survive its SIGABRT."""
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
        if marker in cmdline:
            try:
                os.kill(pid, signal.SIGKILL)
                reaped.append(pid)
            except OSError:
                pass
    return reaped


def tail_has_fault(log_path: Path, markers=FAULT_MARKERS) -> str | None:
    try:
        with open(log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 8192))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    for marker in markers:
        if marker in tail:
            return marker
    return None


def kernel_from_log(log_path: Path) -> str | None:
    """The `Cijk_...` name the KFD abort reported, which is the Tensile kernel that overran."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"kernel: (Cijk_\S+)", text)
    return match.group(1) if match else None


def vram_used_bytes() -> int | None:
    """VRAM in use, straight from sysfs (~1 ms, no driver call).

    Needed because this bug's neighbours are OOMs: the step runs at ~15 GiB of a 16 GiB card, so a
    row that drops an activation-saving setting can wedge the whole machine (the compositor loses its
    allocation). The watchdog below kills such a row before the allocator exhausts the card.
    """
    for path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")):
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def run_row(label: str, overrides: dict, *, args: argparse.Namespace, work: Path, index: int) -> dict:
    import tomllib

    with open(RESOURCES / "config.toml", "rb") as handle:
        sections = tomllib.load(handle)

    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    run_root = work / f"{index:02d}_{slug}"
    mirror, runtime_dir = run_root / "mirror", run_root / "runtime"
    output_dir, logging_dir = run_root / "outputs", run_root / "logs"
    for path in (run_root, runtime_dir, output_dir, logging_dir):
        path.mkdir(parents=True, exist_ok=True)
    make_mirror(mirror)

    # The previous row's teardown can still be draining VRAM; starting into a full card is how a
    # "just one more row" turns into a wedged desktop.
    for _ in range(60):
        used = vram_used_bytes()
        if used is None or used < 1.5 * 1024 ** 3:
            break
        time.sleep(1.0)
    else:
        print(f"    VRAM still at {vram_used_bytes() / 1024 ** 3:.1f} GiB before this row, skipping",
              flush=True)
        return {"row": label, "overrides": overrides, "vram_killed": True, "fault": None,
                "exit": None, "steps_reached": 0, "images": 0, "seconds": 0.0,
                "peak_vram_gb": round((vram_used_bytes() or 0) / 1024 ** 3, 2),
                "kernel": None, "killed_after_fault": False, "timed_out": False, "oom": False,
                "reaped_workers": 0, "run_root": str(run_root)}

    source = Path(args.dataset)
    if not source.is_dir():
        raise SystemExit(f"dataset not found: {source}")
    # A row that names its own images, or asks for their alpha stripped, needs a staged copy; every
    # other row runs against `--dataset` itself, whose warm `.latents_cache/` it only reads (the
    # cache key is the absolute path, so entries exist only for the images already encoded there).
    staged = any(key in overrides for key in ("dataset_subset", "dataset_names", "dataset_alpha"))
    if args.in_place and not staged:
        data_dir = source
        images = len([p for p in source.iterdir()
                      if p.suffix.lower() in IMAGE_EXTENSIONS
                      and not p.name.lower().endswith(".mask.png")])
    else:
        data_dir = run_root / "data"
        images = stage_dataset(data_dir, source, overrides.get("dataset_subset", args.dataset_subset),
                               overrides.get("dataset_alpha", True),
                               overrides.get("dataset_names"), keep=args.keep_data)

    variant = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "output_name": "fix3",
        "model": model_path(args.model),
        "seed": overrides.get("seed", 1145141919),
        "batch_size": overrides.get("train_batch_size", 2),
        "network_dim": overrides.get("network_dim"),
        "set": overrides.get("set"),
    }
    write_config(mirror / "config.toml", sections, variant)

    cli_env = {}
    for item in args.env:
        key, _, value = item.partition("=")
        cli_env[key] = value
    cli_env.update(overrides.get("env") or {})

    log_path = run_root / "train.out"
    env = child_env(runtime_dir, no_hip=bool(overrides.get("no_hip_memory_caching")),
                    extra=cli_env or None)
    steps_wanted = overrides.get("steps", args.steps)

    # `--child-script` swaps the trainer's own entry point for an instrumented one (see `probe_op.py`),
    # copied into the mirror so it runs with the same cwd and the same config resolution.
    entry = "trainer/main.py"
    if args.child_script:
        entry = Path(args.child_script).name
        shutil.copy2(args.child_script, mirror / entry)

    started = time.time()
    if args.prewarm and data_dir != source:
        # A throwaway child that only walks the cache pass, so the measured child starts from the same
        # warm `.latents_cache/` the real run has instead of encoding in-process — the encode allocates
        # and frees VAE workspace, which is part of the allocation history this fault depends on.
        prewarm_log = run_root / "prewarm.out"
        with open(prewarm_log, "wb") as log:
            warm = subprocess.Popen([sys.executable, "-u", entry], cwd=mirror, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 300
        while time.time() < deadline and warm.poll() is None:
            payload = {}
            state = runtime_dir / "state.json"
            if state.is_file():
                try:
                    payload = json.loads(state.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    payload = {}
            if payload.get("encoding", {}).get("done"):
                break
            time.sleep(0.3)
        try:
            os.killpg(os.getpgid(warm.pid), signal.SIGKILL)
        except OSError:
            pass
        warm.wait(timeout=60)
        reap_run_processes(str(mirror))
        for stale in (runtime_dir / "state.json", runtime_dir / "train.lock"):
            stale.unlink(missing_ok=True)
        print(f"    prewarmed in {time.time() - started:.1f}s", flush=True)
        started = time.time()

    with open(log_path, "wb") as log:
        child = subprocess.Popen([sys.executable, "-u", entry], cwd=mirror, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    marker = str(mirror)
    fault, reached, killed = None, 0, False
    peak_vram, vram_killed, timed_out = 0, False, False
    limit = int(args.vram_limit_gb * 1024 ** 3)
    deadline = started + args.row_timeout
    # The watchdog only arms once encoding is done. The latent cache pass spikes VRAM on its own —
    # the largest bucket (512x2304) encoded at batch sizes that reach ~12.7 GiB — and it is the same
    # pass every successful run on this box already performs, so killing a row there would be a false
    # positive. The step is where the unexplained growth lives.
    encoding_done = False
    try:
        while True:
            if child.poll() is not None:
                break
            fault = tail_has_fault(log_path)
            state = runtime_dir / "state.json"
            if state.is_file():
                try:
                    payload = json.loads(state.read_text(encoding="utf-8"))
                    encoding_done = encoding_done or bool(payload.get("encoding", {}).get("done"))
                    reached = max(reached, int(payload.get("training", {}).get("step") or 0))
                    if reached >= steps_wanted:
                        try:
                            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                        except OSError:
                            pass
                        break
                except (ValueError, OSError):
                    pass
            used = vram_used_bytes()
            if used:
                peak_vram = max(peak_vram, used)
                if encoding_done and used > limit:
                    print(f"    VRAM watchdog: {used / 1024 ** 3:.1f} GiB > "
                          f"{args.vram_limit_gb:.1f} GiB, killing", flush=True)
                    vram_killed = True
            if time.time() > deadline:
                print(f"    row timeout after {args.row_timeout}s, killing", flush=True)
                timed_out = True
            if fault or vram_killed or timed_out:
                try:
                    os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                except OSError:
                    pass
                killed = killed or fault is not None
                break
            time.sleep(0.4)
        child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except OSError:
            pass
    finally:
        reaped = reap_run_processes(marker)

    if fault is None:
        fault = tail_has_fault(log_path)
    oom = "OutOfMemoryError" in log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "row": label,
        "overrides": {k: v for k, v in overrides.items() if k not in ("set", "skip")}
        | (overrides.get("set") or {}),
        "images": images,
        "steps_reached": reached,
        "fault": fault,
        "oom": oom,
        "exit": child.returncode,
        "killed_after_fault": killed,
        "vram_killed": vram_killed,
        "timed_out": timed_out,
        "peak_vram_gb": round(peak_vram / 1024 ** 3, 2),
        "kernel": kernel_from_log(log_path) if fault else None,
        "seconds": round(time.time() - started, 1),
        "reaped_workers": len(reaped),
        "run_root": str(run_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default="default", choices=sorted(GRIDS))
    parser.add_argument("--row", default=None, help="exact row label")
    parser.add_argument("--only", default=None, help="substring match over row labels")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1, help="stop the child after this step")
    parser.add_argument("--dataset", default=None, help="defaults to config.toml's train_data_dir")
    parser.add_argument("--in-place", action="store_true",
                        help="run against --dataset directly instead of a staged copy (reads its "
                             "warm latent cache; writes nothing as long as every entry is cached)")
    parser.add_argument("--keep-data", action="store_true",
                        help="reuse a staged data dir and its latent cache instead of re-staging")
    parser.add_argument("--prewarm", action="store_true",
                        help="run a throwaway child first so the measured child starts from a warm "
                             "latent cache, like the real run does (staged rows only)")
    parser.add_argument("--dataset-subset", type=int, default=None,
                        help="stage only the first N images (list_images order)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for the child")
    parser.add_argument("--child-script", default=None,
                        help="run this file instead of trainer/main.py inside the mirror "
                             "(used with probe_op.py)")
    parser.add_argument("--vram-limit-gb", type=float, default=13.5,
                        help="kill a row whose VRAM use crosses this, once encoding is done. A step "
                             "at this config reaches ~12.7 GiB on a 16 GiB card (measured: the second "
                             "batch's 1408x768 bucket), and 12.5 GiB clipped a row that was fine")
    parser.add_argument("--row-timeout", type=float, default=240.0,
                        help="wall-clock kill per row, seconds")
    parser.add_argument("--force", action="store_true", help="run the rows marked skip")
    parser.add_argument("--work", default="/tmp/axl-fix3-repro")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", default=None, help="write the results to this file")
    args = parser.parse_args()

    if args.dataset is None:
        import tomllib

        with open(REPO_ROOT / "config.toml", "rb") as handle:
            args.dataset = str(tomllib.load(handle)["environment"]["train_data_dir"])
    if not model_path(args.model):
        raise SystemExit("no pretrained_model_name_or_path")
    if not Path(model_path(args.model)).is_dir():
        raise SystemExit(f"model path not found: {model_path(args.model)}")

    rows = GRIDS[args.grid]
    if args.row:
        rows = [row for row in rows if row[0] == args.row]
    elif args.only:
        rows = [row for row in rows if args.only.lower() in row[0].lower()]
    if not rows:
        raise SystemExit("no rows selected; use --list")

    skipped = [(label, overrides["skip"]) for label, overrides in rows if overrides.get("skip")]
    if skipped and not args.force:
        rows = [row for row in rows if not row[1].get("skip")]
    if args.list:
        for label, overrides in rows:
            mark = " [skip]" if overrides.get("skip") else ""
            print(f"  {label}{mark}   {json.dumps({k: v for k, v in overrides.items() if k != 'skip'})}")
        for label, reason in skipped:
            print(f"  (skipped) {label}: {reason}")
        return 0
    for label, reason in skipped:
        print(f"skipping {label!r}: {reason}", flush=True)

    work = Path(args.work)
    print(json.dumps({"dataset": args.dataset, "model": model_path(args.model),
                      "grid": args.grid, "rows": len(rows), "repeats": args.repeats,
                      "steps": args.steps, "work": str(work)}), flush=True)

    results, index = [], 0
    for _ in range(args.repeats):
        for label, overrides in rows:
            index += 1
            print(f"\n[{index}] {label}", flush=True)
            result = run_row(label, overrides, args=args, work=work, index=index)
            print("    " + json.dumps(result), flush=True)
            results.append(result)

    print("\n=== summary ===")
    print(f"{'row':<58} {'steps':>5} {'exit':>5}  verdict")
    for result in results:
        verdict = f"FAULT ({result['fault']})" if result["fault"] else "ok"
        if not result["fault"] and result["exit"] not in (0, -15):
            # A child that dies without a fault and without reaching a step is a broken row, not a
            # survivor — probe_op.py failing to import is exactly this shape.
            verdict = f"exited {result['exit']} without a fault (see {result['run_root']}/train.out)"
        print(f"{result['row']:<58} {result['steps_reached']:>5} {str(result['exit']):>5}  {verdict}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
