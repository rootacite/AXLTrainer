# Overview

This document explains what the project is made of and how the pieces talk to each other.

## What this is

AXLTrainer is a complete, local-first pipeline for training **SDXL LoRA** models on a custom image dataset:

- a **Python training engine** (`trainer/`) that does the actual training,
- a **desktop dashboard** (`ranko/`, Kotlin/Compose Multiplatform) that manages the dataset, edits `config.toml`, and controls training from a GUI,
- a **JSON-lines IPC helper** (`api.py`) that bridges the two,
- a **read-only web viewer** (`ui.py`, Streamlit),
- and a set of **dataset utility scripts** (`tools/`, `tagger/`).

There is no HTTP server and no inference/generation service — `api.py` is a local helper, and training runs on your machine.

## Process model

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Ranko (desktop GUI)        │         │  bash start_train.sh         │
│  ranko/  (Kotlin/JVM)       │         │  └─ python -u trainer/main.py│
│                             │         │     (detached, setsid)       │
│  ┌──────────────┐  NDJSON   │         │     │                        │
│  │ api.py       │◄──────────┤         │     ▼                        │
│  │ (IPC helper) │  stdin/   │         │  trainer/control.py          │
│  └──────┬───────┘  stdout   │         │  state.json / command.json / │
│         │                   │         │  train.lock  (runtime dir)   │
│         │                   │         └──────────────┬───────────────┘
│         └── reads ── TensorBoard logs (logging_dir)  │
│         └── reads ── sample PNGs (output_dir/*_samples)              │
└─────────────────────────────┘
```

- **Ranko** starts `api.py` as a subprocess and talks to it with newline-delimited JSON over stdin/stdout. All control goes through `api.py`.
- **Training** is spawned by `api.py` via `bash start_train.sh` in a new session (`setsid`). It is **detached**: closing Ranko does not stop training.
- The running trainer publishes its state to a **runtime directory** (`$AXL_RUNTIME_DIR` → `$XDG_RUNTIME_DIR/axltrainer` → `/tmp/axltrainer-$UID`) as `state.json` (status + progress), `command.json` (one-shot pause/resume/stop commands), and `train.lock` (single-run lock). `api.py` reads `state.json` and writes `command.json` on the trainer's behalf.
- Training metrics go to **TensorBoard** under `logging_dir/{output_name}/`, and sample images land in `output_dir/{output_name}_samples/`. `api.py` reads both to serve `dashboard` / `list_samples`.

## Repository layout

```
.
├── api.py                     # NDJSON IPC helper (stdin/stdout)
├── clean.py                   # interactive CLI cleanup
├── environment.yml            # conda env manifest (env name: axl)
├── start_train.sh             # trainer launcher (AMD/ROCm env vars)
├── start_api.sh               # debug launcher for api.py
├── text_processing.py         # long-prompt chunking + SDXL dual-encoder encoding
├── ui.py                      # Streamlit read-only viewer
├── API.md                     # IPC protocol reference
├── trainer/
│   ├── main.py                # training entry point (no CLI args)
│   ├── config.py              # TrainConfig dataclass + TOML loading
│   ├── config.toml            # the actual configuration
│   ├── dataset.py             # SDXLLoraDataset, bucketing, captions
│   ├── cache.py               # pipelined latent pre-encode (warm_latent_cache)
│   ├── loop.py                # train_one_epoch, per-bucket loss, logging
│   ├── sampling.py            # sample generation (interruptible denoising)
│   ├── models.py              # pipeline load, PEFT→kohya conversion, checkpoints
│   ├── setup.py               # one-stop build of all training objects
│   ├── control.py             # state machine + runtime-dir IPC + run lock
│   ├── device_swap.py         # GPU↔CPU offload/restore, safe points
│   ├── loss_log.py            # Kohya-style epoch-window avg-loss recorder
│   ├── cleanup.py             # discovery + deletion of run artifacts
│   ├── env.py                 # ROCm cache dirs, memory flushing
│   ├── utils.py               # image/caption/bucket/time-id helpers
│   └── test_warm_latent_cache.py
├── ranko/                     # Kotlin/Compose desktop app (AxlRanko)
│   ├── desktopApp/            # window entry point
│   ├── shared/                # UI, viewmodels, IPC client, config editing
│   └── tools/agent.py         # machine-friendly dataset CLI (for scripts/agents)
├── tools/                     # dataset utility scripts (see doc/dataset-tools.md)
├── tagger/                    # ONNX caption generator (WD-tagger style)
├── fixes/                     # resolved issue reports (e.g. ROCm bucket step)
└── tests: test_api_ipc.py, test_train_control.py
```

## Training data flow

```
trainer/config.toml ──► TrainConfig (trainer/config.py)
                              │
                              ▼
trainer/main.py ──► begin_run (acquires train.lock)
     │              set_seed
     │              build_train_objects (trainer/setup.py)
     │                ├─ load SDXL pipeline (models.py) → vae / unet / te1 / te2
     │                ├─ apply PEFT LoRA (network_dim/alpha/dropout) + flash attention
     │                ├─ SDXLLoraDataset + DataLoader (dataset.py)
     │                │    └─ images + .txt captions, aspect-ratio bucketing
     │                ├─ UNet optimizer (Schedule-Free AdamW) + TE optimizer (AdamW)
     │                └─ Accelerator (TensorBoard → logging_dir)
     │
     ├─ warm_latent_cache (cache.py)      [if cache_latents + cache_latents_to_disk]
     │    CPU decode → batched VAE encode → atomic .pt files in <data>/.latents_cache
     │
     ├─ epoch loop → train_one_epoch (loop.py)
     │    batch → group by bucket → build inputs (cached latent or on-demand encode)
     │    → encode prompts (text_processing.py) → noise + timesteps → UNet → MSE
     │    → backward → clip grads → both optimizers step
     │    → every save_every_n_steps: save LoRA checkpoint + generate sample
     │    → at_safe_point() every step (handles pause / resume / stop)
     │
     ├─ final checkpoint: {output_name}_final/{name}.safetensors
     └─ end_run → release train.lock
```

## Control plane: statuses

The trainer exposes a status state machine that both the dashboard and scripts can observe via `state.json`:

```
idle ──► starting ──► encoding ──► training ──► sampling ──► finished
   ▲         │            │            │            │            │
   │         └────────────┴────────────┴────────────┴────────────┘
   │                        (early stop / completion)
   └── reset (train_reset / clean.py)
```

Pause/resume adds `pausing` and `paused` (GPU weights offloaded to CPU) and `resuming` between the current phase and itself; early stop adds `stopping`. See [Training](training.md) for the full lifecycle and [API.md](../API.md) for the wire format.

## What reads what

| Consumer | Reads | Writes |
| --- | --- | --- |
| `trainer/main.py` | `config.toml`, dataset images/captions, `.latents_cache/` | `state.json`, `train.lock`, TensorBoard logs, checkpoints, samples |
| `api.py` | TensorBoard logs, `{output_dir}/{name}_samples/`, `state.json` | `command.json` (on pause/resume/stop), spawned trainer process |
| Ranko (dashboard) | `api.py` responses | `api.py` requests |
| `ui.py` (Streamlit) | TensorBoard logs, sample PNGs | — (read-only) |
| `clean.py` | output/log dirs | deletes run artifacts |

## Design notes worth knowing

- **Configuration is TOML-only.** `trainer/main.py` takes no command-line arguments; `trainer/config.toml` (with hardcoded fallbacks in `trainer/config.py`) is the single source of truth. The TOML file always wins over the Python defaults.
- **Checkpoints are ComfyUI-ready.** PEFT state dicts are remapped to kohya `lora_unet_*` / `lora_te1_*` / `lora_te2_*` keys, converted to bf16, and saved with `modelspec.*` + `ss_*` metadata.
- **The dashboard never touches your GPU.** All GPU work happens in the detached trainer process; Ranko only spawns `api.py` and renders what it returns.
- **Safety rails:** a run lock prevents two concurrent runs; a dead-PID reconciliation marks stale `state.json` as `error`; dataset scans abort on orphan caption files; Reset confirms exact paths before deleting.
