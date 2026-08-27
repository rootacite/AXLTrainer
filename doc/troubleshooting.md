# Troubleshooting

## Known issue: ROCm bucket-step crash (FIXED — read this)

**Symptom:** on AMD ROCm, training on datasets with diverse image resolutions crashes with a hard segfault at a random step — `IOT instruction (core dumped)`, `amdgpu ... [gfxhub] page fault`, `GCVM_L2_PROTECTION_FAULT_STATUS`, faulty client TCP. Randomness came from shuffled batch order (crashes at step 5, 25, or hundreds of steps in).

**Root cause:** with `bucket_reso_steps = 64`, some aspect-ratio buckets produced latent dimensions that are multiples of 8 but **not 16** (the VAE downsamples by 8; e.g. 832×448 → 104×56 latents). ROCm/MIOpen/Flash-Attention kernels require 16-byte-aligned dimensions, so misaligned vectorized loads caused GPU-side out-of-bounds access → MMU page fault → process kill.

**Fix:** keep `bucket_reso_steps = 128` in `trainer/config.toml` (128 ÷ 8 = 16, so latent dims stay divisible by 16/32). A full 1200+ step run completed cleanly after this change.

> **Golden rule on AMD ROCm: always use `bucket_reso_steps = 128`.**

Details are documented in [`fixes/fix1.txt`](../fixes/fix1.txt).

## Common failure modes

| Symptom | Cause / fix |
| --- | --- |
| `FileNotFoundError` or permission errors at startup | The shipped `config.toml` contains the author's local paths. Edit `[environment]` paths (see [Configuration](configuration.md)). |
| `RuntimeError: ... another run is holding the lock` (or similar) | A previous run didn't exit cleanly. Check `train.lock` in the runtime dir; the trainer releases it on `end_run`. Use `train_reset` / the dashboard Reset, or remove the stale lock file. |
| Dashboard shows "training process is no longer running" / status flips to `error` | The trainer PID died. Read `train.log` in the runtime dir for the traceback, then Reset to clear the state. |
| Dashboard can't start / "set AXL_PYTHON" | The trainer deps aren't on `PATH` as `python3`. Point `AXL_PYTHON` at your environment's interpreter, e.g. `AXL_PYTHON=$CONDA_PREFIX/bin/python ./gradlew :desktopApp:run`. |
| `train_start` fails with "a live training PID already exists" | A process marked `finished` hasn't exited yet, or a stale PID is listed. Wait for it to exit (or kill it), then Reset. |
| Status stuck at `starting` with no progress | The trainer failed right after spawn. Check `train.log`; reconcile marks dead PIDs to `error` after a grace window. |
| Dataset scan aborts with "Found isolated tag file" | An orphan `.txt` with no matching image exists. Delete the orphan (or pass `--allow-orphans` to `agent.py`). |
| Training finishes but no `{name}_final` checkpoint | Early-stop during encoding saves no LoRA; during training it saves only if that step had no checkpoint yet. |
| GPU OOM during training | Lower `train_batch_size` / `gradient_accumulation_steps`, or disable `cache_latents_to_disk` batching changes. Pause (offload) before doing other GPU work. |
| Chart zoom doesn't zoom | Zoom needs the mouse over the chart: `Ctrl`+wheel = X axis, `Shift`+wheel = Y axis. |
| `sample_seed = 0` images look random across runs | That's intentional — `0` means "random seed per repeat" (printed in the log). Set a fixed seed for reproducibility. |

## Environment variables

| Variable | Where it matters | Meaning |
| --- | --- | --- |
| `AXL_PYTHON` | Ranko | Interpreter used to run `api.py` / the trainer (default `python3` on `PATH`). |
| `AXL_RUNTIME_DIR` | `api.py`, trainer | Overrides the runtime dir for `state.json` / `command.json` / `train.lock` / `train.log`. |
| `XDG_RUNTIME_DIR` | `api.py`, trainer | Used for the default runtime dir (`$XDG_RUNTIME_DIR/axltrainer`). |
| `PYTHONUNBUFFERED` | launchers | Set to `1` by `start_*.sh` and Ranko so logs flush immediately. |
| `AMD_LOG_LEVEL`, `CK_LOG_LEVEL`, `MIOPEN_*` | `start_*.sh` | Suppress ROCm/MIOpen driver log noise and pin the MIOpen cache to `~/.cache/miopen`. |
| `PYTORCH_CUDA_ALLOC_CONF` | `start_train.sh` | `max_split_size_mb:128,garbage_collection_threshold:0.8` — reduces fragmentation. |
| `ORT_MIGRAPHX_MODEL_CACHE_PATH` / `ORT_MIGRAPHX_CACHE_PATH` | `tagger/`, `trainer/env.py` | Compiled ONNX/MIGraphX cache location (`migraphx_cache/`). |

## Runtime state directory

If you need to inspect or reset state by hand, the runtime dir resolves in this order:

```bash
$AXL_RUNTIME_DIR
$XDG_RUNTIME_DIR/axltrainer
/tmp/axltrainer-$UID
```

Files: `state.json` (status/progress), `command.json` (one-shot pause/resume/stop), `train.lock` (single-run lock), `train.log` (trainer output).

```bash
cat "${XDG_RUNTIME_DIR:-/tmp}/axltrainer/state.json"   # current status
```

## Noisy terminal output

`start_train.sh` filters known ROCm driver chatter (`grid_desc`, `CandidateSelectionModel`, `metadata`) from stdout, and `start_api.sh` pins the MIOpen cache + log levels. If you run `python -u trainer/main.py` directly and see noise, it's harmless driver logging — or use the scripts.

## Getting help from the logs

- **Trainer tracebacks** → `<runtime_dir>/train.log`.
- **Run status / progress** → `<runtime_dir>/state.json`.
- **Metrics** → TensorBoard: `tensorboard --logdir <logging_dir>`.
- **IPC traffic** → Ranko inherits `api.py`'s stderr to its console; `start_api.sh` lets you drive the helper manually on stdin/stdout.
