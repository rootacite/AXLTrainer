# Troubleshooting

## Known issue: ROCm bucket-step crash (FIXED — read this)

**Symptom:** on AMD ROCm, training on datasets with diverse image resolutions crashes with a hard segfault at a random step — `IOT instruction (core dumped)`, `amdgpu ... [gfxhub] page fault`, `GCVM_L2_PROTECTION_FAULT_STATUS`, faulty client TCP. Randomness came from shuffled batch order (crashes at step 5, 25, or hundreds of steps in).

**Root cause:** with `bucket_reso_steps = 64`, some aspect-ratio buckets produced latent dimensions that are multiples of 8 but **not 16** (the VAE downsamples by 8; e.g. 832×448 → 104×56 latents). ROCm/MIOpen/Flash-Attention kernels require 16-byte-aligned dimensions, so misaligned vectorized loads caused GPU-side out-of-bounds access → MMU page fault → process kill.

**Fix:** keep `bucket_reso_steps = 128` in `trainer/config.toml` (128 ÷ 8 = 16, so latent dims stay divisible by 16/32). A full 1200+ step run completed cleanly after this change.

> **Golden rule on AMD ROCm: always use `bucket_reso_steps = 128`.**

Details are documented in [`fixes/fix1.txt`](../fixes/fix1.txt).

## Known issue: gfx1201 Tensile page fault (not the bucket-step bug)

**Symptom:** `bucket_reso_steps` is already 128, encoding finishes, then at some training step (sometimes the first backward, sometimes step 4, sometimes step 101) the process dies with `-6` / `HSA_STATUS_ERROR_MEMORY_FAULT` and no Python traceback. dmesg: `amdgpu ... [gfxhub] page fault`, `GCVM_L2_PROTECTION_FAULT_STATUS:0x0080113B` or `0x0090113B` (TCP client, `RW: 0x0`). Kernel names look like `Cijk_Ailk_Bjlk_…_MT32x32x128_…_ISA1201` or `…_MT64x128x16_…_ISA1201`.

**Root cause:** RX 9070 XT (gfx1201) Tensile GEMM reads one page past a torch allocation during LoRA backward (TE LoRA → UNet cross-attn `d(encoder_hidden_states)` is enough). Fatality depends on whether the next page is unmapped (`HSA_SVM_GUARD_PAGES` defaults to 1). The same overrun against a mapped neighbour is silent. Isolated UNet backward with detached embeds, and a self-contained SDXL loop with the same GEMM shapes (`fixes/fix2/crash.py`), do **not** abort — they never sit the operand next to a hole.

On this box, unmodified `trainer/main.py` with the packaged kanae set:

| config | result |
| --- | --- |
| `network_dim` 12 / 24 / 32, batch 3, seed 1145141920 | **FAULT at step 4**, ~25 s, no sampling |
| `network_dim` 36, batch 3, seed 1145141920, sample every 30 | **FAULT at step 101** (6/6 after reboot) |
| same, `save_every_n_steps = 0` | 120/120 ok |
| same, seed 1145141919 | 120/120 ok |
| `train_batch_size = 2` + `network_dim = 36` (the original `fixes/fix2.txt` recipe) | **does not** abort here |

The seed does not change tensor *values* (`lora_B` starts at 0). It changes caption-shuffle CLIP chunk count, hence `M = batch × chunks × 77` ∈ {231, 462}, hence *where* a 462↔231 switch lands versus the allocator.

`accelerator.prepare()` of UNet + both TEs after the trainer's VAE-cache/`empty_cache` history is the moment the fatal layout is created. Replacing that call with `.to(device)` lets a 12-step dim-32 run finish; that was measured with copies under `/tmp` and is **not** applied in `trainer/`. Dropping `prepare` also drops Accelerate's autocast wrap, so it is not a numeric no-op.

**Workaround:** `PYTORCH_NO_HIP_MEMORY_CACHING=1` before importing torch avoids both abort routes (~2.2× slower; every alloc is `hipMalloc`). Avoid `network_dim` 12 / 24 / 32 on this card if you can. Do **not** set `HSA_SVM_GUARD_PAGES=0` — that only makes the overrun miss the guard page.

Repro, tables, integrity diffs: [`fixes/fix2/README.md`](../fixes/fix2/README.md). First write-up: [`fixes/fix2.txt`](../fixes/fix2.txt). Second sighting: [`fixes/fix2-ex.md`](../fixes/fix2-ex.md).

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
| `PYTORCH_NO_HIP_MEMORY_CACHING` | trainer (workaround) | Set to `1` before importing torch to skip the HIP caching allocator. Avoids the gfx1201 Tensile abort; ~2.2× slower. See [Known issue: gfx1201](#known-issue-gfx1201-tensile-page-fault-not-the-bucket-step-bug). |
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
