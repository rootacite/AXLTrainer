# Troubleshooting

## Known issue: ROCm bucket-step crash (FIXED — read this)

**Symptom:** on AMD ROCm, training on datasets with diverse image resolutions crashes with a hard segfault at a random step — `IOT instruction (core dumped)`, `amdgpu ... [gfxhub] page fault`, `GCVM_L2_PROTECTION_FAULT_STATUS`, faulty client TCP. Randomness came from shuffled batch order (crashes at step 5, 25, or hundreds of steps in).

**Root cause:** with `bucket_reso_steps = 64`, some aspect-ratio buckets produced latent dimensions that are multiples of 8 but **not 16** (the VAE downsamples by 8; e.g. 832×448 → 104×56 latents). ROCm/MIOpen/Flash-Attention kernels require 16-byte-aligned dimensions, so misaligned vectorized loads caused GPU-side out-of-bounds access → MMU page fault → process kill.

**Fix:** keep `bucket_reso_steps = 128` in `config.toml` (128 ÷ 8 = 16, so latent dims stay divisible by 16/32). A full 1200+ step run completed cleanly after this change.

> **Golden rule on AMD ROCm: always use `bucket_reso_steps = 128`.**

The field report this was diagnosed from is sealed in `archive/` — 涉及负责任披露流程，暂不公开.

## Known issue: gfx1201 Tensile page fault (not the bucket-step bug)

**Symptom:** `bucket_reso_steps` is already 128, encoding finishes, then at some training step (sometimes the first backward, sometimes tens of steps in) the process dies with `-6` / `HSA_STATUS_ERROR_MEMORY_FAULT` and no Python traceback. dmesg: `amdgpu ... [gfxhub] page fault`, `GCVM_L2_PROTECTION_FAULT_STATUS:0x0080113B` or `0x0090113B` (TCP client, `RW: 0x0`), and the kernel name ends in `_ISA1201`.

**Root cause:** RX 9070 XT (gfx1201) Tensile GEMM reads one page past a torch allocation during LoRA backward (TE LoRA → UNet cross-attn `d(encoder_hidden_states)` is enough). Fatality depends on whether the next page is unmapped (`HSA_SVM_GUARD_PAGES` defaults to 1); the same over-read against a mapped neighbour is silent. The read is bounded (at most 4 KiB past the operand) and its values are discarded inside the kernel — the tail loop zeroes the registers it loaded — so it moves no numbers; the unmapped page is what turns it into a process kill. Which shapes and configurations lose that particular lottery is a scatter rather than a rule, and the experiment records behind it are sealed in `archive/` — 涉及负责任披露流程，暂不公开.

The HIP toolchain notes for this stack (the wheel layout, the four environment fixes that make `hipcc` work, the VMM API pitfalls) and the probes they go with are sealed in `archive/` — 涉及负责任披露流程，暂不公开.

**Workarounds:** nothing trainer-side fixes this — these kernels read past their operands whatever the configuration, and no stack has been immune: the current pin and the one before it both die, on different shapes. The measures below change the *outcome*, not the over-read, and are ranked by cost:

- **Treat it as a restart event** — what this repo does by default. The process exits `-6` and leaves `state.json` at `training` with a dead PID; a Reset clears it. Keep the checkpoint cadence tight enough that losing one interval is acceptable, and restart with `[training].resume_lora_path` (weights only — step/epoch counters restart at 0).
- **The allocation patch** (`[environment].amdfq = "tail"` or `"vmm"`, see [Configuration](configuration.md)) — the supported answer on RDNA 4. The over-read still happens; the page behind the block is mapped, so it lands in memory that exists instead of the hole that kills the process. This is what the repo runs with by default.
- **`HSA_SVM_GUARD_PAGES=0`** in the training process's environment (the shell that runs `start_train.sh`, or the script itself) removes the page the over-read lands on: measured fault → run completes. The catch is that it hides *every* SVM over-read in that process, so do not leave it on while chasing another memory problem.
- **`PYTORCH_NO_HIP_MEMORY_CACHING=1`** before importing torch gives every tensor its own `hipMalloc`, so no operand ends at the edge of a cached segment: also measured to complete, at a large step-time cost.

## Common failure modes

| Symptom | Cause / fix |
| --- | --- |
| `FileNotFoundError` or permission errors at startup | The shipped `config.toml` (repo root) contains the author's local paths. Edit `[environment]` paths (see [Configuration](configuration.md)). |
| Dashboard says another client owns the helper | The helper serves **one client at a time**: the first Ranko (or web companion) to connect owns it, and every other one is told who has it. Close the other session, or wait five seconds after it disconnects, then Retry. |
| `RuntimeError: ... another run is holding the lock` (or similar) | A previous run didn't exit cleanly. Check `train.lock` in the runtime dir; the trainer releases it on `end_run`. Use `train_reset` / the dashboard Reset, or remove the stale lock file. |
| Dashboard shows "training process is no longer running" / status flips to `error` | The trainer PID died. Read `train.log` in the runtime dir for the traceback, then Reset to clear the state. |
| Dashboard can't start / "set AXL_PYTHON" | The trainer deps aren't on `PATH` as `python3`. Point `AXL_PYTHON` at your environment's interpreter, e.g. `AXL_PYTHON=$CONDA_PREFIX/bin/python ./gradlew :desktopApp:run`. |
| `train_start` fails with "a live training PID already exists" | A process marked `finished` hasn't exited yet, or a stale PID is listed. Wait for it to exit (or kill it), then Reset. |
| Status stuck at `starting` with no progress | The trainer failed right after spawn. Check `train.log`; reconcile marks dead PIDs to `error` after a grace window. |
| Dataset scan aborts with "Found isolated tag file" | An orphan `.txt` with no matching image exists. Delete the orphan (or pass `--allow-orphans` to `agent.py`). |
| Training finishes but no `{name}_final` checkpoint | Early-stop during encoding saves no LoRA; during training it saves only if that step had no checkpoint yet. |
| GPU OOM during training | Lower `train_batch_size` / `gradient_accumulation_steps`, or disable `cache_latents_to_disk` batching changes. Pause (offload) before doing other GPU work. |
| Starting a run or the tagger hangs with no output, and the machine has no proxy | A Hugging Face Hub request is waiting on a socket with no route. Neither path depends on the Hub any more: `load_sdxl_pipeline` pins a single-file base model's component configs cache-only, and `tagger2/main.py` resolves its model locally and never asks. If you still see a stall, it is something else reaching out — read the last line of `tagger2/main.py`'s stderr, which names the step it is in. |
| Chart zoom doesn't zoom | Zoom needs the mouse over the chart: `Ctrl`+wheel = X axis, `Shift`+wheel = Y axis. |
| Browse / Save As opens a plain Java-style dialog instead of the desktop's own | No XDG desktop portal is reachable, so FileKit falls back to the AWT/Swing dialog. Install and run `xdg-desktop-portal` plus a backend (`xdg-desktop-portal-kde` / `-gtk`) for the session and relaunch. |
| Panel **Generate sample** is greyed out / `generate_sample` says "the GPU is in use" | The trainer process is still alive — running *or* paused. A second SDXL would have to load into the same VRAM, so generation waits until the run has finished or been stopped. |
| Generated sample fails with "was trained on X but config.toml targets Y" | The checkpoint's `ss_base_model_version` differs from `[model_spec].base_model_version`. Point `[model_spec]` at the checkpoint's family (or pick a checkpoint of the current one) so the LoRA is applied to the model it was trained on. |
| Generated sample fails with "no LoRA tensors ... match this SDXL LoRA layout" | The checkpoint is not a kohya SDXL LoRA for this base (or is corrupt). Its `.log` next to the PNG has the traceback. |
| Sample generation runs out of VRAM | Close other GPU users, reduce the run's `sample_width`/`sample_height`, or generate after the training process has exited. |
| `sample_seed = 0` images look random across runs | That's intentional — `0` means "random seed per repeat" (printed in the log). Set a fixed seed for reproducibility. |
| Samples of a **v-pred base** (e.g. NoobAI XL vPred) come out as noise | The sampler resolved epsilons for a model trained on velocities. Point `[model_spec]` at the original checkpoint: its `v_pred` / `ztsnr` marker tensors (or a diffusers directory's `scheduler/scheduler_config.json`) are read automatically — there is no switch to set (see [Configuration](configuration.md)). A base whose markers were stripped by a converter must be replaced by the original file. |
| Samples of a v-pred base are near-black or blow out, while the same prompt is fine in ComfyUI | Usually the prompt, not the trainer. NoobAI XL vPred reads lighting tags such as `soft lighting, warm light` as "there is no background light" and collapses the scene to black in **ComfyUI too** — drop the tag (`warm atmosphere` and no lighting tag both render normally). The contrast itself is what `guidance_rescale` is for: the shipped `0.6` (ComfyUI's `RescaleCFG`) took a sample's black-pixel share from 15.3 % to 7.7 % at the same seed. |

## Environment variables

| Variable | Where it matters | Meaning |
| --- | --- | --- |
| `AXL_PYTHON` | Ranko | Interpreter used to run `api.py` / the trainer (default `python3` on `PATH`). |
| `AXL_RUNTIME_DIR` | `api.py`, trainer | Overrides the runtime dir for `state.json` / `command.json` / `train.lock` / `train.log`. |
| `XDG_RUNTIME_DIR` | `api.py`, trainer | Used for the default runtime dir (`$XDG_RUNTIME_DIR/axltrainer`). |
| `PYTHONUNBUFFERED` | launchers | Set to `1` by `start_*.sh` and Ranko so logs flush immediately. |
| `AMD_LOG_LEVEL`, `CK_LOG_LEVEL`, `MIOPEN_*` | `start_*.sh`, `tagger2/` | Suppress ROCm/MIOpen driver log noise and pin the MIOpen cache (`~/.cache/miopen`, or the repo's `tagger2/miopen_cache/` for the tagger). |
| `PYTORCH_CUDA_ALLOC_CONF` | `start_train.sh` | `max_split_size_mb:128,garbage_collection_threshold:0.8` — reduces fragmentation. |
| `HSA_SVM_GUARD_PAGES` | trainer (workaround) | ROCr's SVM guard pages (default `1`). Setting `0` stops the gfx1201 Tensile over-read from faulting, at the price of hiding any other SVM over-read in that process. See [Known issue: gfx1201](#known-issue-gfx1201-tensile-page-fault-not-the-bucket-step-bug). |
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
- **IPC traffic** → Ranko inherits `api.py`'s stderr to its console; `start_api.sh` starts the same loopback WebSocket helper.
