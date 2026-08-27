# Training

This guide covers running training, what happens during a run, and how the pause / resume / early-stop machinery works.

## Running training

The trainer is launched with `start_train.sh`, which sets the AMD/ROCm environment (MIOpen cache dirs, log suppression, allocator settings) and runs:

```bash
bash start_train.sh
```

Equivalent to `python -u trainer/main.py` with stdout filtered of noisy driver lines. The working directory must be the repo root so `trainer/config.toml` resolves.

The dashboard starts it the same way — `api.py`'s `train_start` spawns `bash start_train.sh` detached (`setsid`), so **closing the GUI does not stop training**.

## What a run does, phase by phase

1. **Startup (`starting`)** — loads `config.toml`, acquires the run lock (`train.lock`, fails fast if another run holds it), seeds RNG, builds the pipeline (SDXL model, LoRA adapters via PEFT, dataset + DataLoader, optimizers).
2. **Encoding (`encoding`)** — if `cache_latents` and `cache_latents_to_disk` are on, all images are pre-encoded to latents by a 3-stage pipeline (CPU decode/resize → batched VAE encode per bucket → atomic `.pt` writes into `<train_data_dir>/.latents_cache/`). Already-cached images are skipped. The VAE is moved back to CPU afterwards and GPU memory is flushed. Progress is published as `encoding.current/total`.
3. **Training (`training`)** — the epoch loop: batches are grouped by aspect-ratio bucket, prompts are encoded (chunked for long prompts, `clip_skip` applied), noise + timesteps are added, and the UNet predicts the noise target (with optional `noise_offset`). Per-bucket losses are weighted by bucket size. After gradient accumulation, UNet grads are clipped to `max_grad_norm`, TE grads to `te_max_grad_norm`, and both optimizers step. Every `save_every_n_steps` steps the run saves a checkpoint and generates samples.
4. **Sampling (`sampling`)** — validation images are generated with the current LoRA weights (scheduler swapped to Euler-A with hand-built sigmas, aggressive VAE offloading, interruptible denoising). One checkpoint's samples are saved as `<output_name>_<step:06d>_<repeat>.png` in `{output_dir}/{output_name}_samples/`.
5. **Finish (`finished`)** — the final LoRA is saved (unless stopped early), and the lock is released. On exception the status becomes `error` and the traceback is recorded.

TensorBoard metrics are written to `{logging_dir}/{output_name}/`:

| Tag | Meaning |
| --- | --- |
| `Train/Loss` | Per-step MSE loss. |
| `Train/Avg_Loss` | Kohya-style epoch-window moving average. |
| `UNet/LR/Effective_Actual_LR` | Schedule-Free UNet effective LR. |
| `TE/LR/Base_Scheduled` / `TE/LR/Effective_Actual_LR` | Text-encoder scheduled LR. |

## Pause / resume / early stop

Control is file-based: an external caller (the dashboard or `api.py`) writes a one-shot command to `command.json` (`pause` / `resume` / `stop`), and the trainer consumes it at the next **swap-safe point** — after each encoding item, after each optimizer step, and after each denoising step during sampling.

### Pause (GPU offload)

`pause` offloads everything to CPU in stages — UNet → text encoders → optimizer state (including Schedule-Free's `z` / `exp_avg_sq`) → VAE — then calls `empty_cache`. Progress is published as `swap.{stage, detail, current, total}` while the status is `pausing`; when done the status becomes `paused` with `paused_from` recording which phase (encoding / training / sampling) was interrupted.

While paused, the trainer process sleeps at the safe point and **keeps running** (no GPU memory in use). GPU memory is released so you can use the card for something else.

### Resume

`resume` reloads what the paused phase needs (only the VAE for `encoding`; UNet + text encoders + optimizers for `training`/`sampling`), restores train/eval modes, and returns to the phase it left. Status transitions `resuming` → `encoding` / `training` / `sampling`.

### Early stop

`stop` sets status `stopping`; the current phase aborts at the next safe point. The behavior depends on where the stop lands:

- **During encoding** — no LoRA is saved (`stopped_during = encoding`).
- **During training** — if `global_step > 0` and that step has no checkpoint yet, an emergency final checkpoint is saved as `{output_dir}/{output_name}_final/{output_name}.safetensors` (`stopped_during = training`).
- **During sampling** — leftover repeats are skipped; the step's checkpoint already exists.

In all cases the run ends in `finished` (with a `detail` of `stopped_during_*`), not `error`.

## Checkpoint and artifact layout

All paths derive from `output_name` (sanitized):

| Artifact | Path |
| --- | --- |
| Per-step LoRA | `{output_dir}/{name}_s{step:06d}/{name}.safetensors` |
| Final LoRA | `{output_dir}/{name}_final/{name}.safetensors` |
| Sample images | `{output_dir}/{name}_samples/{name}_{step:06d}_{repeat}.png` |
| TensorBoard logs | `{logging_dir}/{name}/` |
| Latent cache | `{train_data_dir}/.latents_cache/<sha1>.pt` |
| Runtime state / commands / lock / log | `$AXL_RUNTIME_DIR` → `$XDG_RUNTIME_DIR/axltrainer` → `/tmp/axltrainer-$UID` (`state.json`, `command.json`, `train.lock`, `train.log`) |

**Checkpoint format:** PEFT state dicts are remapped to kohya keys (`lora_unet_*`, `lora_te1_*`, `lora_te2_*`), converted to bf16, and saved with alpha scalars plus `modelspec.*` and `ss_*` metadata — directly loadable in ComfyUI or with kohya sd-scripts.

## Watching progress

- **Dashboard** (recommended): live metric cards, charts, progress bars, and sample gallery.
- **Streamlit viewer**: `streamlit run ui.py` — read-only, same data sources.
- **TensorBoard**: `tensorboard --logdir <logging_dir>`.
- **Runtime state**: `cat $XDG_RUNTIME_DIR/axltrainer/state.json` (or the equivalent resolved path).
- **Logs**: `tail -f <runtime_dir>/train.log` (trainer stdout/stderr; driver log noise is filtered by `start_train.sh`).

## Cleanup

A run leaves samples, TensorBoard logs, and checkpoints behind. Two ways to clean up (same targets, same underlying helper `trainer/cleanup.py`):

```bash
python clean.py        # interactive; asks before deleting checkpoints (safe by default)
```

Or the dashboard's **Reset** button, which additionally clears the `finished`/`error` state so a new run can start. Both delete:

1. `{output_dir}/{name}_samples/`
2. `{logging_dir}/{name}/`
3. Optionally (with confirmation / `delete_weights`) all `{output_dir}/{name}_*` checkpoint dirs.

The latent cache is **not** deleted — it's reusable across runs.

## Notes and gotchas

- **One run at a time.** `train.lock` makes a second concurrent run fail immediately.
- **Stale state.** If a training process dies hard, the next status read reconciles the dead PID to `error` ("training process is no longer running"). Reset to clear.
- **Stop during sampling** keeps the already-saved step checkpoint; the partially-denoised image is discarded.
- **`sample_seed = 0`** gives each repeat a fresh random seed (printed to the log); set a fixed seed for reproducibility.
- **Schedule-Free optimizer** requires `train()`/`eval()` mode toggling around sampling; the code does this automatically.
