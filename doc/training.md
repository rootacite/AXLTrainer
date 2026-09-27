# Training

This guide covers running training, what happens during a run, and how the pause / resume / early-stop machinery works.

## Running training

The trainer is launched with `start_train.sh`, which sets the AMD/ROCm environment (MIOpen cache dirs, log suppression, allocator settings) and runs:

```bash
bash start_train.sh
```

Equivalent to `python -u trainer/main.py` with stdout filtered of noisy driver lines. The working directory must be the repo root so `config.toml` resolves.

The dashboard starts it the same way — `api.py`'s `train_start` spawns `bash start_train.sh` detached (`setsid`), so **closing the GUI does not stop training**. `[environment].amdfq` (`none` / `tail` / `vmm`) is applied here as `LD_PRELOAD` (Ranko Utils → ROCm). A missing patch `.so` fails the start.

## What a run does, phase by phase

1. **Startup (`starting`)** — loads `config.toml`, creates this run's timestamped output + log directory, acquires the run lock (`train.lock`, fails fast if another run holds it), seeds RNG, resolves `[model_spec].base_model_version` to a model family, and builds that family's pipeline (LoRA adapters via PEFT, dataset + DataLoader, optimizers). When `[training].resume_lora_path` is set, the LoRA weights are loaded here (before `accelerator.prepare`). Unknown or inconsistent spec strings fail here; `sd3.5-large` is a catalogued family but training is not implemented yet.
2. **Encoding (`encoding`)** — if `cache_latents` and `cache_latents_to_disk` are on, all images are pre-encoded to latents by a 3-stage pipeline (CPU decode/resize/fit+pad → batched VAE encode per bucket → atomic `.pt` writes into that folder's `<folder>/.latents_cache/`). Already-cached images are skipped — a file that cannot be read, or that does not hold the keyed bucket's latent, is not one of them: it is re-encoded over the file and reported on stderr. The `.pt` name hashes the absolute path, the bucket **and** the fit geometry, so changing the bucket rule or the `[bucketing]` clamps re-encodes the dataset once instead of silently serving latents built from differently placed pixels, and two dataset folders never share a cache. Only the VAE is on the GPU (UNet + text encoders stay on CPU); encode batches are small and the VAE uses tiling. The VAE is moved back to CPU afterwards and GPU memory is flushed. Progress is published as `encoding.current/total`.
3. **Training (`training`)** — the epoch loop. The DataLoader's batch sampler draws each batch from a **single aspect-ratio bucket** so `train_batch_size` images share a resolution and stack in one UNet step (remainders smaller than the batch size are kept). Prompts are encoded in one batched CLIP-L + CLIP-G forward (chunked only as far as the longest caption in that batch, up to `max_token_length`; `clip_skip` applied), noise + timesteps are added, and the UNet predicts the noise target (with optional `noise_offset`). After gradient accumulation, UNet grads are clipped to `max_grad_norm`, TE grads to `te_max_grad_norm`, and both optimizers step. Every `save_every_n_steps` steps the run saves a checkpoint and generates samples.

   **Bucketing and padding.** Each image's bucket comes from its aspect ratio and the `train_resolution²` area budget, and the image is then **fitted whole into the bucket** (contain, centred) rather than centre-cropped: leftover bars are pad, and their loss weight is exactly 0. So no sample loses its head or feet, at the cost of a few percent of the bucket being unsupervised (`Letterbox: n/m samples padded, mean x%` is printed at startup). The pad is a neutral 127 grey. See [Configuration](configuration.md#bucketing--aspect-ratio-buckets).

   **Masked loss (optional).** Per-pixel MSE is multiplied by a 0–1 mask after area-downsampling to latent size (1/8). Content source, in order: `{stem}.mask.png` if present (white=train, black=ignore, gray=partial); else the training image's **alpha channel** (transparent=ignore); else all ones. Every sample carries the letterbox pad at weight 0, so the mask is always a full-bucket tensor, but only content masks are counted in the `Loss masks: n/m samples` line. This is *not* inpainting-model training: the UNet still sees the full noisy latent (pad included); only the loss is spatially weighted. Fine details smaller than one latent cell (hair, earrings) need a slightly larger painted region. Masks are not stored in `.latents_cache`. A sidecar always wins over embedded alpha. Because the mask is an unnormalized weight, a reported loss is scaled by the mean mask weight: a spatial mask of coverage *c* reports roughly *c* × the full-frame unmasked loss and produces a correspondingly smaller step (even with no sidecar and no alpha, the letterbox pad already lowers that mean below 1). `python test/verify_mask_pipeline.py --tiers all` (run it in the env from `environment.yml`, currently `axl`) re-runs the whole chain on your own machine — mask plumbing and fit geometry, the exact loss identities, and masked-vs-unmasked `trainer/main.py` runs with per-region probes — and writes a report under its `--report-dir`.
4. **Sampling (`sampling`)** — validation images are generated with the current LoRA weights (scheduler swapped to Euler-A with hand-built sigmas, interruptible denoising). After prompt encode the text encoders are offloaded; after each denoise pass the UNet is offloaded before VAE decode (slicing + tiling). Returning to the training loop restores UNet + both TEs to the train device. Each `[[validation.samples]]` entry is rendered in turn (its own size, steps, CFG and seed), and one checkpoint's samples are saved as `<output_name>_<step:06d>_p<set>_<repeat>.png` in `{output_dir}/{run_id}/{output_name}_samples/`.
5. **Finish (`finished`)** — the final LoRA is saved (unless stopped early), and the lock is released. On exception the status becomes `error` and the traceback is recorded.

TensorBoard metrics are written to `{logging_dir}/{run_id}/`:

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

Every run creates its own directory, named `{output_name}_{YYYYMMDD_HHMMSS}` (a `_2` / `_3` suffix is appended if that name is taken). Inside it the artifact names are unchanged:

```
run_id = {output_name}_{YYYYMMDD_HHMMSS}
```

| Artifact | Path |
| --- | --- |
| Per-step LoRA | `{output_dir}/{run_id}/{name}_s{step:06d}/{name}.safetensors` |
| Per-epoch LoRA (unused today) | `{output_dir}/{run_id}/{name}_e{epoch:03d}_s{step:06d}/{name}.safetensors` |
| Final LoRA | `{output_dir}/{run_id}/{name}_final/{name}.safetensors` |
| Sample images | `{output_dir}/{run_id}/{name}_samples/{name}_{step:06d}_p{set}_{repeat}.png` |
| TensorBoard logs | `{logging_dir}/{run_id}/` |
| Latent cache | one per dataset folder: `<folder>/.latents_cache/<sha1(abs_path::bucket::fit geometry)>.pt` |
| Runtime state / commands / lock / log | `$AXL_RUNTIME_DIR` → `$XDG_RUNTIME_DIR/axltrainer` → `/tmp/axltrainer-$UID` (`state.json`, `command.json`, `train.lock`, `train.log`) |

`state.json` carries the current `run_id`, and the dashboard / `list_samples` / `train_reset` resolve a run as: explicit `run_id` argument → `state.json`'s `run_id` → the newest `{name}_<timestamp>` directory under `logging_dir`. Runs created before this layout (flat `{output_dir}/{name}_s000010/`, `{logging_dir}/{name}/`) are **not** resolved anymore; their files stay on disk and can be cleaned with `python clean.py --legacy-flat`.

**Checkpoint format:** PEFT state dicts are remapped to kohya keys (`lora_unet_*`, `lora_te1_*`, `lora_te2_*`), converted to bf16, and saved with alpha scalars plus `modelspec.*` and `ss_*` metadata — directly loadable in ComfyUI or with kohya sd-scripts.

## Resuming from a checkpoint

Set `[training].resume_lora_path` to a LoRA `.safetensors` (or to a directory containing exactly one) and start a run. The weights are loaded into the wrapped UNet and both text encoders right after the LoRA adapters are created, before the accelerator prepares the models.

What carries over and what does not:

| Carried over | Restarts from zero |
| --- | --- |
| UNet / TE1 / TE2 LoRA weights (`lora_down`, `lora_up`) | Optimizer state (Schedule-Free AdamW on the UNet, AdamW on the TEs) |
| — | LR schedules (`lr_warmup_steps`, TE cosine, `unet_warmup_steps`) |
| — | `global_step` / `epoch` counters, sample filenames, TensorBoard step axis |
| — | Dataset order (caption shuffle is reseeded per epoch) |

Because the counters restart, the run writes into its own `{output_dir}/{run_id}/` directory — resuming from a run that ended at step 300 does not overwrite that run's `{name}_s000300/`.

Rules and failure modes:

- `network_type` must match the checkpoint. The trainer reads `ss_network_type`; a file without that field is `standard`, unless `ss_network_args` has `conv_dim > 0` (kohya LoCon). A locon file into a standard wrap (or the reverse) raises `ValueError` before any weights load. A locon wrap also refuses a locon file that has no TE `mlp_fc1` / `mlp_fc2` keys (pre-MLP locon files). `train_start` runs the type check, so the dashboard fails without touching the GPU.
- `network_dim` / `network_alpha` must match the checkpoint's rank. Locon `conv_dim` must match `ss_network_args` when that field is present. A rank mismatch is rejected at startup with a message naming `network_dim` or `conv_dim`; a differing alpha only logs a warning (the checkpoint's alpha scalars are ignored — this run uses `network_alpha` / `conv_alpha`).
- Tensors in the checkpoint that this LoRA does not use (e.g. modules outside the current `network_type` targets) are counted and listed in the run log; if **no** tensor maps, the run refuses to start.
- TE1 keys saved today are `lora_te1_text_model_encoder_layers_*`. Files written before that spelling (`lora_te1_encoder_layers_*`) still resume.
- `train_start` (and the trainer itself) validates the path before doing any GPU work, so a missing or ambiguous path surfaces as an immediate error in the dashboard.
- Any kohya-format LoRA works as long as its type and rank match, including ones trained by other tools; `ss_steps` / `ss_epoch` metadata are only reported for information.

## Watching progress

- **Dashboard** (recommended): live metric cards, charts, progress bars, and sample gallery.
- **TensorBoard**: `tensorboard --logdir <logging_dir>`.
- **Runtime state**: `cat $XDG_RUNTIME_DIR/axltrainer/state.json` (or the equivalent resolved path).
- **Logs**: `tail -f <runtime_dir>/train.log` (trainer stdout/stderr; driver log noise is filtered by `start_train.sh`).

## Cleanup

A run leaves samples, TensorBoard logs, and checkpoints behind. Two ways to clean up (same targets, same underlying helper `trainer/cleanup.py`), both scoped to **one run**:

```bash
python clean.py                    # interactive: lists run directories, asks which to clean
python clean.py --run rein_20260911_120000
python clean.py --legacy-flat      # old flat layout ({output_dir}/{name}_*, {logging_dir}/{name})
```

Or the dashboard's **Reset** button, which additionally clears the `finished`/`error` state so a new run can start. Both delete:

1. `{output_dir}/{run_id}/{name}_samples/`
2. `{logging_dir}/{run_id}/`
3. Optionally (with confirmation / `delete_weights`) all `{output_dir}/{run_id}/{name}_*` checkpoint dirs.

The run directory itself is removed once it is empty. If no run directory can be resolved, Reset only clears the state — it does not touch legacy flat artifacts.

The latent cache is **not** deleted — it's reusable across runs.

## Notes and gotchas

- **One run at a time.** `train.lock` makes a second concurrent run fail immediately.
- **Stale state.** If a training process dies hard, the next status read reconciles the dead PID to `error` ("training process is no longer running"). Reset to clear.
- **Stop during sampling** keeps the already-saved step checkpoint; the partially-denoised image is discarded.
- **`sample_seed = 0`** gives each repeat a fresh random seed (printed to the log); set a fixed seed for reproducibility.
- **Schedule-Free optimizer** requires `train()`/`eval()` mode toggling around sampling; the code does this automatically.
- **One metric window per run.** Each run writes its own TensorBoard event directory, so the dashboard shows the current/latest run only; a resumed run starts a new curve at step 1 rather than continuing the old one.
- **`ui.py` is deprecated** and still reads the old flat paths, so it will not show runs written in the new layout.
