# Python trainer — AXLTrainer

> Detail behind `AGENT.md` §5. `AGENT.md` keeps the condensed rules; this file carries the full text.


Entry: `bash start_train.sh` → `python -u trainer/main.py` with ROCm log filters and MIOpen cache pins.

| File | Responsibility |
| --- | --- |
| `trainer/main.py` | Lifecycle: run dir, config snapshot, lock, seed, `build_train_objects`, optional `warm_latent_cache`, epoch loop, final checkpoint, `end_run`. |
| `trainer/setup.py` | `TrainArtifacts`: `resolve_family`, pipeline, PEFT LoRA, dataloader, the two Schedule-Free AdamW optimizers, Accelerator. |
| `trainer/config.py` | `TrainConfig` + TOML flatten; `resolve_sample_sets` (prompt sets), `tracker_hparams` (tracker-safe config view), the run's own config snapshot (`save_run_config` / `run_config_mapping` / `TrainConfig.from_mapping`, see `config-contract.md`) and the prompt override that layers over it (`sample_override_path` / `read_sample_override` / `write_sample_override` / `clear_sample_override` / `run_log_dir` / `active_sample_sets`); `__post_init__` validates `[model_spec]` against the family catalog and derives `prediction_type` / `zero_terminal_snr` from the base via `base_model_prediction_flags` (the `v_pred` / `ztsnr` marker tensors ComfyUI reads in `supported_models.py:229`, or a diffusers directory's `scheduler/scheduler_config.json`; cached by path/mtime/size). Those two are not config keys — there is no switch, so training targets and sample rendering cannot disagree with the file. |
| `trainer/family.py` | Catalog (`sdxl_base_v1-0`, `sd3.5-large`), `resolve_family`, `require_trainable`. |
| `trainer/family_sdxl.py` | SDXL load/unpack/LoRA/encode/loss/save/sample; PEFT → kohya remap **and** the reverse map used by resume (`load_lora`). `denoise_loss` scales each sample's loss by Min-SNR's `min(SNR, γ)/SNR` (`min_snr_weight`, module level) when `cfg.min_snr_gamma > 0`, with SNR off the training scheduler's own `alphas_cumprod` — kohya's `apply_snr_weight`; a v-prediction base reads 0 there because `TrainConfig.__post_init__` derives it away (same place as `prediction_type`/`zero_terminal_snr`), and `0` means off, never "γ = 0". `load_sdxl_pipeline` pins a single-file base model cache-only (`local_files_only=True` on `from_single_file`): a checkpoint carries no pipeline config, so diffusers otherwise asks the Hub for `stabilityai/stable-diffusion-xl-base-1.0`'s component configs on every load (0.6 s and one connection here, a stall without a proxy). A diffusers directory and a repo id are untouched — the latter may legitimately download. |
| `trainer/family_sd35.py` | Stub; every method raises `UnsupportedFamilyError`. |
| `trainer/dataset.py` | `LoraImageDataset`: images + sidecar captions, buckets + per-record fit geometry, latent `.pt` lookup. |
| `trainer/cache.py` | Plans which images still need encoding (`dataset.cache_entries()` → `_plan_batches`: buckets filled in record order, encoded front to back in first-appearance order), then runs the three stages around that plan — a thread pool of `os.cpu_count()` verifying the files on disk and decoding/batching the plan's images, this thread encoding each planned batch as its images arrive, a writer thread persisting atomically to `<data>/.latents_cache`. Each image's posterior sample comes from a generator seeded by its own cache key (`_latent_generator`), so the file is a function of (image, geometry, seed) and the global generator — training's noise, timesteps and dropout masks — is left untouched. GPU holds only the VAE; UNet/TEs are offloaded first. |
| `trainer/loop.py` | `train_one_epoch`: group by bucket, family `compute_loss`, both optimizers, `at_safe_point`, `adopt_live_settings` then the save cadence. A save writes the checkpoint and (only when `settings.sampling_enabled`) its samples — sampling has no cadence of its own — and `save_stopped_lora` writes the one a stop during training owes (a step with no checkpoint yet) inside the same `optimizers_eval` the cadence and the final save use. |
| `trainer/sampling.py` | Interruptible SDXL sample gen (called from `SdxlFamily.generate_sample`); the sets come from `active_sample_sets(cfg)`, i.e. this run's edited prompts when it has them, else the config it started from. Each `[[validation.samples]]` set is encoded and rendered on its own (its own scheduler sigmas / size / steps / seed, so no set's chunk padding depends on another's prompt); prompt encode → TE offload; denoise → UNet offload then VAE decode; restore UNet+TEs before returning to the train loop. `_configure_scheduler` builds Euler a (`EulerAncestralDiscreteScheduler`, `timestep_spacing="linspace"` = ComfyUI `euler_ancestral` + `normal`) and takes the prediction type from `models.sample_scheduler_kwargs`, which reads `cfg.prediction_type` (derived from the base) plus the base pipeline's own scheduler config — a v-pred model sampled with epsilons renders noise. Each set's `guidance_rescale` reaches the pipeline as diffusers' `guidance_rescale` (ComfyUI's `RescaleCFG`): 0 = off, and the shipped `0.6` is what keeps a v-pred/zero-SNR base from crushing its shadows. |
| `trainer/models.py` | Flash attn, optimizers, checkpoint **paths**, kohya metadata helper (incl. `ss_min_snr_gamma`, which records the *effective* gamma — 0 for a v-prediction base, where the weighting is off — next to `ss_v_pred`), `sample_scheduler_kwargs` (the `prediction_type` / `rescale_betas_zero_snr` pair a sample pass renders with: `cfg.prediction_type` + `cfg.zero_terminal_snr` — ComfyUI's `ModelSamplingDiscrete = v_prediction, zsnr = true` pair, derived from the base — or the base pipeline's own scheduler config). |
| `trainer/control.py` | State machine, atomic JSON, lock, command poll, and `is_pid_alive` (zombies are gone — see `status-machine.md`). Also `LiveSettings` (cadence + sampling switch + `next_save_step`) and the `settings.json` channel: `read_settings` / `request_settings` / `publish_settings` / `clear_settings`. |
| `trainer/device_swap.py` | GPU↔CPU offload; `at_safe_point`. |
| `trainer/loss_log.py` | Kohya-style `Train/Avg_Loss` window (`LossRecorder`). |
| `trainer/cleanup.py` | Discover/delete one run's samples, TB logs, optional weight dirs (`delete_samples` / `delete_logs` default True, `delete_weights` default False). `clean.py` is the only caller that may pass `delete_weights=True` (after asking); `api.py` `train_reset` passes all three off, so Reset clears the state and leaves every artifact on disk. |
| `trainer/runs.py` | Run id naming (`{name}_{YYYYMMDD_HHMMSS}`), `create_run_dirs`, `find_latest_run`, `list_runs` (name-scoped, or every run with `output_name=None` plus per-run `output_name`/`samples`/`last_step`/`checkpoints`), `run_output_name`. torch-free. |
| `trainer/orphans.py` | Reaps what a signal-killed trainer leaves behind: `start_train.sh` starts it detached before `exec`ing the trainer, it waits for that PID (start-time guarded, `is_running` counts zombies as gone) and then kills the trainer's session, or the forkservers matching a `trainer/main.py` path when it is not the session leader. Torch-free, and its `is_running` / `PROC` are what `control.is_pid_alive` uses, so the two agree on what "still running" means. |
| `trainer/checkpoints.py` | `resolve_resume_path`, `read_lora_metadata` (cached by path+mtime+size, because the Dashboard polls `list_checkpoints`), `discover_checkpoints` (run-scoped) for resume + the Ranko picker and the Checkpoints section, and a run's pinned checkpoints (`pins_path` / `read_pins` / `write_pins` / `pin_entry` / `unpin_entry`): the JSON file `<logging_dir>/<run_id>/checkpoint_pins.json`, written by the API only. |
| `trainer/genjob.py` | Job records for one-off sample generation (`{name}_samples/generated/*.json`): naming, `mode` (`single`/`sets`/`batch`/`evaluate`), `files`/`images_done`/`total_images` progress, the `batch` plan record (`new_batch_job`), the evaluation record (`new_evaluation_job` + `PHASE_*`, carrying the `tags` the scoring is narrowed to), the `config_log_dir` and the `sample_sets` a range batch renders with (recorded like a single-checkpoint pass's; the runner prefers them and only a record without them resolves the run's own config), the `sample_sets` each `sets` job renders with, `cancel_requested`, request validation, atomic write, listing. torch-free. |
| `trainer/generate_sample.py` | `python -u trainer/generate_sample.py --spec <job.json>`: loads the base + a kohya LoRA (reusing `SdxlFamily.apply_lora`/`load_lora`) and renders the job's `mode`. `single` = one image with the job's own prompt; `sets` = the prompt sets recorded on the job (`sample_sets`, resolved by api.py from that run's own prompts — its saved config or the sets edited for it — else `resolve_sample_sets` of the run's `config_log_dir`; `_p{set}_{repeat}.png` into the run's `_samples/generated/`, seed `0` = random per image, model side from the checkpoint metadata, prediction type from its `modelspec.prediction_type` / `ss_v_pred` when it carries one); `batch` = that `sets` pass for each checkpoint of a step range, one pipeline for the whole range and a new one only when a checkpoint's LoRA shape changes (`_shape_key`), one `sets` job per checkpoint, a failure recorded on its own job and the range carrying on; `evaluate` (`run_evaluation`) = top the checkpoint's sample images up to the spec's depth with the config its run saved, tag every image and score it (see `evaluation.py`), rendering only the `(set, repeat)` slots `plan` lists (`_render_sets(plan=…)`, `_plan_slots`) and skipping the diffusion model entirely when nothing is missing. Writes PNGs + progress into the job file. Detached, never touches `state.json`/the lock. Handles SIGTERM/SIGINT: the check points are between sets, between repeats, inside `_on_step_end` and — for an evaluation — between tags (`tag_paths`' `on_entry`), so a cancel lands within one step or one image. |
| `trainer/evaluation.py` | What the Checkpoints section's evaluation is built from, torch-free: `collect_images` (a checkpoint's sample images — the run's own at its step plus the recorded generations for it, each with the prompt it was rendered from), `expansion_plan` (depth is a floor counted in *images*: enough of them mean nothing is rendered, else `ceil((depth - existing) / N)` whole copies of the config's pass, each set numbering its new repeats after the highest index it already uses so nothing existing is overwritten), `prompt_tags` / `normalize_tag` / `selected_tags` (the tagger's own `_`→space, lowercase form, so `(anal:1.2)` compares with a tagger label), `prompt_tag_counts` (the picker's frequency list: one row per tag a prompt set asks for, most frequent first), and `score_images(images, tags=…)` (per-image micro and per-prompt union precision/recall/F1, `images_failed`/`images_skipped`, top false positives/negatives; a non-empty `tags` narrows the whole comparison to those tags — the rest are neither misses nor extras — and the selection is echoed as `tags`). It also holds the run's own tag selection: `evaluation_tags_path` / `read_evaluation_tags` / `write_evaluation_tags` (`{logging_dir}/{run_id}/evaluation_tags.json`, written whole by the API when a pass is started, an empty list included, and read leniently — a broken file is a warning, never a raise) and `last_selected_tags(jobs)` as the fallback for a run from before that file (the newest `evaluate` record that is no longer running and carries a selection — `tags`, else `scores.tags`). |
| `trainer/comfy.py` | ComfyUI HTTP client + local-server discovery. Torch-free. Every request bypasses `http_proxy` (a loopback call through this machine's proxy answers 502); discovery walks `/proc/net/tcp{,6}` and accepts only a `system.comfyui_version` answer. |
| `trainer/automation.py` | The Automation page's state: `automation/settings.json`, uploaded workflows, prompt sets, job records (`jobs/<id>/job.json`, including the `pass` block a targeted pass is running), workflow validation + the model pre-check against the live `/object_info` (both combo shapes 0.35 reports). Torch-free. |
| `trainer/run_automation.py` | `python -u trainer/run_automation.py --spec <job.json> [--only-failed \| --image <name> \| --append <index> --images N \| --append-all --images N]`: pushes each prompt through the workflow (positive node, batch size, its own seed), polls history, downloads `SaveImage` outputs as `p0003_01.png` + a sidecar `.txt`, updates the job file. The four selectors are what the Gallery's actions spawn: `--only-failed` skips the prompts that already have images, `--image` redraws that one image in place (new random seed, sidecar rewritten, `image_seeds` updated, the prompt's other images untouched), `--append <index>` queues that one prompt N times, and `--append-all` queues every prompt N times — each image with its own seed, appending names from the next free number on (record and directory both consulted, so a deleted name is never handed out twice). The three targeted forms keep the images the entry already has instead of resetting its list. Detached; SIGTERM cancels between prompts and inside a poll. |
| `trainer/env.py` | MIGraphX cache dir, `flush_memory`. |
| `trainer/hardware.py` | Ranko hardware panel: nvtop snapshot, AMD edge/junction, CPU util/temp, RAM. |
| `trainer/utils.py` | Image list, caption shuffle, bucket math (`pick_bucket_size`), fit geometry (`fit_geometry`/`fit_to_bucket`), loss masks, `build_time_ids`. |
| `trainer/blobcodec.py` | Torch-free resize/re-encode + `/tmp` LRU cache + spawn process pool for Ranko `blob_*`. |
| `trainer/fsrpc.py` | Torch-free dataset/config/profile/mask/export IO behind IPC. |
| `text_processing.py` | **Repo root**, not under `trainer/`. Long-prompt chunking + dual CLIP encode. `family_sdxl.py` adds `os.getcwd()` to `sys.path` to import it. |

### Import dualism (easy to break)

`python trainer/main.py` puts `trainer/` on `sys.path[0]`, so modules use `from config import TrainConfig`.

`python -m unittest` / `import api` treat `trainer` as a **package**, so `api.py` uses `from trainer.config import …`. Several trainer modules already have:

```python
try:
    import control
    from device_swap import SwapContext
except ImportError:
    from trainer import control
    from trainer.device_swap import SwapContext
```

When adding a trainer module, support **both** import styles, or you will pass CLI training and fail unit tests (or the reverse). Do not move `text_processing.py` into `trainer/` without updating `loop.py` and both import paths.

### Training loop invariants

- Mixed precision default **bf16**.
- Both optimizers are **Schedule-Free AdamW**: one over the UNet's parameters (its own warmup via `unet_warmup_steps`), one over TE1+TE2's (`te_warmup_steps` in `[te_optimizer]` — it used to be `[training].lr_warmup_steps`, which the loader still reads when the new key is absent). No external LR scheduler is built for either, so `train()`/`eval()` mode toggling is what keeps a save (and a sample pass) on the averaged weights.
- LoRA targets (`SdxlFamily.apply_lora`): Standard UNet `to_q/to_k/to_v/to_out.0`, TE `q_proj/k_proj/v_proj/out_proj`. Locon UNet uses two PEFT adapters — Linear extras at `network_dim`, Conv2d (`conv1/conv2/conv_shortcut/conv`) at `conv_dim` — and TE also wraps `fc1/fc2`. See `doc/locon.md`.
- When `[optimization].gradient_checkpointing_unet` / `gradient_checkpointing_te` are true (the defaults), UNet and both TEs enable gradient checkpointing after PEFT wrap (TEs also `enable_input_require_grads` because embeddings stay frozen).
- Batches are **regrouped by `(bucket_w, bucket_h)`** before stacking — never stack mixed spatial sizes.
- `at_safe_point` is called every step (and during cache/sample). New long GPU work must call it or pause/stop will hang until the phase ends.
- VAE is moved to CPU after latent warm-cache; on-demand encode during training is the fallback.

### Checkpoints (ComfyUI / kohya)

`SdxlFamily.save_lora` remaps PEFT keys to:

- `lora_unet_*` / `lora_te1_*` / `lora_te2_*`
- `lora_down.weight` / `lora_up.weight` / `alpha`
- tensors **bf16**
- metadata `modelspec.*` + `ss_*`

Layout (`lora_checkpoint_file`, rooted at `artifact_root(cfg)` = `cfg.run_dir` or `cfg.output_dir`):

| Kind | Directory |
| --- | --- |
| step | `{output_dir}/{run_id}/{output_name}_s{step:06d}/{safe_name}.safetensors` |
| epoch | `{output_dir}/{run_id}/{output_name}_e{epoch:03d}_s{step:06d}/…` |
| final | `{output_dir}/{run_id}/{output_name}_final/…` |

Samples: `{output_dir}/{run_id}/{output_name}_samples/` filenames matching `_(\d+)_(\d+)\.png$` → `(step, repeat_idx)`. `api.scan_samples` uses that regex; unmatched files go under step `"-1"`.

Cleanup treats every child dir of the run dir whose name **starts with** `output_name` except `{name}_samples` as a weight dir, and removes the run dir once it is empty. Flat artifacts from before the run-directory layout are no longer resolved by the API/Ranko — `clean.py --legacy-flat` still cleans them.

### Resume (weights only)

`[training].resume_lora_path` (file, or a directory holding exactly one `.safetensors`) is loaded in `build_train_objects` **after** `family.apply_lora` and **before** `accelerator.prepare`, via `ModelFamily.load_lora`. `SdxlFamily.load_lora` builds the kohya→PEFT key map with `build_kohya_to_peft_map`, which derives it from `adapter_parameter_names(module)` (i.e. `named_parameters()`, **not** `get_peft_model_state_dict` — that one strips the `.default` adapter name and produces keys `load_state_dict` cannot use). Rank/alpha must match `network_dim`/`network_alpha`; unmatched tensors are counted and logged, zero matches raise. Step/epoch counters restart at 0 — there is no optimizer/scheduler state. `[train_start]` validates the path up front; `main.py` publishes `artifacts.resume` into `state.json` via `control.set_resume`.

### ROCm (non-negotiable)

`bucket_reso_steps` must keep VAE latents (spatial / 8) **divisible by 16**. Default **128**. `64` causes random GPU page faults on AMD (the field report is sealed: 涉及负责任披露流程，暂不公开). Do not “optimize” this down. `start_train.sh` also sets `PYTORCH_CUDA_ALLOC_CONF` and MIOpen log/cache env; keep those if you touch the launcher.

