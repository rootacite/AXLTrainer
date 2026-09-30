# AGENT.md — AXLTrainer

Working notes for coding agents. Human-facing docs live under `doc/` and `README.md`. Prefer this file when changing, extending, or refactoring code.

**What this is:** a local-first **LoRA** training stack (SDXL implemented; SD 3.5 catalogued but not trainable): Python engine (`trainer/`) + JSON-RPC helper (`api.py`) + Kotlin/Compose desktop dashboard (`ranko/`, product name **AxlRanko**) + dataset scripts (`tools/`, `tagger/`, `ranko/tools/agent.py`).

**What this is not:** an HTTP API, a generation/inference server, or a kohya `sd-scripts` fork. There is no network control plane.

---

## 0. Working agreement (read first)

The maintainer drives this repo one step at a time. Do exactly what the current instruction asks, and nothing more:

- No unrelated fixes, refactors, cleanups or "while I'm here" edits — not even small ones.
- Stop at the end of the requested step. Do not run ahead into the step after it.
- Work beyond the request is a proposal, not an action: report it (what it would touch, why it seems useful) and leave it undone until asked.
- **Never drive Ranko's window with `xdotool`** (or any other synthetic-input tool): the maintainer runs the app and verifies its interface by hand. What has to be checked automatically belongs in the test suites (`ranko/shared/src/jvmTest/`), which compose the real components in a window rather than clicking a running app.
- Terminology: **"the hook"** means `amdfq/amdfq-vmm-rs/` in its peralloc mode — the Rust `LD_PRELOAD` interposer that serves `hipMalloc` from address ranges it reserves itself (`amdfq/amdfq-vmm-rs/DESIGN.md`). Say **"the tail hook"** (or `amdfq/amdfq-tail-rs/`) when the tail-guard implementation is meant (`amdfq/amdfq-tail-rs/DESIGN.md`). The original C tail tree is `amdfq/amdfq-tail/`. The older C VMM tree (`amdfq-vmm/`) was deleted.
- Two of the hook's behaviours are **optional workarounds for driver bugs the 2026-09 kernel fixed**, and both default to off: `amdfq_va_never_reuse` (a freed range keeps its VA forever) and `amdfq_vram_reserve_gib` (the hook leaves that many GiB of the amdgpu free counter untouched). They travel config.toml → `trainer/amdfq_patch.py` → `AMDFQ_VA_NEVER_REUSE` / `AMDFQ_VRAM_RESERVE` → the hook, and the Dashboard's VA bar changes meaning with the first one. Changing either default is a behaviour change: say why, and touch the config row, the Python default, the hook default and the Dashboard text together.
- The hook's third knob, `amdfq_pool_mib` (`0`, off, or `16`–`512` MiB; the shipped config asks for `64`), is **not** a workaround but an allocation pool (DESIGN.md D12): one `hipMemCreate` builds a pool of that size, a request of at most half of it is carved out of one without any driver call, and a pool is released once the upper layer has freed everything carved out of it. It travels the same path (`trainer/amdfq_patch.py` → `AMDFQ_POOL_SIZE`, in bytes), and the hook itself stays off when that variable is unset — a hand-preload is not given a pool. It is on in the shipped config because that is what the maintainer chose, *not* because it is faster: `test/bench_alloc_pool.py` measured the pool off vs 16/32/64/128 MiB on this repo's own workload and found step rate flat to 1 % slower, with a pool-free run aborting 1 time in 9 against 2 out of 2 at 256 MiB. Re-run that script, in the env `environment.yml` names, before repeating or contradicting those numbers; the reasons it cannot help much (a per-allocation cost worth ≤0.2 % of a step, and pool create/teardown work of its own) are in the report it writes.
- **A hypothesis may come from intuition; a conclusion needs corroboration — no conclusion from a single witness.** Reading source (quote it as `file:line`) earns a hypothesis worth testing, not a verdict: say which of the two you are handing over, and label the inferred part as inference. When the question is "does this actually break", the experiment comes first; reading the code and agreeing with yourself is still one witness. (2026-09-18: the CLR reverse-pointer hazard written up as F3 in `amdfq/amdfq-vmm-rs/` (DIFF.md, since removed) was read out as a likely cause of the hook's NaN/hang. A purpose-built HIP probe that rebuilds the same shape and churns it — 200 rounds of map/unmap over the shared pad, all three access states, free-and-reclaim — did not reproduce it, and closed the alignment worry read out of the same source. Two hypotheses died to one experiment; both had looked convincing on paper.)

---

## 1. First 60 seconds

| Need | Go here |
| --- | --- |
| Architecture / data flow | `doc/overview.md` |
| Every TOML key | `doc/configuration.md` |
| Pause / resume / stop / artifacts | `doc/training.md` |
| Ranko tabs and IPC usage | `doc/dashboard.md` |
| Wire protocol (methods, shapes) | `API.md` |
| Dataset CLIs | `doc/dataset-tools.md` |
| Mask verification status + restart runbook | `doc/mask-verification.md` |
| Kohya LoCon (C3Lier) | `doc/locon.md` |
| ROCm pitfalls | `doc/troubleshooting.md` (the field reports behind it are sealed in `archive/`: 涉及负责任披露流程，暂不公开) |

Verify after a change (pick the layer you touched):

```bash
# Python IPC + control plane (cwd = repo root, env `axl`)
python -m unittest discover -s test
python -m unittest discover -s test -p 'test_validation.py'   # one file only

# Latent-cache pipeline (mock VAE)
python test/test_warm_latent_cache.py

# Ranko serialization / IPC models
cd ranko && ./gradlew :shared:jvmTest
```

Do **not** start a real training run to “see if it compiles” unless the task requires GPU behavior. `config.toml` contains **author-local paths** and will fail on other machines.

---

## 2. Process model (do not invent a new one)

```
Ranko (JVM)  --WebSocket JSON-RPC-->  api.py  --reads/writes-->  config, datasets, TB, samples
                                      |  writes command.json
                                      |  spawns (setsid) bash start_train.sh
                                      v
                         python -u trainer/main.py
                                      |
                                      +--> runtime dir: state.json, command.json, settings.json, train.lock, train.log
                                      +--> logging_dir/{run_id}/          TensorBoard
                                      +--> output_dir/{run_id}/{name}_*   checkpoints + samples
```

`run_id` = `{output_name}_{YYYYMMDD_HHMMSS}`, created by `trainer/main.py` through `trainer/runs.py` (`create_run_dirs`). Every run gets its own pair of directories, so a later run (whose step counter restarts at 0) never overwrites an earlier one. `output_name` must be filename-safe — letters and digits (any script), `-`, `_`, `.` — because it becomes the run id **and** the artifact directory names: `validate_output_name` (`trainer/runs.py`) refuses a space or a slash, and it is called from `TrainConfig.__post_init__` (so the trainer and `train_start` refuse such a config at startup) and from `fsrpc.config_save` (so a hand-written one cannot be saved); the Utils form mirrors the same rule and shows `OUTPUT_NAME_HINT`. `api.py` resolves the run for `dashboard` / `list_samples` / `train_reset` / `list_runs` as: explicit `run_id` param → `state.json`'s `run_id` → the newest run directory of an explicitly named `output_name`. A request that names nothing means *the run the trainer is on* and stops at `state.json`: there is deliberately no "newest run directory overall" fallback, which used to make a dashboard with no run recorded show a stopped run's step count and size under its `Current run` heading (a run the trainer never recorded is picked from the history list instead). Passing `run_id` alone is enough: `trainer/runs.py` `run_output_name` recovers the name the run id was built from, which is how a run created under another `output_name` (or one whose `logging_dir` directory is gone) still resolves its `{name}_samples` and weight dirs.

Hard rules:

- Training is **detached**. `api.py` `train_start` uses `start_new_session=True` (`setsid`). Closing Ranko must not kill the run.
- Ranko **never** talks to the GPU. After connect, `commonMain` does not read or write trainer files; it only speaks JSON-RPC. Desktop `jvmMain` may spawn `api.py --websocket` and pick paths with FileKit.
- The trainer is `exec`'d by `start_train.sh`, so that shell's PID and **session** become the trainer's. A GPU fault aborts the trainer from inside HIP (see `doc/troubleshooting.md`) without running Python's `atexit`; its DataLoader forkserver then keeps the workers it forked alive, reparented to init, each holding `/dev/kfd` and ~0.5 GB. `start_train.sh` therefore starts `trainer/orphans.py` first, detached, to reap that session once the trainer is gone — keep it, and keep it unable to touch a session that is not the trainer's.
- Ranko uses `python -u api.py --websocket` (default `127.0.0.1:18765`). LAN bind is `--host 0.0.0.0` plus `--allow-ip` / `AXL_WS_ALLOW`; loopback is always admitted. WebSocket is the only control channel; there is no stdin NDJSON fallback. Logs / tracebacks go to stderr.
- Working directory for `api.py` and `start_train.sh` is the **repo root** (directory that contains `api.py` and `trainer/`).
- Ranko finds that root by walking up from the executable / `user.dir` until a directory looks like one: `api.py` present, or `config.toml` next to the `trainer/` package (`TrainerRepo.looksLikeRepoRoot`). A lone `config.toml` must not qualify — a stranger's file would otherwise be edited. That walk is desktop bootstrap only.

Runtime dir resolution (same in `trainer/control.py` and `api.py`):

1. `$AXL_RUNTIME_DIR`
2. `$XDG_RUNTIME_DIR/axltrainer`
3. `/tmp/axltrainer-$UID`

Files: `state.json`, `command.json`, `settings.json`, `train.lock`, `train.log`. Tests **must** set `AXL_RUNTIME_DIR` to a temp dir (see `test_train_control.py`).

Interpreter override: Ranko uses `$AXL_PYTHON` if set, else `python3`. Training deps live in the conda env `environment.yml` names — currently `axl` (torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`). There is **no** `requirements.txt`.

---

## 3. Status machine

Defined in `trainer/control.py`. Do not add statuses without updating `API.md`, Ranko `TrainStatus`, and tests.

```
idle → starting → encoding → training → sampling → finished
                      ↑          ↑          ↑
                   pausing ↔ paused ↔ resuming
                              ↓
                           stopping → finished
dead PID while "live" → error   (reconcile)
reset (PID dead)      → idle    (keeps samples + TB logs; optional weight delete)
```

`LIVE_STATUSES`: starting, encoding, training, sampling, pausing, paused, resuming, stopping.

`is_pid_alive` (`control.py`) reads the process state through `orphans.is_running`, so a **zombie** counts as gone — `os.kill(pid, 0)` alone succeeds on one. `api.py` never `wait()`s the trainer (or a generator) it spawns, so a finished run's trainer stays a zombie until api.py exits (which is when Ranko closes), and the old signal-only check told the dashboard `alive: true` for a run that was over: the checkpoint panel's `Generate sample` stayed disabled in that session and only became usable after a restart. Everything that reads a PID (`reconcile`, `train_start`, `_require_alive`, `_gpu_busy`, `_running_generation`, `_reconcile_generated`) goes through it.

`PHASE_STATUSES` (pause/resume attach here): encoding, training, sampling.

Commands (`command.json`): `pause` | `resume` | `stop`. One-shot, sequenced. Trainer consumes them **only** at `device_swap.at_safe_point(phase, swap_ctx)`.

Live settings (`settings.json`): `{save_every_n_steps, sampling_enabled}`, written by `api.py` (`train_settings`, and `train_start` seeding it from `config.toml`), read by the trainer on every optimizer step (`control.read_settings` → `LiveSettings.adopt` in `loop.adopt_live_settings`, after `at_safe_point` so a change made while paused applies to the step the run resumes with). `state.json` carries the **effective** values in its `settings` block (`{save_every_n_steps, sampling_enabled, next_save_step}`, published by the trainer through `control.publish_settings`), which is what Ranko displays. `reset_to_idle` deletes the file, so a change never leaks into the next run. `next_save_step` replaces the old `global_step % N == 0` rule: the first checkpoint is at step N, each save sets `next = step + N`, and a cadence change at step S sets `next = S + N` ("every N steps from now"). Untouched, the sequence is identical to the modulo rule; flipping only the switch leaves the schedule alone. `0` still means "write no checkpoints".

Pause **must** offload the denoise network, all text encoders, both optimizers (including Schedule-Free state), and VAE to CPU, then `empty_cache`. Resume reloads what the current phase needs. Do not “pause” by sleeping on GPU. `SwapContext` uses `denoise` + `text_encoders: list` so a future family can add a third encoder without a new swap shape.

Early-stop semantics (keep these):

| Phase | Stop behavior |
| --- | --- |
| encoding | no LoRA written |
| training | save `{output_name}.safetensors` only if this step has no checkpoint yet |
| sampling | skip leftover repeats (step checkpoint already exists) |

`train_start` fails if a live PID exists, including a process that marked `finished` but has not exited. `train_reset` fails while status is in `_RESET_BLOCKED` **and** PID is alive.

`state.json` also carries `run_id` (the run directory created for this run, written by `control.begin_run`) and `resume` (`null`, or `{path, filename, step, epoch, loaded, skipped}` for a run seeded from a checkpoint via `control.set_resume`).

---

## 4. Configuration contract

**Single source of truth:** `config.toml` at the **repo root**. `trainer/main.py` takes **no CLI args**.

`_load_toml_config()` resolves that path **relative to the working directory**, which is what lets
`verify_mask_pipeline.py` runs an unmodified `trainer/main.py` against a
throwaway mirror of the repo. Every entry point therefore runs with cwd = repo root.

Load path:

- Python: `trainer/config.py` flattens **all TOML tables into one dict**. Section names do not exist at runtime on the Python side — only keys. `TrainConfig` fields default via `get_val(key, hardcoded)`. **TOML wins** over Python defaults.
- Kotlin: `AxlTrainerConfig` is **sectional** (`environment`, `model_spec`, `training`, …). Utils tab saves via `TomlDocumentPatcher`: in-place replace of uncommented `key = value` inside named tables. Comments, blank lines, and unknown tables (e.g. `[bookkeeping]`) stay intact. **Do not rewrite the whole file.**
- Kotlin load: ktoml refuses an integer literal for a `Double`, so `TomlIntegerLiterals` rewrites `key = 0` to `0.0` for the keys `AxlTrainerConfig` declares as `Double` before decoding. A hand-edited `amdfq_vram_reserve_gib = 0` must not cost Ranko its startup; the save side already writes `0.0` (`TomlDocumentPatcher.float`).

Adding a hyperparameter (all four, or the GUI will drift):

1. `config.toml` — pick an existing table or add one.
2. `TrainConfig` in `trainer/config.py` — same **flat** key name.
3. Kotlin `AxlTrainerConfig` + nested data class (`ConfigModel.kt`), `TrainingConfigForm`, Utils UI bind/save map (section name → key → encoded value).
4. `doc/configuration.md`.

If only the trainer needs it, you can skip (3) but document that the Utils editor will not see it until the Kotlin model is updated. ktoml parse of the full file will fail if a **required** Kotlin field is missing — new optional keys are safer as Kotlin defaults.

`run_dir` is the one field that is **not** a user-facing key: `main.py` writes the run directory it created into `cfg.run_dir` at startup, and `artifact_root(cfg)` roots every artifact path at it. Leave it empty in `config.toml`.

### Validation prompt sets (`[[validation.samples]]`)

Validation renders N prompt sets per sampling point, each producing its own `repeat` images. The
array of tables is the only list-shaped config, and it deliberately leans on the flattening rule: it
must stay under `[validation]` (a top-level `[[samples]]` would be dropped, because
`_load_toml_config` only copies **tables**), so the Python side reads it as the flat key `samples`.

- `resolve_sample_sets(cfg)` (`trainer/config.py`, torch-free, accepts a `TrainConfig` *or* the
  flattened mapping) resolves each entry; a key an entry omits falls back to the flat `sample_*`
  scalar of the same shape, and **no entries at all yield one set built from those scalars** — the
  single-prompt behaviour, which is why the `validation.sample_*` overrides in
  `test/verify_mask_pipeline.py` still work. Ranges and the per-entry error message live there.
- Seed rule: inside a set the nth image uses `seed + n` (`0` = random per image). Two sets sharing a
  seed start from the same noise; that is the point (only the prompt differs).
- Images: `{output_name}_{step:06d}_p{set}_{repeat}.png`, `set` counting from 0. `api.scan_samples`
  also parses the old two-number name as set 0, and returns `set_index` for the Ranko `Pn` badges.
  `control.set_sampling` reports a global image counter plus `prompt_set`/`prompt_sets`.
- Ranko: `SampleSetForm` in `TrainingConfigForm`, tabs in the Utils Validation section,
  `TomlDocumentPatcher.replaceArrayOfTables` for the blocks. The form writes `[validation]` from the
  **first** set, so the file never holds two contradictory prompts.

### Train data entries (`[[environment.train_data]]`)

The datasets a run trains on, one block per folder with a per-epoch `repeat` (kohya `num_repeats`
semantics). It is the second list-shaped config and follows the same trick as the sample sets: the
blocks stay under `[environment]`, so the Python side reads them as the flat key `train_data`, and
the flat `train_data_dir` scalar stays beside them as the **mirror of the first entry**.

- `resolve_train_data_entries(cfg)` (`trainer/config.py`, torch-free, accepts a `TrainConfig` *or*
  the flattened mapping) resolves each entry; the blocks win over the scalar, `path` is required in
  a block, `repeat` defaults to `1` and must be in `1..512`. **No blocks yield one entry built from
  `train_data_dir` with repeat 1** — the single-folder behaviour, which is why every existing config
  and every `cfg.train_data_dir = str(dir)` test still trains the same thing.
- `train_data_dir` is what everything expecting *one* path reads: `models.py`'s `ss_train_data_dir`,
  the `dataset_tag` fallback in `api.py` (which resolves the list and takes the first entry),
  `agent.py --config`, and a config with no blocks. `main.py` resolves the list before loading the
  model, so a malformed entry fails early.
- Repeats reach training in one place: `LoraImageDataset` appends each record's index to its bucket
  **`repeat`** times. The train loop reads nothing but `artifacts.dataloader`, so the epoch length,
  `len(dataloader)` (hence `steps_per_epoch`, the total step count and the TE cosine schedule) and
  the progress bars all follow that list length without a loop change. `__len__` stays the unique
  image count — `warm_latent_cache` walks `range(len(dataset))` — and `total_samples` carries the
  per-epoch figure.
- Ranko: `TrainDataDirForm` in `TrainingConfigForm` (rows in the Utils Environment section),
  `TomlDocumentPatcher.replaceArrayOfTables` for the blocks, `train_data_dir` written from the
  **first** row so the file never holds two contradictory folders. `DatasetSelection` (a
  `@SingleIn(AppScope)` holder of one index) is what the single-folder pages — Images, Statistics,
  the tag card — share, so the folder picked on one is the folder the others open.

### Bucketing + fit geometry (the geometry contract)

`pick_bucket_size(w, h, min_reso, max_reso, step, no_upscale, area=train_resolution²)` is an **area
budget** rule: the bucket aims at `area` pixels and takes its aspect ratio from the image, with
`min/max_bucket_reso` as per-axis clamps and `no_upscale` refusing an axis larger than the source's
(floored to one step for sources thinner than a step). It replaced a rule that pinned the *short* side
to `min_bucket_reso` and capped the long side, which forced every portrait into one 0.6-aspect bucket
and cropped the overflow — 250 of 640 images in the author's dataset lost a mean 54 % of their long
edge that way. Shipped defaults are therefore `min_bucket_reso = 384`, `max_bucket_reso = 2688`; keep
`min ≤ train_resolution ≤ max` (the dataset warns on stderr otherwise).

`fit_geometry(src_w, src_h, bucket_w, bucket_h)` then places the whole image inside the bucket
(contain, centred) and `fit_to_bucket` renders it with a `FIT_PAD_VALUE` (127) fill. There is **no
crop variant in the training path** — `resize_and_center_crop` survives only for the
verification harness's independent implementation. The pad is loss weight exactly 0, produced by
`load_loss_mask`, which now always returns a full-bucket mask (content resized, then pasted onto a
zero canvas — never pre-pad-then-resample, LANCZOS ringing leaks weight into the pad rows).

Any change to the rule, the clamps or the interpolators changes every sample's pixels: the latent
cache key carries the fit geometry for exactly that reason (see §8).

Shipped `config.toml` and `TrainConfig` fallbacks contain **author machine paths**. Never “fix” them to placeholders as part of an unrelated PR unless asked; consumers already know they must edit `[environment]`.

---

## 5. Python trainer map

Entry: `bash start_train.sh` → `python -u trainer/main.py` with ROCm log filters and MIOpen cache pins.

| File | Responsibility |
| --- | --- |
| `trainer/main.py` | Lifecycle: run dir, lock, seed, `build_train_objects`, optional `warm_latent_cache`, epoch loop, final checkpoint, `end_run`. |
| `trainer/setup.py` | `TrainArtifacts`: `resolve_family`, pipeline, PEFT LoRA, dataloader, dual optimizers, Accelerator. |
| `trainer/config.py` | `TrainConfig` + TOML flatten; `resolve_sample_sets` (prompt sets), `tracker_hparams` (tracker-safe config view); `__post_init__` validates `[model_spec]` against the family catalog and derives `prediction_type` / `zero_terminal_snr` from the base via `base_model_prediction_flags` (the `v_pred` / `ztsnr` marker tensors ComfyUI reads in `supported_models.py:229`, or a diffusers directory's `scheduler/scheduler_config.json`; cached by path/mtime/size). Those two are not config keys — there is no switch, so training targets and sample rendering cannot disagree with the file. |
| `trainer/family.py` | Catalog (`sdxl_base_v1-0`, `sd3.5-large`), `resolve_family`, `require_trainable`. |
| `trainer/family_sdxl.py` | SDXL load/unpack/LoRA/encode/loss/save/sample; PEFT → kohya remap **and** the reverse map used by resume (`load_lora`). |
| `trainer/family_sd35.py` | Stub; every method raises `UnsupportedFamilyError`. |
| `trainer/dataset.py` | `LoraImageDataset`: images + sidecar captions, buckets + per-record fit geometry, latent `.pt` lookup. |
| `trainer/cache.py` | Pipelined CPU decode → batched VAE encode → atomic `.pt` in `<data>/.latents_cache`. GPU holds only the VAE; UNet/TEs are offloaded first. |
| `trainer/loop.py` | `train_one_epoch`: group by bucket, family `compute_loss`, both optimizers, `at_safe_point`, `adopt_live_settings` then the save cadence. A save writes the checkpoint and (only when `settings.sampling_enabled`) its samples — sampling has no cadence of its own. |
| `trainer/sampling.py` | Interruptible SDXL sample gen (called from `SdxlFamily.generate_sample`). Each `[[validation.samples]]` set is encoded and rendered on its own (its own scheduler sigmas / size / steps / seed, so no set's chunk padding depends on another's prompt); prompt encode → TE offload; denoise → UNet offload then VAE decode; restore UNet+TEs before returning to the train loop. `_configure_scheduler` builds Euler a (`EulerAncestralDiscreteScheduler`, `timestep_spacing="linspace"` = ComfyUI `euler_ancestral` + `normal`) and takes the prediction type from `models.sample_scheduler_kwargs`, which reads `cfg.prediction_type` (derived from the base) plus the base pipeline's own scheduler config — a v-pred model sampled with epsilons renders noise. Each set's `guidance_rescale` reaches the pipeline as diffusers' `guidance_rescale` (ComfyUI's `RescaleCFG`): 0 = off, and the shipped `0.6` is what keeps a v-pred/zero-SNR base from crushing its shadows. |
| `trainer/models.py` | Flash attn, optimizers, checkpoint **paths**, kohya metadata helper, `sample_scheduler_kwargs` (the `prediction_type` / `rescale_betas_zero_snr` pair a sample pass renders with: `cfg.prediction_type` + `cfg.zero_terminal_snr` — ComfyUI's `ModelSamplingDiscrete = v_prediction, zsnr = true` pair, derived from the base — or the base pipeline's own scheduler config). |
| `trainer/control.py` | State machine, atomic JSON, lock, command poll, and `is_pid_alive` (zombies are gone — see §3). Also `LiveSettings` (cadence + sampling switch + `next_save_step`) and the `settings.json` channel: `read_settings` / `request_settings` / `publish_settings` / `clear_settings`. |
| `trainer/device_swap.py` | GPU↔CPU offload; `at_safe_point`. |
| `trainer/loss_log.py` | Kohya-style `Train/Avg_Loss` window (`LossRecorder`). |
| `trainer/cleanup.py` | Discover/delete one run's samples, TB logs, optional weight dirs (`delete_samples` / `delete_logs` default True, `delete_weights` default False). `clean.py` is the only caller that may pass `delete_weights=True` (after asking); `api.py` `train_reset` passes all three off, so Reset clears the state and leaves every artifact on disk. |
| `trainer/runs.py` | Run id naming (`{name}_{YYYYMMDD_HHMMSS}`), `create_run_dirs`, `find_latest_run`, `list_runs` (name-scoped, or every run with `output_name=None` plus per-run `output_name`/`samples`/`last_step`/`checkpoints`), `run_output_name`. torch-free. |
| `trainer/orphans.py` | Reaps what a signal-killed trainer leaves behind: `start_train.sh` starts it detached before `exec`ing the trainer, it waits for that PID (start-time guarded, `is_running` counts zombies as gone) and then kills the trainer's session, or the forkservers matching a `trainer/main.py` path when it is not the session leader. Torch-free, and its `is_running` / `PROC` are what `control.is_pid_alive` uses, so the two agree on what "still running" means. |
| `trainer/checkpoints.py` | `resolve_resume_path`, `read_lora_metadata` (cached by path+mtime+size, because the Dashboard polls `list_checkpoints`), `discover_checkpoints` (run-scoped) for resume + the Ranko picker and the Checkpoints section, and a run's pinned checkpoints (`pins_path` / `read_pins` / `write_pins` / `pin_entry` / `unpin_entry`): the JSON file `<logging_dir>/<run_id>/checkpoint_pins.json`, written by the API only. |
| `trainer/genjob.py` | Job records for one-off sample generation (`{name}_samples/generated/*.json`): naming, `mode` (`single`/`sets`/`batch`), `files`/`images_done`/`total_images` progress, the `batch` plan record (`new_batch_job`), `cancel_requested`, request validation, atomic write, listing. torch-free. |
| `trainer/generate_sample.py` | `python -u trainer/generate_sample.py --spec <job.json>`: loads the base + a kohya LoRA (reusing `SdxlFamily.apply_lora`/`load_lora`) and renders the job's `mode`. `single` = one image with the job's own prompt; `sets` = every `resolve_sample_sets(config.toml)` set for the checkpoint (`_p{set}_{repeat}.png` into the run's `_samples/generated/`, seed `0` = random per image, model side from the checkpoint metadata, prediction type from its `modelspec.prediction_type` / `ss_v_pred` when it carries one); `batch` = that `sets` pass for each checkpoint of a step range, one pipeline for the whole range and a new one only when a checkpoint's LoRA shape changes (`_shape_key`), one `sets` job per checkpoint, a failure recorded on its own job and the range carrying on. Writes PNGs + progress into the job file. Detached, never touches `state.json`/the lock. Handles SIGTERM/SIGINT: the check points are between sets, between repeats and inside `_on_step_end`, so a cancel lands within one step. |
| `trainer/comfy.py` | ComfyUI HTTP client + local-server discovery. Torch-free. Every request bypasses `http_proxy` (a loopback call through this machine's proxy answers 502); discovery walks `/proc/net/tcp{,6}` and accepts only a `system.comfyui_version` answer. |
| `trainer/automation.py` | The Automation page's state: `automation/settings.json`, uploaded workflows, prompt sets, job records (`jobs/<id>/job.json`, including the `pass` block a targeted pass is running), workflow validation + the model pre-check against the live `/object_info` (both combo shapes 0.35 reports). Torch-free. |
| `trainer/run_automation.py` | `python -u trainer/run_automation.py --spec <job.json> [--only-failed \| --image <name> \| --append <index> --images N]`: pushes each prompt through the workflow (positive node, batch size, its own seed), polls history, downloads `SaveImage` outputs as `p0003_01.png` + a sidecar `.txt`, updates the job file. The three selectors are what the Gallery's per-image actions spawn: `--only-failed` skips the prompts that already have images, `--image` redraws that one image in place (new random seed, sidecar rewritten, `image_seeds` updated, the prompt's other images untouched), and `--append N` queues that one prompt N times, each with its own seed, appending names from the next free number on (record and directory both consulted, so a deleted name is never handed out twice). Both targeted forms keep the images the entry already has instead of resetting its list. Detached; SIGTERM cancels between prompts and inside a poll. |
| `trainer/env.py` | MIGraphX cache dir, `flush_memory`. || `trainer/hardware.py` | Ranko hardware panel: nvtop snapshot, AMD edge/junction, CPU util/temp, RAM. |
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
- Dual optimizers: **Schedule-Free AdamW** on UNet (no LR scheduler), **AdamW** on TE1+TE2 with cosine/warmup via Accelerator.
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

---

## 6. IPC (`api.py`)

Framing: one JSON object per WebSocket text frame. `{id, method, params}` → `{id, ok: true, result}` or `{id, ok: false, error}`.

Handlers (`_HANDLERS` — add here **and** in `API.md` **and** `TrainerIpcClient.kt`):

| Method | Side effect |
| --- | --- |
| `ping` | none |
| `dashboard` | read TB scalars + flattened config (run-scoped: `run_id`) |
| `list_runs` | every run directory under `output_dir` / `logging_dir` (any `output_name`) with brief figures + `current` / `live`; backs the Dashboard run history |
| `list_samples` | scan the run's sample PNGs (`path` is a blob key) |
| `list_checkpoints` | list LoRA files under `output_dir/{name}_<timestamp>/` (read-only) |
| `checkpoint_pins` | read-only: one run's pinned checkpoints, from `<logging_dir>/<run_id>/checkpoint_pins.json` |
| `checkpoint_pin_set` | pin / unpin one checkpoint of one run and answer with the whole list |
| `train_status` | `control.status_payload()` + dead-PID reconcile |
| `train_start` | spawn `start_train.sh` (rejects a bad `resume_lora_path` up front) |
| `train_pause` / `train_resume` / `train_stop` | write `command.json`; `train_resume` is refused while a generation job runs (one card) |
| `train_settings` | write `settings.json` (cadence / sampling switch for the run in progress); refused without a live PID |
| `train_reset` | `run_cleanup` (nothing deleted: samples, logs and weights all kept) + `reset_to_idle` |
| `dataset_tag` | spawn `tagger/main.py` (GPU ONNX); overwrites sidecar `.txt` |
| `generate_sample` | spawn `trainer/generate_sample.py` **detached** (`mode: single`; returns immediately; refuses while a live trainer is using the GPU — `paused` is allowed — and while another job, of any run, is running) |
| `generate_checkpoint_samples` | the same, `mode: sets`: the config's whole `[[validation.samples]]` list for one checkpoint, into the run's `_samples/generated/` |
| `generate_checkpoint_samples_batch` | the same pass for every checkpoint of a step range, oldest first, in one process (`mode: batch`); it writes one `sets` job per checkpoint, so the cards fill in as it goes |
| `cancel_generation` | SIGTERM the running generation (batch included) and set its `cancel_requested`; no GPU gate, and `train_resume` is what usually matters next |
| `list_generated_samples` | read-only: the run's `generated/*.json` jobs, newest first; a `running` job whose PID died is rewritten to `error` |
| `hardware_status` | `nvtop -s` JSON + DRM hwmon temps + `/proc` CPU (read-only) |
| `config_get` / `config_save` | read / atomic-write repo `config.toml` |
| `profile_list` / `_get` / `_save` / `_delete` | `configs/` presets |
| `prompt_matrix` | read-only: repo-root `input_matrix.txt` (the prompt wizard's tag matrix) |
| `prompt_profile_list` / `_get` / `_save` / `_delete` | `prompt_profiles/*.json`; the store is dumb (text + a `spec` object), the v1/v2 upgrades happen in Ranko |
| `automation_config_get` / `_save` | `automation/settings.json` (server, workflow, positive node, count, poll, output dir) |
| `automation_discover` | probe the machine's loopback listeners for a ComfyUI (`/proc/net/tcp{,6}` + `system.comfyui_version`), proxy-free |
| `automation_workflow_list` / `_validate` / `_save` / `_delete` | uploaded API-format workflows + the model pre-check against the live `/object_info` |
| `automation_prompt_list` / `_get` / `_save` / `_delete` | `automation/prompts/*.txt` prompt sets |
| `automation_job_start` | write `automation/jobs/<id>/job.json`, spawn `trainer/run_automation.py` **detached** (refuses while another job runs) |
| `automation_job_list` / `_get` | read-only, newest first; a `running` job whose PID died is rewritten to `error`; `_get` adds `summary` + `log_tail` |
| `automation_job_cancel` / `_retry_failed` / `_delete` | SIGTERM the runner's group / rerun only the prompts without images / remove the job directory |
| `automation_image_regenerate` / `_delete` / `automation_prompt_extend` / `automation_job_prompt_edit` | the Gallery's per-image actions: redraw one image in place (`--image`, new random seed), remove one image (its sidecar too; the last image of a prompt drops that entry and the rest renumber), append N images each from its own seed (`--append`/`--images`), rewrite one prompt's text in the record. The first three spawn the runner and answer with the job detail; the edit writes the record only. Their triggers are all on the Gallery page — `ImagePreviewOverlay` is a viewer with no actions: the ↻ disc in a thumbnail's top-left, the ✕ in its top-right, the ⤓ (save) in its bottom-right, and `Copy prompt` / `改提示词…` / `再加几张…` on the prompt row. The three that render go inert while the job runs. A targeted pass writes a `pass` block into the record (`{mode, prompt_index, images_done, total_images, image}`, also in `job_summary`) and clears it when it ends: a redraw does not move the job's own counters, so this is what the job row and the record being worked on show as `Adding image 2/4` — the line sits under that record's own header, above its images, not at the top of the page |
| `dataset_list` | non-recursive folder scan + tags + sizes |
| `caption_write` | `{stem}.txt` |
| `dataset_drop` / `dataset_shuffle` | trash / renumber (Python owns IO) |
| `mask_get` / `_write` / `_delete` | `{stem}.mask.png` PNG |
| `blob_stat` / `blob_batch` | resized JPEG/WebP/PNG, process pool, `/tmp` cache |
| `tag_lexicon` | `tagger/selected_tags.csv` text |
| `checkpoint_export` | server-local copy of a `.safetensors` |
| `fs_listdir` / `fs_roots` | directory listing for the Web path picker (no file bytes) |

`dashboard` synthesizes `Train/Avg_Loss` from `Train/Loss` via `synthesize_avg_loss` when the tag is missing (old runs). Do not rename TensorBoard tags without updating Ranko chart cards.

`generate_sample` / `generate_checkpoint_samples` share one gate (`_claim_generation`: `_gpu_busy` — the `_TAG_BLOCKED` statuses with a live PID, so `paused` passes — and `_running_generation`, which scans **every** run's `generated/*.json` because the GPU is single-tenant and closes a job whose PID is gone). The API's own `dataset_tag` guard uses the same `_gpu_busy`.

Every job record carries the `step` of the checkpoint it renders (`_checkpoint_step`: the artifact directory name, else `ss_steps` in its metadata), because that is what the Dashboard attaches the images to; a job with none renders into `generated/` and is shown nowhere. Its counters (`current_step`, `total_steps`, `images_done`, `total_images`) are always integers — the client declares them `Int`, and an explicit `null` fails its decode and takes the whole `list_generated_samples` reply down with it. `TrainerIpcClient`'s `Json` sets `coerceInputValues = true` for the same reason: one bad record must not hide every generated image.

A `batch` job (`new_batch_job`) is the plan of a step-range pass and holds **no images at all**: it lists the checkpoints in render order and tracks `checkpoint_index` / `images_done` / `failed`, while each checkpoint's images live in its own `sets` job (with `batch_id` / `batch_index` / `batch_total`), which is what puts them on the right card. `mode: sets` and `mode: single` records may carry `cancel_requested`: `cancel_generation` signals the process and sets that flag, the job stays `running` until the process is really gone (the card is not handed over before that), and then closes as `cancelled` with whatever `files` it wrote. `JOB_CANCELLED` / `jobHasImages` in Ranko mean those images still show — the rule is "no longer running and has images", which also keeps a failed pass's partial images visible.

Kotlin client: match replies on `id`. Control methods are serialized; blob calls may run next to `train_status`. `ignoreUnknownKeys = true`. Keep that when adding fields. Do not add a generic `read_file` / `write_file`.

---

## 7. Ranko (`ranko/`)

Compose Multiplatform **desktop JVM** plus a **wasmJs** local/LAN companion (`:webApp`). Kotlin 2.4.10, Compose 1.12.0, Material 3, Metro DI, ktoml, Coil 3, haze 2.0. Visual style is KataHana **Sky & Sakura**: `RankoTheme` + Nunito + porcelain cards. Raw hex lives only in `ui/theme/Color.kt`. Screens read `rankoColors` / `PorcelainCard` / `CapsuleButton`; Canvas helpers take colors as parameters.

User-facing look-and-feel (background: Solid / Glow / Image, independent card vs background blur, font/icon scale, thumbnail JPEG quality) lives in the **Appearance** section of the Utils tab and is persisted by `AppearanceRepository` (Java Preferences, key `com/acite/axlranko/appearance`). `App.kt` consumes it and feeds `LocalDensity` so **font scale only affects sp** and **icon scale only affects dp** (`density * iconScale`, `fontScale * font / iconScale`). `RankoBackdrop` renders glow orbs for `Glow` and a cropped, dimmed photo for `Image`. A full-window haze layer uses `backgroundBlurRadiusDp` (gaps); `PorcelainCard` / `FrostedSurface` use `cardBlurRadiusDp`.

The Utils **WM** tab is the same kind of UI-only section (no config key) and is desktop-only by construction: `App.kt` provides `LocalAppWindow` (`util/AppWindow.kt`) from its `appWindow` argument, the desktop entry point passes `DesktopAppWindow` (`util/AppWindow.jvm.kt`) — Maximize applies the screen's own bounds to the window and remembers what it had, Restore puts that back, and Exit calls the entry point's own quit lambda, i.e. `ProcessExitGuard.armOnce()` plus `exitApplication()`, the same two steps the close request takes — and the wasm entry point passes nothing, so `visibleSections` (`pages/UtilsScreen.kt`) leaves the tab out of the web build. It exists for a session whose compositor draws no decorations (cage): there is no title bar to maximize or close from, and no window manager to act on a maximize request either, which is why the size is applied rather than asked for.

| Path | Role |
| --- | --- |
| `ranko/desktopApp/…/main.kt` | Window; `createGraph<AppGraph>()` |
| `ranko/webApp/…/main.kt` | `ComposeViewport` `#webApp`; same graph; no spawn |
| `…/util/ProcessExitGuard.kt` (jvmMain) | One watcher per quit, armed from `main.kt`'s close request and from a shutdown hook: it kills the JVM if it is still there `DEFAULT_GRACE_SECONDS` later, start-time guarded so a reused PID is left alone. A hang inside the VM cannot be undone from inside the VM, and a windowless Ranko keeps the GPU render nodes and its `api.py` child. |
| `ranko/shared/src/commonMain/…/App.kt`, `Stage.kt` | Shell + four screens |
| `…/ui/theme/` | Sky & Sakura palette, tokens, Nunito, `RankoTheme` |
| `…/ui/components/` | Backdrop, porcelain/frosted surfaces, capsule controls |
| `…/pages/*Screen.kt` + `*ViewModel.kt` | UI + state |
| `…/data/TrainerIpcClient.kt` | WebSocket JSON-RPC client |
| `…/data/TrainerRepo.kt` (jvmMain) | Repo-root discovery for spawning the helper |
| `…/data/ConfigModel.kt` | Sectional TOML model |
| `…/data/TomlDocumentPatcher.kt` | Comment-preserving in-memory patch |
| `…/data/TomlIntegerLiterals.kt` | Bare-integer tolerance on load |
| `…/data/ConfigImporter.kt` | In-memory parse only; load/save go through IPC |
| `…/data/ConfigProfileStore.kt` | Name rules + apply merge; disk IO is `profile_*` |
| `…/data/BlobStore.kt` | Hash cache + coalesced `blob_stat`/`blob_batch`. The fast path serves a ref straight from memory; a ref with a non-empty `rev` skips it and is **revalidated** through the stat's hash instead, which is what lets a caller who knows a file can be rewritten in place (the Gallery's `p0001_01.png` after a redraw — the rev is the record's seed for that image) show the new pixels rather than the bytes read the first time. The server's cache already keys on the file's mtime and size, so a rewritten file answers with a different hash |
| `…/pages/AutomationScreen.kt` + `…ViewModel.kt` | The fifth tab: Prompts (wizard/profiles/results), ComfyUI (discovery, workflows, batch, log) and Gallery (jobs, thumbnails, save). The Gallery follows a running job by itself: `refreshJobs` re-reads the **selected** job's detail on every poll (and the poll takes one last look after the job stops), because a pass writes images, swaps a redrawn one and ends the run while the page is open — without it new images stayed invisible and the per-image buttons stayed disabled until the job was clicked again |
| `…/pages/components/automation/` | `PromptEditor`/`PromptPanes` (the ported wizard's pages, step rail, manifest, results), `ComfyPanes`, `GalleryPane`, `UiText` (this page's own chrome, both languages) |
| `…/pages/components/SamplePreview.kt` | The fullscreen image preview, shared by the Dashboard's sample grid and the Gallery (`PreviewImage` + `ImagePreviewOverlay`). It must be given a window-sized box — inside a `verticalScroll` it collapses onto its header row |
| `…/prompt/` | The Kotlin port of `tools/gen_prompts.py`: matrix parser, face groups, spec, profile codec (v1/v2 → v3), generator, wizard pages, manifest, the ported string table. A `fingering` stage on a pose that holds its own legs (`LEGS_HELD_MARKERS`: `mating press`, `anvil position`, `full nelson`, or the tags `legs up` / `folded` / `knees to chest` / `legs over head` — 9 rows of the shipped matrix, matched on tags only) writes `1boy, hetero` instead of `solo`: the pose already uses both arms, so a hand action there made the model draw a third one. `HAND_ONLY_STAGES` in `PromptGenerator` is the list of stages that rule applies to (fingering today; object insertion is the same shape but was left as it was). The generator writes the anal channel word **weighted** — `PromptLimits.ANAL_CHANNEL_TAG`, `(anal:1.2)`, from `PromptGenerator.channelTag` — because a bare `anal` beside a pose the model reads as vaginal is the element that goes missing; the vaginal channel word stays plain, and the tests assert both forms. A pose that names its own place — `POSE_PLACES` in `prompt/PromptTypes.kt`, read through `PromptGenerator.posePlaces` — is paired only with a scene carrying it (`sceneCompatible` consults the places before `poseLocus`, which reads `sitting, resting head on desk, sleeping` as lying), and a row naming two places accepts either. The pool is per mode: SFW → `SFW_POSES`, SEX → `POSES`, NSFW → `SFW_POSES` + `QUESTIONABLE_POSES` (NSFW must never draw a sex row; the SEX stage page is what covers a sex act's run-up). A `QUESTIONABLE_POSES` row parses with `MatrixEntry.selfStated`, states its own clothing and exposure, and so gets no outfit, no `nude`, no `open clothes` and no chest/belly level; `PromptGenerator.stateFill` fills only the half it leaves open (aggressively: `topless, nipples` / `breasts hanging, nipples` / `sideboob, nipples`, nothing when the chest is not in frame, `bottomless` for an unstated bottom) from `POSE_CHEST_WORDS` / `POSE_BOTTOM_WORDS`. An unknown `UPPER_CASE:` section header is a parse error rather than a tag row of the section above it. `tools/gen_prompts.py` is deliberately **not** kept in step — it is legacy and frozen: the CLI knows none of these rules, and its parser does not even know the `QUESTIONABLE_POSES` section, so its SFW pool silently takes those rows. |
| `…/util/SaveClientFile.kt` (+ jvm/wasm actuals) | Hands a finished file to the user: desktop save dialog, web download |
| `…/util/ClientFilePicker.kt` (+ jvm/wasm actuals) | Reads a text file the user picks on their machine (a workflow JSON); `openLocalDirectory` is desktop-only |
| `…/jvmMain/` | WebSocket transport, helper spawn, FileKit, Coil fetcher |
| `Graphs.kt` / `Factory.kt` | Metro `AppGraph` + ViewModel factory |

Screens: `Images` | `Statistics` | `Utils` | `Dashboard` | `Automation` (`Stage.kt` enum).

Path pickers go through `PathPicker`. Desktop (`JvmPathPicker`) is FileKit (XDG portal on Linux). Web (`WasmPathPicker`) is an in-app porcelain dialog over `fs_listdir` / `fs_roots`, because the browser cannot return a POSIX path the trainer can open. Do not reintroduce `JFileChooser`. `initialDirectoryFor` seeds FileKit from the current field value; Save As on desktop may create a 0-byte placeholder that `deleteEmptyPlaceholder` removes.

Dashboard run history: `RunSelector` (`pages/components/DashboardWidgets.kt`, with the `runStampLabel` / `runDetailLabel` / `runStateLabel` / `displayedRun` helpers it renders) lists `list_runs` newest first and is what the whole page follows. Its first entry, **Current run**, clears the pin (`selectedRun = null`) so the page follows `state.json`; every other entry pins `DashboardUiState.selectedRun`. The run the page shows is `displayedRun` = the pin, else the entry whose id matches `dashboard`'s `run_id` (the trainer's own run) — with no run recorded there is none, so the box reads `Current run` / `Nothing started yet` and shows no run's figures. While following, the collapsed box keeps the title `Current run` and shows the run it resolved to underneath; a pinned entry titles that run's id instead. The only badge vocabulary is `runStateLabel`: **Live** while that run's process is running, **Stopped** otherwise, nothing at all when no run resolves (a `current` run is *not* labelled — it is Live or Stopped like any other). `fetchOnce` passes the shown run's `name` + `run_id` to `dashboard` / `list_samples` / `list_checkpoints` / `list_generated_samples` so charts, thumbnails and the checkpoint panel all come from that run; `TrainControlCard`'s identity line is that same run's (`controlRunName` / `controlRunId` in `pages/components/TrainControlCard.kt`, which fall back to `train_status`), so a history run with no logs and no samples is still named on the card. `trainingControlsEnabled` (`DashboardScreenViewModel.kt`) is the rule behind the control bar: the five buttons act on the run `state.json` is on, so they are all off while a past run is pinned (`TrainControlCard(controlsEnabled = …)`); `liveSettingsEnabled` adds "and that run is live" for the cadence/sampling controls, and without that the row shows `nextRunSummary` — `Next run · save every 100 steps · sampling on` from `config.toml` — because `state.json`'s `settings` block belongs to the run that published it (a runtime dir no run touched carries the placeholder `0` = "no checkpoints", which is not a configuration). `train_start` publishes the new run's values immediately after `mark_starting` (which clears the state), so the `starting` window shows the real cadence too. Selecting a run drops the previous run's `chartPick` / `previewIndex`.

The page's last section is **Checkpoints**, not a sample gallery: `pages/components/CheckpointList.kt` (`checkpointRows` / `CheckpointRow` / `checkpointRowLabel`) joins `list_checkpoints` (filtered to the run being shown) with `list_samples` and the generated jobs, so every card is a checkpoint with the images of its step. A job joins the card whose `checkpoint` path it names — the step is only the fallback for a record that names none — and any pass no card claimed gets a `samples only` row of its own rather than vanishing — or a `samples only` card for a step whose weights are gone (Reset keeps samples). The fullscreen preview cycles `previewList(state.checkpoints, samples, jobs)` — the same
`checkpointRows` → `sectionImages` list the section draws — so a thumbnail the page rendered always
resolves to an index when it is clicked; a preview built separately from the rows is how a
step-less pass ended up shown but inert. The Ctrl+click panel's own row uses `panelJobsForStep`,
which adds a pass rendered from the checkpoint it is showing to the jobs recorded at that step.

The user can **pin** a checkpoint (the card's pin button) and the pinned cards lead the section,
drawn on `PorcelainCard(emphasized = true)` — the accent tint and border that say "this card is not
like the others" — with a `Pinned` badge and a `CheckpointsDivider` (`N pinned · M more`) between the
two groups; the state is one run's, kept by the helper in
`<logging_dir>/<run_id>/checkpoint_pins.json` and read back through `checkpoint_pins` — Ranko never
writes that file. `checkpointRows(…, pinned = …)` is a stable partition of the section's own order,
so the pinned block keeps the newest-step-first order and a pinned card does not move when another
checkpoint is saved; a pin whose file is gone stays in the file and draws no card. The list is
cleared when the shown run changes, and `DashboardScreenViewModel.pinsWriteInFlight` keeps a poll
that started before a pin write from putting the old list back over the reply.

**Save As** is one flow with two entry points — the Ctrl+click panel and every checkpoint card —
because the copy is the helper's job (`checkpoint_export`; Ranko has no path into `output_dir`).
`saveCheckpointAs(checkpoint)` asks the OS for a destination, copies off-thread and reports into
`DashboardUiState.exportInFlightPath` / `exportResult` (a `CheckpointExport` carrying the source path,
so the line shows only on the card or panel it belongs to; `CheckpointExportStatus` renders it for
both). One export at a time: the second request is ignored and every other button is off while the
dialog or the copy is open.

Above the cards, `SampleRangeRow` (in `DashboardScreen.kt`, with `checkpointSteps` / `checkpointsInRange` / `batchRangeError` from `CheckpointList.kt`) is the step-range control: it prefills the run's own steps, says how many checkpoints the range covers, and starts `generate_checkpoint_samples_batch`. While a batch runs the row shows `runningBatch`'s progress (`batchProgressLabel`) with a **Stop** button that calls `cancel_generation`, and `runningJob != null` is what switches every per-card button off (a batch holds the card like any other generation). A checkpoint with no images gets a **Generate samples** button that calls `generate_checkpoint_samples`; it is enabled only while no live trainer is using the GPU (`generationAllowed` = `!alive || status !in GPU_BUSY_STATUSES`, mirrored by `api._gpu_busy`), which is what makes a **paused** run samplable. The Ctrl+click panel's own `Generate sample` button passes that same `gpuFree` rather than reading `alive` directly: the two used to disagree for a paused run, and the raw PID check is what left the panel disabled over a run that had finished (see §3 on zombies). Generated jobs live on `DashboardUiState.generatedJobs` (they used to sit inside `ChartPickState`), and `generatedSampleItems(job)` expands a `sets` job to one `SampleItem` per file, reading the `Pn` set from the file name.

Dashboard charts: the five training charts draw an always-on hover cursor with the exact step under the pointer, and mark the clicked step (dashed) plus the step a pick matched (bold, flagged). The **Train / Avg Loss** card additionally owns the checkpoint panel, opened by `Ctrl`+left click or by a left double click — the pick fires on the *picking* click, so a double click anchors at the second click, and the 400 ms window rule lives in `completesDoubleClick` (`pages/components/ChartPick.kt`). The panel shows the matched checkpoint highlighted, the clicked step's `Avg Loss`/`Loss`/UNet+TE LR, that step's samples with any generated ones, and can be dismissed by a click outside / close / `Esc`. It is resizable by dragging its bottom-right grip: placement is decided once from the click and the *default* size so a drag can never move the panel (`placePanelOrigin`/`clampPanelOrigin`), and the slots fill the dragged width (`sampleSlotWidth`, no 400 dp cap) with extra images wrapping instead of scrolling. `Save As` asks the OS for a dest path then calls `checkpoint_export` — the same flow a checkpoint card's own `Save As` uses (see the Checkpoints paragraph above); `Generate sample` renders one extra image per §5. Sample and dataset images load through `blob_batch` (Coil `BlobRef`), never `java.io.File`.

Dataset scan in the GUI is **non-recursive**, one folder, image + same-stem `.txt`. Orphan captions **abort** the statistics scan. `ranko/tools/agent.py` mirrors this (`--allow-orphans` to inspect anyway). Trash for GUI/agent drops: `/tmp/axlranko/trash` (not the dataset’s own `trash/` used by some `tools/` scripts).

Statistics → Control Panel **Shuffle & Renumber** is IPC `dataset_shuffle` (`trainer/fsrpc.py`): one group per stem (image(s) + `.txt` + `.mask.png`), temp `axl-shuffle-*` names, then `0001…`. The group keeps a mask on its image. It refuses a folder with an orphan `.txt`. Kotlin `DatasetShuffle.kt` remains a jvmTest helper. The standalone `tools/suf.py` still splits on the last dot and therefore still separates `{stem}.mask.png` from its image.

Every page that changes dataset files on disk **must** call `DatasetRefreshHub.notifyDatasetChanged()` — Statistics' tag edits, drop and shuffle, Utils' tagger, Images' caption save. Images and Statistics both cache the folder and reload off that hub (`reloadFromDisk`, `scanDataset`), and its buffer drops the oldest signal rather than refusing a burst. The other direction needs no signal: the nav rail scans Statistics on entry. The Statistics → Images thumbnail jump has to rescan too (`selectItemByTxtPath` always reloads with the jump remembered) instead of selecting out of the cached list, which is what used to show captions and samples the other page had already replaced. Images' reload keeps an unsaved caption draft only while the file still holds the text the draft was based on.

Mask edits stay in memory until **Save mask**. Invert / Fill / Clear / strokes set `maskDirty`; Clear is a pending sidecar delete; **Reset mask** restores the last loaded/saved snapshot. Navigation never writes a mask (`flushPendingMask` is gone). Raster is a `ByteArray` (`MaskCanvas`). Desktop `maskPaintInput` is AWT + 4 ms `MouseInfo` poll; wasm is document pointer events + a 4 ms interval. Compose pointer APIs still report the primary button only, which is why those actuals exist. The same listener reports every pointer position (`onPointerMoved`, throttled to 16 ms by `MaskPreview` for the brush cursor) and handles Alt+wheel brush resizing (`onBrushResize`; `util/MaskBrush.nudgeBrushRadius` owns the step and range). Coordinates come from `LayoutCoordinates.boundsInWindow()` in that modifier.

Screen→window conversion for the sampled pointer must go through a component **inside** the window (`window.contentPane`), never through the `Window` itself: a `Window`'s screen position is its frame origin including decorations, so converting through it lands `insets.top` pixels off — 41 px under KWin/XWayland — and half the samples then paint a parallel line offset from the other half. `MaskPreview` maps box coordinates to image coordinates and drops samples outside the drawn image, calling `onStrokeLeaveImage` so a stroke that leaves the image resumes as a new segment instead of smearing along the border. Brush math lives in `util/MaskBrush.kt` (falloff, path interpolation, blending) and raster ops in `util/MaskCanvas.kt`; both are unit-tested without a display.

DI: Metro `@Inject` / `@SingleIn(AppScope)` / `@ContributesBinding`. ViewModels via `metroViewModel()`. New ViewModels need constructor injection and to be reachable from the graph (follow existing `*ScreenViewModel`).

Hot reload: `./gradlew :desktopApp:hotRun --auto`. Normal: `./gradlew :desktopApp:run`. Web: `./gradlew :webApp:wasmJsBrowserDevelopmentRun` (helper must already be listening).

---

## 8. Dataset contract

Sidecar captions, comma-separated tags, extensions: jpg/jpeg/png/webp/bmp. The dataset is the list of `[[environment.train_data]]` folders (`train_data_dir` alone when there is no list), each with its own per-epoch `repeat`; records stay unique per image and only the bucket lists carry the repeats (see §4). Optional loss mask: `{stem}.mask.png` next to `{stem}.png` (always PNG). If the sidecar exists, MSE is weighted by that mask (white=train, black=ignore). If it is missing and the training image has an alpha channel, that alpha is the mask (0=ignore, 255=train). Either way the letterbox pad is weight 0, so **every** sample returns a full-bucket mask from `load_loss_mask`; only sidecar/alpha images count into `n_masked` (`Loss masks: n/m samples`). A silhouette's alpha is a hard 0/255 step, so `tools/mask_blur.py` batch-writes that sidecar from the alpha instead: a Gaussian blur of it, `--radius` in source-image pixels, same size as the training image (the loader resizes a sidecar to the source size NEAREST-first, so any other size would come back stepped), marked with an `axl_mask_blur` PNG text chunk. A sidecar without that chunk (a mask painted in Ranko) is left alone unless `--overwrite`; the training images are never written. **Exclude** `*.mask.png` from every image listing (`list_images`, Ranko Images/Statistics, `agent.py`, `tagger/`). Drop/trash moves the sidecar with the pair.

Python trainer `list_images` / Ranko / `agent.py` should stay consistent on extensions and “same stem” pairing. Ranko `parse_tags` = split `,` → trim → drop empty. Duplicates preserved in captions; stats dedupe per file.

Latent cache: one per dataset folder, `<folder>/.latents_cache/{sha1(abs_path::bucket_w x bucket_h::left,top,fit_w x fit_h)}.pt`. The key carries the absolute image path, so folders never collide, and the fit geometry, so any change to the bucket rule, the clamps or the fit path forces a one-time re-encode instead of silently serving latents built from differently placed pixels (a mask edit does *not* move the key — masks are not cached). A file is a hit only when it holds the keyed bucket's latent (4 channels, spatial / 8): an unreadable file, or one holding anything else — a stray write, a half-written file left by a killed run — is a miss, warned on stderr and re-encoded over, and `warm_latent_cache` skips only what the dataset itself calls a hit. Do not hand-edit cache files.

`tagger/` is an ONNX WD-tagger (`python tagger/main.py DIR --threshold 0.35`). Model files sit next to the script (`model.onnx`, `selected_tags.csv`). `migraphx_cache/` is generated — do not treat as source. Ranko Utils → Environment **Tag dataset** uses IPC `dataset_tag`.

`tools/` scripts are mostly **in-place / destructive** (`tools/vpred_reference.py` is the exception: a read-only renderer that implements ComfyUI's own sampling recipe, for checking a sample the trainer produced against ComfyUI's code path). Prefer `ranko/tools/agent.py --dry-run` for agent-driven edits. `ui.py` is a **deprecated** Streamlit viewer; do not extend it.

---

## 9. Recipes

### Add an IPC method

1. `handle_*` + `_HANDLERS` in `api.py`.
2. Request/response in `API.md`.
3. `TrainerIpcClient` method + kotlinx.serialization models.
4. Test in `test/test_api_ipc.py` (and Kotlin `DashboardIpcTest.kt` if the payload is parsed).
5. Never print to stdout from the handler.

### Add a base family

1. Catalog row in `trainer/family.py` `CATALOG` **and** Ranko `ModelSpecCatalog.kt` (same strings).
2. Implement `trainer/family_<id>.py` (load / unpack / LoRA / encode / loss / save / sample).
3. Register it in `resolve_family`. Set `trainable=True` only when the methods work.
4. Tests in `test_family.py` + Kotlin catalog test.
5. `doc/configuration.md` table.

Do not add a fifth TOML key. `base_model_version` is the dispatch key; the other `[model_spec]` fields must match the catalog row.

### Change training math / sampling

- Family loss / encode: `trainer/family_sdxl.py` (`compute_loss`, `encode_prompts`).
- Loop plumbing: `trainer/loop.py` (`build_group_inputs`, cadence).
- Prompt encoding (SDXL): root `text_processing.py`.
- Sample images: `trainer/sampling.py` via `SdxlFamily.generate_sample`.
- Keep `control.set_training` / `set_sampling` / `set_encoding` in sync so the dashboard progress bars stay honest.

### Change pause/offload

Only `trainer/device_swap.py` + call sites of `at_safe_point`. Iterate `SwapContext.text_encoders`; do not re-hardcode two TEs. Cover with `test_train_control.py` (optimizer tensor moves, command seq, stale PID). Do not skip Schedule-Free optimizer state when offloading.

### Change cleanup targets

Single helper: `trainer/cleanup.py`, always scoped to one run (`run_id`), with `run_id=None` meaning the legacy flat layout. `clean.py` is the interactive CLI (`--run`, `--legacy-flat`) and deletes samples + logs, plus weights when the user confirms; `train_reset` is the API and does nothing when no run resolves — it calls the same helper with all three flags off, because a reset run stays in the Dashboard's history list with its charts, samples and checkpoints. Do not give the API a way to delete weights again: Reset is the button a user reaches for at the end of a run.

### Change resume / checkpoint loading

`ModelFamily.load_lora` (implemented in `family_sdxl.py`) + `trainer/checkpoints.py`. Keep the kohya key map derived from the save-side `_convert_peft_to_kohya_bf16`, cover with `test_family.py` (map, round trip, rank mismatch, zero matches). Weights only — do not promise optimizer-state resume without implementing it.

---

## 10. Tests and how to run locally

| Suite | Command | Covers |
| --- | --- | --- |
| IPC | `python -m unittest discover -s test -p 'test_api_ipc.py'` | ping, dashboard empty logs, sample grouping (incl. `_p{set}_` names), avg-loss, dataset_tag, hardware_status, run-scoped dashboard/samples/checkpoints/reset (reset keeps samples + logs), `list_runs` history (brief figures, `current`/`live`, another `output_name`, a run with no logs resolving by id), `sample_sets` payload |
| Blob / FS RPC | `python -m unittest discover -s test -p 'test_blob_ipc.py'` | encode params, cache hit, hash, LRU cap, process pool, dataset list/shuffle/drop/mask/config/profile (incl. `config_save` refusing a non-filename-safe `output_name`), WS ping |
| Validation sets | `python -m unittest discover -s test -p 'test_validation.py'` | `resolve_sample_sets`: no entries → one set from the scalars, per-key fallback, name defaulting, ranges with the entry index (incl. `guidance_rescale` 0–1), matching seed sequences. `tracker_hparams` against a real `SummaryWriter` (a list-valued key must not reach `add_hparams`) |
| Tagger | `python -m unittest discover -s test -p 'test_tagger.py'` | CLI parse, dummy-session sidecar writes |
| Control | `python -m unittest discover -s test -p 'test_train_control.py'` | runtime dir, atomic state, commands, lock, swap tensors, run_id/resume state, `LiveSettings` (anchoring, the untouched modulo sequence, a missed boundary, `0`, a cadence change restarting the clock, a switch-only change leaving it, `read_settings` on a missing/corrupt file, `request_settings` merging, the `settings` state block, reset clearing the request) |
| Runs | `python -m unittest discover -s test -p 'test_runs.py'` | run id format/collision, run dir creation, latest-run lookup, run listing, `run_output_name`, `validate_output_name`, sample-dir lookup by name or fallback, all-names listing with per-run samples / newest step / checkpoints |
| Orphans | `python -m unittest discover -s test -p 'test_orphans.py'` | start-time identity, an unreaped child counting as gone, session membership, reaping a session, watching a leader die, the forkserver command-line fallback (no GPU) |
| Gen jobs | `python -m unittest discover -s test -p 'test_genjob.py'` | job naming/stem (incl. the `_sets`/`_batch` markers and `set_image_path`), request validation ranges, atomic write, listing order, done/error transitions, the sets-job fields, a job file without a mode reading as `single`, the batch plan record, no counter ever null (no GPU) |
| Checkpoint pins | `python -m unittest discover -s test -p 'test_checkpoint_pins.py'` | the pin file (`pins_path` in the run's log dir, atomic write, a missing/corrupt/hand-written file read leniently, pin idempotence and order, unpin), and both methods through `api.dispatch`: per-run files, the `file` the reply names, pinning a path that is not a file refused, a stale pin still removable, pinning a past run creating its log directory, a read never writing the file |
| The generator | `python -m unittest discover -s test -p 'test_generate_sample.py'` | the SIGTERM/SIGINT flag and its handler, `_shape_key` (what forces a pipeline rebuild), an empty batch plan refused, a missing checkpoint failing its own entry while the batch records it (no GPU; importing the module pulls torch in) |
| Automation | `python -m unittest discover -s test -p 'test_automation.py'` | ComfyUI client (proxy bypass, queue/poll/download, cancel), discovery (port scan, signature check, `$AXL_COMFY_URL`), workflow validation + both combo schemas of the model pre-check, settings/prompt-set stores, job records and dead-PID reconcile, the record helpers behind the per-image actions (`drop_image` keeping/removing an entry and renumbering, `next_image_number`, `prompt_entry_index`), the real runner against a stub ComfyUI (images + sidecars, seeds, batch size, one failing prompt, consecutive-failure stop, `--only-failed`, a redraw overwriting one image with a new seed, an append adding N with N seeds, a targeted pass leaving other prompts alone, a pass naming nothing refused, SIGTERM), the `automation_*` handlers through `api.dispatch` (including that a job's image passes `blob_stat`, the four per-image actions end to end, their refusals and the name-traversal guard), and two read-only checks against the machine's own ComfyUI when one is listening (no GPU work) |
| Family | `python -m unittest discover -s test -p 'test_family.py'` | catalog, spec mismatch, SD 3.5 refuse, `TrainConfig` refusing an unusable `output_name`, v-pred metadata, `sample_scheduler_kwargs` and `TrainConfig`'s derived prediction type (an epsilon base keeps epsilons; the base's `v_pred`/`ztsnr` markers or a v-pred/zero-SNR scheduler config switch both training and sampling), `build_noise_scheduler` following the same file, TE checkpoint helper, resume key map / round trip, the checkpoint-metadata cache (re-read after a rewrite, caller cannot corrupt it) |
| Sample offload | `python -m unittest discover -s test -p 'test_sampling_offload.py'` | S1/S2 device helpers, restore-after-sample, pause/resume re-offload, `_configure_scheduler` (Euler a + linspace, and that a v-pred model really gets `v_prediction` with the zero-terminal-SNR betas) |
| GPU smoke | `python -m unittest discover -s test -p 'test_vram_gpu.py'` | TE LoRA backward with checkpointing; sample offload on ROCm (conda `axl`) |
| Latent cache | `python test/test_warm_latent_cache.py` | pipelined vs serial; a cache file that is not the keyed latent is re-encoded; `--real` needs a VAE |
| Buckets | `python -m unittest discover -s test -p 'test_bucket_sampler.py'` | sampler batching/remainders, fit-geometry sweep (step alignment, clamps, no upscale, zero pad, cache key), a cache file holding another shape (or nothing readable) counting as a miss |
| Train data repeats | `python -m unittest discover -s test -p 'test_train_data_repeat.py'` | `resolve_train_data_entries` (blocks vs. the scalar, per-entry errors), per-folder latent caches, repeats expanding the buckets, an epoch drawing every image `repeat` times, `build_dataloader`'s length (what `steps_per_epoch` is derived from), collating one index twice |
| Masked loss | `python -m unittest discover -s test -p 'test_masked_loss.py'` | sidecar exclusion, ones/zero/gray weights, alpha fallback, fit+pad alignment, zero-weight pad, a tall sample keeping both end bands |
| Masked loss GPU | `python -m unittest discover -s test -p 'test_masked_loss_gpu.py'` | real SDXL encode+loss on a 2-image clone of `train_data_dir` (skipped without CUDA) |
| Mask blur | `python -m unittest discover -s test -p 'test_mask_blur.py'` | sidecar naming/extensions, alpha extraction (RGBA/LA/palette), uniform-alpha skips, blur written at the source size with the `axl_mask_blur` marker and a monotone ramp, training image untouched, hand-painted sidecar protected vs `--overwrite`, rerun replaces its own output, `--dry-run`, worker-pool vs in-process runs agreeing byte for byte and keeping the input order, CLI end to end, one masked loader check (skipped without torch) |
| Mask verifier | `python test/verify_mask_pipeline.py --tiers all` | closed loop for masks: CPU plumbing (sidecar pairing, crop/bucket geometry, cache independence), exact loss identities on GPU (all-ones == no mask, all-black == zero grads, mask linearity, coverage→loss), then real `trainer/main.py` runs (masked vs unmasked, 2 seeds, duplicate-run noise floor, resume) with per-region error probes. Report in `<report-dir>/mask_verify_report.md`; run it in the env `environment.yml` names (`axl`), ~41 min measured (52 checks, 0 failed on 2026-09-15). Its children are the runs the gfx1201 fault used to kill; it retries and escalates to `PYTORCH_NO_HIP_MEMORY_CACHING=1` if one dies. Refuses to start while a training run looks live; results, cost and the two deliberately unresolved observations: `doc/mask-verification.md` |
| Ranko | `cd ranko && ./gradlew :shared:jvmTest` | IPC models (incl. the `list_runs` history payload, the `settings` block, a `sets` job's `files`/counters, a `batch` record's progress, `cancel_requested`), the range helpers (`checkpointSteps` / `checkpointsInRange` / `batchRangeError`) and the batch/progress labels, run-history labels (stamp, step/sample/checkpoint/size line, `Live`/`Stopped`), the control-bar rule for a pinned past run plus `liveSettingsEnabled` / `generationAllowed` and the settings summary, the Checkpoints grouping (`checkpointRows`: a checkpoint with no samples, a `samples only` step, generated images riding with their step, a cancelled or failed pass keeping the images it wrote, `step N · final` labels, pinned rows leading as a stable partition while a pin with no checkpoint draws nothing), the pin payload the section reads (`CheckpointPinsResponse`, the `checkpoint_pin_set` request), generated-set expansion (`generatedSampleItems` → one item per file with its `Pn`), the `train_reset` request (no `delete_weights`), the output-name rule (`OutputNameTest`), TOML patch (incl. `[[validation.samples]]` blocks), catalog form, sample-set form/labels, image headers, mask sidecar names, `MaskCanvas` stroke math, `MaskBrush` falloff/cursor radii/wheel nudge, AWT mask input (buttons, hover, Alt+wheel; needs a display), dataset shuffle (mask/caption pairing, padded renumbering, seeded order, untouched directories and foreign files, orphan refusal, rollback on a failed rename), the prompt port (matrix/profile/generator/wizard/manifest) against the repo's own `input_matrix.txt` and `prompt_profiles/` — including that only the 9 folded/pinned poses count as holding their legs and that a fingering draw on one of them gets `1boy, hetero`, never `solo`, that a pose naming its own place pairs only with a scene carrying it (and every such row still has one), that the three pose pools stay disjoint (NSFW never draws a `POSES` row; SFW and SEX never draw a `QUESTIONABLE_POSES` one), that a self-stated row brings its own clothes and only gets the half it leaves open filled — the Automation IPC payloads (incl. the per-image seed list and its fallback to a record's single seed, the four image/prompt action requests, and the "add N images" draft bounds), the wizard's choice pills (`PromptChoiceLayoutTest`: every pill keeps the width it has on a wide pane when the pane narrows to 360 dp, so a long label wraps to the next line instead of measuring the last pill — the `SEX` mode button — down to nothing), and a render smoke test that composes each Automation pane — the per-image dialogs included — in a real window with fixture state (it catches the layout crash class that only shows up at measurement time; needs a display) |

Python suites live in `test/` — a plain namespace directory, deliberately **without**
`__init__.py`, so `import test` still resolves to the standard library package. Run them from the
repo root; `unittest discover -s test` puts `test/` on `sys.path` while `python -m` keeps the repo
root there, which is what the suites' `import api` / `from trainer…` need. A single file:
`python -m unittest discover -s test -p 'test_runs.py'`.

Cwd for Python tests: **repo root**. Use the env named in `environment.yml` (`axl`) so `torch` / `tensorboard` import.

Do not hit a real GPU in unit tests except `test_vram_gpu`, which is skipped when `torch.cuda.is_available()` is false. `test_train_control` may import `torch` for tensor device checks.

---

## 11. Style and refactor constraints

- Match the file you are in: trainer code is straightforward PyTorch, type hints on new public functions, no new abstraction layers “for cleanliness” unless a third call site exists.
- Dual-import `try/except ImportError` is intentional, not dead code.
- Kotlin: existing screens use ViewModel + Compose Material 3; do not introduce a second architecture (no extra navigation libraries).
- Comments: short, only for non-obvious constraints (ROCm alignment, stdout vs stderr, TOML flatten). Do not narrate the change.
- Do not add `requirements.txt`, HTTP servers, extra config formats, or a second control protocol.
- Do not format/rewrite unrelated Kotlin/Python files. Do not commit `ranko/build/`, `__pycache__/`, `tagger/migraphx_cache/`, or local `config.toml` path churn.
- License: MIT (`LICENSE`). Ranko still contains Compose template leftovers (`Greeting.kt`); ignore unless the task is cleanup.

---

## 12. Environment cheatsheet

| Var | Who | Meaning |
| --- | --- | --- |
| `AXL_PYTHON` | Ranko | Interpreter for `api.py` |
| `AXL_WS_HOST` / `AXL_WS_PORT` | api.py / Ranko | WebSocket bind (default `127.0.0.1:18765`) |
| `AXL_WS_ALLOW` | api.py | Comma-separated client IPs/CIDRs; loopback always allowed |
| `AXL_BLOB_WORKERS` | api.py | Blob encode pool size (`0` = inline) |
| `AXL_BLOB_CACHE_DIR` / `AXL_BLOB_CACHE_BYTES` | api.py | Processed-image cache |
| `AXL_RUNTIME_DIR` | trainer + api | Override runtime dir (required in tests) |
| `XDG_RUNTIME_DIR` | trainer + api | Default parent for `axltrainer/` |
| `PYTHONUNBUFFERED` | launchers / Ranko | Set to `1` |
| `MIOPEN_*` / `AMD_LOG_LEVEL` | `start_train.sh` | Quiet ROCm, pin cache |

Python: 3.14, PyTorch `2.13.0+rocm10.0.0` (HIP `7.15.26333`) per `environment.yml` (CUDA torch also works if you swap the wheel). That is also the stack on which the gfx1201 Tensile page fault reproduces most readily, and the pin that preceded it, `2.12.0+rocm7.14.1`, faults as well under other configurations: the pin changes which shapes and allocator layouts lose the guard-page lottery, not whether the kernels over-read (`doc/troubleshooting.md`). The measurements behind that sentence are sealed in `archive/` — 涉及负责任披露流程，暂不公开. Desktop: JDK 17+; Gradle wrapper provisions JDK 21.

Author reference GPU: AMD RX 9070 XT 16 GB, ROCm 7.2. Primary target is **AMD ROCm**, not NVIDIA.

---

## 13. Out of scope unless explicitly asked

- Publishing Ranko as a public site, TLS, and auth tokens. LAN with an IP allowlist is in. Process: `doc/ranko-web-target.md`.
- Replacing PEFT/diffusers with kohya sd-scripts internals.
- Serving checkpoints, Civitai upload, or remote training.
- Changing default `bucket_reso_steps` away from 128.
- Reviving or expanding `ui.py`.
