# AGENT.md — AXLTrainer

Working notes for coding agents. Human-facing docs live under `doc/` and `README.md`. Prefer this file when changing, extending, or refactoring code.

**What this is:** a local-first **LoRA** training stack (SDXL implemented; SD 3.5 catalogued but not trainable): Python engine (`trainer/`) + NDJSON IPC (`api.py`) + Kotlin/Compose desktop dashboard (`ranko/`, product name **AxlRanko**) + dataset scripts (`tools/`, `tagger/`, `ranko/tools/agent.py`).

**What this is not:** an HTTP API, a generation/inference server, or a kohya `sd-scripts` fork. There is no network control plane.

---

## 0. Working agreement (read first)

The maintainer drives this repo one step at a time. Do exactly what the current instruction asks, and nothing more:

- No unrelated fixes, refactors, cleanups or "while I'm here" edits — not even small ones.
- Stop at the end of the requested step. Do not run ahead into the step after it.
- Work beyond the request is a proposal, not an action: report it (what it would touch, why it seems useful) and leave it undone until asked.
- Terminology: **"the hook"** means `amdfq/amdfq-vmm-rs/` in its peralloc mode — the Rust `LD_PRELOAD` interposer that serves `hipMalloc` from address ranges it reserves itself (`amdfq/amdfq-vmm-rs/DESIGN.md`). Say **"the tail hook"** (or `amdfq/amdfq-tail-rs/`) when the tail-guard implementation is meant (`amdfq/amdfq-tail-rs/DESIGN.md`). The original C tail tree is `amdfq/amdfq-tail/`. The older C VMM tree (`amdfq-vmm/`) was deleted.
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
| ROCm pitfalls | `doc/troubleshooting.md`, `fixes/fix1.txt`, `fixes/fix2/` |

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
Ranko (JVM)  --NDJSON stdin/stdout-->  api.py  --reads-->  TensorBoard + sample PNGs
                                      |  writes command.json
                                      |  spawns (setsid) bash start_train.sh
                                      v
                         python -u trainer/main.py
                                      |
                                      +--> runtime dir: state.json, command.json, train.lock, train.log
                                      +--> logging_dir/{run_id}/          TensorBoard
                                      +--> output_dir/{run_id}/{name}_*   checkpoints + samples
```

`run_id` = `{output_name}_{YYYYMMDD_HHMMSS}`, created by `trainer/main.py` through `trainer/runs.py` (`create_run_dirs`). Every run gets its own pair of directories, so a later run (whose step counter restarts at 0) never overwrites an earlier one. `api.py` resolves the run for `dashboard` / `list_samples` / `train_reset` as: explicit `run_id` param → `state.json`'s `run_id` → newest `{name}_<timestamp>` under `logging_dir`.

Hard rules:

- Training is **detached**. `api.py` `train_start` uses `start_new_session=True` (`setsid`). Closing Ranko must not kill the run.
- Ranko **never** talks to the GPU. It only spawns `api.py` and renders responses.
- The trainer is `exec`'d by `start_train.sh`, so that shell's PID and **session** become the trainer's. A GPU fault aborts the trainer from inside HIP (`conclusions/bf16-kernel-overrun.md`) without running Python's `atexit`; its DataLoader forkserver then keeps the workers it forked alive, reparented to init, each holding `/dev/kfd` and ~0.5 GB. `start_train.sh` therefore starts `trainer/orphans.py` first, detached, to reap that session once the trainer is gone — keep it, and keep it unable to touch a session that is not the trainer's.
- `api.py` stdout is **NDJSON only**. Logs / tracebacks go to stderr (`run_ipc_loop` redirects `sys.stdout` to stderr after keeping the real stdout for replies).
- Working directory for `api.py` and `start_train.sh` is the **repo root** (directory that contains `api.py` and `trainer/`).
- Ranko finds that root by walking up from the executable / `user.dir` until a directory looks like one: `api.py` present, or `config.toml` next to the `trainer/` package (`TrainerRepo.looksLikeRepoRoot`). A lone `config.toml` must not qualify — a stranger's file would otherwise be edited.

Runtime dir resolution (same in `trainer/control.py` and `api.py`):

1. `$AXL_RUNTIME_DIR`
2. `$XDG_RUNTIME_DIR/axltrainer`
3. `/tmp/axltrainer-$UID`

Files: `state.json`, `command.json`, `train.lock`, `train.log`. Tests **must** set `AXL_RUNTIME_DIR` to a temp dir (see `test_train_control.py`).

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
reset (PID dead)      → idle    (also deletes samples + TB logs; optional weights)
```

`LIVE_STATUSES`: starting, encoding, training, sampling, pausing, paused, resuming, stopping.

`PHASE_STATUSES` (pause/resume attach here): encoding, training, sampling.

Commands (`command.json`): `pause` | `resume` | `stop`. One-shot, sequenced. Trainer consumes them **only** at `device_swap.at_safe_point(phase, swap_ctx)`.

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
`verify_mask_pipeline.py` / `fixes/fix2/repro_real.py` run an unmodified `trainer/main.py` against a
throwaway mirror of the repo. Every entry point therefore runs with cwd = repo root.

Load path:

- Python: `trainer/config.py` flattens **all TOML tables into one dict**. Section names do not exist at runtime on the Python side — only keys. `TrainConfig` fields default via `get_val(key, hardcoded)`. **TOML wins** over Python defaults.
- Kotlin: `AxlTrainerConfig` is **sectional** (`environment`, `model_spec`, `training`, …). Utils tab saves via `TomlDocumentPatcher`: in-place replace of uncommented `key = value` inside named tables. Comments, blank lines, and unknown tables (e.g. `[bookkeeping]`) stay intact. **Do not rewrite the whole file.**

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
  single-prompt behaviour, which is why `validation.sample_*` overrides in `fixes/` and
  `test/verify_mask_pipeline.py` still work. Ranges and the per-entry error message live there.
- Seed rule: inside a set the nth image uses `seed + n` (`0` = random per image). Two sets sharing a
  seed start from the same noise; that is the point (only the prompt differs).
- Images: `{output_name}_{step:06d}_p{set}_{repeat}.png`, `set` counting from 0. `api.scan_samples`
  also parses the old two-number name as set 0, and returns `set_index` for the Ranko `Pn` badges.
  `control.set_sampling` reports a global image counter plus `prompt_set`/`prompt_sets`.
- Ranko: `SampleSetForm` in `TrainingConfigForm`, tabs in the Utils Validation section,
  `TomlDocumentPatcher.replaceArrayOfTables` for the blocks. The form writes `[validation]` from the
  **first** set, so the file never holds two contradictory prompts.

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
crop variant in the training path** — `resize_and_center_crop` survives only for `fixes/` and the
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
| `trainer/config.py` | `TrainConfig` + TOML flatten; `resolve_sample_sets` (prompt sets), `tracker_hparams` (tracker-safe config view); `__post_init__` validates `[model_spec]` against the family catalog. |
| `trainer/family.py` | Catalog (`sdxl_base_v1-0`, `sd3.5-large`), `resolve_family`, `require_trainable`. |
| `trainer/family_sdxl.py` | SDXL load/unpack/LoRA/encode/loss/save/sample; PEFT → kohya remap **and** the reverse map used by resume (`load_lora`). |
| `trainer/family_sd35.py` | Stub; every method raises `UnsupportedFamilyError`. |
| `trainer/dataset.py` | `LoraImageDataset`: images + sidecar captions, buckets + per-record fit geometry, latent `.pt` lookup. |
| `trainer/cache.py` | Pipelined CPU decode → batched VAE encode → atomic `.pt` in `<data>/.latents_cache`. GPU holds only the VAE; UNet/TEs are offloaded first. |
| `trainer/loop.py` | `train_one_epoch`: group by bucket, family `compute_loss`, both optimizers, save+sample cadence, `at_safe_point`. |
| `trainer/sampling.py` | Interruptible SDXL sample gen (called from `SdxlFamily.generate_sample`). Each `[[validation.samples]]` set is encoded and rendered on its own (its own scheduler sigmas / size / steps / seed, so no set's chunk padding depends on another's prompt); prompt encode → TE offload; denoise → UNet offload then VAE decode; restore UNet+TEs before returning to the train loop. |
| `trainer/models.py` | Flash attn, optimizers, checkpoint **paths**, kohya metadata helper. |
| `trainer/control.py` | State machine, atomic JSON, lock, command poll. |
| `trainer/device_swap.py` | GPU↔CPU offload; `at_safe_point`. |
| `trainer/loss_log.py` | Kohya-style `Train/Avg_Loss` window (`LossRecorder`). |
| `trainer/cleanup.py` | Discover/delete one run's samples, TB logs, optional weight dirs. Shared by `api.py` `train_reset` and `clean.py`. |
| `trainer/runs.py` | Run id naming (`{name}_{YYYYMMDD_HHMMSS}`), `create_run_dirs`, `find_latest_run`, `list_runs`. torch-free. |
| `trainer/orphans.py` | Reaps what a signal-killed trainer leaves behind: `start_train.sh` starts it detached before `exec`ing the trainer, it waits for that PID (start-time guarded, zombies count as gone) and then kills the trainer's session, or the forkservers matching a `trainer/main.py` path when it is not the session leader. torch-free. |
| `trainer/checkpoints.py` | `resolve_resume_path`, `read_lora_metadata`, `discover_checkpoints` (run-scoped) for resume + the Ranko picker. |
| `trainer/genjob.py` | Job records for one-off sample generation (`{name}_samples/generated/*.json`): naming, request validation, atomic write, listing. torch-free. |
| `trainer/generate_sample.py` | `python -u trainer/generate_sample.py --spec <job.json>`: loads the base + a kohya LoRA (reusing `SdxlFamily.apply_lora`/`load_lora`), samples with the run's own scheduler/settings taken from the checkpoint metadata, writes the PNG + progress into the job file. Detached, never touches `state.json`/the lock. |
| `trainer/env.py` | MIGraphX cache dir, `flush_memory`. |
| `trainer/hardware.py` | Ranko hardware panel: nvtop snapshot, AMD edge/junction, CPU util/temp, RAM. |
| `trainer/utils.py` | Image list, caption shuffle, bucket math (`pick_bucket_size`), fit geometry (`fit_geometry`/`fit_to_bucket`), loss masks, `build_time_ids`. |
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
- LoRA targets: UNet `to_q/to_k/to_v/to_out.0`; TE `q_proj/k_proj/v_proj/out_proj` (`setup.apply_lora_modules`).
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

`bucket_reso_steps` must keep VAE latents (spatial / 8) **divisible by 16**. Default **128**. `64` causes random GPU page faults on AMD (see `fixes/fix1.txt`). Do not “optimize” this down. `start_train.sh` also sets `PYTORCH_CUDA_ALLOC_CONF` and MIOpen log/cache env; keep those if you touch the launcher.

---

## 6. IPC (`api.py`)

Framing: one JSON object per line. `{id, method, params}` → `{id, ok: true, result}` or `{id, ok: false, error}`.

Handlers (`_HANDLERS` — add here **and** in `API.md` **and** `TrainerIpcClient.kt`):

| Method | Side effect |
| --- | --- |
| `ping` | none |
| `dashboard` | read TB scalars + flattened config (run-scoped: `run_id`) |
| `list_samples` | scan the run's sample PNGs |
| `list_checkpoints` | list LoRA files under `output_dir/{name}_<timestamp>/` (read-only) |
| `train_status` | `control.status_payload()` + dead-PID reconcile |
| `train_start` | spawn `start_train.sh` (rejects a bad `resume_lora_path` up front) |
| `train_pause` / `train_resume` / `train_stop` | write `command.json` |
| `train_reset` | `run_cleanup` (resolved run) + `reset_to_idle` |
| `dataset_tag` | spawn `tagger/main.py` (GPU ONNX); overwrites sidecar `.txt` |
| `generate_sample` | spawn `trainer/generate_sample.py` **detached** (returns immediately; refuses while any trainer PID is alive, and while another job is running) |
| `list_generated_samples` | read-only: the run's `generated/*.json` jobs, newest first; a `running` job whose PID died is rewritten to `error` |
| `hardware_status` | `nvtop -s` JSON + DRM hwmon temps + `/proc` CPU (read-only) |

`dashboard` synthesizes `Train/Avg_Loss` from `Train/Loss` via `synthesize_avg_loss` when the tag is missing (old runs). Do not rename TensorBoard tags without updating Ranko chart cards.

Kotlin client: one request at a time (`Mutex` in `TrainerIpcClient`). `ignoreUnknownKeys = true`. Keep that when adding fields.

---

## 7. Ranko (`ranko/`)

Compose Multiplatform **desktop JVM only** (not Android/iOS). Kotlin 2.4.10, Compose 1.12.0, Material 3, Metro DI, ktoml, Coil 3, haze 2.0. Visual style is KataHana **Sky & Sakura**: `RankoTheme` + Nunito + porcelain cards. Raw hex lives only in `ui/theme/Color.kt`. Screens read `rankoColors` / `PorcelainCard` / `CapsuleButton`; Canvas helpers take colors as parameters.

User-facing look-and-feel (background: Solid / Glow / Image, independent card vs background blur, font/icon scale) lives in the **Appearance** section of the Utils tab and is persisted by `AppearanceRepository` (Java Preferences, key `com/acite/axlranko/appearance`). `App.kt` consumes it and feeds `LocalDensity` so **font scale only affects sp** and **icon scale only affects dp** (`density * iconScale`, `fontScale * font / iconScale`). `RankoBackdrop` renders glow orbs for `Glow` and a cropped, dimmed photo for `Image`. A full-window haze layer uses `backgroundBlurRadiusDp` (gaps); `PorcelainCard` / `FrostedSurface` use `cardBlurRadiusDp`.

| Path | Role |
| --- | --- |
| `ranko/desktopApp/…/main.kt` | Window; `createGraph<AppGraph>()` |
| `…/util/ProcessExitGuard.kt` (jvmMain) | One watcher per quit, armed from `main.kt`'s close request and from a shutdown hook: it kills the JVM if it is still there `DEFAULT_GRACE_SECONDS` later, start-time guarded so a reused PID is left alone. A hang inside the VM cannot be undone from inside the VM, and a windowless Ranko keeps the GPU render nodes and its `api.py` child. |
| `ranko/shared/src/commonMain/…/App.kt`, `Stage.kt` | Shell + four screens |
| `…/ui/theme/` | Sky & Sakura palette, tokens, Nunito, `RankoTheme` |
| `…/ui/components/` | Backdrop, porcelain/frosted surfaces, capsule controls |
| `…/pages/*Screen.kt` + `*ViewModel.kt` | UI + state |
| `…/data/TrainerIpcClient.kt` | NDJSON child process |
| `…/data/TrainerRepo.kt` | Repo-root discovery |
| `…/data/ConfigModel.kt` | Sectional TOML model |
| `…/data/TomlDocumentPatcher.kt` | Comment-preserving save |
| `…/data/ConfigImporter.kt` | expect/actual load/save |
| `…/jvmMain/` | TOML IO, `getAppExecutionPath` |
| `Graphs.kt` / `Factory.kt` | Metro `AppGraph` + ViewModel factory |

Screens: `Images` | `Statistics` | `Utils` | `Dashboard` (`Stage.kt` enum).

Every file/folder/save dialog goes through `util/FileDialogs.kt` (FileKit: XDG desktop portal on Linux, so the KDE/GNOME picker, `IFileDialog` / `NSOpenPanel` elsewhere). Do not reintroduce `JFileChooser`: it is Swing-drawn and ignores the desktop theme, and FileKit only falls back to it when no portal is reachable. The functions are suspend and are called from a ViewModel's `viewModelScope` (no parent-window handle is passed, matching the reference setup). `initialDirectoryFor` seeds the dialog from the current field value; `saveFileDialog` lets FileKit create the destination file, so the Save As path overwrites it and `deleteEmptyPlaceholder` removes the leftover when the appended `.safetensors` renamed it.

Dashboard charts: the five training charts draw an always-on hover cursor with the exact step under the pointer, and mark the clicked step (dashed) plus the step a pick matched (bold, flagged). The **Train / Avg Loss** card additionally owns the checkpoint panel, opened by `Ctrl`+left click or by a left double click — the pick fires on the *picking* click, so a double click anchors at the second click, and the 400 ms window rule lives in `completesDoubleClick` (`pages/components/ChartPick.kt`). The panel shows the matched checkpoint highlighted, the clicked step's `Avg Loss`/`Loss`/UNet+TE LR, that step's samples with any generated ones, and can be dismissed by a click outside / close / `Esc`. It is resizable by dragging its bottom-right grip: placement is decided once from the click and the *default* size so a drag can never move the panel (`placePanelOrigin`/`clampPanelOrigin`), and the slots fill the dragged width (`sampleSlotWidth`, no 400 dp cap) with extra images wrapping instead of scrolling. `Save As` copies the LoRA file out through the OS save dialog with progress; `Generate sample` renders one extra image per §5. Pure helpers for the mapping, the nearest-checkpoint/sample selection, the click timing, the panel sizing/placement, the training-stat lookup and the generate-form validation live in `pages/components/ChartPick.kt`; the checkpoint list is scanned on click (2 s for 60+ files) and cached per session; the copy lives in `util/FileCopy.kt` and refuses to overwrite the source.

Dataset scan in the GUI is **non-recursive**, one folder, image + same-stem `.txt`. Orphan captions **abort** the statistics scan. `ranko/tools/agent.py` mirrors this (`--allow-orphans` to inspect anyway). Trash for GUI/agent drops: `/tmp/axlranko/trash` (not the dataset’s own `trash/` used by some `tools/` scripts).

Mask painting does **not** use Compose pointer APIs: `maskPaintInput` (`pages/components/MaskPaint.kt` expect, `jvmMain/.../MaskPaint.jvm.kt` actual) attaches a global AWT mouse listener to the host window (both buttons are reported) plus a 4 ms `MouseInfo` sampler while a stroke is active, because AWT coalesces motion events and fast strokes used to land as separate dots. The same listener reports every pointer position (`onPointerMoved`, throttled to 16 ms by `MaskPreview` for the brush cursor) and handles Alt+wheel brush resizing (`onBrushResize`; `util/MaskBrush.nudgeBrushRadius` owns the step and range). Coordinates come from `LayoutCoordinates.boundsInWindow()` in that modifier.

Screen→window conversion for the sampled pointer must go through a component **inside** the window (`window.contentPane`), never through the `Window` itself: a `Window`'s screen position is its frame origin including decorations, so converting through it lands `insets.top` pixels off — 41 px under KWin/XWayland — and half the samples then paint a parallel line offset from the other half. `MaskPreview` maps box coordinates to image coordinates and drops samples outside the drawn image, calling `onStrokeLeaveImage` so a stroke that leaves the image resumes as a new segment instead of smearing along the border. Brush math lives in `util/MaskBrush.kt` (falloff, path interpolation, blending) and raster ops in `util/MaskCanvas.kt`; both are unit-tested without a display.

DI: Metro `@Inject` / `@SingleIn(AppScope)` / `@ContributesBinding`. ViewModels via `metroViewModel()`. New ViewModels need constructor injection and to be reachable from the graph (follow existing `*ScreenViewModel`).

Hot reload: `./gradlew :desktopApp:hotRun --auto`. Normal: `./gradlew :desktopApp:run`.

---

## 8. Dataset contract

Sidecar captions, comma-separated tags, extensions: jpg/jpeg/png/webp/bmp. Optional loss mask: `{stem}.mask.png` next to `{stem}.png` (always PNG). If the sidecar exists, MSE is weighted by that mask (white=train, black=ignore). If it is missing and the training image has an alpha channel, that alpha is the mask (0=ignore, 255=train). Either way the letterbox pad is weight 0, so **every** sample returns a full-bucket mask from `load_loss_mask`; only sidecar/alpha images count into `n_masked` (`Loss masks: n/m samples`). **Exclude** `*.mask.png` from every image listing (`list_images`, Ranko Images/Statistics, `agent.py`, `tagger/`). Drop/trash moves the sidecar with the pair.

Python trainer `list_images` / Ranko / `agent.py` should stay consistent on extensions and “same stem” pairing. Ranko `parse_tags` = split `,` → trim → drop empty. Duplicates preserved in captions; stats dedupe per file.

Latent cache: `<train_data_dir>/.latents_cache/{sha1(abs_path::bucket_w x bucket_h::left,top,fit_w x fit_h)}.pt`. The key carries the fit geometry, so any change to the bucket rule, the clamps or the fit path forces a one-time re-encode instead of silently serving latents built from differently placed pixels (a mask edit does *not* move the key — masks are not cached). Do not hand-edit cache files.

`tagger/` is an ONNX WD-tagger (`python tagger/main.py DIR --threshold 0.35`). Model files sit next to the script (`model.onnx`, `selected_tags.csv`). `migraphx_cache/` is generated — do not treat as source. Ranko Utils → Environment **Tag dataset** uses IPC `dataset_tag`.

`tools/` scripts are mostly **in-place / destructive**. Prefer `ranko/tools/agent.py --dry-run` for agent-driven edits. `ui.py` is a **deprecated** Streamlit viewer; do not extend it.

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

Single helper: `trainer/cleanup.py`, always scoped to one run (`run_id`), with `run_id=None` meaning the legacy flat layout. `clean.py` is the interactive CLI (`--run`, `--legacy-flat`); `train_reset` is the API and does nothing when no run resolves. Keep them identical.

### Change resume / checkpoint loading

`ModelFamily.load_lora` (implemented in `family_sdxl.py`) + `trainer/checkpoints.py`. Keep the kohya key map derived from the save-side `_convert_peft_to_kohya_bf16`, cover with `test_family.py` (map, round trip, rank mismatch, zero matches). Weights only — do not promise optimizer-state resume without implementing it.

---

## 10. Tests and how to run locally

| Suite | Command | Covers |
| --- | --- | --- |
| IPC | `python -m unittest discover -s test -p 'test_api_ipc.py'` | ping, dashboard empty logs, sample grouping (incl. `_p{set}_` names), avg-loss, dataset_tag, hardware_status, run-scoped dashboard/samples/checkpoints/reset, `sample_sets` payload |
| Validation sets | `python -m unittest discover -s test -p 'test_validation.py'` | `resolve_sample_sets`: no entries → one set from the scalars, per-key fallback, name defaulting, ranges with the entry index, matching seed sequences. `tracker_hparams` against a real `SummaryWriter` (a list-valued key must not reach `add_hparams`) |
| Tagger | `python -m unittest discover -s test -p 'test_tagger.py'` | CLI parse, dummy-session sidecar writes |
| Control | `python -m unittest discover -s test -p 'test_train_control.py'` | runtime dir, atomic state, commands, lock, swap tensors, run_id/resume state |
| Runs | `python -m unittest discover -s test -p 'test_runs.py'` | run id format/collision, run dir creation, latest-run lookup, run listing |
| Orphans | `python -m unittest discover -s test -p 'test_orphans.py'` | start-time identity, an unreaped child counting as gone, session membership, reaping a session, watching a leader die, the forkserver command-line fallback (no GPU) |
| Gen jobs | `python -m unittest discover -s test -p 'test_genjob.py'` | job naming/stem, request validation ranges, atomic write, listing order, done/error transitions (no GPU) |
| Family | `python -m unittest discover -s test -p 'test_family.py'` | catalog, spec mismatch, SD 3.5 refuse, v-pred metadata, TE checkpoint helper, resume key map / round trip |
| Sample offload | `python -m unittest discover -s test -p 'test_sampling_offload.py'` | S1/S2 device helpers, restore-after-sample, pause/resume re-offload |
| GPU smoke | `python -m unittest discover -s test -p 'test_vram_gpu.py'` | TE LoRA backward with checkpointing; sample offload on ROCm (conda `axl`) |
| Latent cache | `python test/test_warm_latent_cache.py` | pipelined vs serial; `--real` needs a VAE |
| Buckets | `python -m unittest discover -s test -p 'test_bucket_sampler.py'` | sampler batching/remainders, fit-geometry sweep (step alignment, clamps, no upscale, zero pad, cache key) |
| Masked loss | `python -m unittest discover -s test -p 'test_masked_loss.py'` | sidecar exclusion, ones/zero/gray weights, alpha fallback, fit+pad alignment, zero-weight pad, a tall sample keeping both end bands |
| Masked loss GPU | `python -m unittest discover -s test -p 'test_masked_loss_gpu.py'` | real SDXL encode+loss on a 2-image clone of `train_data_dir` (skipped without CUDA) |
| Mask verifier | `python test/verify_mask_pipeline.py --tiers all` | closed loop for masks: CPU plumbing (sidecar pairing, crop/bucket geometry, cache independence), exact loss identities on GPU (all-ones == no mask, all-black == zero grads, mask linearity, coverage→loss), then real `trainer/main.py` runs (masked vs unmasked, 2 seeds, duplicate-run noise floor, resume) with per-region error probes. Report in `<report-dir>/mask_verify_report.md`; run it in the env `environment.yml` names (`axl`), ~41 min measured (52 checks, 0 failed on 2026-09-15). Its children are the runs the gfx1201 fault used to kill; it retries and escalates to `PYTORCH_NO_HIP_MEMORY_CACHING=1` if one dies. Refuses to start while a training run looks live; results, cost and the two deliberately unresolved observations: `doc/mask-verification.md` |
| Ranko | `cd ranko && ./gradlew :shared:jvmTest` | IPC models, TOML patch (incl. `[[validation.samples]]` blocks), catalog form, sample-set form/labels, image headers, mask sidecar names, `MaskCanvas` stroke math, `MaskBrush` falloff/cursor radii/wheel nudge, AWT mask input (buttons, hover, Alt+wheel; needs a display) |

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
| `AXL_RUNTIME_DIR` | trainer + api | Override runtime dir (required in tests) |
| `XDG_RUNTIME_DIR` | trainer + api | Default parent for `axltrainer/` |
| `PYTHONUNBUFFERED` | launchers / Ranko | Set to `1` |
| `MIOPEN_*` / `AMD_LOG_LEVEL` | `start_train.sh` | Quiet ROCm, pin cache |

Python: 3.14, PyTorch `2.13.0+rocm10.0.0` (HIP `7.15.26333`) per `environment.yml` (CUDA torch also works if you swap the wheel). That is also the stack on which the gfx1201 Tensile page fault reproduces most readily — the `fixes/fix2` repros die on demand on it; the pin that preceded it, `2.12.0+rocm7.14.1`, faults as well under other configurations (`fixes/fix3`) — the pin changes which shapes and allocator layouts lose the guard-page lottery, not whether the kernels over-read (`conclusions/bf16-overrun-mitigations.md`). Desktop: JDK 17+; Gradle wrapper provisions JDK 21.

Author reference GPU: AMD RX 9070 XT 16 GB, ROCm 7.2. Primary target is **AMD ROCm**, not NVIDIA.

---

## 13. Out of scope unless explicitly asked

- Porting Ranko off desktop JVM.
- Replacing PEFT/diffusers with kohya sd-scripts internals.
- Serving checkpoints, Civitai upload, or remote training.
- Changing default `bucket_reso_steps` away from 128.
- Reviving or expanding `ui.py`.
