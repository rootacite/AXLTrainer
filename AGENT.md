# AGENT.md — AXLTrainer

Working notes for coding agents. Human-facing docs live under `doc/` and `README.md`. Prefer this file when changing, extending, or refactoring code.

**What this is:** a local-first **LoRA** training stack (SDXL implemented; SD 3.5 catalogued but not trainable): Python engine (`trainer/`) + NDJSON IPC (`api.py`) + Kotlin/Compose desktop dashboard (`ranko/`, product name **AxlRanko**) + dataset scripts (`tools/`, `tagger/`, `ranko/tools/agent.py`).

**What this is not:** an HTTP API, a generation/inference server, or a kohya `sd-scripts` fork. There is no network control plane.

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
| ROCm pitfalls | `doc/troubleshooting.md`, `fixes/fix1.txt` |

Verify after a change (pick the layer you touched):

```bash
# Python IPC + control plane (cwd = repo root, env `axl`)
python -m unittest test_api_ipc test_train_control test_family

# Latent-cache pipeline (mock VAE)
python trainer/test_warm_latent_cache.py

# Ranko serialization / IPC models
cd ranko && ./gradlew :shared:jvmTest
```

Do **not** start a real training run to “see if it compiles” unless the task requires GPU behavior. `trainer/config.toml` contains **author-local paths** and will fail on other machines.

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
                                      +--> logging_dir/{output_name}/   TensorBoard
                                      +--> output_dir/{output_name}_*   checkpoints + samples
```

Hard rules:

- Training is **detached**. `api.py` `train_start` uses `start_new_session=True` (`setsid`). Closing Ranko must not kill the run.
- Ranko **never** talks to the GPU. It only spawns `api.py` and renders responses.
- `api.py` stdout is **NDJSON only**. Logs / tracebacks go to stderr (`run_ipc_loop` redirects `sys.stdout` to stderr after keeping the real stdout for replies).
- Working directory for `api.py` and `start_train.sh` is the **repo root** (directory that contains `api.py` and `trainer/`).
- Ranko finds that root by walking up from the executable / `user.dir` until it sees `api.py` **or** `trainer/config.toml` (`TrainerRepo`).

Runtime dir resolution (same in `trainer/control.py` and `api.py`):

1. `$AXL_RUNTIME_DIR`
2. `$XDG_RUNTIME_DIR/axltrainer`
3. `/tmp/axltrainer-$UID`

Files: `state.json`, `command.json`, `train.lock`, `train.log`. Tests **must** set `AXL_RUNTIME_DIR` to a temp dir (see `test_train_control.py`).

Interpreter override: Ranko uses `$AXL_PYTHON` if set, else `python3`. Training deps live in conda env `axl` (`environment.yml`). There is **no** `requirements.txt`.

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

---

## 4. Configuration contract

**Single source of truth:** `trainer/config.toml`. `trainer/main.py` takes **no CLI args**.

Load path:

- Python: `trainer/config.py` flattens **all TOML tables into one dict**. Section names do not exist at runtime on the Python side — only keys. `TrainConfig` fields default via `get_val(key, hardcoded)`. **TOML wins** over Python defaults.
- Kotlin: `AxlTrainerConfig` is **sectional** (`environment`, `model_spec`, `training`, …). Utils tab saves via `TomlDocumentPatcher`: in-place replace of uncommented `key = value` inside named tables. Comments, blank lines, and unknown tables (e.g. `[bookkeeping]`) stay intact. **Do not rewrite the whole file.**

Adding a hyperparameter (all four, or the GUI will drift):

1. `trainer/config.toml` — pick an existing table or add one.
2. `TrainConfig` in `trainer/config.py` — same **flat** key name.
3. Kotlin `AxlTrainerConfig` + nested data class (`ConfigModel.kt`), `TrainingConfigForm`, Utils UI bind/save map (section name → key → encoded value).
4. `doc/configuration.md`.

If only the trainer needs it, you can skip (3) but document that the Utils editor will not see it until the Kotlin model is updated. ktoml parse of the full file will fail if a **required** Kotlin field is missing — new optional keys are safer as Kotlin defaults.

Shipped `config.toml` and `TrainConfig` fallbacks contain **author machine paths**. Never “fix” them to placeholders as part of an unrelated PR unless asked; consumers already know they must edit `[environment]`.

---

## 5. Python trainer map

Entry: `bash start_train.sh` → `python -u trainer/main.py` with ROCm log filters and MIOpen cache pins.

| File | Responsibility |
| --- | --- |
| `trainer/main.py` | Lifecycle: lock, seed, `build_train_objects`, optional `warm_latent_cache`, epoch loop, final checkpoint, `end_run`. |
| `trainer/setup.py` | `TrainArtifacts`: `resolve_family`, pipeline, PEFT LoRA, dataloader, dual optimizers, Accelerator. |
| `trainer/config.py` | `TrainConfig` + TOML flatten; `__post_init__` validates `[model_spec]` against the family catalog. |
| `trainer/family.py` | Catalog (`sdxl_base_v1-0`, `sd3.5-large`), `resolve_family`, `require_trainable`. |
| `trainer/family_sdxl.py` | SDXL load/unpack/LoRA/encode/loss/save/sample; PEFT → kohya remap. |
| `trainer/family_sd35.py` | Stub; every method raises `UnsupportedFamilyError`. |
| `trainer/dataset.py` | `LoraImageDataset`: images + sidecar captions, buckets, latent `.pt` lookup. |
| `trainer/cache.py` | Pipelined CPU decode → batched VAE encode → atomic `.pt` in `<data>/.latents_cache`. |
| `trainer/loop.py` | `train_one_epoch`: group by bucket, family `compute_loss`, both optimizers, save+sample cadence, `at_safe_point`. |
| `trainer/sampling.py` | Interruptible SDXL sample gen (called from `SdxlFamily.generate_sample`). |
| `trainer/models.py` | Flash attn, optimizers, checkpoint **paths**, kohya metadata helper. |
| `trainer/control.py` | State machine, atomic JSON, lock, command poll. |
| `trainer/device_swap.py` | GPU↔CPU offload; `at_safe_point`. |
| `trainer/loss_log.py` | Kohya-style `Train/Avg_Loss` window (`LossRecorder`). |
| `trainer/cleanup.py` | Discover/delete samples, TB logs, optional `{output_name}_*` weight dirs. Shared by `api.py` `train_reset` and `clean.py`. |
| `trainer/env.py` | MIGraphX cache dir, `flush_memory`. |
| `trainer/utils.py` | Image list, caption shuffle, bucket math, `build_time_ids`. |
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
- Batches are **regrouped by `(bucket_w, bucket_h)`** before stacking — never stack mixed spatial sizes.
- `at_safe_point` is called every step (and during cache/sample). New long GPU work must call it or pause/stop will hang until the phase ends.
- VAE is moved to CPU after latent warm-cache; on-demand encode during training is the fallback.

### Checkpoints (ComfyUI / kohya)

`SdxlFamily.save_lora` remaps PEFT keys to:

- `lora_unet_*` / `lora_te1_*` / `lora_te2_*`
- `lora_down.weight` / `lora_up.weight` / `alpha`
- tensors **bf16**
- metadata `modelspec.*` + `ss_*`

Layout (`lora_checkpoint_file`):

| Kind | Directory |
| --- | --- |
| step | `{output_dir}/{output_name}_s{step:06d}/{safe_name}.safetensors` |
| epoch | `{output_dir}/{output_name}_e{epoch:03d}_s{step:06d}/…` |
| final | `{output_dir}/{output_name}_final/…` |

Samples: `{output_dir}/{output_name}_samples/` filenames matching `_(\d+)_(\d+)\.png$` → `(step, repeat_idx)`. `api.scan_samples` uses that regex; unmatched files go under step `"-1"`.

Cleanup treats every `{output_dir}` child dir whose name **starts with** `output_name` except `{name}_samples` as a weight dir. Do not invent output folder names that collide with that prefix rule.

### ROCm (non-negotiable)

`bucket_reso_steps` must keep VAE latents (spatial / 8) **divisible by 16**. Default **128**. `64` causes random GPU page faults on AMD (see `fixes/fix1.txt`). Do not “optimize” this down. `start_train.sh` also sets `PYTORCH_CUDA_ALLOC_CONF` and MIOpen log/cache env; keep those if you touch the launcher.

---

## 6. IPC (`api.py`)

Framing: one JSON object per line. `{id, method, params}` → `{id, ok: true, result}` or `{id, ok: false, error}`.

Handlers (`_HANDLERS` — add here **and** in `API.md` **and** `TrainerIpcClient.kt`):

| Method | Side effect |
| --- | --- |
| `ping` | none |
| `dashboard` | read TB scalars + flattened config |
| `list_samples` | scan sample PNGs |
| `train_status` | `control.status_payload()` + dead-PID reconcile |
| `train_start` | spawn `start_train.sh` |
| `train_pause` / `train_resume` / `train_stop` | write `command.json` |
| `train_reset` | `run_cleanup` + `reset_to_idle` |

`dashboard` synthesizes `Train/Avg_Loss` from `Train/Loss` via `synthesize_avg_loss` when the tag is missing (old runs). Do not rename TensorBoard tags without updating Ranko chart cards.

Kotlin client: one request at a time (`Mutex` in `TrainerIpcClient`). `ignoreUnknownKeys = true`. Keep that when adding fields.

---

## 7. Ranko (`ranko/`)

Compose Multiplatform **desktop JVM only** (not Android/iOS). Kotlin 2.4, Compose 1.11.1, Material 3, Metro DI, ktoml, Coil 3.

| Path | Role |
| --- | --- |
| `ranko/desktopApp/…/main.kt` | Window; `createGraph<AppGraph>()` |
| `ranko/shared/src/commonMain/…/App.kt`, `Stage.kt` | Shell + four screens |
| `…/pages/*Screen.kt` + `*ViewModel.kt` | UI + state |
| `…/data/TrainerIpcClient.kt` | NDJSON child process |
| `…/data/TrainerRepo.kt` | Repo-root discovery |
| `…/data/ConfigModel.kt` | Sectional TOML model |
| `…/data/TomlDocumentPatcher.kt` | Comment-preserving save |
| `…/data/ConfigImporter.kt` | expect/actual load/save |
| `…/jvmMain/` | TOML IO, `getAppExecutionPath` |
| `Graphs.kt` / `Factory.kt` | Metro `AppGraph` + ViewModel factory |

Screens: `Images` | `Statistics` | `Utils` | `Dashboard` (`Stage.kt` enum).

Dataset scan in the GUI is **non-recursive**, one folder, image + same-stem `.txt`. Orphan captions **abort** the statistics scan. `ranko/tools/agent.py` mirrors this (`--allow-orphans` to inspect anyway). Trash for GUI/agent drops: `/tmp/axlranko/trash` (not the dataset’s own `trash/` used by some `tools/` scripts).

DI: Metro `@Inject` / `@SingleIn(AppScope)` / `@ContributesBinding`. ViewModels via `metroViewModel()`. New ViewModels need constructor injection and to be reachable from the graph (follow existing `*ScreenViewModel`).

Hot reload: `./gradlew :desktopApp:hotRun --auto`. Normal: `./gradlew :desktopApp:run`.

---

## 8. Dataset contract

Sidecar captions, comma-separated tags, extensions: jpg/jpeg/png/webp/bmp.

Python trainer `list_images` / Ranko / `agent.py` should stay consistent on extensions and “same stem” pairing. Ranko `parse_tags` = split `,` → trim → drop empty. Duplicates preserved in captions; stats dedupe per file.

Latent cache: `<train_data_dir>/.latents_cache/{sha1(abs_path::WxH)}.pt`. Changing bucket math invalidates keys; do not hand-edit cache files.

`tagger/` is an interactive ONNX WD-tagger (`cd tagger && python main.py`). Model files are relative (`model.onnx`, `selected_tags.csv`). `migraphx_cache/` is generated — do not treat as source.

`tools/` scripts are mostly **in-place / destructive**. Prefer `ranko/tools/agent.py --dry-run` for agent-driven edits. `ui.py` is a **deprecated** Streamlit viewer; do not extend it.

---

## 9. Recipes

### Add an IPC method

1. `handle_*` + `_HANDLERS` in `api.py`.
2. Request/response in `API.md`.
3. `TrainerIpcClient` method + kotlinx.serialization models.
4. Test in `test_api_ipc.py` (and Kotlin `DashboardIpcTest.kt` if the payload is parsed).
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

Single helper: `trainer/cleanup.py`. `clean.py` is the interactive CLI; `train_reset` is the API. Keep them identical.

---

## 10. Tests and how to run locally

| Suite | Command | Covers |
| --- | --- | --- |
| IPC | `python -m unittest test_api_ipc` | ping, dashboard empty logs, sample grouping, avg-loss |
| Control | `python -m unittest test_train_control` | runtime dir, atomic state, commands, lock, swap tensors |
| Family | `python -m unittest test_family` | catalog, spec mismatch, SD 3.5 refuse, v-pred metadata |
| Latent cache | `python trainer/test_warm_latent_cache.py` | pipelined vs serial; `--real` needs a VAE |
| Ranko | `cd ranko && ./gradlew :shared:jvmTest` | IPC models, TOML patch, catalog form, image headers |

Cwd for Python tests: **repo root**. Use conda env `axl` so `torch` / `tensorboard` import.

Do not hit a real GPU in unit tests. `test_train_control` may import `torch` for tensor device checks.

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

Python: 3.12, PyTorch `2.12.0+rocm7.2` (CUDA torch also works if you swap the wheel). Desktop: JDK 17+; Gradle wrapper provisions JDK 21.

Author reference GPU: AMD RX 9070 XT 16 GB, ROCm 7.2. Primary target is **AMD ROCm**, not NVIDIA.

---

## 13. Out of scope unless explicitly asked

- Porting Ranko off desktop JVM.
- Replacing PEFT/diffusers with kohya sd-scripts internals.
- Serving checkpoints, Civitai upload, or remote training.
- Changing default `bucket_reso_steps` away from 128.
- Reviving or expanding `ui.py`.
