# AGENT.md — AXLTrainer

Working notes for coding agents. Human-facing docs live under `doc/` and `README.md`.

**This file is the entry point.** It keeps the working agreement, the rules a change has to respect, and the macro architecture. The detail behind each section — rationale, measurements, module-by-module text — lives in `doc/agent/`, one file per topic (§1 lists them).

**What this is:** a local-first **LoRA** training stack (SDXL implemented; SD 3.5 catalogued but not trainable): Python engine (`trainer/`) + JSON-RPC helper (`api.py`) + Kotlin/Compose desktop dashboard (`ranko/`, product name **AxlRanko**) + dataset scripts (`tools/`, `tagger2/`, `tagger/`, `ranko/tools/agent.py`).

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

Topic detail, one file per AGENT.md section:

| Topic | Detail file |
| --- | --- |
| §2 Process model — run ids, detached training, runtime dirs, orphan reaping | `doc/agent/process-model.md` |
| §3 Status machine — commands, live settings, pause/offload, early stop | `doc/agent/status-machine.md` |
| §4 Configuration contract — flattening, run snapshot, prompt sets, train-data entries, bucketing | `doc/agent/config-contract.md` |
| §5 Python trainer — module map, resume mechanics | `doc/agent/trainer.md` |
| §6 IPC — handler table, job records, client lanes and resource locks | `doc/agent/ipc.md` |
| §7 Ranko — screens, Dashboard/Checkpoints rules, masking, prompt port | `doc/agent/ranko.md` |
| §8 Dataset contract — captions, masks, latent cache, taggers | `doc/agent/dataset.md` |
| §9 Recipes — add an IPC method / a base family, change training math, pause, cleanup, resume | `doc/agent/recipes.md` |
| §10 Tests — what each suite covers | `doc/agent/tests.md` |

Human-facing docs:

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
                                      +--> logging_dir/{run_id}/          TensorBoard + the run's config.toml
                                      +--> output_dir/{run_id}/{name}_*   checkpoints + samples
```

`run_id` = `{output_name}_{YYYYMMDD_HHMMSS}`, created by `trainer/main.py` through `trainer/runs.py` (`create_run_dirs`); every run gets its own pair of directories, so a later run never overwrites an earlier one. `output_name` must be filename-safe (letters, digits, `-`, `_`, `.`): `validate_output_name` (`trainer/runs.py`) enforces it from `TrainConfig.__post_init__`, from `fsrpc.config_save` and in the Utils form (`OUTPUT_NAME_HINT`). `api.py` resolves the run for `dashboard` / `list_samples` / `train_reset` / `list_runs` as: explicit `run_id` param → `state.json`'s `run_id` → the newest run directory of an explicitly named `output_name`; a request that names nothing means *the run the trainer is on* and stops at `state.json`. Passing `run_id` alone is enough (`runs.run_output_name` recovers the name the id was built from).

Hard rules:

- Training is **detached** (`train_start` uses `start_new_session=True`). Closing Ranko must not kill the run.
- Ranko **never** talks to the GPU. After connect, `commonMain` only speaks JSON-RPC; desktop `jvmMain` may spawn `api.py --websocket` and pick paths with FileKit.
- `start_train.sh` `exec`s the trainer, so that shell's PID and session become the trainer's; a HIP abort leaves forked DataLoader workers behind. Keep `trainer/orphans.py` started first and detached, and keep it unable to touch a session that is not the trainer's.
- Working directory for `api.py` and `start_train.sh` is the **repo root** (the directory holding `api.py` and `trainer/`).
- WebSocket is the only control channel (default `127.0.0.1:18765`; LAN bind is `--host 0.0.0.0` plus `--allow-ip` / `AXL_WS_ALLOW`, loopback always admitted; when neither is set the allowlist is `192.168.0.0/16`). Logs and tracebacks go to stderr. The helper serves **one client session**: the first instance to `hello` owns it, its own later connections join it, and any other client is refused with `CLIENT_BUSY` and closed.
- Ranko finds the repo root by walking up for `api.py`, or `config.toml` next to the `trainer/` package (`TrainerRepo.looksLikeRepoRoot`); a lone `config.toml` does not qualify.

Runtime dir resolution (same in `trainer/control.py` and `api.py`):

1. `$AXL_RUNTIME_DIR`
2. `$XDG_RUNTIME_DIR/axltrainer`
3. `/tmp/axltrainer-$UID`

Files: `state.json`, `command.json`, `settings.json`, `train.lock`, `train.log`. Tests **must** set `AXL_RUNTIME_DIR` to a temp dir (see `test_train_control.py`).

Interpreter override: Ranko uses `$AXL_PYTHON` if set, else `python3`. Training deps live in the conda env `environment.yml` names — currently `axl` (torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`). There is **no** `requirements.txt`.

Detail (rationale and the measured numbers): `doc/agent/process-model.md`.

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

`LIVE_STATUSES`: starting, encoding, training, sampling, pausing, paused, resuming, stopping. `PHASE_STATUSES` (pause/resume attach here): encoding, training, sampling.

- `is_pid_alive` (`control.py`) reads the process state through `orphans.is_running`, so a **zombie** counts as gone; `reconcile`, `train_start`, `_require_alive`, `_gpu_busy`, `_running_generation` and `_reconcile_generated` all use it.
- Commands (`command.json`: `pause` | `resume` | `stop`) are one-shot and consumed **only** at `device_swap.at_safe_point(phase, swap_ctx)`.
- Pause **must** offload the denoise network, all text encoders, both optimizers (Schedule-Free state included) and the VAE to CPU, then `empty_cache`; resume reloads what the current phase needs. `SwapContext` carries `denoise` + `text_encoders: list`.
- Live settings (`settings.json`: `save_every_n_steps`, `sampling_enabled`) are written by `api.train_settings` and adopted by the trainer on the next optimizer step (`control.read_settings` → `LiveSettings.adopt`, after `at_safe_point`). `state.json` carries the **effective** values plus `next_save_step`, and `train_status` adds a `requested` block until the trainer adopts a change. `reset_to_idle` deletes the file. `0` means "write no checkpoints"; `next_save_step` replaces the modulo rule (first save at step N, then `step + N`, and a cadence change at step S sets `next = S + N`).
- `train_start` fails if a live PID exists; `train_reset` fails while status is in `_RESET_BLOCKED` **and** the PID is alive.

Early-stop semantics (keep these):

| Phase | Stop behavior |
| --- | --- |
| encoding | no LoRA written |
| training | save `{output_name}.safetensors` only if this step has no checkpoint yet |
| sampling | skip leftover repeats (step checkpoint already exists) |

`state.json` also carries `run_id` and `resume` (`null`, or `{path, filename, step, epoch, loaded, skipped}`).

Detail: `doc/agent/status-machine.md`.

---

## 4. Configuration contract

**Single source of truth:** `config.toml` at the **repo root**. `trainer/main.py` takes **no CLI args**, and every entry point runs with cwd = repo root — `_load_toml_config()` resolves that path relative to the working directory, which is what lets `verify_mask_pipeline.py` run an unmodified `trainer/main.py` against a throwaway mirror of the repo.

Load path:

- Python: `trainer/config.py` flattens **all TOML tables into one dict**; section names do not exist at runtime on the Python side, only keys. `TrainConfig` fields default via `get_val(key, hardcoded)`, and **TOML wins** over Python defaults.
- Kotlin: `AxlTrainerConfig` is **sectional** (`environment`, `model_spec`, `training`, …). Utils saves through `TomlDocumentPatcher` (in-place replace of uncommented `key = value` inside named tables — comments, blank lines and unknown tables stay intact; do not rewrite the whole file). `TomlIntegerLiterals` rewrites `key = 0` to `0.0` for the keys the model declares `Double`, so a hand-edited `amdfq_vram_reserve_gib = 0` cannot cost Ranko its startup.

Adding a hyperparameter (all four, or the GUI will drift):

1. `config.toml` — pick an existing table or add one.
2. `TrainConfig` in `trainer/config.py` — same **flat** key name.
3. Kotlin `AxlTrainerConfig` + nested data class (`ConfigModel.kt`), `TrainingConfigForm`, Utils UI bind/save map (section name → key → encoded value).
4. `doc/configuration.md`.

If only the trainer needs it, you can skip (3) but document that the Utils editor will not see it until the Kotlin model is updated. ktoml parse of the full file will fail if a **required** Kotlin field is missing — new optional keys are safer as Kotlin defaults.

`run_dir` is the one field that is **not** a user-facing key: `main.py` writes the run directory it created into `cfg.run_dir` at startup, and `artifact_root(cfg)` roots every artifact path at it. Leave it empty in `config.toml`.

Two list-shaped configs lean on the flattening rule, each staying under its parent table and read on the Python side as a flat key: `[[validation.samples]]` → `samples` (prompt sets; `resolve_sample_sets` falls back per key to the flat `sample_*` scalars, and **no entries yield one set built from those scalars**), and `[[environment.train_data]]` → `train_data` (one folder per block with a per-epoch `repeat`; `resolve_train_data_entries` falls back to a single entry from `train_data_dir` with repeat 1, and the `train_data_dir` scalar stays the mirror of the first entry).

Bucketing + fit geometry: `pick_bucket_size` is an **area budget** rule (aim at `train_resolution²`, take the aspect from the image, clamp per axis with `min/max_bucket_reso`, honour `no_upscale`); `fit_geometry` contains the image centred in the bucket and `fit_to_bucket` pads it with `FIT_PAD_VALUE` = 127. There is **no crop variant in the training path**. Keep `min ≤ train_resolution ≤ max` (shipped: 384 / 2688). Any change to the rule, the clamps or the interpolators changes every sample's pixels — the latent cache key carries the fit geometry for exactly that reason.

Shipped `config.toml` and `TrainConfig` fallbacks contain **author machine paths**. Never “fix” them to placeholders as part of an unrelated PR unless asked; consumers already know they must edit `[environment]`.

Detail (run snapshot, prompt sets, train-data entries, the geometry contract): `doc/agent/config-contract.md`.

---

## 5. Python trainer map

Entry: `bash start_train.sh` → `python -u trainer/main.py` with ROCm log filters and MIOpen cache pins. The module-by-module map is in `doc/agent/trainer.md`.

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

`[training].resume_lora_path` (file, or a directory holding exactly one `.safetensors`) is loaded in `build_train_objects` after `family.apply_lora` and before `accelerator.prepare`, through `ModelFamily.load_lora`. Rank/alpha must match `network_dim`/`network_alpha`; unmatched tensors are counted and logged, and zero matches raise. Step/epoch counters restart at 0 — there is no optimizer/scheduler state. `[train_start]` validates the path up front; `main.py` publishes `artifacts.resume` into `state.json` via `control.set_resume`.

### ROCm (non-negotiable)

`bucket_reso_steps` must keep VAE latents (spatial / 8) **divisible by 16**. Default **128**. `64` causes random GPU page faults on AMD (the field report is sealed: 涉及负责任披露流程，暂不公开). Do not “optimize” this down. `start_train.sh` also sets `PYTORCH_CUDA_ALLOC_CONF` and MIOpen log/cache env; keep those if you touch the launcher.

Detail (module map, load_lora key-map mechanics): `doc/agent/trainer.md`.

---

## 6. IPC (`api.py`)

Framing: one JSON object per WebSocket text frame. `{id, method, params}` → `{id, ok: true, result}` or `{id, ok: false, error}` (plus `code`/`holder` on a refusal).

- **One client at a time.** A connection is admitted by `hello {client, instance, lane}`: the first instance owns the helper, the same instance may open as many connections as it likes (the desktop's four lanes), and another instance is answered `CLIENT_BUSY` and closed. There is no takeover request — the helper is released five seconds after the owner's last connection is gone. A first request without `hello` claims a free helper; a connection that never sends a request claims nothing.
- **No server-side locking.** `dispatch` runs the handler directly, and nothing on the server orders two calls. The app's locks live in the client (`data/IpcResources.kt`): a per-resource read/write table plus a per-method policy row saying what a call claims and how long it may wait, with anything a long job can hold (`gpu`, `dataset:<dir>`, `path:<dest>`, `automation:job:<id>`) refused at once, naming the holder.
- The client keeps **four connections** (`control` / `poll` / `blob` / `long`), each admitted with `hello` and each call with its own timeout; match replies on `id`, keep `ignoreUnknownKeys = true`, and drop a reply that lands after its timeout.
- A handler is added in `_HANDLERS` **and** in `API.md` **and** `TrainerIpcClient.kt` **and** as a policy row in `IpcResources.kt` (recipe: `doc/agent/recipes.md`). Never print to stdout from a handler, and do not add a generic `read_file` / `write_file`.
- The generation entries (`generate_sample`, `generate_checkpoint_samples`, `evaluate_checkpoint`) share one GPU gate (`_claim_generation`, which allows a **paused** run and refuses while another job of any run is running); `dataset_tag` uses the same `_gpu_busy`. Every job record carries integer counters and the `step` of the checkpoint it renders.

The handler table, the job-record shapes (`single` / `sets` / `batch` / `evaluate`), the evaluation plan and the TensorBoard reader cache are in `doc/agent/ipc.md`.

---

## 7. Ranko (`ranko/`)

Compose Multiplatform **desktop JVM** plus a **wasmJs** local/LAN companion (`:webApp`). Kotlin 2.4.10, Compose 1.12.0, Material 3, Metro DI, ktoml, Coil 3, haze 2.0. Visual style follows KataHana's porcelain cards and Nunito, on an amber night palette (`RankoPalette.Amber` in `ui/theme/Color.kt`; the brand accent is the logo amber `#F8A818`). Screens read `rankoColors` / `PorcelainCard` / `CapsuleButton`.

Screens: `Home` | `Images` | `Statistics` | `Utils` | `Dashboard` | `Automation` (`Stage.kt` enum). The app opens on Home. Look-and-feel (background, blur, font/icon scale) lives in Utils → **Appearance** and is persisted by `AppearanceRepository`; `App.kt` feeds it to `LocalDensity`, so font scale affects sp only and icon scale dp only.

Conventions that bite:

- Path pickers go through `PathPicker` (FileKit on desktop, an in-app dialog over `fs_listdir` / `fs_roots` on web). Do not reintroduce `JFileChooser`.
- Every page that changes dataset files on disk must call `DatasetRefreshHub.notifyDatasetChanged()`.
- Mask edits stay in memory until **Save mask**; navigation never writes a mask.
- The Dashboard's pollers start on entry and stop on leave (`onEnter` / `onLeave`).
- DI: Metro `@Inject` / `@SingleIn(AppScope)` / `@ContributesBinding`; ViewModels via `metroViewModel()` with constructor injection, reachable from the graph.
- Run: `./gradlew :desktopApp:run` (hot reload `./gradlew :desktopApp:hotRun --auto`; web `./gradlew :webApp:wasmJsBrowserDevelopmentRun` with the helper already listening).

Detail (file map, run history and control rules, Checkpoints / Sampling Prompts / Evaluate, chart rules, masking, the prompt port): `doc/agent/ranko.md`.

---

## 8. Dataset contract

Sidecar captions are comma-separated tags; extensions jpg/jpeg/png/webp/bmp. The dataset is the list of `[[environment.train_data]]` folders (`train_data_dir` alone when there is no list). The optional loss mask `{stem}.mask.png` (always PNG) weights the MSE (white=train, black=ignore); a missing sidecar falls back to the image's alpha, and the letterbox pad is weight 0 either way, so `load_loss_mask` returns a full-bucket mask for every sample. **Exclude** `*.mask.png` from every image listing (`list_images`, Ranko Images/Statistics, `agent.py`, `tagger/`), and move it with the pair on drop/trash.

- Latent cache: one per dataset folder, `<folder>/.latents_cache/{sha1(abs_path::bucket_w x bucket_h::left,top,fit_w x fit_h)}.pt`. Only a file holding the keyed bucket's latent counts as a hit (anything else is a miss, warned about and re-encoded over). The posterior sample is drawn from a generator seeded by `sha1(f"{cfg.seed}:{key}")`, so the global generator is untouched. Changing the scheme changes the values it writes without changing any key — bump the key if a cache has to be rebuilt. Do not hand-edit cache files.
- `tagger2/` is the captioner Ranko runs (Pixai tagger v1, resolved cache-only; `--categories` defaults to `general`, `--only-tags` is the partial add-only pass). `tagger/` is the legacy ONNX tagger, reachable from the CLI only, and its `selected_tags.csv` is what `tag_lexicon` and the Statistics tag card read.
- `tools/` scripts are mostly **in-place / destructive** (`tools/vpred_reference.py` is the read-only exception); prefer `ranko/tools/agent.py --dry-run` for agent-driven edits. `ui.py` is a deprecated Streamlit viewer — do not extend it.

Detail (mask blur, the cache verdict, tagger flags, `tag_directory` / `tag_paths`): `doc/agent/dataset.md`.

---

## 9. Recipes

Step-by-step recipes live in `doc/agent/recipes.md`:

| Task | Recipe |
| --- | --- |
| Add an IPC method | *Add an IPC method* |
| Add a base family | *Add a base family* |
| Change training math / sampling | *Change training math / sampling* |
| Change pause/offload | *Change pause/offload* |
| Change cleanup targets | *Change cleanup targets* |
| Change resume / checkpoint loading | *Change resume / checkpoint loading* |

---

## 10. Tests and how to run locally

Run the suite for the layer you touched (commands in §1), from the repo root, in the env `environment.yml` names (`axl`) so `torch` / `tensorboard` import.

- `test/` is a plain namespace directory deliberately **without** `__init__.py`, so `import test` still resolves to the standard library package.
- Do not hit a real GPU in unit tests except `test_vram_gpu`, which is skipped when `torch.cuda.is_available()` is false; `test_train_control` may import `torch` for tensor device checks.

What each suite covers — Python and Kotlin, one row per suite — is in `doc/agent/tests.md`.

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
| `AXL_WS_ALLOW` | api.py | Comma-separated client IPs/CIDRs; loopback always allowed. Unset, and no `--allow-ip`, defaults to `192.168.0.0/16` |
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
