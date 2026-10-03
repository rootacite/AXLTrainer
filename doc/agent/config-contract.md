# Configuration contract — AXLTrainer

> Detail behind `AGENT.md` §4. `AGENT.md` keeps the condensed rules; this file carries the full text.


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

### The run's own config snapshot

`trainer/main.py` copies the repo's `config.toml` verbatim into `{logging_dir}/{run_id}/config.toml` (`config.save_run_config`, right after `create_run_dirs`), because a run's samples are its config's and the next run is free to edit the repo file. It is the one place a *past* run's prompts and sampling values survive, and it is what the Checkpoints section's **Evaluate** action renders and scores against:

- `config.run_config_mapping(log_dir)` answers `(flattened mapping, the file it came from)`: the sets saved for that run (`sample_sets.json`, see below) when there are any, else the run's own snapshot, else the hparams it recorded at startup, else the repo's `config.toml` (a run from before snapshots — a real possibility, so the caller records which file it used and says so in the UI).
- `TrainConfig.from_mapping(mapping)` builds a config from such a mapping instead of the module-level `_CONFIG`: only fields the dataclass declares are taken, missing ones keep their defaults — **except the two list-shaped keys** (`samples`, `train_data`), whose default is whatever `config.toml` the process sits beside: a mapping that omits them reads as empty, so a run trained with only the flat `sample_*` scalars (or a single `train_data_dir`) cannot pick up today's `[[validation.samples]]` blocks or dataset folders — and `__post_init__` still derives `prediction_type` / `zero_terminal_snr` / `min_snr_gamma` from the base, so a snapshot cannot disagree with its own checkpoint.
- The file is a copy, not a rewrite: `cleanup.py` removes it with the rest of the log directory, and nothing reads it except the evaluation path (the training path keeps using the repo file, cwd = repo root).
- **One layer sits on top of it**: `{logging_dir}/{run_id}/sample_sets.json`, the prompt sets a run was edited to sample with, written only by `sample_prompts_set` (the Dashboard's Sampling Prompts section). `run_config_mapping` replaces the mapping's `samples` with it and reports it as `source`, so every reader of a run's prompts — `evaluate_checkpoint`, `evaluation_prompts`, `TrainConfig.from_mapping`, a manual sample pass — says the same thing. `read_sample_override` is lenient (no file / bad JSON / an entry that does not resolve → warn + "no override"), and the entries are stored complete so a reader never falls back to the repo's scalars for a missing key. The trainer reads it through `active_sample_sets(cfg)` (its own run's directory) before every sample point, which is how an edit made while the run is live lands on the next checkpoint — no runtime file and no run-id guard. Absent, every path behaves exactly as it did before the file existed; the snapshot and the hparams record are never rewritten (they stay the record of what training used).

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
  `len(dataloader)` (hence `steps_per_epoch`, the total step count and the progress bars) and
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
cache key carries the fit geometry for exactly that reason (see `dataset.md`).

Shipped `config.toml` and `TrainConfig` fallbacks contain **author machine paths**. Never “fix” them to placeholders as part of an unrelated PR unless asked; consumers already know they must edit `[environment]`.

