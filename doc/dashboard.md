# Dashboard (Ranko)

Ranko ("AxlRanko") is the desktop GUI, built with Kotlin Multiplatform + Compose Multiplatform (JVM desktop target). It is a **controller, not a trainer**: it manages the dataset, edits the config, and drives the detached Python training process through the IPC helper (`api.py`).

The chrome is the same **Sky & Sakura** night palette as KataHana (deep purple, sakura pink, sky blue, Nunito, porcelain cards). Utils → **Appearance** can switch the backdrop (solid / glow orbs / a local image), **card blur** vs **background blur** (independent; cards/nav vs the wallpaper in the gaps), font scale (text only), icon scale (icons, padding, component size), and **thumbnail quality** (JPEG 1–100 for dataset/sample thumbs over IPC). Those prefs live in Java Preferences, not `config.toml`.

![Dashboard — live training run](screenshots/dashboard-training.png)

## Requirements

- JDK 17+ (the Gradle wrapper auto-provisions a JDK 21 toolchain through the foojay resolver if needed).
- Python with the trainer deps on `PATH` as `python3`, or set `AXL_PYTHON` to the interpreter to use (recommended when using the `axl` conda env).
- The trainer repo must be discoverable: Ranko walks up from the executable and from the working directory looking for a folder containing `api.py` (or `config.toml` next to the `trainer/` package). Running `./gradlew :desktopApp:run` from inside the repo satisfies this.

## Build and run

```bash
cd ranko
./gradlew :desktopApp:run            # standard dev run
./gradlew :desktopApp:hotRun --auto  # Compose hot reload
./gradlew :shared:jvmTest            # unit tests
./gradlew :desktopApp:packageDeb     # package an installer (also: packageDmg / packageMsi)
```

## The four tabs

The app opens with a floating, draggable navigation rail (Images / Statistics / Utils / Dashboard). It snaps to the nearest window edge, collapses to a ball after a short idle (tap to expand), and stays inside the window when dragged or when the window is resized. State is app-scoped, so switching tabs never loses your place.

### Images — dataset caption editor

![Images tab](screenshots/images-tab.png)

- Left: scrollable thumbnail list of every image in the selected dataset folder (jpg/jpeg/png/webp/bmp) — the entries of `[[environment.train_data]]`, picked by the chip row above the panels when the config lists more than one folder (`train_data_dir` alone when it lists one). `*.mask.png` sidecars are not listed. The selected image is highlighted; images with unsaved caption edits get a **red border**. A small pink corner dot marks images that already have a loss mask (`{stem}.mask.png` or an alpha channel on the training image); the dot turns red while the in-memory mask is unsaved.
- Right, top: a **Mask** toolbar (toggle paint, mask-only view, brush size, feather as % of diameter default 20%, strength default 100%, invert / fill white / fill black, clear, save mask), then the large preview. With **Mask** on, left-drag paints white (train) and right-drag paints black (ignore; Ctrl+left also erases). Strokes are sampled along the pointer path, so fast movement stays continuous. The cursor shows the brush footprint: the solid circle is the full-strength core, the dashed circle is the outer edge of the feather, and both scale with the image. **Alt+wheel** over the canvas resizes the brush (same range as the slider, wheel up larger). The brush only paints while the pointer is over the image: dragging into the letterbox or past the edges stops painting (nothing is smeared along the border), and re-entering the image starts a new segment rather than a line back to where the pointer left. The overlay dims ignored regions. If there is no sidecar, a transparent training image is previewed using its alpha as the mask. **Save mask** writes `{stem}.mask.png` (takes precedence over alpha). **Clear** deletes only the sidecar and falls back to alpha (or full-image training). Switching thumbs auto-saves a dirty mask. Caption **Save** does not write the mask.
- Right, bottom: caption/tag editor with **Reset** and **Save** — both enabled only while there are unsaved caption edits.
- The vertical divider (thumbnails ↔ preview) and horizontal divider (preview ↔ editor) are draggable.
- Saving tags writes the caption text to the image's `.txt` file. Re-entering the tab re-scans the disk without discarding in-progress caption drafts.

### Statistics — tag analysis & bulk cleanup

![Statistics tab](screenshots/statistics-tab.png)

- Left: every tag with a **frequency bar** colored by occurrence rate (blue → green → red as frequency rises), plus count and percentage. A tag that has a row in `tagger/selected_tags.csv` is shown as `english [chinese]`; search matches either side. Captions on disk stay English. A copy button after the label puts the English tag on the clipboard. Selected tags slide right.
- Right top: a staggered grid of thumbnails matching the current filter. **Click a thumbnail to jump to the Images tab with that image preselected.**
- Right bottom controls:
  - **Logic mode**: Intersection (AND) / Union (OR), plus a **Not** negation toggle.
  - **Clear / Invert selection**.
  - **Remove Selected**: strip the selected tags from the captions of all matching images.
  - **Drop Selected Samples**: with probability `r` (0.001–1.0), move each matching image + caption (+ `{stem}.mask.png` if present) to `/tmp/axlranko/trash`.
  - **Batch Add**: prepend/append a new tag to matching captions, skipping images that already contain it.
  - **Shuffle Dataset**: the `tools/suf.py` behaviour, on the folder the chip row selects. After a confirmation it renames every sample to a shuffled `0001…` sequence, moving each `{stem}.txt` and `{stem}.mask.png` with its image (a whole group is renamed through one temporary name, so a mask can never land on another sample and an already-numbered folder shuffles without collisions). Directories (`.latents_cache`, `trash`) and files outside the dataset contract keep their names; an orphan `.txt` refuses the shuffle, like the scan does. The status line under the controls reports the count, or the failure. The latent cache re-encodes once afterwards, because its key is the image path.
- Dataset scanning is strict: any orphan caption file (`.txt` with no matching image) aborts with an error card. It scans the same selected dataset folder as the Images tab (one chip row, shared selection), and the chip row stays visible while an error card is up, so a folder that no longer exists can be switched away from.

### Utils — config editor

![Utils tab](screenshots/utils-tab.png)

- A validated, structured editor for `config.toml` (repo root) — no hand-editing TOML.
- Environment section lists the **Train data directories**: one row per `[[environment.train_data]]` entry — a folder path with **Browse** and its per-epoch **Repeat** (1–512), `+ Add folder`, and a delete button per row (the last row stays). `train_data_dir` is written as the first row's path. Then **Auto-tag dataset**: a confidence slider / threshold (default `0.35`) and a **Tag dataset** button that tags the selected folder (the chip row above it when there is more than one). That calls IPC `dataset_tag`, which runs `tagger/main.py` on GPU (MIGraphX) against that folder and overwrites sidecar `.txt` captions. When it finishes, Images and Statistics reload from disk.
- Training section ends with **Resume from LoRA checkpoint**: a path field with **Browse** (OS file dialog filtered to `.safetensors` — type or paste a directory holding a single checkpoint to use that form), **Pick from run checkpoints** (a dialog listing `list_checkpoints` results for the current output name: run id, step, `r/α`, size, newest first), and **Clear**. Saving writes `[training].resume_lora_path`; the summary line then shows `· resume`. Selecting from the dialog only fills the field — save to apply.
- Left: the config sections (Environment, ROCm, Model Spec, Training, Network, Bucketing, Optimization, UNet Optimizer, Text Encoder, Infrastructure, Validation, Appearance), with a warning badge on sections containing invalid fields. **ROCm** picks `[environment].amdfq` (`none` / `tail` / `vmm`) for the next Train start. Appearance is UI-only (not written to `config.toml`): Solid / Glow / Image backdrop, independent card/background blur, font scale, icon scale, thumbnail JPEG quality.
- Right: fields per section — path fields with a **Browse** button (OS file dialog), switches for booleans, segmented buttons for `mixed_precision`, chips for `lr_scheduler`, and numeric fields with inline validation and helper hints (effective batch size, LoRA scale α/dim, bucket-step divisibility, sample aspect ratio).
- **Validation** is a tabbed editor over `[[validation.samples]]`: a horizontal strip of set chips (label, warning icon while the set has an invalid field), a `+` that clones the open set, and a small `×` that deletes a set after a confirmation dialog (never the last one). The open tab shows Label, Positive/Negative prompt, Width/Height/Steps, Guidance scale/Seed/Repeat. Saving writes the `[validation]` scalars from the first tab plus one explicit block per tab.
- Header shows the config path, a summary line (`name · resolution · epochs · batch`), and an **Unsaved** indicator. **Save** validates the whole form (auto-jumping to the first invalid section), then patches the TOML in place, preserving comments and formatting. **Reload** is blocked while the form is dirty.
- **Profiles** saves and applies named `config.toml` presets. A name in the **Profile name** field writes the editor's values to `configs/<name>.toml` next to `config.toml` — the folder is tracked by git, the presets in it are ignored — and a name already taken asks before it overwrites. Clicking a saved profile applies it: `config.toml` is patched in place (comments, blank lines, unknown tables and every key the profile does not carry stay as they are) and the editor reloads from the result, with a confirmation first when the editor has unsaved changes. A profile written here holds the whole config, so applying it replaces the environment paths too; a hand-trimmed file changes only the keys it lists, and a section `config.toml` has no table for is named in the status line instead of being dropped silently. **Delete** removes the preset file and leaves `config.toml` alone. A run already in flight keeps the settings it started with.

See [Configuration](configuration.md) for the meaning of every field.

### Dashboard — training monitor & control

The heart of the app. It spawns `api.py` on first use and polls it (every 1 s while a run is live, otherwise 3 s). The hardware panel polls `hardware_status` on its own 1 s cadence while the tab is visible.

- **Header**: connected/disconnected indicator, auto-refresh switch, Refresh button, dataset / target / base-model compact metrics, and sliders for **Curve Smoothing** (EMA 0–0.99), **Chart Line** (stroke 1–8), and **Sample Size** (80–360 px thumbnails).
- **Training control card**:
  - Status chip (idle / starting / encoding / training / sampling / pausing / paused / resuming / stopping / finished / error), plus transient **gpu-out** / **gpu-in** chips with swap progress while offloading/loading.
  - Run info: output name, run id, PID, elapsed time, alive flag, and the run's `detail` / `error` lines. When `[training].resume_lora_path` is set, the card also shows what the next run will resume from, or — while running — the checkpoint this run was seeded from (`Resumed from … · checkpoint step N · M tensors`).
  - Three phase progress bars: **latent encode** (`encoding.current/total`), **training** (step/total, epoch, loss, avg-loss), **sampling** (image r/R, denoise step d/D; with several `[[validation.samples]]` sets it also shows `set s/S` and `r` counts the images of the whole pass).
  - Buttons: **Start** (enabled only when terminal: idle/finished/error), **Pause** / **Resume** (with in-flight spinner states), **Early Stop** (phase-aware confirmation dialog — warns whether a checkpoint will be saved), and **Reset** (clears Finished/Error state; deletes the resolved run's samples + TensorBoard logs with exact paths shown; optional checkbox to also delete that run's LoRA checkpoints).
- **Hardware**: live GPU / CPU panel under Training Control. GPU numbers come from `nvtop -s` (JSON snapshot); AMD edge/junction temps from DRM hwmon; CPU util/temp from `/proc` and thermal zones; RAM from `/proc/meminfo`. When `[environment].amdfq` is `vmm`, a separate VA bar shows used / total GPU virtual address space: **GPU VA (live)** while the hook gives a freed range's address back (the default, and what the 2026-09 kernel allows), or **GPU VA (not returned)** while `amdfq_va_never_reuse` keeps it for the process lifetime (the pre-fix workaround). Info line + current-value cards + four charts in two rows: **GPU** (util / VRAM), **Temp** (edge / junction / CPU), **Power** (GPU watts), **CPU** (util / RAM). Ranko keeps a ~6 minute ring buffer. Missing nvtop shows an error on this section only — training controls and TensorBoard charts keep working.
- **Path chips**: current run id, `{logging_dir}/{run_id}`, `{output_dir}/{run_id}` — `—` when no run directory resolves yet.
- **Metric cards**: Current Step, Latest Loss, UNet LR, TE Effective LR.
- **Training charts**: Train/Avg_Loss, Train/Loss, UNet/LR/Effective_Actual_LR, TE/LR/Base_Scheduled, TE/LR/Effective_Actual_LR — interactive line charts with an always-on hover readout (see interactions below). On **Train / Avg Loss**, `Ctrl` + left click or a left double click opens the checkpoint panel described below.
- **Generated samples**: thumbnails grouped by step (newest first). Click to open a fullscreen dark preview with prev/next, keyboard (Esc closes, ←/→ navigate), and drag/swipe paging. When a run used several prompt sets, every thumbnail carries a small `P1`/`P2` badge saying which `[[validation.samples]]` entry rendered it (a single-set run and a run from before this feature show no badge).

If the helper process can't be reached (and no data has loaded), a full-screen error card with **Retry** (restarts `api.py`) is shown. The error message suggests setting `AXL_PYTHON` if the interpreter wasn't found.

## How it talks to the trainer

1. **Discovery** — `TrainerRepo.findRoot()` walks up from the app's executable and `user.dir` looking for `api.py` or a `config.toml` that sits next to the `trainer/` package.
2. **Spawn** — Ranko connects to `ws://127.0.0.1:18765`. If nothing is listening it runs `$AXL_PYTHON` (if set) or `python3 -u api.py --websocket` with the working directory at the repo root, stderr inherited, `PYTHONUNBUFFERED=1`. A JVM shutdown hook kills the helper **this process spawned**.
3. **Protocol** — JSON-RPC on that WebSocket: requests are `{"id": n, "method": "...", "params": {...}}`, responses are `{"id": n, "ok": true, "result": {...}}` or `{"id": n, "ok": false, "error": "..."}`. Replies match on `id`. Dataset images travel as resized JPEG blobs (`blob_batch`), not `java.io.File`.
4. **Methods** — train control plus `config_*`, `dataset_*`, `blob_*`, `mask_*`, `profile_*`. Full reference: [API.md](../API.md).

**Important**: `train_start` spawns the trainer **detached** (`setsid`). Closing Ranko does not stop training; use Pause/Early Stop (or the runtime `command.json`) to control it.

## Chart interactions

- **Hover**: a vertical cursor follows the pointer inside the plot and a small label shows the exact step under it (snapped to a logged step). The readout is always on for the five training charts, not the hardware ones.
- **Pan**: drag horizontally/vertically.
- **Zoom X**: `Ctrl` + mouse wheel (anchored at the cursor).
- **Zoom Y**: `Shift` + mouse wheel.
- **Checkpoint pick**: `Ctrl` + left click on **Train / Avg Loss** (a click, not a drag) resolves the checkpoint nearest to the clicked step and opens a floating panel (see below). A left **double click** does the same and is the trigger without a keyboard: two clicks within 400 ms, *wherever* they land — only the interval counts, not the distance between them — and the panel opens at the **second** click. A `Ctrl`+click picks immediately and never pairs with a following plain click; a third quick click starts a new pair rather than picking again; the click that fires must be inside the plot (the first one may be anywhere on the chart). The chart then marks the clicked step with a dashed line and the step the pick actually matched with a bold accent line, a dot, an axis flag and a `ckpt <step>` label, so the snapping is visible. Both marks disappear when the panel is closed.
- Series are EMA-smoothed (slider), downsampled with LTTB to ≤500 points, and the initial viewport clips outlier percentiles.

### Checkpoint panel

`Ctrl` + left click, or a left double click, on the Avg Loss chart opens a panel anchored next to the cursor (a double click positions it at the second click). It leads with what the click actually matched:

- Header: the clicked step, a scan spinner and a close button.
- **Matched checkpoint** (highlighted block, `MATCHED CHECKPOINT`): directory name, a `step N` badge with the checkpoint's own step, how far it is from the click (`24 steps before the click point`), `run · step · rank/α · size · final`, and the full path.
- **Training at step N**: `Avg Loss`, `Loss`, `UNet LR` and `TE LR` at the clicked step (`Train/Avg_Loss`, `Train/Loss`, `UNet/LR/Effective_Actual_LR`, `TE/LR/Base_Scheduled`). A series the run never logged shows `—`.
- **Save As**: opens the OS save dialog (the KDE/GNOME picker on Linux; suggested name `{checkpoint_dir}.safetensors`, e.g. `lllj_s003050.safetensors`, starting in the run's directory) and copies the LoRA there with a progress bar and a `Saved → …` / failure line. A name typed without an extension gets `.safetensors` appended; copying a checkpoint onto itself is refused.
- **Generate sample**: expands a form (prompt, negative prompt, CFG, steps, seed — prefilled from the first `[[validation.samples]]` set, or from `sample_prompts`, `sample_negative`, `guidance_scale`, `sample_steps` when the file has no sets) that renders one extra image from the matched checkpoint, see below.
- **Samples at step N**: the step's own sample images followed by any generated ones, three per row; on a multi-set run each thumbnail is labelled with its `Pn` set. Clicking one opens the usual fullscreen preview (which then cycles through the generated images too, in step order). If the checkpoint's step has no samples, the nearest sampled step is shown and labelled as such.
- **Size**: the panel opens at a content-derived size (three 400 dp-wide sample slots, ≈1268 dp wide for a three-image run) and can be dragged by the grip in its bottom-right corner. The slots — and the text — follow the panel width, so dragging it bigger shows bigger samples; extra generated images wrap onto the next row instead of scrolling. A drag never moves the panel: placement is decided once from the click and the default width, and a panel that outgrows the window is slid back inside rather than flipped to the other side of the cursor. The dragged size is clamped to the window and kept for the rest of the session.

Checkpoints are rescanned on every click (`list_checkpoints`, ~2 s for 60+ checkpoints because every safetensors header is read), so the panel shows the previous scan immediately and refreshes in place. Dismiss with a click outside, the close button, or `Esc`. The panel is read-only for the run itself — resume selection stays in the Utils tab; **Save As** only copies the file out.

#### Generating a sample from the matched checkpoint

`Generate sample` → fill in the form → `Generate` runs one image (the run's `sample_width` × `sample_height`) through the checkpoint's own LoRA and appends it to that step's row, highlighted with an accent border, a `GENERATED` (or `NEW`, for this session) badge and a `CFG 5 · 20 steps · seed 12345` caption. Progress (`denoising 12/20`) comes from the job file, which the generator rewrites on every denoise step.

- Prompt handling, CFG, step count, seed (`0` = random, the seed actually used is written back), scheduler and CLIP settings mirror the run's own sampling (the first prompt set), so a generated image is comparable with the training samples. `clip_skip`, `max_token_length`, `network_dim`/`network_alpha` and the base model come from the checkpoint's kohya metadata, not from today's `config.toml` — sampling an old checkpoint uses the settings it was trained with. If the checkpoint was trained on a different `base_model_version` than `[model_spec]` says, the job fails with that message instead of producing a mismatched image.
- Files land in `{output_dir}/{run_id}/{output_name}_samples/generated/` — one PNG plus one JSON job record (prompt, CFG, steps, seed, checkpoint, state, error) per generation. The directory is *inside* the sample dir, so the training sample strip and Images tab ignore it, and resetting the run deletes it along with the samples. A generated image belongs to no prompt set, so it never carries a `Pn` badge.
- Generation needs the GPU to itself: while the trainer process is alive (running or paused) the button is disabled and `generate_sample` refuses with `the GPU is in use`. Only one generation runs at a time. The generator is a detached process (`trainer/generate_sample.py`), so closing Ranko does not kill it; it never touches `state.json`, the lock or the training loop.
- Failures (a rank mismatch, a dead base model, an OOM) are reported in the panel and in the job's `.log` next to the PNG.

## Keyboard shortcuts

| Where | Key | Action |
| --- | --- | --- |
| Sample preview overlay | `Esc` | Close preview |
| Sample preview overlay | `←` / `→` | Previous / next sample |
| Checkpoint panel | `Esc` | Close the panel |
| Checkpoint panel | Drag the bottom-right grip | Resize (kept for the session) |
| Checkpoint panel | `Save As` | Copy the checkpoint to a chosen file |
| Checkpoint panel | `Generate sample` → `Generate` | Render one extra sample from the matched checkpoint |
| Charts | Hover | Step readout under the cursor |
| Charts | `Ctrl` + click | Pick the nearest checkpoint (Avg Loss only) |
| Charts | Double click | Same pick without a keyboard; the second click sets the position (Avg Loss only) |
| Charts | `Ctrl` + wheel | Zoom X axis |
| Charts | `Shift` + wheel | Zoom Y axis |

There are no global app-level shortcuts.

## Environment variables

| Variable | Effect |
| --- | --- |
| `AXL_PYTHON` | Interpreter used to run `api.py` (and thus the trainer). Defaults to `python3` on `PATH`. Set this when the trainer deps live in a conda/venv, e.g. `AXL_PYTHON=$CONDA_PREFIX/bin/python`. |

## Tech stack

Kotlin Multiplatform / Compose Multiplatform (Desktop JVM) · Material 3 · Metro + metrox-viewmodel (DI) · ktoml + kotlinx.serialization (TOML/config) · Coil 3 (images) · FileKit (OS file dialogs) · okio / kotlinx.coroutines. Charts are hand-drawn on `Canvas` (LTTB downsampling, EMA smoothing, percentile outlier clipping, cursor-anchored zoom, hover step readout, checkpoint pick markers).

## Web target

`:webApp` is a Compose Multiplatform `wasmJs` companion. It does not spawn `api.py`. Set host/port in Utils → Helper (defaults `127.0.0.1:18765`; `?host=` / `?port=` override). For another machine, bind `python -u api.py --host 0.0.0.0 --allow-ip <client-ip-or-cidr>`. Path pickers are an in-app dialog over `fs_listdir`. Mask paint uses DOM pointer events; Save mask is the only sidecar write. Image wallpaper is desktop-only. Gradle notes: [ranko-web-target.md](ranko-web-target.md). The Streamlit `ui.py` viewer is separate and deprecated.
