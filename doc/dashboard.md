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

## The five tabs

The app opens with a floating, draggable navigation rail (Images / Statistics / Utils / Dashboard / Automation). It snaps to the nearest window edge, collapses to a ball after a short idle (tap to expand), and stays inside the window when dragged or when the window is resized. State is app-scoped, so switching tabs never loses your place.

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
- Training section holds **Save every N steps** and the **Sampling** switch (`[training].sampling_enabled`) — what the next run starts with; a run in progress is retuned from the Dashboard — and ends with **Resume from LoRA checkpoint**: a path field with **Browse** (OS file dialog filtered to `.safetensors` — type or paste a directory holding a single checkpoint to use that form), **Pick from run checkpoints** (a dialog listing `list_checkpoints` results for the current output name: run id, step, `r/α`, size, newest first), and **Clear**. Saving writes `[training].resume_lora_path`; the summary line then shows `· resume`. Selecting from the dialog only fills the field — save to apply.
- Left: the config sections (Environment, ROCm, Model Spec, Training, Network, Bucketing, Optimization, UNet Optimizer, Text Encoder, Infrastructure, Validation, Appearance, WM), with a warning badge on sections containing invalid fields. **ROCm** picks `[environment].amdfq` (`none` / `tail` / `vmm`) for the next Train start. Appearance is UI-only (not written to `config.toml`): Solid / Glow / Image backdrop, independent card/background blur, font scale, icon scale, thumbnail JPEG quality. **WM** is offered by the desktop build only and is UI-only as well: **Maximize** sizes the window to its screen — the size is applied, not requested, because a compositor that hosts no window manager has nothing that acts on a maximize request — and reads **Restore** once it has, which puts the size the window had before back. **Exit** quits through the same path as closing the window. Both exist for a cage or other single-window session, whose compositor draws no title bar to maximize or close from.
- Right: fields per section — path fields with a **Browse** button (OS file dialog), switches for booleans, segmented buttons for `mixed_precision`, chips for `lr_scheduler`, and numeric fields with inline validation and helper hints (effective batch size, LoRA scale α/dim, bucket-step divisibility, sample aspect ratio).
- **Validation** is a tabbed editor over `[[validation.samples]]`: a horizontal strip of set chips (label, warning icon while the set has an invalid field), a `+` that clones the open set, and a small `×` that deletes a set after a confirmation dialog (never the last one). The open tab shows Label, Positive/Negative prompt, Width/Height/Steps, Guidance scale/Seed/Repeat. Saving writes the `[validation]` scalars from the first tab plus one explicit block per tab.
- Header shows the config path, a summary line (`name · resolution · epochs · batch`), and an **Unsaved** indicator. **Save** validates the whole form (auto-jumping to the first invalid section), then patches the TOML in place, preserving comments and formatting. **Reload** is blocked while the form is dirty.
- **Profiles** saves and applies named `config.toml` presets. A name in the **Profile name** field writes the editor's values to `configs/<name>.toml` next to `config.toml` — the folder is tracked by git, the presets in it are ignored — and a name already taken asks before it overwrites. Clicking a saved profile applies it: `config.toml` is patched in place (comments, blank lines, unknown tables and every key the profile does not carry stay as they are) and the editor reloads from the result, with a confirmation first when the editor has unsaved changes. A profile written here holds the whole config, so applying it replaces the environment paths too; a hand-trimmed file changes only the keys it lists, and a section `config.toml` has no table for is named in the status line instead of being dropped silently. **Delete** removes the preset file and leaves `config.toml` alone. A run already in flight keeps the settings it started with.

See [Configuration](configuration.md) for the meaning of every field.

### Dashboard — training monitor & control

The heart of the app. It spawns `api.py` on first use and polls it (every 1 s while a run is live, otherwise 3 s). The hardware panel polls `hardware_status` on its own 1 s cadence while the tab is visible.

- **Header**: connected/disconnected indicator, auto-refresh switch, Refresh button, dataset / target / base-model compact metrics, and sliders for **Curve Smoothing** (EMA 0–0.99), **Chart Line** (stroke 1–8), and **Sample Size** (80–360 px thumbnails, default 120).
- **Run history**: the selector under the header is what the page is showing. Its first entry, **Current run**, follows `state.json` — a freshly started run is picked up on its own, and the collapsed box keeps the title `Current run` while it does, with the run it resolved to on the second line — and every other entry pins the page to that run (the box then titles that run's id). A trainer that has no run recorded (a cleaned runtime directory, or one Reset cleared) shows **Current run** with `Nothing started yet` and no badge, and no run's step count or size under it: the page follows the trainer, so a run the trainer is not on comes from the list. The list is built from the run directories of `output_dir` and `logging_dir` whatever `output_name` they were created with, newest first, and each entry carries the run's timestamp, its newest step, how many samples and checkpoints it has, its size, and one of two badges: **Live** while that run's process is running, **Stopped** once it is not (no badge at all while there is no run to show yet). Switching runs reloads the charts, the sample thumbnails and the checkpoint panel for that run; the Training Control card names the run being shown, so a run with no logs and no samples is still identified by the name its run id was built from. A run whose TensorBoard directory is gone — Reset used to delete it, or `logging_dir` has moved since — is still listed and still shows its samples, because a run's step count is read from its sample filenames and checkpoint directories, not from an event file.
- **Training control card**:
  - Status chip (idle / starting / encoding / training / sampling / pausing / paused / resuming / stopping / finished / error), plus transient **gpu-out** / **gpu-in** chips with swap progress while offloading/loading.
  - Run info: output name, run id, PID, elapsed time, alive flag, and the run's `detail` / `error` lines. When `[training].resume_lora_path` is set, the card also shows what the next run will resume from, or — while running — the checkpoint this run was seeded from (`Resumed from … · checkpoint step N · M tensors`).
  - Three phase progress bars: **latent encode** (`encoding.current/total`), **training** (step/total, epoch, loss, avg-loss), **sampling** (image r/R, denoise step d/D; with several `[[validation.samples]]` sets it also shows `set s/S` and `r` counts the images of the whole pass).
  - Buttons: **Start** (enabled only when terminal: idle/finished/error), **Pause** / **Resume** (with in-flight spinner states), **Early Stop** (phase-aware confirmation dialog — warns whether a checkpoint will be saved), and **Reset** (clears the Finished/Error state so Start can launch a new run). Reset **keeps** the resolved run's sample images and TensorBoard logs — that run stays in the history list with its charts and pictures — and the dialog's optional checkbox only deletes that run's LoRA checkpoints. `python clean.py` is still what wipes a run.
  - **Live settings**: `Save every N steps` (a field + **Apply**) and the **Sampling** switch, under the three phase bars. They act on the run in progress — the request goes to the runtime `settings.json`, the trainer adopts it at its next optimizer step, and the line underneath reads what is actually in force (`Save every 100 steps · next at step 250 · sampling on`). A cadence change restarts the countdown from the step that adopted it, so "every N steps" means "N steps from now" and no save point is silently skipped; the sampling switch does not touch the schedule, and a pass already rendering is not interrupted. Nothing is written back to `config.toml`: the file stays what the **next** run starts from (Utils → Training). The controls are shown only while the card may act on the run being viewed (see the pinning rule below) and that run is live — a paused one counts. A failed request (e.g. no live trainer) shows its message in place. With **no** live run the row has nothing to retune and reads the next run instead: `Next run · save every 100 steps · sampling on`, straight from `config.toml` (Utils → Training is where those are edited). The `settings` block in `state.json` describes the run that published it, and a runtime directory no run has touched carries the placeholder `0` — which means "no checkpoints", not "the configuration says so" — so a stopped card never shows it.
  - All five buttons are off while a past run is pinned in the run history: they act on the run `state.json` is on, and the card says so. Charts, hardware, sample images and the checkpoint panel keep working for the run being viewed. The live settings row follows the same rule, so a pinned past run shows its values read-only.
- **Hardware**: live GPU / CPU panel under Training Control. GPU numbers come from `nvtop -s` (JSON snapshot); AMD edge/junction temps from DRM hwmon; CPU util/temp from `/proc` and thermal zones; RAM from `/proc/meminfo`. When `[environment].amdfq` is `vmm`, a separate VA bar shows used / total GPU virtual address space: **GPU VA (live)** while the hook gives a freed range's address back (the default, and what the 2026-09 kernel allows), or **GPU VA (not returned)** while `amdfq_va_never_reuse` keeps it for the process lifetime (the pre-fix workaround). Info line + current-value cards + four charts in two rows: **GPU** (util / VRAM), **Temp** (edge / junction / CPU), **Power** (GPU watts), **CPU** (util / RAM). Ranko keeps a ~6 minute ring buffer. Missing nvtop shows an error on this section only — training controls and TensorBoard charts keep working.
- **Path chips**: current run id, `{logging_dir}/{run_id}`, `{output_dir}/{run_id}` — `—` when no run directory resolves yet.
- **Metric cards**: Current Step, Latest Loss, UNet LR, TE Effective LR.
- **Training charts**: Train/Avg_Loss, Train/Loss, UNet/LR/Effective_Actual_LR, TE/LR/Base_Scheduled, TE/LR/Effective_Actual_LR — interactive line charts with an always-on hover readout (see interactions below). On **Train / Avg Loss**, `Ctrl` + left click or a left double click opens the checkpoint panel described below.
- **Checkpoints**: one card per LoRA checkpoint the run being shown wrote, newest step first, each with the step badge (`step 3050`, or `step 3050 · final`), the artifact directory, the run / step / r-α / size line, the checkpoint's path, and the images that belong to that step: the samples the run wrote for it (in `(set, repeat)` order) followed by any generated pass, each marked `GENERATED` (and the ones started in this session `NEW`). A pass that was stopped or that failed part way is shown with the images it did write — files on
disk belong on the card — and only a job that is still running is the progress line instead.

Clicking a thumbnail opens the same fullscreen preview as everywhere else (Esc closes, ←/→ navigate). The preview cycles the section's own images in the section's own order — a card's training samples, then the images of the passes that joined it, then the `samples only` rows — so whatever the section draws can be opened, a generated pass recorded without a step included. When a run used several prompt sets, every thumbnail carries a small `P1`/`P2` badge saying which `[[validation.samples]]` entry rendered it (a single-set run and a run from before this feature show no badge).
  - **Sample range**: above the cards, `from` / `to` step fields (prefilled with the run's own
    steps, so "all of them" is one click), the **Sample range** button, and how many checkpoints the
    range covers. Pressing it renders that whole pass for every checkpoint whose step is inside the
    range, oldest first, as one detached job: it says `Sampling steps 100–600 · 3/8 checkpoints ·
    12/48 images` with a progress bar and a **Stop** button while it runs. A range that matches
    nothing says so instead of starting; the button is off while the GPU is busy with training or
    another generation. **Stop** asks the generator to finish the step it is in and stop: the cards
    keep what was already rendered, the batch closes as cancelled (not as a failure), and the
    checkpoint it was in the middle of shows the images it had written.
  - A checkpoint with **no** images yet is listed too — that is what a run with `sampling_enabled = false`, or an early checkpoint, looks like — with a **Generate samples** button instead. Pressing it renders the config's whole `[[validation.samples]]` list for that checkpoint (one image per set and repeat, the prompt/size/steps/CFG/seed/repeat from `config.toml`, the network settings from the checkpoint's own metadata) as a detached job, writing `{...}_p{set}_{repeat}.png` into the run's `{output_name}_samples/generated/`. The card shows `set s/S · image i/N · denoising k/K` while it runs, and the images appear under that checkpoint when it finishes. Nothing is overwritten: the run's own samples stay the record of what training produced.
  - The button is enabled only while no live trainer is using the GPU — the run is **paused** (pause offloaded every module, so the card is free), **stopped** or **over**. While a run is training the card says so instead; while another generation is running the button is off, because the GPU is single-tenant. `train_resume` refuses while a generation is running, so resuming cannot put a second SDXL on the card.
  - A step whose images outlived its weights (Reset deleted the checkpoint dirs) keeps a **samples only** card, so those pictures never become unreachable.
  - The section is fed by `list_checkpoints` + `list_samples` + `list_generated_samples` on the page's own poll; the checkpoint list is what the section needs, so it is refreshed with every fetch (the helper caches each checkpoint's safetensors header by path/mtime/size, otherwise a finished run would re-read 40+ headers per poll).

If the helper process can't be reached (and no data has loaded), a full-screen error card with **Retry** (restarts `api.py`) is shown. The error message suggests setting `AXL_PYTHON` if the interpreter wasn't found.

### Automation — prompt wizard, ComfyUI batch, gallery

The tab that writes prompts (or takes them from a saved set) and pushes them through a ComfyUI
workflow, then shows what came out. Its three sections sit in a rail on the left; the language
chips (中文 / EN) at the top right switch the whole page, and it opens in English.

Everything the section remembers lives under `automation/` in the repo root (gitignored):
`settings.json`, `workflows/*.json` (workflows you uploaded), `prompts/*.txt` (saved prompt sets)
and `jobs/<job_id>/` (one run's `job.json`, `log.txt` and `images/`). `AXL_AUTOMATION_DIR`
overrides that root; a job's images are served to the tab through the same `blob_*` calls the rest
of the app uses, so no new image path exists.

**Prompts** — the wizard from `tools/gen_prompts.py`, in Kotlin:

- The tag matrix is the repo's `input_matrix.txt` (read over IPC, never edited here); the status
  line shows its path and line count, and **Reload** re-reads it after you edit the file by hand.
- **Profiles** are the `prompt_profiles/*.json` files, listed with their format version, size and
  mtime. Loading one opens the configuration list; a v1 or v2 profile is upgraded in memory (the
  row then shows what changed, e.g. `upgraded from v2 to v3`) and is only rewritten when you save.
  **Save as** writes a v3 profile, with an overwrite switch for a name that is taken; **Delete**
  removes the file.
- **Wizard**: 13 steps (character, mode, exposure, clothing, chest, belly, face, scene, family,
  ratio, stages, pose, count) with a step list on the left, back/next at the bottom, and pages the
  current mode or exposure does not use skipped (`nude` drops clothing; only `sex` has family /
  ratio / stages). The face page is five groups (总表情, 视线, 眼睛状态, 嘴状态, 脸红, 眼泪 here named
  Expression / Gaze / Eye state / Mouth / Blush / Tears), each with **any** (roll one of the whole
  mode pool), **off**, or a tick list that becomes that group's candidate pool — one tag per group
  per prompt, never two. Chest and belly take a level or "follow the pose". The stage page sets the
  eight per-prompt weights (`during` starts at 1); all-zero falls back to `during`. SFW with a high
  exposure, an exposed chest, or a sex face tag shows a yellow banner instead of the CLI's question.
- **List**: the one-page configuration list (总清单) with the current value of every item; clicking
  a row opens that step's editor in a dialog and returns to the list. **Generate**, **Save as
  profile** and **Back to the wizard** sit at the bottom.
- **Generated prompts**: the list, with a copy button per line, **Copy all**, **Download .txt** (the
  desktop save dialog or a browser download), and **Send to batch**, which hands the list to the
  ComfyUI section. The seed field is on the count step: blank means a fresh random seed, and the
  result header shows the seed that was used so the same batch can be reproduced.

**ComfyUI** — connect, pick a workflow, run the batch:

- **Server**: probes the machine that runs the helper. With the address blank it asks the helper to
  find a ComfyUI itself (it walks the loopback listeners and accepts only a server that reports a
  `comfyui_version`); a filled address is probed as given. The card shows the version and the queue
  depth, or the ports that were tried and why each was rejected.
- **Workflow**: **Upload JSON…** takes a file from *your* machine (the desktop reads it after the OS
  dialog; the web target sends it from a file input) and stores it under `automation/workflows/`.
  Each stored workflow shows its node count, its `SaveImage` count and its numeric `batch_size`
  count, and can be checked or deleted. The pre-check compares every model-like input against the
  live `/object_info` and names what is missing (both combo shapes ComfyUI 0.35 reports are read).
  **Positive-prompt node** lists every `CLIPTextEncode` with a snippet of its current text; the one
  the sampler's `positive` link points at is picked for you and marked as guessed.
- **Batch**: prompts come from the generated list, a saved prompt set, or a box you type into. Set
  images per prompt (1–16; refused when the workflow has no numeric `batch_size`), the history poll
  interval, and the output folder (defaults to `automation/jobs`; **Browse** picks it). **Save
  settings** writes `settings.json`, **Start** queues the job, and while one runs the card shows its
  progress with **Cancel**. `Save as prompt set` stores the current list under `automation/prompts/`
  for reuse. The log card shows the tail of the job's `log.txt`.
- Generation runs detached (`trainer/run_automation.py`), so closing Ranko does not stop it. Each
  prompt gets its own random seed (written into every numeric seed input, including one that is
  fed through a linked seed node), the positive node's text is replaced, and `batch_size` is set on
  every numeric one. One bad prompt is recorded and the batch continues; three failures in a row
  stop it. A `SaveImage` output is downloaded as `p0003_01.png` with a sidecar `.txt` holding the
  seed, `prompt_id` and prompt text — `PreviewImage` nodes are ignored. Cancelling keeps whatever
  already landed.

**Gallery** — the jobs and their images:

- Newest first, with state, `done/total`, image count and elapsed time; filter chips (all / running
  / done / failed / cancelled) and a search box over the job id and workflow path.
- The selected job offers **Cancel** (SIGTERMs the runner), **Retry failed** (runs only the prompts
  that produced no image, in the same folder), **Save records .txt** (one line per image: name,
  seed, `prompt_id`, prompt), **Open folder** (desktop) and **Delete** (asks first; removes the job
  directory).
- Thumbnails are grouped per prompt and wrap instead of scrolling sideways; the slider sets their
  size (80–360 px). Clicking one opens the same fullscreen preview the Dashboard uses, with the
  caption (seed, `prompt_id`, prompt), prev/next, **Save this image…** (the image at full size) and
  **Copy prompt**.
- A job only knows the images it recorded; a prompt that failed shows its error instead of thumbs.

## How it talks to the trainer

1. **Discovery** — `TrainerRepo.findRoot()` walks up from the app's executable and `user.dir` looking for `api.py` or a `config.toml` that sits next to the `trainer/` package.
2. **Spawn** — Ranko connects to `ws://127.0.0.1:18765`. If nothing is listening it runs `$AXL_PYTHON` (if set) or `python3 -u api.py --websocket` with the working directory at the repo root, stderr inherited, `PYTHONUNBUFFERED=1`. A JVM shutdown hook kills the helper **this process spawned**.
3. **Protocol** — JSON-RPC on that WebSocket: requests are `{"id": n, "method": "...", "params": {...}}`, responses are `{"id": n, "ok": true, "result": {...}}` or `{"id": n, "ok": false, "error": "..."}`. Replies match on `id`. Dataset images travel as resized JPEG blobs (`blob_batch`), not `java.io.File`.
4. **Methods** — train control plus `config_*`, `dataset_*`, `blob_*`, `mask_*`, `profile_*`, `prompt_*` and `automation_*`. Full reference: [API.md](../API.md).

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
- **Generate sample**: expands a form (prompt, negative prompt, CFG, steps, seed — prefilled from the first `[[validation.samples]]` set, or from `sample_prompts`, `sample_negative`, `guidance_scale`, `sample_steps` when the file has no sets) that renders one extra image from the matched checkpoint, see below. Its button follows the same rule as the Checkpoints section's — nothing may be using the GPU, a paused run counting as free — so a run that is over enables it even while the helper still has its PID on record.
- **Samples at step N**: the step's own sample images followed by any generated ones, three per row; on a multi-set run each thumbnail is labelled with its `Pn` set. Clicking one opens the usual fullscreen preview (which then cycles through the generated images too, in step order). If the checkpoint's step has no samples, the nearest sampled step is shown and labelled as such.
- **Size**: the panel opens at a content-derived size (three 400 dp-wide sample slots, ≈1268 dp wide for a three-image run) and can be dragged by the grip in its bottom-right corner. The slots — and the text — follow the panel width, so dragging it bigger shows bigger samples; extra generated images wrap onto the next row instead of scrolling. A drag never moves the panel: placement is decided once from the click and the default width, and a panel that outgrows the window is slid back inside rather than flipped to the other side of the cursor. The dragged size is clamped to the window and kept for the rest of the session.

The checkpoint list comes from the page's own poll (`list_checkpoints` against the run being shown, with each safetensors header cached by path/mtime/size), so a click resolves immediately and a background rescan refreshes it in place. Dismiss with a click outside, the close button, or `Esc`. The panel is read-only for the run itself — resume selection stays in the Utils tab; **Save As** only copies the file out.

#### Generating a sample from the matched checkpoint

This form renders **one** image with its own prompt, which is handy for a quick look. The
Checkpoints section's **Generate samples** button is the other direction: the config's whole
`[[validation.samples]]` list for a checkpoint, which is what to use for a checkpoint the run did
not sample (or to re-render one after changing the prompts in `config.toml`).

`Generate sample` → fill in the form → `Generate` runs one image (the run's `sample_width` × `sample_height`) through the checkpoint's own LoRA and appends it to that step's row, highlighted with an accent border, a `GENERATED` (or `NEW`, for this session) badge and a `CFG 5 · 20 steps · seed 12345` caption. Progress (`denoising 12/20`) comes from the job file, which the generator rewrites on every denoise step.

- Prompt handling, CFG, step count, seed (`0` = random, the seed actually used is written back), scheduler and CLIP settings mirror the run's own sampling (the first prompt set), so a generated image is comparable with the training samples. `clip_skip`, `max_token_length`, `network_dim`/`network_alpha` and the base model come from the checkpoint's kohya metadata, not from today's `config.toml` — sampling an old checkpoint uses the settings it was trained with. If the checkpoint was trained on a different `base_model_version` than `[model_spec]` says, the job fails with that message instead of producing a mismatched image.
- Files land in `{output_dir}/{run_id}/{output_name}_samples/generated/` — one PNG plus one JSON job record (prompt, CFG, steps, seed, checkpoint, mode, state, error) per generation. The directory is *inside* the sample dir, so the training sample strip and Images tab ignore it, and it stays with the run's samples when the run is reset. This one-image form belongs to no prompt set, so it never carries a `Pn` badge; the whole-set pass of the Checkpoints section does (its files are named `_p{set}_{repeat}.png`, and its job record lists them in `files`).
- Generation needs the GPU to itself: while a live trainer is using it — `starting`, `encoding`, `training`, `sampling`, or a swap in flight — the button is disabled and `generate_sample` refuses, naming the reason. A **paused** run is allowed (`pause` has offloaded the UNet, both text encoders, the optimizers and the VAE), and so is a run that is stopped or over; `train_resume` refuses while a generation is running so the two can never share the card. Only one generation runs at a time, whichever run it belongs to. The generator is a detached process (`trainer/generate_sample.py`), so closing Ranko does not kill it; it never touches `state.json`, the lock or the training loop.
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
