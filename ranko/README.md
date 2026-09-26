# AxlRanko

AxlRanko is a tool for managing AI model training datasets, built with Kotlin Multiplatform and Compose Multiplatform. It targets Desktop (JVM) and a wasmJs browser companion that talks to the same `api.py` helper.

It focuses on the day-to-day maintenance of Stable Diffusion / LoRA training datasets: browsing and editing image captions, analyzing tag distribution, and cleaning up datasets in bulk based on tags.

## Features

### Image caption editing (Images)
- Browse the training dataset as a thumbnail list (jpg / jpeg / png / webp / bmp supported)
- Large preview plus a caption (tag) editor; unsaved edits are highlighted with a red border
- Save or reset the .txt caption file of the current image with one click
- Left/right and top/bottom pane ratios are adjustable by dragging the dividers

### Dataset statistics (Statistics)
- Scans the dataset and reports the occurrence count and percentage of every tag as a color-coded bar chart (higher frequency renders redder)
- Click tags to filter images, with both intersection (AND) and union (OR) logic modes
- Click any thumbnail in the result grid to jump to the caption editor, with that image preselected
- Dataset integrity check: aborts with an error if an orphan caption file (with no matching image) is found

### Training dashboard (Dashboard)
- Ranko starts `api.py --websocket` as a local helper (JSON-RPC on loopback) and polls TensorBoard metrics plus training sample images
- Start / pause / resume / early-stop / reset controls. Pause offloads GPU weights to CPU; the trainer process is detached so closing Ranko does not stop it. Reset clears Finished state and deletes this run's samples + TensorBoard logs (optional checkpoint wipe), same targets as `clean.py`
- Progress bars for latent encoding, training steps, and sample generation
- Live step / loss / LR cards, interactive training charts, and sample previews grouped by step
- Auto-refresh every 3s (1s while a run is live), with a curve-smoothing slider
- Optional `AXL_PYTHON` to point at the trainer interpreter (otherwise `python3` on PATH)

### Bulk cleanup tools
- Remove selected tags: strips the selected tags from the captions of all matching images
- Add a tag in bulk: prepends or appends a tag to the captions of matching images (skips images that already have it)
- Drop samples: moves a random subset of matching images and their captions to `/tmp/axlranko/trash`, controlled by a probability rate r

### UI
- Material 3 design
- Draggable floating navigation rail; page transitions use a slide + fade animation

## Requirements and configuration

- JDK 17+ and Gradle (the project ships a wrapper)
- On startup the app locates its config automatically: it walks up from the executable and working directory until it finds the trainer repo root (`api.py`, or a `config.toml` next to the `trainer/` package)
- The Dashboard helper is `api.py --websocket` at the trainer repo root. Desktop Ranko uses the same upward search, then connects to `ws://127.0.0.1:18765` (spawning the helper if needed). The wasm UI does not spawn: set host/port in Utils → Helper (or `?host=` / `?port=`). LAN: `python -u api.py --host 0.0.0.0 --allow-ip <client>`. After connect, dataset and config IO go through that socket.
- The config is a TOML file; `[environment].train_data_dir` points to the training dataset directory
- Dataset layout: image files and same-named `.txt` caption files stored side by side; caption content is a comma-separated list of tags

## Running

```bash
# Standard run (desktop app)
./gradlew :desktopApp:run

# Hot reload run
./gradlew :desktopApp:hotRun --auto

# Browser companion (start api.py first)
./gradlew :webApp:wasmJsBrowserDevelopmentRun

# Run tests
./gradlew :shared:jvmTest
```

## Project structure

- `desktopApp/` - desktop app entry point (Compose Desktop window)
- `webApp/` - wasmJs entry (`ComposeViewport` + `#webApp`)
- `shared/` - shared code (page UI, data models, config parsing)
  - `commonMain/` - common cross-platform code
  - `jvmMain/` - FileKit, AWT mask input, helper spawn
  - `wasmJsMain/` - browser WebSocket, path-picker dialog, DOM mask input

## Tech stack

- Kotlin Multiplatform / Compose Multiplatform (Desktop JVM)
- Material 3
- Metro (dependency injection) + metrox-viewmodel
- ktoml (TOML parsing) + kotlinx.serialization
- Coil 3 (image loading)
- okio / kotlinx.coroutines

## Note

- An example config lives in `config.toml` at the root of this repository.
