<p align="center">
  <img src="axltrainer.png" alt="AXLTrainer logo" width="168"/>
</p>

<h1 align="center">AXLTrainer</h1>

<p align="center">
  <strong>Local LoRA training · AMD first</strong><br/>
  A local-first LoRA training stack: a TOML-driven Python engine, a JSON-RPC helper, and a Compose
  Multiplatform dashboard that drives them — one machine, one app, no cloud control plane.
</p>

<p align="center">
  SDXL LoRA + Kohya LoCon today · SD 3.5 catalogued · Desktop JVM + wasmJs web companion · ROCm
</p>

## System Capabilities & What AXLTrainer Can Do

AXLTrainer is a local, end-to-end SDXL LoRA training system designed to handle the entire lifecycle of model creation on a single personal machine. Rather than treating training as merely running a script, it provides a complete ecosystem for dataset preparation, execution, real-time control, and downstream generation.

### Core Capabilities

- **Local SDXL LoRA & Kohya LoCon Training**: Train standard SDXL LoRAs as well as Kohya LoCon (LoRA-C3Lier) adapters with multi-encoder (CLIP-L + CLIP-G) support, v-prediction auto-detection, and Schedule-Free AdamW + AdamW optimization.
- **AMD ROCm First-Class Support**: Native support for AMD ROCm GPUs with hardware-specific fixes (such as 128-step bucket resolution alignment for MIOpen/FlashAttention stability and custom allocation patch interposers for RDNA 4 Tensile GEMM bounds protection), alongside CUDA support.
- **Dataset Curation & Auto-Tagging**: Tag images automatically with the embedded Pixai ViTDet tagger (30,000+ tags across 6 categories), perform partial/add-only tagging, inspect tag frequency distributions, execute bulk filtering/replacements, and paint loss masks down to individual pixels.
- **True-Offload Pause & Resume**: Pause training runs at swap-safe boundaries and offload all model weights (UNet, text encoders, VAE, and Schedule-Free optimizer states) to CPU memory, releasing GPU VRAM completely for other applications.
- **Live Dynamic Parameter Tuning**: Change save intervals, toggle test sampling, and edit validation prompts on a running training process in real time without restarting or corrupting configuration files.
- **Automated Checkpoint Evaluation**: Benchmark and score generated LoRA checkpoints against dataset prompt expectations with micro precision, recall, F1 metrics, and tag coverage reports.
- **Prompt Generation & ComfyUI Automation Batching**: Generate prompts using a 16-step matrix wizard, dispatch jobs directly to local ComfyUI instances, track progress, and manage result galleries with individual image actions (redraw, append variations, edit prompt, delete).

## Highlights: Designed Around Ranko & Creator Workflow

Mainstream LoRA training stacks often force creators to jump between separate tools: command-line scripts for captioning, external web UIs for TensorBoard, manual file managers for sample inspection, and ComfyUI for testing. **Ranko**, the Kotlin/Compose Multiplatform control center, unifies this entire workflow into a single, cohesive desktop or web application.

### Human-Centric Interface & Unified Dataset Curation
- **Integrated Dataset Studio**: Browse thumbnails, paint per-pixel loss masks with feathering and invert options, edit `.txt` captions, and run the automated dataset tagger, without leaving the application.
- **Tag Analytics & Cleaning**: Inspect tag frequency charts (with English label matching and Chinese dictionary translations), perform AND/OR logic filtering, batch add or remove tags, and safely renumber/shuffle datasets without breaking image-mask pairings.
- **No-GPU UI Overhead**: Ranko acts purely as a controller. It speaks JSON-RPC over a loopback WebSocket to the `api.py` helper, which drives the detached training process, so the UI itself holds no VRAM and never touches the GPU.

### Live Tuning & Dynamic Run Control
- **On-the-Fly Adjustments**: Change `save_every_n_steps` or toggle sample rendering while training is running — the trainer adopts those at its next optimizer step. Rewrite the validation prompts too: a live run reads them before every sample point. Neither touches the baseline `config.toml`.
- **Checkpoint Inspection & Direct Sampling**: Select any point on the loss curve to open its associated checkpoint, view historical samples, adjust CFG/step parameters using stored checkpoint metadata, and render new sample images on demand.

### Automated Generation & ComfyUI Pipeline
- **16-Step Prompt Matrix Wizard**: Generate balanced prompt batches covering character, exposure, clothing, pose, face expression, and anatomical details using customizable profile presets.
- **ComfyUI Batch Integration**: Connect to local ComfyUI instances, upload API workflows, automatically bind positive prompt nodes, and execute automated batch generations.
- **Interactive Gallery Suite**: Manage generated outputs with per-image operations including single-image redraws with new seeds, variation expansion, prompt editing, and error tracking.

## What this is

AXLTrainer trains LoRAs on your own machine, and it treats the whole loop as the product rather than just the training step: curating images, writing and cleaning captions, painting loss masks, picking the hyperparameters, watching a run, inspecting samples, comparing and scoring checkpoints, and pushing prompts through a ComfyUI workflow when you want more pictures of what you just trained.

It consists of four primary components:

- **The training engine** (`trainer/`) — a headless Python process driven by one TOML file. `trainer/main.py` takes no CLI arguments.
- **The control plane** (`api.py`) — a JSON-RPC helper over a loopback WebSocket. It reads TensorBoard scalars, sample images, and checkpoints, writes configs and datasets, and manages run lifecycle operations (start, pause, resume, stop, reset).
- **Ranko** (`ranko/`, product name **AxlRanko**) — the Kotlin/Compose app you interact with. Five tabs: Images, Statistics, Utils, Dashboard, Automation. Ships as a desktop JVM application and a Kotlin/Wasm web companion.
- **Dataset and prompt tooling** (`tools/`, `tagger2/`, `tagger/`, `ranko/tools/agent.py`) — command-line utilities for scripting and dataset automation.

There is no HTTP API, no inference server, and no kohya `sd-scripts` fork. Training runs in a detached process; Ranko serves strictly as a controller.

### Core Architecture Decisions

- **TOML-Only Configuration**: `config.toml` at the repository root is the single source of truth. Form values in the Utils tab patch the file in place while preserving comments and layout.
- **Detached Execution**: `api.py` launches training via `bash start_train.sh` in a detached session (`setsid`). Closing Ranko does not interrupt training. Communication occurs via `command.json` and live `settings.json` channels.
- **Run-Centric Isolation**: Every run creates a unique `{output_name}_{YYYYMMDD_HHMMSS}` folder in `output_dir` and `logging_dir`. Historical runs retain verbatim copies of their initial `config.toml`, samples, and checkpoints.
- **Full VRAM Offloading**: Pausing offloads the UNet, text encoders, optimizers (including Schedule-Free states), and VAE to system memory, followed by GPU memory cache clearing.
- **ComfyUI-Compatible Checkpoints**: PEFT state dicts are remapped to kohya `lora_unet_*` / `lora_te1_*` / `lora_te2_*` format in bf16 with `modelspec.*` and `ss_*` metadata for direct loading in ComfyUI or kohya scripts.

## Ranko — The Control Center

The visual theme uses an amber night palette (warm near-black, logo amber `#F8A818`) with Nunito typography and porcelain cards. The application opens on a Home page and keeps a floating navigation rail that snaps to window edges and collapses when idle.

Appearance settings live under Utils -> Appearance: backdrop styles (Solid, Glow-orbs, Image), blur effects, text/icon scaling, and thumbnail quality settings, all stored in Java Preferences.

Below is a detailed walkthrough of Ranko's five core tabs.

### Images — Captioning & Loss Mask Painter

<p align="center">
  <img src="doc/screenshots/images-tab.webp" alt="Images tab — thumbnail list, mask painter and caption editor"/>
</p>

The primary dataset editing interface. The left panel shows image thumbnails, while the right panel contains the mask painter canvas and caption editor.

- **Folder Selection**: The top chip bar allows switching between `[[environment.train_data]]` folders. Selection syncs across Images, Statistics, and Utils tabs.
- **Thumbnail Grid**: Displays dataset images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) while hiding `*.mask.png` sidecars. Visual indicators highlight unsaved captions (red border) and existing loss masks (pink corner dot, turning red when unsaved).
- **Mask Painter Canvas**: Features controls for Mask toggle, Mask Only preview, Brush size, Feathering (0-100%, default 20%), Strength (default 100%), Invert, Fill White/Black, Clear, Reset, and Save. Left-click paints white (train region), right-click paints black (ignore region), and `Ctrl`+left erases. Cursor rendering displays brush core and feather radius. **Alt+wheel** resizes brush size. Painting automatically halts at image borders.
- **Mask Storage**: Edits reside in memory until saved to `{stem}.mask.png`. Transparent PNGs preview using alpha channels as fallback masks when no sidecar exists.
- **Caption Editor**: Modifies comma-separated tag lists in `{stem}.txt` with explicit Save and Reset controls.

### Statistics — Tag Analytics & Bulk Cleaning

<p align="center">
  <img src="doc/screenshots/statistics-tab.webp" alt="Statistics tab — tag distribution, filter controls, shuffle and renumber"/>
</p>

Provides dataset-wide caption analytics and batch cleaning tools.

- **Tag Frequency Distribution**: Displays tags sorted by occurrence frequency with color-coded bars (blue to green to red). Displays English tags alongside Chinese translations when listed in `tagger/selected_tags.csv`.
- **Filter Grid**: Renders a staggered grid of images matching the selected tag filters. Clicking an image navigates directly to it in the Images tab.
- **Filtering Logic**: Supports Intersection (AND), Union (OR), and Negation (NOT) modes, with quick selection controls.
- **Batch Dataset Actions**: 
  - **Remove Selected**: Strips selected tags across matching captions.
  - **Batch Add**: Prepends or appends specified tags to matching captions.
  - **Drop Selected Samples**: Moves matching images, captions, and masks into `/tmp/axlranko/trash` based on a probability threshold (`r` = 0.001 to 1.0).
  - **Shuffle & Renumber**: Executes folder-wide sequential renumbering (`0001...`) via IPC (`trainer/fsrpc.py`), safely preserving pairings between images, `.txt` captions, and `.mask.png` files.

### Utils — Configuration & Profiles

<p align="center">
  <img src="doc/screenshots/utils-training.webp" alt="Utils — the Training section, with the run's step estimate"/>
  <img src="doc/screenshots/utils-rocm.webp" alt="Utils — the ROCm section, with the allocation patch"/>
</p>

A structured form editor for `config.toml` that validates input fields before starting a run.

- **Section Hierarchy**: Environment, ROCm, Model Spec, Training, Network, Bucketing, Optimization, UNet Optimizer, Text Encoder, Infrastructure, Validation, Appearance, WM, Helper, Profiles.
- **Environment & Auto-Tagging**: Configure dataset directory paths and per-folder epoch repeats. Includes the **Auto-tag dataset** interface driving `tagger2/main.py` with category toggles, confidence thresholds, and **Partial Tagging** (add-only mode for specific tag lists).
- **ROCm Configuration**: Select memory allocation interposers (`none`, `tail`, `vmm`), toggle hardware workarounds, and set pool sizes for AMD graphics cards.
- **Training Estimator**: Configures epochs, batch sizes, gradient accumulation, learning rates, and save intervals. Provides real-time step and sample calculation previews (`≈ 3,920 steps · 280/epoch × 14 epochs`).
- **Validation Prompts**: Tabbed prompt editor for `[[validation.samples]]` specifying prompts, resolutions, inference steps, guidance scales, and seeds.
- **Profile Presets**: Save and apply named `config.toml` presets stored under `configs/`.

### Dashboard — Live Monitor & Control

<p align="center">
  <img src="doc/screenshots/dashboard-overview.webp" alt="Dashboard — run history, Training Control, hardware and path chips"/>
</p>

Provides real-time training telemetry, dynamic run management, interactive loss charts, and sample inspection.

- **Run History**: Switch between active and completed historical runs. Loading a historical run populates its corresponding charts, sample images, and checkpoint records.
- **Training Control**: Displays run status (`idle`, `training`, `sampling`, `pausing`, `paused`, etc.), process IDs, and memory swap progress. Contains controls for Start, Pause/Resume, Early Stop, and Reset.
- **Live Settings Panel**: Allows dynamically changing `save_every_n_steps` and toggling `sampling_enabled` during an active training run.
- **Hardware Telemetry**: Displays live GPU utilization, VRAM usage, clock speeds, fan speeds, power draw, temperatures (edge/junction/CPU), system RAM, and GPU Virtual Address space metrics via `nvtop` and system sources.

<p align="center">
  <img src="doc/screenshots/dashboard-charts.webp" alt="Dashboard — training charts, hover readout and metric cards"/>
</p>

#### Interactive Charts & Checkpoint Inspection

- **Rendered Metrics**: Three interactive canvas charts — `Train/Avg_Loss`, `Train/Loss`, and a dual-axis `Learning Rate` chart (`UNet/LR/Effective_Actual_LR` on the left axis, `TE/LR/Effective_Actual_LR` on the right). Supports EMA smoothing (including Avg Loss), zooming (`Ctrl`+wheel for the X axis, `Shift`+wheel for Y), panning, and a default window of the newest 800 steps. Sliders above the charts change that window and the y-axis clip.
- **Checkpoint Selection**: Double-clicking or `Ctrl`-clicking a point on the loss curve locates the nearest checkpoint, displaying its step details, parameters, loss values, and associated sample outputs.
- **On-Demand Sample Generation**: Renders new test images directly from selected historical checkpoints using metadata extracted from the checkpoint itself.

<p align="center">
  <img src="doc/screenshots/dashboard-sampling-prompts.webp" alt="Dashboard — the Sampling Prompts section"/>
</p>

#### Sampling Prompts Management

- Inspect and modify prompt sets used for validation sampling. Edits save to `sample_sets.json` for the targeted run without altering the global `config.toml`.

<p align="center">
  <img src="doc/screenshots/dashboard-checkpoints.webp" alt="Dashboard — checkpoint cards with samples, Save As, Generate samples, Evaluate and Clear samples"/>
</p>

#### Checkpoint Gallery & Evaluation

- **Card View**: Displays all generated `.safetensors` checkpoints with associated sample images.
- **Actions**: Pin key checkpoints, export files via **Save As**, render sample ranges across multiple checkpoints, or delete sample artifacts via **Clear samples**.
- **Automated Evaluation**: Evaluates checkpoint quality by generating samples, tagging outputs with the auto-tagger, and calculating micro precision, recall, F1 scores, and tag union statistics against prompt definitions.

### Automation — Prompt Wizard, ComfyUI & Gallery

Provides prompt generation tools, batch ComfyUI dispatching, and output gallery management. Stores data locally under `automation/`.

<p align="center">
  <img src="doc/screenshots/automation-prompts.webp" alt="Automation — the prompt wizard, profiles and generated list"/>
</p>

#### Prompt Wizard

- **Matrix Generator**: A 16-step wizard (character, mode, exposure, clothing, body figure, face expression, scene, pose, etc.) that parses tag matrix configurations to produce structured prompt lists.
- **Profiles**: Load and save prompt configurations as reusable JSON profiles.

<p align="center">
  <img src="doc/screenshots/automation-comfyui.webp" alt="Automation — ComfyUI server, workflows, positive-prompt node and batch"/>
</p>

#### ComfyUI Batch Manager

- **Server Detection**: Auto-detects local ComfyUI instances running on loopback ports.
- **Workflow Binding**: Upload API-format ComfyUI workflow JSONs, inspect required nodes, select target `CLIPTextEncode` positive prompt nodes, and configure batch counts.
- **Detached Execution**: Dispatches batch jobs via `trainer/run_automation.py`, injecting randomized seeds and prompt text into workflow definitions.

<p align="center">
  <img src="doc/screenshots/automation-gallery.webp" alt="Automation — the Gallery: jobs, thumbnails and per-image actions"/>
</p>

#### Gallery & Image Operations

- **Job History**: Browse completed and active generation jobs with filtering options.
- **Per-Image Actions**: Features interactive overlays on individual thumbnails:
  - **Redraw**: Re-queues the prompt with a new seed and overwrites the existing image file and metadata.
  - **Add Images**: Generates additional variations for a selected prompt.
  - **Edit Prompt**: Modifies the prompt text used for subsequent redraws or additions.
  - **Delete**: Removes a single image and its sidecar; deleting a prompt's last image drops that prompt entry and the remaining prompts renumber.

### Web Companion Target

The `:webApp` target compiles the Ranko interface to Kotlin/Wasm, allowing control over local networks via a web browser. Communicates with `api.py` via WebSockets without requiring local JVM execution on the client device.

## Training Engine Technical Specifications

The core engine under `trainer/` is a PyTorch/diffusers implementation driven strictly by `config.toml`.

### Execution Phases

1. **`starting`**: Loads config, creates output directories, acquires lock (`train.lock`), initializes pipelines, and loads resume checkpoints if specified.
2. **`encoding`**: Pre-encodes dataset images to `.pt` files in `<folder>/.latents_cache/` using VAE tiled encoding. VAE weights offload to CPU once caching completes.
3. **`training`**: Main training loop. Iterates through aspect-ratio buckets, encodes captions using CLIP-L/CLIP-G, adds noise, executes forward passes, accumulates gradients, clips norms, and updates optimizers.
4. **`sampling`**: Renders test prompts using `EulerAncestralDiscreteScheduler` at specified checkpoint intervals. Offloads text encoders and UNet during decoding phases.
5. **`finished`**: Writes final LoRA weights, releases file locks, and updates run state.

### LoRA & Kohya LoCon Architecture

- **Standard LoRA**: Wraps UNet attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`) and Text Encoder projections.
- **Kohya LoCon (`network_type = "locon"`)**: Adds a second adapter whose Linear extras and TE MLP layers (`fc1`, `fc2`) take `network_dim`, while the Conv2d layers (`conv1`, `conv2`, `conv_shortcut`, downsample/upsample `conv`) take `conv_dim` and `conv_alpha`.
- **Optimizers**: Schedule-Free AdamW on UNet layers paired with standard AdamW (and cosine/linear schedules) on Text Encoders.

### Bucketing & Loss Mask Weighting

- **Aspect Ratio Bucketing**: Calculates pixel area budgets based on `train_resolution²` (~1.05 MP for 1024x1024). Fits full images inside buckets with neutral padding (loss weight = 0). `bucket_reso_steps` must stay at 128 on AMD ROCm, which keeps every VAE latent dimension divisible by 16.
- **Spatial Loss Masks**: Multiplies per-pixel MSE loss by downscaled 0-1 mask weights provided by `{stem}.mask.png` sidecars or image alpha channels.

## Dataset & CLI Tooling

### Dataset Format Standard

Datasets consist of images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`) paired with corresponding `{stem}.txt` tag files and optional `{stem}.mask.png` loss masks.

### Included Command Line Scripts

- `tagger2/main.py`: Captioning tool using `pixai-tagger-v1.0` (ViTDet architecture, 30,000+ Danbooru tags).
- `ranko/tools/agent.py`: Non-interactive CLI for inspecting, adding, removing, and filtering `.txt` dataset tags programmatically.
- `tools/caper.py`: Bulk tag removal tool.
- `tools/dropper.py`: Random sample reduction utility.
- `tools/mask_blur.py`: Generates blurred loss masks from alpha channel silhouettes.
- `tools/suf.py`: Legacy file renumbering script.
- `tools/inspect_lora.py`: Inspects LoRA `.safetensors` structure and metadata.

## Architecture & IPC Protocol

Ranko operates independently from GPU execution by communicating with `api.py` over WebSockets using JSON-RPC frame formats.


```

┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Ranko (desktop / web)      │         │  bash start_train.sh         │
│  ranko/ (Kotlin)            │         │  └─ python -u trainer/main.py│
│                             │         │     (detached, setsid)       │
│  ┌──────────────┐  WebSocket│         │     │                        │
│  │ api.py       │◄──────────┤         │     ▼                        │
│  │ (JSON-RPC)   │  loopback │         │  trainer/control.py          │
│  └──────┬───────┘           │         │  state.json / command.json / │
│         │                   │         │  settings.json / train.lock  │
│         └── reads ── TensorBoard logs (logging_dir/{run_id})         │
│         └── reads ── sample PNGs (output_dir/{run_id}/*_samples)     │
└─────────────────────────────┘         └──────────────────────────────┘

```

- **Lanes**: Four parallel connection channels (`control`, `poll`, `blob`, `long`) isolate long-running operations (such as auto-tagging) from UI polling and image transfers.
- **Image Protocol**: Images and thumbnails transfer via binary blob requests (`blob_stat` / `blob_batch`) with hash caching.

Full IPC details are documented in [API.md](API.md).

## Getting Started

### Prerequisites

- Python 3.11+ (Python 3.14 recommended).
- GPU with sufficient VRAM for SDXL training (NVIDIA CUDA or AMD ROCm).
- JDK 17+ for building or running the Ranko desktop app.

### Installation

1. Clone the repository and setup the Conda environment:

```bash
conda env create -f environment.yml
conda activate axl

```

2. Configure `config.toml` at the repository root to set base model paths (`pretrained_model_name_or_path`), dataset directories (`train_data_dir`), and output paths (`output_dir`, `logging_dir`).
3. Launch training or open the control interface:

```bash
# Headless engine execution
bash start_train.sh

# Launch Ranko desktop UI
cd ranko && ./gradlew :desktopApp:run

```

## Deployment

### Desktop Application

```bash
cd ranko
./gradlew :desktopApp:run              # Launch dev build
./gradlew :desktopApp:packageDeb       # Package Linux .deb package (or packageDmg / packageMsi)

```

Run `./install.sh` at the repository root to create desktop environment shortcuts and application menu launchers.

### Web Companion

```bash
cd ranko
./gradlew :webApp:wasmJsBrowserDistribution

```

Host the output directory (`ranko/webApp/build/dist/wasmJs/productionExecutable/`) on any static web server and point it to a running `api.py` helper instance.

`python install_daemon.py` builds that wasm site, copies it into the current user's data directory, and installs a `systemctl --user` unit that serves the site on `0.0.0.0:18766` together with `api.py` on `0.0.0.0:18765` (allowlist `192.168.0.0/16`). The static site has no allowlist of its own. `--uninstall` removes the unit and the copy.

## Documentation Reference

* [Overview Guide](doc/overview.md)
* [Installation Details](doc/installation.md)
* [Configuration Reference](doc/configuration.md)
* [Training Lifecycle & Features](doc/training.md)
* [Dashboard & UI Guide](doc/dashboard.md)
* [Dataset Tools Reference](doc/dataset-tools.md)
* [Mask Verification Suite](doc/mask-verification.md)
* [LoCon Technical Details](doc/locon.md)
* [Troubleshooting & ROCm Workarounds](doc/troubleshooting.md)
* [Web Target Architecture](doc/ranko-web-target.md)
* [API IPC Protocol Specifications](API.md)

## License

MIT License — see [LICENSE](LICENSE).
