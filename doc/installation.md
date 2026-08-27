# Installation & Setup

## Prerequisites

- **Python 3.11+** (3.12 recommended; `trainer/config.toml` parsing uses `tomllib` and `ranko/tools/agent.py` requires 3.11+).
- **A GPU with enough VRAM for SDXL LoRA training.** The shipped environment targets **AMD ROCm** (MIOpen/MIGraphX). CUDA works too if you install a CUDA build of PyTorch instead (see below).
- **JDK 17+** only if you want to build/run the Ranko desktop dashboard (the Gradle wrapper auto-provisions a JDK 21 toolchain via the foojay resolver).

> There is **no `requirements.txt`** and no `pyproject.toml`. The only dependency manifest is `environment.yml` (a conda environment named `axl`).

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate axl
```

The manifest pins the ROCm stack used during development, including:

| Component | Version (in `environment.yml`) | Notes |
| --- | --- | --- |
| Python | 3.12.13 | |
| PyTorch | `2.12.0+rocm7.2` | torch / torchvision / torchaudio |
| diffusers | 0.38.0 | pipeline + schedulers |
| transformers | 4.57.6 | CLIP text encoders |
| peft | 0.19.1 | LoRA adapters |
| accelerate | 1.13.0 | mixed precision + TensorBoard |
| schedulefree | 1.4.1 | UNet optimizer (`AdamWScheduleFree`) |
| safetensors | 0.8.0rc1 | checkpoint I/O |
| tensorboard | 2.20.0 | metric logging (read by `api.py` / `ui.py`) |
| streamlit + plotly | 1.58.0 / 6.8.0 | `ui.py` viewer |
| onnxruntime | 1.27.0 | `tagger/` ONNX captioning |

### NVIDIA / CUDA instead of ROCm

The code is plain PyTorch/diffusers — only the environment differs. Install the equivalent CUDA build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# plus the rest: diffusers transformers peft accelerate schedulefree safetensors tensorboard ...
```

If you skip MIOpen, the ROCm-specific env vars in `start_train.sh` / `start_api.sh` are harmless no-ops.

## 2. Configure the project

Open `trainer/config.toml` and set at least:

- `[environment].pretrained_model_name_or_path` — SDXL model (a diffusers directory, or a single-file checkpoint path).
- `[environment].train_data_dir` — your dataset folder (images + same-named `.txt` captions).
- `[environment].output_name` — the run name (used for every artifact path).
- `[environment].output_dir` / `[environment].logging_dir` — where checkpoints and TensorBoard logs go.

**The shipped values are the author's local machine** (`/home/acite/...`, `/opt/models/...`) and will fail on any other machine. There are commented-out alternative dataset examples in the file.

See [Configuration](configuration.md) for the complete reference, or edit the file from the Ranko dashboard's **Utils** tab (validated, no hand-editing of TOML required).

## 3. Prepare your dataset

```
train_data_dir/
├── 00001.png
├── 00001.txt      # caption: comma-separated tags, e.g. "1girl, solo, long hair"
├── 00002.png
├── 00002.txt
└── ...
```

- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (recursive scan).
- A caption file must share the image's basename. If a `.txt` is missing, the filename (underscores → spaces) is used as the caption.
- Caption content is a comma-separated list of tags. With `shuffle_caption = true`, tokens after the first `keep_tokens` are shuffled deterministically per epoch.

No caption files yet? Run the ONNX tagger to generate them (see [Dataset tools](dataset-tools.md#tagger---onnx-caption-generator)).

## 4. Verify the install

```bash
# The IPC helper should answer a ping and exit cleanly
printf '%s\n' '{"id":1,"method":"ping","params":{}}' | python -u api.py
# expected output:
# {"id": 1, "ok": true, "result": {"status": "ok"}}
```

## Running the tests

```bash
# Python: IPC dispatch, sample scanning, avg-loss synthesis
python -m unittest test_api_ipc

# Python: control-plane state machine, commands, run lock, device-swap helpers
python -m unittest test_train_control

# Python: latent-cache equivalence against the serial reference implementation
python trainer/test_warm_latent_cache.py          # mock VAE
python trainer/test_warm_latent_cache.py --real   # + real SDXL VAE smoke test

# Kotlin: Ranko unit tests (IPC request/response models)
cd ranko && ./gradlew :shared:jvmTest
```

All Python tests use `unittest` and are runnable under `pytest` as well.

## Building the desktop dashboard

```bash
cd ranko

# Run (dev)
./gradlew :desktopApp:run

# Hot-reload dev run (Compose)
./gradlew :desktopApp:hotRun --auto

# Package installers (Dmg / Msi / Deb)
./gradlew :desktopApp:packageDmg   # or packageMsi / packageDeb
```

Ranko finds the trainer repo automatically by walking up from the executable and the working directory looking for `api.py` or `trainer/config.toml`, so either run it from inside the repo, or place it so the trainer repo is an ancestor. See [Dashboard](dashboard.md).
