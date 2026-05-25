
# AXLTrainer

A lightweight, no-nonsense PyTorch/Accelerator script to train LoRA models with custom image datasets. Supports automatic bucketing, smart caption processing, dynamic network dropout, and periodic sample generation.

---

## 🚨 CRITICAL WARNING BEFORE YOU START

**DO NOT USE THE DEFAULT PATHS DEFINED IN `config.py`!**
The default values (such as `/home/acite/...`) are hardcoded for the author's local environment. They **will not work** on your machine and will cause `FileNotFoundError` or permission issues. 

Always review and modify `config.py` to match your own directory layout before starting!

---

## Key Features

* **Network Dropout Support:** Control model regularization and prevent overfitting directly via `network_dropout` in config.
* **Smart Sample Sampling:** Supports generating multiple evaluation samples (`sample_repeat`) per checkpoint. Set `sample_seed = 0` to automatically use unique random seeds for each sample.
* **Auto-Bucketing:** Dynamically groups images into optimal resolution buckets to handle variable aspect ratios.
* **Easy Cleanup:** Includes a specialized cleanup script to safely wipe logs, samples, or unwanted checkpoints.
* **Web UI Mode:** Optional Gradio interface (`ui.py`) for users who prefer a graphical control panel.

---

## Installation & Setup

1. **Install Dependencies:**
   Make sure you have PyTorch (with CUDA) installed, then run:
```bash
pip install -r requirements.txt
```

2. **Configure Accelerator:**
Initialize Hugging Face Accelerator configuration for your system:
```bash
accelerate config
```

---

## Configuration (`config.py`)

Open `config.py` and modify the core parameters. Remember to set your own paths:

* `img_dir`: Absolute path to your raw dataset folder.
* `output_dir`: Path where trained safetensors checkpoints and samples will be saved.
* `logging_dir`: Path for TensorBoard training logs.
* `network_dropout`: Set between `0.0` and `1.0` (e.g., `0.1`) to enable LoRA dropout.
* `sample_seed`: Set to `0` for randomized evaluation images, or a fixed number for reproducibility.
* `sample_repeat`: Number of images to generate at each validation step.

---

## How to Run

### Option 1: Command Line / Script

You can launch the training using the provided helper shell script:

```bash
bash start.sh
```

*(Make sure to update any custom paths inside `start.sh` if needed!)*

### Option 2: Graphical Web UI

If you prefer a visual interface:

```bash
streamlit run ui.py
```

---

## Project Cleanup (`clean.py`)

Training runs leave behind lots of TensorBoard logs and intermediate sample images. To reset your workspace, run the dedicated utility script:

```bash
python clean.py
```

**What it does:**

1. Automatically detects and deletes sample images belonging to the current `output_name`.
2. Automatically removes TensorBoard event logs for the current project.
3. **Asks for confirmation** before wiping heavy `.safetensors` model weight checkpoints. Safe by default!
