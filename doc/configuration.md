# Configuration (`trainer/config.toml`)

All training settings live in a single TOML file, **`trainer/config.toml`**, read at startup by `trainer/config.py`. `trainer/main.py` accepts **no command-line arguments** — the TOML file (plus hardcoded fallbacks in `config.py`) is the only way to configure a run. Where both exist, **the TOML value always wins**.

You can edit this file by hand or with the Ranko dashboard's **Utils** tab, which validates values and preserves comments/formatting.

> ⚠️ The shipped `config.toml` contains the author's local paths (`/home/acite/...`, `/opt/models/...`). Replace them before running.

## Reading order

1. `trainer/config.toml` is parsed with `tomllib` and flattened into one dict (sections are just namespaces).
2. A `TrainConfig` dataclass is built from the flattened values; keys absent from the file fall back to hardcoded defaults in `trainer/config.py`.
3. Any key missing from both falls back to `None`.

## Sections and keys

### `[environment]` — paths

| Key | Example | Meaning |
| --- | --- | --- |
| `pretrained_model_name_or_path` | `"/opt/models/diffusers/waillu_170"` | SDXL base model. A diffusers directory, or a single-file checkpoint path (`from_single_file`). |
| `output_dir` | `"/home/acite/LLM/axltrainer/outputs"` | Where LoRA checkpoints and sample images are saved. Created if missing. |
| `logging_dir` | `"/home/acite/LLM/axltrainer/logs"` | Where TensorBoard event files go (`{logging_dir}/{output_name}/`). Created if missing. |
| `train_data_dir` | `"/home/acite/LLM/Character/rein/"` | Dataset folder: images + same-named `.txt` captions. |
| `output_name` | `"rein"` | Run name; used in every artifact path and as the TensorBoard project name. Sanitized to `[A-Za-z0-9._-]`. |

### `[model_spec]` — checkpoint metadata (written into every `.safetensors`)

| Key | Meaning |
| --- | --- |
| `base_model_version` | e.g. `"sdxl_base_v1-0"` |
| `modelspec_architecture` | e.g. `"stable-diffusion-xl-v1-base/lora"` |
| `modelspec_implementation` | e.g. `"https://github.com/Stability-AI/generative-models"` |
| `modelspec_sai_model_spec` | e.g. `"1.0.0"` |

These populate the `modelspec.*` metadata fields so tools like the ComfyUI model manager can identify the checkpoint.

### `[training]` — core training settings

| Key | Default (file) | Notes |
| --- | --- | --- |
| `is_vpred` | `false` | If `true`, the DDIM scheduler uses `v_prediction` + `rescale_betas_zero_snr`; otherwise `epsilon` prediction. |
| `min_snr_gamma` | `5.0` | **Defined but not used in the training math** (kept for metadata compatibility). |
| `seed` | `1145141919` | Global training seed. |
| `mixed_precision` | `"bf16"` | `"bf16"` / `"fp16"` / `"no"`. |
| `train_batch_size` | `4` | Per-device batch size. |
| `gradient_accumulation_steps` | `1` | Effective batch = `train_batch_size × gradient_accumulation_steps`. |
| `learning_rate` | `1.0` | **Metadata only** (`ss_learning_rate`). Actual LRs come from `[unet_optimizer]` / `[te_optimizer]`. |
| `lr_scheduler` | `"cosine"` | One of: `cosine`, `cosine_with_restarts`, `linear`, `constant`, `constant_with_warmup`, `polynomial`, `adafactor`. |
| `lr_warmup_steps` | `100` | Warmup applied to the text-encoder scheduler (Schedule-Free AdamW handles its own warmup via `unet_warmup_steps`). |
| `max_grad_norm` | `1.0` | UNet gradient clipping. |
| `epoch` | `16` | Total epochs. |
| `save_every_n_epochs` | `1` | **Defined but not used**; checkpoints are driven by `save_every_n_steps`. |
| `save_every_n_steps` | `100` | Save a LoRA checkpoint + generate samples every N steps. |

### `[network]` — LoRA network

| Key | Default | Notes |
| --- | --- | --- |
| `network_dim` | `48` | LoRA rank `r`. |
| `network_alpha` | `24` | LoRA alpha. Scale ≈ `alpha / dim` (0.5 here). |
| `network_dropout` | `0.25` | LoRA dropout (`0.0`–`1.0`), regularization / overfitting control. |
| `clip_skip` | `1` | Hidden-state index used from the text encoders. |
| `max_token_length` | `225` | Prompt budget; prompts are chunked into `model_max_length − 2` pieces and padded, so values above 77 enable long prompts. |

### `[bucketing]` — aspect-ratio buckets

| Key | Default | Notes |
| --- | --- | --- |
| `enable_bucket` | `true` | Group images by aspect ratio instead of forcing one resolution. |
| `bucket_no_upscale` | `true` | If true, images smaller than the bucket are used at native size (no upscaling). |
| `train_resolution` | `1024` | Base resolution; buckets are derived around it. |
| `bucket_reso_steps` | `128` | Bucket size granularity. **Keep at 128 on AMD ROCm** (see [Troubleshooting](troubleshooting.md#rocm-bucket-step-crash) — must be divisible by 16 to keep latent dims aligned). |
| `min_bucket_reso` | `768` | Smallest bucket side. |
| `max_bucket_reso` | `1280` | Largest bucket side. |

### `[optimization]` — data & training optimizations

| Key | Default | Notes |
| --- | --- | --- |
| `cache_latents` | `true` | Pre-encode all images to latents before training. |
| `cache_latents_to_disk` | `true` | Persist encoded latents to `<train_data_dir>/.latents_cache/` (SHA1-keyed `.pt` files, atomic writes). Reused across runs. |
| `shuffle_caption` | `true` | Shuffle caption tokens after `keep_tokens`, deterministically per epoch. |
| `keep_tokens` | `2` | Number of leading caption tokens kept in place when shuffling. |
| `caption_extension` | `".txt"` | Caption file extension. |
| `noise_offset` | `0.05` | Adds a small offset to the noise target (aids contrast/color variety). |

### `[unet_optimizer]` — UNet optimizer (Schedule-Free AdamW)

| Key | Default | Notes |
| --- | --- | --- |
| `unet_learning_rate` | `5e-5` | **The actual UNet learning rate.** |
| `unet_weight_decay` | `0.01` | |
| `unet_betas_1` | `0.9` | |
| `unet_betas_2` | `0.99` | |
| `unet_eps` | `1e-8` | |
| `unet_warmup_steps` | `100` | Schedule-Free warmup (no separate LR scheduler is needed for the UNet). |

### `[te_optimizer]` — text-encoder optimizer (plain AdamW)

| Key | Default | Notes |
| --- | --- | --- |
| `te_learning_rate` | `5e-6` | **The actual text-encoder learning rate** (usually 10× lower than UNet). |
| `te_weight_decay` | `0.01` | |
| `te_betas_1` | `0.9` | |
| `te_betas_2` | `0.99` | |
| `te_max_grad_norm` | `0.3` | Gradient clipping for the text encoders. |

### `[infrastructure]` — data loading

| Key | Default | Notes |
| --- | --- | --- |
| `max_data_loader_n_workers` | `20` | DataLoader worker count. |
| `persistent_workers` | `true` | Keep workers alive between epochs. |

### `[validation]` — sample generation

| Key | Default | Notes |
| --- | --- | --- |
| `sample_prompts` | `"(rein_character:1.1), ..."` | Positive prompt used for validation samples. |
| `sample_negative` | `"worst quality, low quality, ..."` | Negative prompt. |
| `sample_width` / `sample_height` | `1280` / `720` | Sample image size. |
| `sample_steps` | `55` | Denoising steps. |
| `sample_seed` | `0` | `0` = unique random seed per repeat (printed to the log); otherwise `seed + repeat_idx`. |
| `sample_repeat` | `3` | Number of samples per checkpoint. |
| `guidance_scale` | `6.0` | CFG scale. |

### `[bookkeeping]`

Intentionally empty. The Python side treats missing keys as `None`; it exists for `ss_*` metadata fields that aren't always present (e.g. `ss_session_id`, `ss_training_comment`, model hashes, dataset dirs, bucket info).

## Derived values worth knowing

- **Effective batch size** = `train_batch_size × gradient_accumulation_steps`.
- **LoRA scale** = `network_alpha / network_dim` (0.5 with the defaults).
- **Steps per epoch** = `⌈len(dataloader) / gradient_accumulation_steps⌉`; **total steps** = steps-per-epoch × `epoch`.
- **UNet LR vs TE LR**: the UNet uses Schedule-Free AdamW (its own warmup via `unet_warmup_steps`); the text encoders use plain AdamW with a warmup + `lr_scheduler` decay. The dashboard's "TE LR" card reflects the scheduled TE LR.

## Editing from the GUI

The Ranko **Utils** tab is a validated form over exactly these sections/keys:

- Path fields have a Browse button (JVM file chooser).
- `mixed_precision` and `lr_scheduler` are segmented buttons / chips.
- Booleans are switches.
- Inline hints show derived values (effective batch, LoRA scale, bucket-step divisibility, sample aspect ratio).
- Save runs full-form validation; on error it jumps to the first section with an invalid field. The writer is a line-preserving TOML patcher, so comments and formatting survive edits.
