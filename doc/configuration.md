# Configuration (`config.toml`)

All training settings live in a single TOML file, **`config.toml` at the repo root**, read at startup by `trainer/config.py`. `trainer/main.py` accepts **no command-line arguments** — the TOML file (plus hardcoded fallbacks in `config.py`) is the only way to configure a run. Where both exist, **the TOML value always wins**.

You can edit this file by hand or with the Ranko dashboard's **Utils** tab, which validates values and preserves comments/formatting.

> ⚠️ The shipped `config.toml` contains the author's local paths (`/home/acite/...`, `/opt/models/...`). Replace them before running.

## Reading order

1. `config.toml` is parsed with `tomllib` and flattened into one dict (sections are just namespaces). The path is resolved **against the working directory**, so the trainer must be started from the repo root.
2. A `TrainConfig` dataclass is built from the flattened values; keys absent from the file fall back to hardcoded defaults in `trainer/config.py`.
3. Any key missing from both falls back to `None`.

## Sections and keys

### `[environment]` — paths

| Key | Example | Meaning |
| --- | --- | --- |
| `pretrained_model_name_or_path` | `"/opt/models/diffusers/waillu_170"` | SDXL base model. A diffusers directory, or a single-file checkpoint path (`from_single_file`). |
| `output_dir` | `"/home/acite/LLM/axltrainer/outputs"` | Root for run directories: each run writes `{output_dir}/{output_name}_{YYYYMMDD_HHMMSS}/…`. Created if missing. |
| `logging_dir` | `"/home/acite/LLM/axltrainer/logs"` | Root for TensorBoard logs: each run writes `{logging_dir}/{output_name}_{YYYYMMDD_HHMMSS}/`. Created if missing. |
| `train_data_dir` | `"/home/acite/LLM/Character/rein/"` | Dataset folder: images + same-named `.txt` captions. Optional `{stem}.mask.png` (white=train, black=ignore) enables masked loss; if missing, a transparent training image uses its alpha as the mask. |
| `output_name` | `"rein"` | Run name; prefix of every artifact path, of the run directory, and of the TensorBoard project. Sanitized to `[A-Za-z0-9._-]` in the run id and checkpoint filename. |

### `[model_spec]` — base-model family + checkpoint metadata

`base_model_version` is the **dispatch key**. The trainer looks it up in a catalog (`trainer/family.py`, mirrored in Ranko `ModelSpecCatalog`) and loads that family's pipeline / LoRA / loss path. The other three keys must match the catalog row for that version (hand-edits that drift are rejected at `TrainConfig` load). Ranko's Utils tab exposes a dropdown; changing it rewrites the three metadata strings.

| `base_model_version` | Trainable | `modelspec_architecture` | `modelspec_implementation` | `modelspec_sai_model_spec` |
| --- | --- | --- | --- | --- |
| `sdxl_base_v1-0` | yes | `stable-diffusion-xl-v1-base/lora` | `https://github.com/Stability-AI/generative-models` | `1.0.0` |
| `sd3.5-large` | **no** (UI slot only) | `stable-diffusion-v3-5-large/lora` | `https://github.com/Stability-AI/sd3.5` | `1.0.0` |

These also populate `modelspec.*` and `ss_base_model_version` on every `.safetensors`. `modelspec.prediction_type` is `v_prediction` when `[training].is_vpred` is true, otherwise `epsilon` (SDXL). Selecting `sd3.5-large` is valid config; `train_start` / `build_train_objects` fail before loading weights.

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
| `epoch` | `16` | Total epochs for this run. |
| `save_every_n_epochs` | `1` | **Defined but not used**; checkpoints are driven by `save_every_n_steps`. |
| `save_every_n_steps` | `100` | Save a LoRA checkpoint + generate samples every N steps. |
| `resume_lora_path` | `""` | Optional. kohya LoRA `.safetensors` (or a checkpoint directory holding exactly one) loaded into the UNet + both text encoders **before** training. Weights only: step/epoch counting still starts at 0 and the run gets its own timestamped directory, so earlier runs are never overwritten. `network_dim` / `network_alpha` must match the checkpoint. See [Training → Resuming from a checkpoint](training.md#resuming-from-a-checkpoint). |

`run_dir` is **not** a config key you should write: the trainer fills it in at runtime with the absolute run directory created for that run.

### `[network]` — LoRA network

| Key | Default | Notes |
| --- | --- | --- |
| `network_dim` | `48` | LoRA rank `r`. |
| `network_alpha` | `24` | LoRA alpha. Scale ≈ `alpha / dim` (0.5 here). |
| `network_dropout` | `0.25` | LoRA dropout (`0.0`–`1.0`), regularization / overfitting control. |
| `clip_skip` | `1` | Hidden-state index used from the text encoders. |
| `max_token_length` | `225` | Upper bound on prompt tokens. Captions longer than CLIP's 75 content tokens are split into `model_max_length − 2` chunks; a batch is padded only to the longest caption in that batch (not always to this cap). Sampling still uses as many chunks as the sample prompt needs, up to this value. |

### `[bucketing]` — aspect-ratio buckets

Each bucket holds about `train_resolution²` pixels and takes its aspect ratio from the image, so a
tall portrait gets a tall bucket instead of being squeezed into a short one and cropped. Whatever
mismatch is left between bucket and image is **letterboxed**: the whole image is fitted into the
bucket and the leftover bars carry loss weight 0, so they neither train nor count as content.

| Key | Default | Notes |
| --- | --- | --- |
| `enable_bucket` | `true` | Group images by aspect ratio instead of forcing one resolution. With it off, every image goes to a single `train_resolution × train_resolution` bucket. |
| `bucket_no_upscale` | `true` | A bucket side is never built larger than the image's own side, so fitting never upscales the image (the one exception: sources thinner than one `bucket_reso_steps`, which are floored at one step). |
| `train_resolution` | `1024` | Area anchor: each bucket aims at `train_resolution²` pixels (~1.05 MP at 1024) at whatever orientation the image has. |
| `bucket_reso_steps` | `128` | Bucket size granularity. **Keep at 128 on AMD ROCm** (see [Troubleshooting](troubleshooting.md#rocm-bucket-step-crash) — must be divisible by 16 to keep latent dims aligned). |
| `min_bucket_reso` | `384` | Smallest bucket side. This is the knob for extreme aspect ratios: lower it to give very tall art more resolution, rather than lowering `train_resolution`. |
| `max_bucket_reso` | `2688` | Largest bucket side. Keep `min_bucket_reso ≤ train_resolution ≤ max_bucket_reso`; otherwise every bucket lands on a clamp, loses the image's aspect ratio, and the trainer warns on stderr. |

### `[optimization]` — data & training optimizations

| Key | Default | Notes |
| --- | --- | --- |
| `cache_latents` | `true` | Pre-encode all images to latents before training. |
| `cache_latents_to_disk` | `true` | Persist encoded latents to `<train_data_dir>/.latents_cache/` (SHA1-keyed `.pt` files, atomic writes). Reused across runs. |
| `gradient_checkpointing_unet` | `true` | After PEFT wrap, call `enable_gradient_checkpointing()` on the UNet. Saves VRAM by recomputing activations in backward; turn off for faster steps if the GPU has headroom. |
| `gradient_checkpointing_te` | `true` | Same for both text encoders (`gradient_checkpointing_enable` / `enable_gradient_checkpointing`, plus `enable_input_require_grads` because embeddings stay frozen). |
| `shuffle_caption` | `true` | Shuffle caption tokens after `keep_tokens`, deterministically per epoch. |
| `keep_tokens` | `2` | Number of leading caption tokens kept in place when shuffling. |
| `caption_extension` | `".txt"` | Caption file extension. |
| `noise_offset` | `0.05` | Adds a small offset to the noise target (aids contrast/color variety). |
| `flush_memory_every_step` | `true` | After each training batch, `gc.collect` + HIP/CUDA `empty_cache`. Disable if step time is dominated by allocator churn. |

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
| `sample_prompts` | `"(rein_character:1.1), ..."` | Positive prompt used for validation samples. Also the fallback prompt of a `[[validation.samples]]` entry that omits `prompt`. |
| `sample_negative` | `"worst quality, low quality, ..."` | Negative prompt (fallback for `negative`). |
| `sample_width` / `sample_height` | `1280` / `720` | Sample image size (fallbacks for `width` / `height`). |
| `sample_steps` | `55` | Denoising steps (fallback for `steps`). |
| `sample_seed` | `0` | `0` = unique random seed per image (printed to the log); otherwise `seed + repeat_idx` (fallback for `seed`). |
| `sample_repeat` | `3` | Number of samples per checkpoint (fallback for `repeat`). |
| `guidance_scale` | `6.0` | CFG scale (fallback for `guidance_scale`). |

#### `[[validation.samples]]` — one block per prompt set

Any number of these blocks turns validation into a multi-prompt pass. Every set renders its own
`repeat` images at each sampling point, in block order.

```toml
[[validation.samples]]
name = "classroom"        # optional tab label; blank = the prompt's first tag
prompt = "1girl, classroom, ..."
negative = "worst quality, ..."
width = 1152
height = 768
steps = 35
guidance_scale = 6.0
seed = 1
repeat = 3
```

| Key | Required | Falls back to |
| --- | --- | --- |
| `prompt` | yes | `sample_prompts` |
| `negative`, `width`, `height`, `steps`, `guidance_scale`, `seed`, `repeat` | no | the `[validation]` scalar of the same shape |
| `name` | no | the prompt's first tag, else `Set N` |

- **No blocks at all** = exactly one set built from the scalars above, i.e. the single-prompt
  behaviour. `validation.sample_*` overrides (used by `fixes/` and `test/verify_mask_pipeline.py`)
  keep working in that case.
- **Ranges** (enforced by `trainer/config.py` and by the Utils form): `width`/`height` 64–4096,
  `steps` 1–150, `guidance_scale` 0–30, `seed` 0–2³²−1, `repeat` 1–32, `prompt` non-empty.
  A violation aborts the run at startup with the offending index (`validation.samples[2]: steps …`).
- **Seed**: inside a set the nth image uses `seed + n` (`0` = a fresh random seed per image). Two
  sets that share a seed therefore start from the same noise, so only the prompt differs.
- **File names**: `{output_name}_{step:06d}_p{set}_{repeat}.png`, with `set` counting from 0. The
  two-number form of older runs (`…_{step}_{repeat}.png`) is still parsed, as set 0.
- Sampling time scales with `Σ repeat`; each set's images are rendered sequentially.
- The run's configuration is recorded in TensorBoard's **HParams** tab. `add_hparams` only
  accepts int/float/str/bool/tensor values, so a list-valued key such as `samples` is written
  as a JSON string (`tracker_hparams()` in `trainer/config.py`) rather than passed through raw.

### `[bookkeeping]`

Intentionally empty. The Python side treats missing keys as `None`; it exists for `ss_*` metadata fields that aren't always present (e.g. `ss_session_id`, `ss_training_comment`, model hashes, dataset dirs, bucket info).

## Derived values worth knowing

- **Effective batch size** = `train_batch_size × gradient_accumulation_steps`.
- **LoRA scale** = `network_alpha / network_dim` (0.5 with the defaults).
- **Steps per epoch** = `⌈len(dataloader) / gradient_accumulation_steps⌉`; **total steps** = steps-per-epoch × `epoch`.
- **UNet LR vs TE LR**: the UNet uses Schedule-Free AdamW (its own warmup via `unet_warmup_steps`); the text encoders use plain AdamW with a warmup + `lr_scheduler` decay. The dashboard's "TE LR" card reflects the scheduled TE LR.
- **Bucket and pad**: `pick_bucket_size` derives the bucket from the image's aspect ratio and the `train_resolution²` budget; `fit_geometry` then places the image inside it as `{fit_w}×{fit_h}` centred at an offset, and the rest of the bucket is pad. The trainer prints the bucket list and the mean pad for a run (`Letterbox: n/m samples padded, mean x%`).

## Editing from the GUI

The Ranko **Utils** tab is a validated form over exactly these sections/keys:

- Path fields have a Browse button (OS file dialog: the desktop portal picker on Linux).
- `mixed_precision` and `lr_scheduler` are segmented buttons / chips.
- Booleans are switches.
- Inline hints show derived values (effective batch, LoRA scale, bucket-step divisibility, sample aspect ratio).
- The **Validation** section edits `[[validation.samples]]` as horizontal tabs: one chip per set
  (its label, a warning icon when the set has an invalid field), `+` clones the open set, and the
  `×` on a chip deletes that set after a confirmation. The last set cannot be deleted. Saving writes
  the `[validation]` scalars from the first set plus one fully explicit block per tab (a blank label
  is left out), so the file never carries two contradictory prompts.
- Save runs full-form validation; on error it jumps to the first section with an invalid field (and to the offending set's tab). The writer is a line-preserving TOML patcher, so comments and formatting survive edits — except inside the replaced `[[validation.samples]]` blocks.
