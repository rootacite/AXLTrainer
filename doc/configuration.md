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
| `train_data_dir` | `"/home/acite/LLM/Character/rein/"` | Dataset folder: images + same-named `.txt` captions. Optional `{stem}.mask.png` (white=train, black=ignore) enables masked loss; if missing, a transparent training image uses its alpha as the mask. Since `[[environment.train_data]]` exists, this key **mirrors that list's first entry** and is the single folder the paths that expect one read (checkpoint metadata, the Utils → Environment tag button, and a config that has no list). The trainer drops it as soon as the list has an entry. |
| `output_name` | `"rein"` | Run name; prefix of every artifact path, of the run directory, and of the TensorBoard project. It has to be one filename-safe token — letters and digits (any script), `-`, `_` and `.` — because the run id is `{output_name}_{YYYYMMDD_HHMMSS}` and the sample/checkpoint directories are named after it. A space, a slash or any other character the sanitizer would rewrite is refused by the Utils form, by `config_save`, and by the trainer at startup (`TrainConfig`), so a run can always be found again from its own id. |
| `amdfq` | `"none"` | Allocation patch for the next Train start: `"none"`, `"tail"` (`amdfq-tail-rs`), or `"vmm"` (`amdfq-vmm-rs`). Ranko Utils → **ROCm**. `start_train.sh` `LD_PRELOAD`s the matching release `.so`; a missing library fails the start instead of running unpatched. On RDNA 4, Tail or VMM is strongly preferred; VMM uses less VRAM system-wide because it bypasses ROCr's Memory Pool. |
| `amdfq_vram_reserve_gib` | `0.0` | GiB of driver-reported free VRAM (`/sys/class/drm/cardN/device/mem_info_vram_total − mem_info_vram_used`) the VMM hook will not consume. This is the amdgpu counter, not `hipMemGetInfo` (which does not see the compositor or RADV). The hooked `hipMemGetInfo` reports that remaining minus this floor so the caching allocator sees other clients. Before `hipMemCreate`, the hook compares the same counter against this floor plus the rounded request; if the allocation would leave less, `hipMalloc` returns OOM instead of creating or forwarding. **Optional**: this was the workaround for the driver handing a process's frames to another one on eviction, which the 2026-09 kernel fixed — leave it at `0` (`0` is also the default and means off) unless you run an older kernel. Ranko Utils → **ROCm**. Takes effect on the next Train start. Ignored unless `amdfq = "vmm"`. |
| `amdfq_va_never_reuse` | `false` | **Optional** VMM-hook switch, ignored unless `amdfq = "vmm"`. `false` (default): a `hipFree` gives its range's GPU virtual address back to the driver, so the address is reusable and the Dashboard's VA bar only shows what the hook holds right now. `true`: the pre-fix behaviour, kept for an older kernel — a range that was mapped once keeps its VA for the process lifetime and is never mapped again, so that bar only grows. The workaround existed because tearing a mapping down used to leave the compute VM's TLB stale, which made same-address reuse unsafe; the 2026-09 kernel fixed that. Ranko Utils → **ROCm**. Takes effect on the next Train start. |
| `amdfq_pool_mib` | `64` | **Optional** VMM-hook allocation pool, in MiB; ignored unless `amdfq = "vmm"`. `0` is off: every `hipMalloc` gets its own `hipMemCreate` + reserve + map. A value in `16`–`512` turns on one `hipMemCreate` per pool of that size: a request of **at most half the pool size** is carved out of a pool that already exists (no driver call at all), and anything larger keeps its own handle. Several pools may exist at once, a new one is built when no existing pool has room, and a pool is released — handle, mapping and (unless `amdfq_va_never_reuse`) its VA — once the upper layer has freed every block carved out of it. Values outside `16`–`512` are clamped by the hook, and a pool that cannot be built completely falls back to the per-request route. Ranko Utils → **ROCm**. Takes effect on the next Train start. The hook's own default, with `AMDFQ_POOL_SIZE` unset, is off; this row is what the shipped config asks for, and it is the value the pool-size sweep measured as 0.6 % *slower* than no pool at all on this repo's own workload (`test/bench_alloc_pool.py`).<br><br>**A pool is not better when bigger.** It is committed VRAM the driver cannot hand to anything else until its last block is freed, so an oversized pool risks OOM and fragmenting the card, and the savings fall off — the small-request traffic it removes is a bounded share of each step. One more consequence of sharing: blocks inside a pool are neighbours, so an over-read past a block still lands in mapped memory but an over-*write* past its end can reach another live block, where a solo allocation would have hit the throwaway pad behind it. |

### `[[environment.train_data]]` — dataset folders and their repeats

The datasets a run trains on: one block per folder, `repeat` = how often that folder's images are
drawn inside a single epoch.

```toml
train_data_dir = "/home/acite/LLM/Character/LLLJ/"   # mirrors the first block
output_name = "lllj"

[[environment.train_data]]
path = "/home/acite/LLM/Character/LLLJ/"
repeat = 3

[[environment.train_data]]
path = "/home/acite/Pictures/05_babara"
repeat = 1
```

| Key | Default | Meaning |
| --- | --- | --- |
| `path` | — (required) | Dataset folder, same shape as `train_data_dir` (images + same-named `.txt` captions, optional `{stem}.mask.png`). |
| `repeat` | `1` | How many times this folder is drawn inside one epoch, `1`–`512`. A missing path and a repeat outside the range are refused at startup with `train_data[<i>]: …`; `train_data` must be an array of tables. |

- **No block at all** means one entry built from the `train_data_dir` scalar with `repeat = 1`, which is
  what every config written before this list keeps doing.
- The repeat reaches training in one place: the folder's images get their record index **repeated** in the
  bucket list the batch sampler draws from. So one epoch is `images × repeat` samples, and `len(dataloader)`
  — from which `steps_per_epoch`, the total step count, the progress bars and the TE cosine schedule are all
  derived — grows with the sum. `len(dataset)` itself stays the number of unique images (the latent warm-up
  walks that index range, and must not reload one latent per repeat).
- A batch is still one aspect-ratio bucket's worth of images, so a repeated image can land in the same batch
  as its own copy (with its own noise draw and its own weight in the batch mean). A small folder with a large
  repeat can therefore fill a batch with near-copies of the same few images.
- Each folder keeps **its own** `<folder>/.latents_cache/`: the cache key hashes the absolute image path
  and the fit geometry, so two folders never collide and a folder's cache is re-encoded on its own.

Ranko's Utils → Environment section is the editor (a row per folder with its repeat), and the Images,
Statistics and Tag dataset surfaces act on the folder selected there. `[[validation.samples]]`'s own
`repeat` is unrelated: that one is how many images a sample point renders.

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
| `save_every_n_steps` | `100` | Steps between LoRA checkpoints (`0` writes none). The value here is what a run **starts** with; the Dashboard's Training Control card can retune it for the run in progress (a change restarts the countdown from the step that adopts it), and a run keeps its own cadence — the file is never rewritten by a live change. |
| `sampling_enabled` | `true` | Whether a checkpoint save also renders the `[[validation.samples]]` images. `false` keeps the same cadence but writes checkpoints only (the expensive part of a save point is the sampling, not the checkpoint). Changeable mid-run like the cadence; a checkpoint with no samples can be rendered later, one pass per checkpoint, from the Dashboard. |
| `resume_lora_path` | `""` | Optional. kohya LoRA `.safetensors` (or a checkpoint directory holding exactly one) loaded into the UNet + both text encoders **before** training. Weights only: step/epoch counting still starts at 0 and the run gets its own timestamped directory, so earlier runs are never overwritten. `network_type` and `network_dim` / `network_alpha` must match the checkpoint. See [Training → Resuming from a checkpoint](training.md#resuming-from-a-checkpoint). |

`run_dir` is **not** a config key you should write: the trainer fills it in at runtime with the absolute run directory created for that run.

### `[network]` — LoRA network

| Key | Default | Notes |
| --- | --- | --- |
| `network_type` | `"standard"` | `"standard"` (attention LoRA) or `"locon"` (Kohya LoRA-C3Lier on the UNet). See [LoCon](locon.md). |
| `network_dim` | `48` | LoRA rank `r`. |
| `network_alpha` | `24` | LoRA alpha. Scale ≈ `alpha / dim` (0.5 here). |
| `network_dropout` | `0.25` | LoRA dropout (`0.0`–`1.0`), regularization / overfitting control. |
| `conv_dim` | `0` | Locon only: rank of Conv2d 3×3 (and ResNet 1×1 shortcuts). May differ from `network_dim`. `0` with `network_type = "locon"` is invalid. |
| `conv_alpha` | `0` | Locon only: conv alpha. Scale is `conv_alpha / conv_dim`. May differ from `network_alpha`. |
| `clip_skip` | `1` | Hidden-state index used from the text encoders. |
| `max_token_length` | `225` | Upper bound on prompt tokens. Captions longer than CLIP's 75 content tokens are split into `model_max_length − 2` chunks; a batch is padded only to the longest caption in that batch (not always to this cap). Sampling still uses as many chunks as the sample prompt needs, up to this value. |

Standard wraps UNet `to_q` / `to_k` / `to_v` / `to_out.0` and TE `q_proj` / `k_proj` / `v_proj` / `out_proj`. Locon adds UNet Linear extras (`proj_in` / `proj_out` / `ff.net.0.proj` / `ff.net.2` / `time_emb_proj`) at `network_dim`, Conv2d extras (`conv1` / `conv2` / `conv_shortcut` / `conv`) at `conv_dim`, and TE MLP `fc1` / `fc2`. `conv_in` and `conv_out` stay unwrapped. Checkpoints write `ss_network_type` (`standard` or `locon`) and, for locon, `ss_network_args = "conv_dim=N conv_alpha=M"`.

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
| `cache_latents_to_disk` | `true` | Persist encoded latents to `<train_data_dir>/.latents_cache/` (SHA1-keyed `.pt` files, atomic writes). Reused across runs while the file holds the keyed bucket's latent; a file that does not is re-encoded. |
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
  behaviour. `validation.sample_*` overrides (used by `test/verify_mask_pipeline.py`) keep working
  in that case.
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
- `mixed_precision`, `network_type`, and `lr_scheduler` are segmented buttons / chips.
- Booleans are switches.
- Inline hints show derived values (effective batch, LoRA scale, bucket-step divisibility, sample aspect ratio).
- The **Validation** section edits `[[validation.samples]]` as horizontal tabs: one chip per set
  (its label, a warning icon when the set has an invalid field), `+` clones the open set, and the
  `×` on a chip deletes that set after a confirmation. The last set cannot be deleted. Saving writes
  the `[validation]` scalars from the first set plus one fully explicit block per tab (a blank label
  is left out), so the file never carries two contradictory prompts.
- The **Training** section carries the **Sampling** switch (`[training].sampling_enabled`) beside
  `save_every_n_steps`. Those two are the *starting* values: the Dashboard's Training Control card
  can change both for the run in progress without touching this file.
- Save runs full-form validation; on error it jumps to the first section with an invalid field (and to the offending set's tab). The writer is a line-preserving TOML patcher, so comments and formatting survive edits — except inside the replaced `[[validation.samples]]` blocks.
