# LoCon integration (decided plan)

Recorded 2026-09-27. Contract for **Kohya LoCon** (LoRA-C3Lier) on the SDXL trainer. Human-facing
keys live in [configuration.md](configuration.md) and [training.md](training.md). This file keeps
the *why*, the layer list, the ComfyUI key findings, and the resume rules.

## Status

| Item | State |
| --- | --- |
| Trainable network | PEFT `LoraConfig` on SDXL UNet + both TEs |
| UNet targets | Standard: `to_q`, `to_k`, `to_v`, `to_out.0`. Locon: those plus C3Lier extras in §2 |
| TE targets | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Metadata | `ss_network_module = "networks.lora"`; `ss_network_type` = `standard` \| `locon` |
| LoCon / `conv_dim` | implemented (`conv_dim` may differ from `network_dim`; second PEFT adapter) |
| Locon TE MLP `fc1`/`fc2` | implemented |
| LyCORIS | absent |

Default stays attention LoRA (`network_type = "standard"`). LoCon is an opt-in on the same PEFT +
kohya `.safetensors` path.

## Base-model pairing (field note)

Reported by the maintainer, 2026-10-02, and not measured in this repo: train the LoCon with
**Illustrious XL v2.0 stable** as the base model and generate with **WAI** as the base model, and the
result is strikingly good — more so the more deliberately the LoCon is allowed to overfit.

## 1. Which LoCon

kohya_ss GUI exposes two things that share the word "LoCon":

| GUI label | `network_module` | Mechanism |
| --- | --- | --- |
| **Kohya LoCon** | `networks.lora` | Same module as Standard. `conv_dim` / `conv_alpha` add Conv2d 3×3 (and ResNet 1×1 shortcuts). kohya docs: **LoRA-C3Lier**. |
| **LyCORIS/LoCon** | `lycoris.kohya` | Third-party LyCORIS, `algo=locon`, optional tucker / preset / `train_norm`. |

This repo takes **Kohya LoCon**. Checkpoints already speak kohya `lora_down` / `lora_up` / `alpha`
with `ss_network_module = "networks.lora"`. PEFT 0.20.0 already wraps `Conv2d` with the same 4D
shapes kohya uses (`lora_A` = `(r, in, k, k)`, `lora_B` = `(out, r, 1, 1)`). LyCORIS is a new
dependency and a different key layout; leave it until asked.

Sources: `/tmp/kohya_ss/kohya_gui/lora_gui.py` (Kohya LoCon → `networks.lora` + `conv_dim`;
LyCORIS/LoCon → `lycoris.kohya` + `algo=locon`); kohya-ss/sd-scripts `networks/lora.py` at
submodule commit `6721028c` (`UNET_TARGET_REPLACE_MODULE` / `UNET_TARGET_REPLACE_MODULE_CONV2D_3X3`,
`conv_dim` gate); `/tmp/kohya_ss/docs/train_network_README-ja.md` (LierLa vs C3Lier).

## 2. Layer coverage

kohya Standard (LoRA-LierLa) is every Linear and Conv2d 1×1 under `Transformer2DModel` plus
`CLIPAttention` / `CLIPMLP`. Kohya LoCon adds parent types `ResnetBlock2D`, `Downsample2D`,
`Upsample2D`. SDXL uses `use_linear_projection=True`, so Transformer 1×1 convs are Linear
(`proj_in` / `proj_out`); the 1×1 tensors LoCon actually adds are ResNet `conv_shortcut`.

AxlTrainer Standard is narrower than kohya LierLa: attention Q/K/V/Out only.

### v1 locon (locked)

When `network_type = "locon"`, wrap **kohya C3Lier on the UNet** and **LierLa MLP on both TEs**:

**UNet (PEFT `target_modules` suffixes)**

- already Standard: `to_q`, `to_k`, `to_v`, `to_out.0`
- kohya Transformer extras: `proj_in`, `proj_out`, `ff.net.0.proj`, `ff.net.2`
- ResNet / sample: `conv1`, `conv2`, `conv_shortcut`, `time_emb_proj`, `conv`

`conv` matches `downsamplers.0.conv` / `upsamplers.0.conv`. UNet `conv_in` / `conv_out` stay
unwrapped (kohya also leaves them alone).

**TE:** locon wraps attention `q_proj`, `k_proj`, `v_proj`, `out_proj` **and** CLIPMLP `fc1` /
`fc2`. Standard keeps attention only.

**Standard** (`network_type = "standard"`) keeps today's four UNet + four TE names. Locon's extra
layers appear only when the type is `locon`.

### Size (measured)

Probe: meta-device UNet + CLIP configs from `/opt/models/diffusers/waillu_170`, rank 16, axl env.
Script: `/tmp/axl_locon_probe.py`.

| Layout | Modules | LoRA params | bf16 weights |
| --- | --- | --- | --- |
| Current AxlTrainer | 736 | 29.6 M | 56.5 MiB |
| kohya Standard (LierLa) | 986 | 57.0 M | 108.7 MiB |
| Kohya LoCon (C3Lier, TE = kohya Standard) | 1052 | 63.6 M | 121.4 MiB |

Locon TE now includes CLIPMLP `fc1` / `fc2` (88 modules on SDXL), so the full locon layout matches
the 1052-module C3Lier+LierLa-TE row above. LoCon UNet is 788 modules / 49.2 M params vs current
UNet 560 / 23.2 M. Extra over kohya Standard UNet is 66 modules (38× 3×3, 11× 1×1 shortcut, 17×
`time_emb_proj`). Adapter weights are small next to activations; 16 GB step-time VRAM for locon is
**unmeasured**.

## 3. Configuration

New keys live in `[network]` (flat `TrainConfig` names, same as every other hyperparameter).

```toml
[network]
network_type = "standard"   # or "locon"
network_dim = 16
network_alpha = 8
network_dropout = 0.08
conv_dim = 8                # locon only; 0 = do not wrap 3×3
conv_alpha = 8
clip_skip = 1
max_token_length = 225
```

| Key | Default | Rules |
| --- | --- | --- |
| `network_type` | `"standard"` | `"standard"` or `"locon"`. |
| `conv_dim` | `0` | Locon: `≥ 1` to wrap 3×3. `0` with `network_type = "locon"` is invalid in v1. |
| `conv_alpha` | `0` | Locon: `≥ 1`. Scale for conv is `conv_alpha / conv_dim`. |

**Conv rank:** locon UNet uses two PEFT adapters so `conv_dim` / `conv_alpha` may differ from
`network_dim` / `network_alpha`. Linear extras (including `time_emb_proj`) take `network_dim`;
Conv2d (`conv1`, `conv2`, `conv_shortcut`, downsample/upsample `conv`) take `conv_dim`. Both
adapters are active in the forward (`base_model.set_adapter(["default", "conv"])`).

Four-place add (or the GUI drifts): `config.toml`, `TrainConfig`, Ranko `NetworkConfig` +
`TrainingConfigForm` + `NetworkFields`, [configuration.md](configuration.md). Kotlin defaults so an
old TOML still parses.

## 4. Key names (ComfyUI)

Export stays kohya:

- `lora_{unet,te1,te2}_<flattened>.lora_down.weight`
- `lora_{unet,te1,te2}_<flattened>.lora_up.weight`
- `lora_{unet,te1,te2}_<flattened>.alpha`
- tensors bf16

ComfyUI measured: tree `/home/acite/LLM/comfyui` at `61de2e98` (`v0.35.0-29`). Loader:
`comfy/sd.py` `load_lora_for_models` → `model_lora_keys_unet` / `_clip` → `load_lora` →
`LoRAAdapter`. Env `comfy_ui`.

### UNet

`model_lora_keys_unet` registers **both** dialects:

- live `diffusion_model.*` → `lora_unet_` + **ldm** path with `.` → `_`
- `unet_to_diffusers(config)` → `lora_unet_` + **diffusers** path with `.` → `_`

ResNet map (`comfy/utils.py` `UNET_MAP_RESNET`): `conv1` ↔ `in_layers.2`, `conv2` ↔ `out_layers.3`,
`time_emb_proj` ↔ `emb_layers.1`, `conv_shortcut` ↔ `skip_connection`. Downsample conv ↔
`input_blocks.*.0.op`; upsample conv ↔ `output_blocks.*.*.conv`.

Probe `/tmp/axl_comfy_lora_keys.py` against the SDXL config in `model_detection.py:1417-1421`:
796 LoRA-relevant stems, **0 misses** for current Axl remap, raw diffusers flatten, and full ldm
flatten. Current `_UNET_PATH_MAP` remaps 8/74 locon conv stems (some `conv_shortcut`, plus a
`down_blocks.2.downsamplers` / `up_blocks.2.upsamplers` pair). Real SDXL downsamples sit on blocks
0 and 1; those stay diffusers names today and still hit ComfyUI.

**Locked:** locon (and the Standard remap cleanup that goes with it) writes **ldm** names using the
ComfyUI / `unet_to_diffusers` table, so a file is one dialect and matches kohya C3Lier. ComfyUI
loads that dialect. A1111 additional-networks and kohya merge/resize were not measured; ldm names
are the ones those tools are written for.

`_convert_peft_to_kohya_bf16` already copies 4D tensors. Completing `_remap_unet_path` is enough
for conv keys; `build_kohya_to_peft_map` follows the live adapter.

Do not emit `lora_mid` (LyCORIS tucker). ComfyUI `LoRAAdapter.calculate_weight` flatten-mm on a
4D Kohya pair matched `up.flatten(1) @ down.flatten(1) * (alpha/rank)` with max abs diff 0
(`/tmp/axl_comfy_conv_apply.py`).

### TE

ComfyUI SDXL CLIP aliases are `lora_te{1,2}_text_model_encoder_layers_{b}_{self_attn_*|mlp_fc*}`.

A real AxlTrainer file
(`…/konomi_20260927_080807/konomi_s000300/konomi.safetensors`, 736 modules):

| Prefix | Count | ComfyUI alias |
| --- | --- | --- |
| `lora_unet_input_blocks_*` / `middle_block_*` / `output_blocks_*` | 560 | all hit |
| `lora_te2_text_model_encoder_layers_*` | 128 | all hit |
| `lora_te1_encoder_layers_*` | 48 | **0 hit** |

TE1 drops `text_model_` because transformers 5.16.1 `CLIPTextModel` children are
`embeddings` / `encoder` / `final_layer_norm`; `CLIPTextModelWithProjection` still nests under
`text_model`. `_remap_te_path` only rewrites `text_model.encoder.layers.`.

**Locked, same pass as the UNet remap:**

- **Save** both TEs as `lora_te{1,2}_text_model_encoder_layers_*` (insert `text_model_` when the
  PEFT path starts at `encoder.layers.`).
- **Load** both spellings for TE1 (`text_model_encoder_layers_*` and `encoder_layers_*`) so existing
  konomi-style files still resume.

## 5. Metadata

`ss_network_module` is `networks.lora` for kohya Standard **and** Kohya LoCon. It cannot be the
discriminator. Axl writes its own field.

| Key | Standard | Locon |
| --- | --- | --- |
| `ss_network_type` | `standard` | `locon` |
| `ss_network_module` | `networks.lora` | `networks.lora` |
| `ss_network_args` | omitted | `conv_dim=N conv_alpha=M` |
| `ss_network_dim` / `ss_network_alpha` | linear rank / alpha | same (v1 = conv rank / alpha) |

`ss_network_type` is what Axl resume trusts. `ss_network_args` is for kohya-shaped readers.

Write the type from the layout **actually applied** (`apply_lora`), not from a stale config field.

## 6. Resume mismatch — hard refuse

Today `SdxlFamily.load_lora` maps whatever kohya keys hit the current adapter and **warns** on the
rest. A locon file into a Standard wrap would load attention and skip conv, then train with random
conv adapters. That is the failure this rule exists to stop.

Refuse **before** building the kohya→PEFT map, in `SdxlFamily.load_lora` (also used by
`trainer/generate_sample.py`):

1. Read `ss_network_type`. Missing → `standard` (files written before the field).
2. If the type is missing and `ss_network_args` has `conv_dim > 0` → `locon` (kohya LoCon files).
3. Compare to `cfg.network_type` (default `standard`). On mismatch raise `ValueError` naming both
   types and the file path.
4. Locon wrap: refuse a file with no `mlp_fc1` / `mlp_fc2` keys, and refuse `conv_dim` that does
   not match `ss_network_args` when that field is present.
5. Then the existing rank-shape check (`network_dim` mismatch already raises).

`api.py` `train_start` runs the same comparison via `read_lora_metadata` before `setsid`, so a
bad `resume_lora_path` dies in the dashboard without touching the GPU.

`generate_sample` already rebuilds `TrainConfig` from checkpoint `ss_network_dim` / `ss_network_alpha`.
It must also take `ss_network_type` / `ss_network_args` so `apply_lora` builds the layout the file
was trained with. A Standard process asked to sample a locon file raises the same mismatch error.

Alpha mismatch stays a warning (current behaviour). Type mismatch never warns-and-continues.

## 7. Where it attaches

### Backend

| Place | Change |
| --- | --- |
| `config.toml` `[network]` | `network_type`, `conv_dim`, `conv_alpha` |
| `trainer/config.py` `TrainConfig` | same keys; locon requires `conv_dim` / `conv_alpha` ≥ 1 (ranks may differ) |
| `trainer/family_sdxl.py` `apply_lora` | **only place that changes the network.** Locon: Linear adapter `default` + Conv adapter `conv`; TE + `fc1`/`fc2`. |
| `trainer/family_sdxl.py` `_UNET_PATH_MAP` / `_remap_unet_path` / `_remap_te_path` | Full ldm table from ComfyUI `unet_to_diffusers`; TE1 `text_model_` on save, both TE1 spellings on load |
| `trainer/models.py` `build_kohya_metadata` | `ss_network_type`; locon also `ss_network_args` |
| `trainer/family_sdxl.py` `load_lora` | type check, locon MLP keys, `conv_dim` |
| `api.py` `train_start` | type check on `resume_lora_path` |
| `trainer/generate_sample.py` | copy type + conv args from checkpoint metadata into `cfg` |
| `trainer/checkpoints.py` `discover_checkpoints` | optional later: surface `network_type` for Ranko |

Leave alone: `device_swap` (adapters live on `modules.denoise` / TEs), dataloader, loss, pause
offload, `family.py` protocol (`apply_lora(cfg, modules)` is enough), `family_sd35.py`.

### Ranko

No new IPC method. `config_get` / `config_save` already ship the whole TOML.

| Place | Change |
| --- | --- |
| `ConfigModel.kt` `NetworkConfig` | `networkType`, `convDim`, `convAlpha` with defaults |
| `TrainingConfigForm.kt` | fields, validation (`locon` ⇒ conv dim/alpha ≥ 1, ranks independent), `toTomlSections()["network"]` |
| `UtilsUiState.kt` `ConfigSection.Network.fieldKeys` | dirty / error ownership |
| `UtilsScreen.kt` `NetworkFields` | type dropdown Standard / LoCon; conv dim/alpha visible for LoCon |
| `doc/configuration.md` `[network]` table | after the keys exist |

Desktop and wasm share `commonMain`. Dashboard checkpoint cards can keep showing dim/alpha only in
v1; a type badge is optional.

## 8. Tests

| Suite | Assert |
| --- | --- |
| `test_family.py` | Standard targets unchanged; locon Linear+Conv adapters and TE `fc1`/`fc2`; conv 4D round-trip; ldm names for `conv1`; TE1 `text_model_encoder_layers`; load accepts old `encoder_layers` |
| `test_family.py` resume | `ss_network_type` mismatch raises before any `load_state_dict`; missing type + no `conv_dim` loads as standard; `ss_network_args conv_dim>0` without type is locon |
| config | flatten + `__post_init__`: locon with `conv_dim=0` fails; `conv_dim != network_dim` is allowed |
| Ranko `jvmTest` | TOML patch of the new keys; Network form validation |
| `test_api_ipc.py` | `train_start` rejects a locon-tagged resume while config is standard (metadata-only file) |

GPU step rate / peak VRAM for locon is a separate experiment, not a unit test.

## 9. Suggested implementation order

1. `ss_network_type` on every new save (`standard`), `TrainConfig.network_type` default `standard`,
   resume / `train_start` refuse a non-standard file. Old files without the key still resume.
2. Completing the UNet ldm map + TE1 `text_model_` save / dual load. Behaviour-preserving for
   Standard UNet keys already in ldm form; TE1 new files become ComfyUI-loadable.
3. `apply_lora` locon branch + `conv_dim` / `conv_alpha` + `ss_network_args`. Config accepts
   `network_type = "locon"` only once this branch exists.
4. Ranko Network dropdown and conv fields.
5. [configuration.md](configuration.md) / [training.md](training.md) user-facing rows.

Stop at the end of the requested step when implementing; this list is ordering, not a licence to
run ahead.

## 10. Out of scope until asked

- LyCORIS (`lycoris.kohya`, tucker, `lora_mid`, presets, LoHa / LoKr / DyLoRA)
- Expanding Standard itself to kohya LierLa (UNet Transformer extras + TE MLP on `network_type = "standard"`)
- SD 3.5
- Changing pause / swap / dataloader / loss
- A1111 / kohya merge as a required gate (ComfyUI is the measured loader)

## 11. Open (intentionally)

- 16 GB locon step time and peak VRAM with the default gradient-checkpointing flags.
- Full `load_lora_for_models` on a live SDXL UNet+CLIP in ComfyUI (aliases and `calculate_weight`
  were measured; an end-to-end node graph was not).
- A1111 additional-networks and kohya `resize_lora` / merge against an Axl locon file.

## 12. How the numbers were produced

All probes are torch-free of a training run. Re-run them if the remap or ComfyUI tree moves.

```bash
# Layer counts (conda env `axl`)
python /tmp/axl_locon_probe.py

# ComfyUI alias hits + TE spellings (conda env `comfy_ui`)
python /tmp/axl_comfy_lora_keys.py
python /tmp/axl_comfy_conv_apply.py
python /tmp/axl_comfy_real_ckpt.py
```

Witnesses: `trainer/family_sdxl.py` `apply_lora` / `_UNET_PATH_MAP` / `_convert_peft_to_kohya_bf16`;
`trainer/models.py` `build_kohya_metadata`; ComfyUI `comfy/lora.py`, `comfy/utils.py`
`unet_to_diffusers`, `comfy/weight_adapter/lora.py` `LoRAAdapter`; kohya `networks/lora.py`
`6721028c`; real checkpoint `konomi_s000300` (2208 tensors, `ss_network_module=networks.lora`).
