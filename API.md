# Training Dashboard IPC

`api.py` is a local helper process. Ranko starts it and talks over **newline-delimited JSON** on stdin/stdout. There is no HTTP server and no generation pipeline.

Logs (TensorBoard, traceback, warnings) go to **stderr**. stdout is only NDJSON.

## Launch

```bash
# From the trainer repo root (directory that contains api.py and trainer/)
python -u api.py
```

Environment:

| Variable | Meaning |
|---|---|
| `AXL_PYTHON` | Optional. Ranko uses this interpreter instead of `python3`. |

Working directory must be the repo root so `config.toml` resolves. Ranko locates `api.py` by walking up from the executable / `user.dir`.

`start_api.sh` is a debug wrapper. The desktop app owns the process in normal use.

## Framing

One JSON object per line, UTF-8.

Request:

```json
{"id": 1, "method": "dashboard", "params": {"name": null, "start_step": null, "end_step": null}}
```

Success:

```json
{"id": 1, "ok": true, "result": { }}
```

Failure:

```json
{"id": 1, "ok": false, "error": "unknown method: foo"}
```

`id` is echoed back. Clients should send one request at a time (or match on `id`). Blank lines are ignored.

## Methods

### `ping`

Params: `{}`

Result:

```json
{ "status": "ok" }
```

### `dashboard`

Reads the latest TensorBoard scalars under `{logging_dir}/{run_id}` and the current training config.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name` from config. |
| `run_id` | string \| null | No | Run directory to read. Defaults to `state.json`'s `run_id`, then the newest `{output_name}_<timestamp>` directory under `logging_dir`. |
| `start_step` | integer \| null | No | Inclusive lower bound on metric steps. |
| `end_step` | integer \| null | No | Inclusive upper bound on metric steps. |

Result:

```json
{
  "config": {
    "train_data_dir": "string",
    "output_name": "string",
    "logging_dir": "string",
    "output_dir": "string",
    "pretrained_model_name_or_path": "string",
    "resume_lora_path": "string"
  },
  "run_id": "rein_20260911_120000",
  "latest_stats": {
    "Train/Loss": 0.0,
    "Train/Avg_Loss": 0.0,
    "UNet/LR/Effective_Actual_LR": 0.0,
    "current_step": 0
  },
  "metrics": {
    "Metric/Tag/Name": [
      { "step": 0, "value": 0.0, "wall_time": 0.0 }
    ]
  },
  "sample_sets": [
    {
      "name": "classroom",
      "prompt": "string",
      "negative": "string",
      "width": 1152,
      "height": 768,
      "steps": 35,
      "guidance_scale": 6.0,
      "seed": 1,
      "repeat": 3
    }
  ]
}
```

`config` is the flattened `TrainConfig` plus a fresh read of `config.toml` (TOML wins). `run_id` is `null` when no run directory can be resolved; metrics / `latest_stats` are then empty rather than an error. Flat artifacts from before the run-directory layout are not resolved.

`sample_sets` is `resolve_sample_sets` over that config: one entry per `[[validation.samples]]` block, or a single entry built from the flat `sample_*` scalars when the file has none. The flat `sample_prompts` / `sample_negative` / `sample_width` / `sample_height` / `sample_steps` / `sample_seed` / `sample_repeat` / `guidance_scale` keys in `config` mirror the first entry, so a client that only reads those keeps working. A block that fails validation is reported on stderr and yields `[]` rather than an IPC error, so the dashboard keeps rendering.

Training logs `Train/Loss` (per-step) and `Train/Avg_Loss` (Kohya-style epoch-window mean). If TensorBoard only has `Train/Loss` (older runs), `dashboard` synthesizes `Train/Avg_Loss` as a Kohya `LossRecorder` over a window of `min(n, 100)` points.

### `list_samples`

Scans `{output_dir}/{run_id}/{output_name}_samples/*.png`.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |
| `run_id` | string \| null | No | Run directory to scan. Resolved like `dashboard`. |

Result:

```json
{
  "run_id": "rein_20260911_120000",
  "samples": {
    "1000": [
      {
        "filename": "sample_1000_p0_0.png",
        "set_index": 0,
        "repeat_idx": 0,
        "path": "/absolute/path/to/sample_1000_p0_0.png"
      }
    ]
  }
}
```

Filename pattern `_(\d+)_p(\d+)_(\d+)\.png$` → `(step, set_index, repeat_idx)`; the two-number form written before `[[validation.samples]]` (`_(\d+)_(\d+)\.png$`) still parses, as set `0`. Unmatched files use step `"-1"` and set `0`. Within a step the samples are ordered by `(set_index, repeat_idx)`. `path` is absolute so the UI can load the file from disk. `samples` is empty (and `run_id` null) when no run resolves.

### `generate_sample`

Generates one extra sample image from a LoRA checkpoint of the resolved run, detached from any
training process. Returns as soon as the generator is spawned; follow it with `list_generated_samples`.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `checkpoint` | string | Yes | Path to a `.safetensors` LoRA file (any run's). |
| `prompt` | string | Yes | Non-empty. |
| `negative_prompt` | string \| null | No | Defaults to the first `[[validation.samples]]` entry's `negative` (or `sample_negative`). |
| `cfg` | number \| null | No | 1–30, defaults to that entry's `guidance_scale`. |
| `steps` | integer \| null | No | 1–150, defaults to that entry's `steps`. |
| `seed` | integer \| null | No | 0–4294967295, `0` = random (the seed actually used is written back to the job). |
| `step` | integer \| null | No | Step the checkpoint belongs to; used by the UI to attach the image to that step's samples. |
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard`. |

`width` / `height` default to the first `[[validation.samples]]` entry's (or `[validation].sample_width`
/ `sample_height`); other image settings (`clip_skip`, `max_token_length`, `network_dim`,
`network_alpha`, base model) come from the checkpoint's own kohya metadata so an old checkpoint is
sampled with the settings it was trained with.

Refused with an error when the trainer process is alive (running **or** paused — the GPU is single
tenant), when a generation for the run is already `running`, or when a value is out of range.

Result:

```json
{
  "job": {
    "id": "rein_s000100_gen_20260915_161123",
    "state": "running",
    "run_id": "rein_20260911_120000",
    "output_name": "rein",
    "checkpoint": "/out/rein_20260911_120000/rein_s000100/rein.safetensors",
    "prompt": "1girl, solo",
    "negative_prompt": "",
    "cfg": 5.0,
    "steps": 20,
    "seed": 12345,
    "width": 1152,
    "height": 768,
    "step": 100,
    "current_step": 0,
    "total_steps": 20,
    "image_path": null,
    "error": null,
    "pid": 12345,
    "started_at": 1757500000.0
  },
  "log_path": "/out/rein_20260911_120000/rein_samples/generated/rein_s000100_gen_20260915_161123.log"
}
```

### `list_generated_samples`

Lists the generated samples of the resolved run, newest first. Read-only; a job whose generator died
(reported `running` but its PID is gone) is rewritten to `error` so it never blocks the next one.

Params: `name` / `run_id` as in `list_samples`.

Result:

```json
{
  "run_id": "rein_20260911_120000",
  "jobs": [
    {
      "id": "rein_s000100_gen_20260915_161123",
      "state": "done",
      "step": 100,
      "cfg": 5.0,
      "steps": 20,
      "seed": 12345,
      "current_step": 20,
      "total_steps": 20,
      "image_path": "/out/rein_20260911_120000/rein_samples/generated/rein_s000100_gen_20260915_161123.png",
      "error": null
    }
  ]
}
```

`state` is `running`, `done` or `error`; `jobs` is empty when the run has no `generated/` directory.

### `list_checkpoints`

Lists LoRA checkpoints written by earlier runs of one `output_name`, newest step first. Read-only: safe while training is running.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |
| `output_dir` | string \| null | No | Overrides `[environment].output_dir`. |

Result:

```json
{
  "checkpoints": [
    {
      "path": "/out/rein_20260911_120000/rein_s000100/rein.safetensors",
      "run_id": "rein_20260911_120000",
      "dir": "rein_s000100",
      "filename": "rein.safetensors",
      "step": 100,
      "epoch": null,
      "final": false,
      "size_bytes": 12345678,
      "modified": 1757500000.0,
      "network_dim": 48,
      "network_alpha": 24,
      "output_name": "rein"
    }
  ]
}
```

`step` / `epoch` come from the directory name (`{name}_s000100`, `{name}_e003_s000100`, `{name}_final`) with `ss_steps` / `ss_epoch` metadata as fallback, so `final` checkpoints still report the step they were saved at. `network_dim` / `network_alpha` come from the safetensors metadata and are `null` when absent. Use `path` as `[training].resume_lora_path`.

### `train_status`

Reads `$AXL_RUNTIME_DIR` or `$XDG_RUNTIME_DIR/axltrainer/` or `/tmp/axltrainer-$UID/` (`state.json`). Reconciles a dead PID into `error`.

Params: `{}`

Result: the on-disk state plus `alive` (PID is running) and `log_path`. Relevant state keys: `run_id` (run directory created for this run), `output_name`, and `resume` — `null` for a fresh run, otherwise

```json
{ "path": "/out/rein_…_final/rein.safetensors", "filename": "rein.safetensors", "step": 300, "epoch": 7, "loaded": 96, "skipped": 0 }
```

`status` is one of: `idle`, `starting`, `encoding`, `training`, `sampling`, `pausing`, `paused`, `resuming`, `stopping`, `finished`, `error`.

Pause/resume is a GPU swap process. While `pausing` or `resuming`, `swap` is `{stage, detail, current, total}`.

While sampling, `sampling` is `{active, repeat, repeats, denoise_step, denoise_steps, global_step, prompt_set, prompt_sets}`: `repeat`/`repeats` count the images of the whole pass (all `[[validation.samples]]` sets) and `prompt_set`/`prompt_sets` are 1-based (both `0` for a run with no sets).

### `train_start`

Spawns `bash start_train.sh` in a new session (`setsid`) so closing Ranko does not stop training. Stdout/stderr append to `train.log` in the runtime dir.

Params: `{}`

Fails if a live training PID already exists, including a process that has already marked `finished` but has not exited yet. Also fails synchronously — before any GPU work — when `[training].resume_lora_path` is set but does not resolve to a `.safetensors` file, and when `[environment].amdfq` is `tail` or `vmm` but the corresponding `target/release/libamdfq_*_rs.so` is missing.

### `train_pause` / `train_resume` / `train_stop`

Writes `command.json` (`pause` | `resume` | `stop`). The trainer consumes it at the next swap-safe point.

Params: `{}`

Fails if no live training PID.

Pause offloads UNet / text encoders / optimizer state / VAE to CPU and `empty_cache`s. Resume reloads what the paused phase needs. Early-stop during encoding does not save a LoRA; during training it saves `{output_name}.safetensors` if that step has no checkpoint yet; during sampling it skips leftover repeats (the step checkpoint already exists).

### `train_reset`

Clears the on-disk trainer state back to `idle` and deletes the resolved run's sample images and TensorBoard logs (same targets as `clean.py`). Optional weight deletion.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |
| `run_id` | string \| null | No | Run to clean. Resolved like `dashboard`. |
| `delete_weights` | bool | No | If true, also remove the run's `{output_name}_*` checkpoint dirs. Default false. |

The result carries `run_id` and `cleanup` (`run_dir`, `samples_dir`, `log_dir`, `weight_dirs`, `removed`, `skipped`, `errors`). The run directory is removed when it becomes empty. When no run directory resolves, `run_id` is `null` and **nothing is deleted** — legacy flat artifacts are only reachable via `python clean.py --legacy-flat`.

Fails if the training PID is still alive. `clean.py` remains the CLI cleaner and uses the same helper.

### `dataset_tag`

Runs `tagger/main.py` with the same interpreter as `api.py` (the `axl` env). Writes WD-tagger captions next to every image in a folder (non-recursive). Overwrites existing `.txt` files. Ranko should reload Images / Statistics after a successful call.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `directory` | string \| null | No | Dataset folder. Defaults to `[environment].train_data_dir`. |
| `threshold` | number | No | Minimum tag confidence, `0.0`–`1.0`. Default `0.35`. |
| `batch_size` | integer | No | ONNX batch size. Default `1` (MIGraphX compiles once per shape; use `8` only after you accept a one-time recompile). |

Result:

```json
{
  "directory": "/abs/path",
  "threshold": 0.35,
  "provider": "MIGraphXExecutionProvider",
  "total": 100,
  "processed": 100,
  "failed": 0,
  "seconds": 12.3,
  "errors": [{ "file": "0001.png", "error": "…" }]
}
```

Fails if the folder is missing, `threshold` is out of range, or a training process is in a GPU-using status (`starting` / `encoding` / `training` / `sampling` / `pausing` / `resuming` / `stopping`). Pause (`paused`) is allowed because weights are offloaded. The tagger is a child process so GPU memory is released when it exits.

### `hardware_status`

Read-only host snapshot for the Ranko Dashboard hardware panel. GPU fields come from `nvtop -s` (JSON snapshot mode in nvtop 3.3.2+). Process lists are dropped. AMD edge / junction / mem temperatures are filled from DRM hwmon when present. CPU util is a `/proc/stat` delta; CPU temp prefers `x86_pkg_temp` then `k10temp`; RAM comes from `/proc/meminfo`. CPU package power is omitted (RAPL / turbostat need root).

Params: `{}`

Result:

```json
{
  "available": true,
  "error": null,
  "ts": 1710000000.12,
  "gpus": [
    {
      "index": 0,
      "name": "AMD Radeon RX 9070 XT",
      "gpu_clock_mhz": 2165.0,
      "mem_clock_mhz": 2500.0,
      "fan_pct": 30.0,
      "gpu_util_pct": 92.0,
      "mem_util_pct": 76.0,
      "power_w": 303.0,
      "temp_c": 72.0,
      "temp_edge_c": 72.0,
      "temp_junction_c": 85.0,
      "temp_mem_c": 80.0,
      "mem_total_bytes": 17095983104,
      "mem_used_bytes": 13000000000,
      "mem_free_bytes": 4095983104
    }
  ],
  "cpu": {
    "name": "Intel Core …",
    "n_logical": 28,
    "util_pct": 41.2,
    "temp_c": 41.0,
    "mem_total_bytes": 67108864000,
    "mem_used_bytes": 22020096000
  },
  "vmm_va": {
    "patch": "vmm",
    "used_bytes": 8388608,
    "total_bytes": 281474976710656,
    "total_source": "journal",
    "pid": 12345,
    "spans": 4
  }
}
```

`available` is false when nvtop is missing, times out, or returns no GPUs; `error` then has a short reason. CPU fields are still filled when possible. This method does not fail the IPC call — Ranko keeps the training UI up if hardware collection fails.

`vmm_va` is present only when `[environment].amdfq` is `"vmm"`. `used_bytes` is the GPU VA the VMM hook has reserved and will not return (`EVER_MAPPED`), read from `$AXL_RUNTIME_DIR/amdfq_vmm_va.<trainer-pid>.json` (0 if the trainer is not running or has not written yet). `total_bytes` is the GPU VM size: `journalctl -k` `vm size is N GB` first (no sudo), then `dmesg`, then `/sys/module/amdgpu/parameters/vm_size` when that value is positive, otherwise 256 TiB. `total_source` is `journal` / `dmesg` / `sysfs` / `default`. The module parameter is often `-1` (auto) and is not the live size.

## Example

```bash
printf '%s\n' '{"id":1,"method":"ping","params":{}}' | python -u api.py
```
