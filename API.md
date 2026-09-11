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

Working directory must be the repo root so `trainer/config.toml` resolves. Ranko locates `api.py` by walking up from the executable / `user.dir`.

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
  }
}
```

`config` is the flattened `TrainConfig` plus a fresh read of `trainer/config.toml` (TOML wins). `run_id` is `null` when no run directory can be resolved; metrics / `latest_stats` are then empty rather than an error. Flat artifacts from before the run-directory layout are not resolved.

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
        "filename": "sample_1000_0.png",
        "repeat_idx": 0,
        "path": "/absolute/path/to/sample_1000_0.png"
      }
    ]
  }
}
```

Filename pattern `_(\d+)_(\d+)\.png$` → `(step, repeat_idx)`. Unmatched files use step `"-1"`. `path` is absolute so the UI can load the file from disk. `samples` is empty (and `run_id` null) when no run resolves.

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

### `train_start`

Spawns `bash start_train.sh` in a new session (`setsid`) so closing Ranko does not stop training. Stdout/stderr append to `train.log` in the runtime dir.

Params: `{}`

Fails if a live training PID already exists, including a process that has already marked `finished` but has not exited yet. Also fails synchronously — before any GPU work — when `[training].resume_lora_path` is set but does not resolve to a `.safetensors` file.

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

## Example

```bash
printf '%s\n' '{"id":1,"method":"ping","params":{}}' | python -u api.py
```
