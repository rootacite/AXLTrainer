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

Reads the latest TensorBoard scalars under `{logging_dir}/{output_name}` and the current training config.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name` from config. |
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
    "pretrained_model_name_or_path": "string"
  },
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

`config` is the flattened `TrainConfig` plus a fresh read of `trainer/config.toml` (TOML wins). Empty log dirs return empty `metrics` / `latest_stats`, not an error.

Training logs `Train/Loss` (per-step) and `Train/Avg_Loss` (Kohya-style epoch-window mean). If TensorBoard only has `Train/Loss` (older runs), `dashboard` synthesizes `Train/Avg_Loss` as a Kohya `LossRecorder` over a window of `min(n, 100)` points.

### `list_samples`

Scans `{output_dir}/{output_name}_samples/*.png`.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |

Result:

```json
{
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

Filename pattern `_(\d+)_(\d+)\.png$` → `(step, repeat_idx)`. Unmatched files use step `"-1"`. `path` is absolute so the UI can load the file from disk.

### `train_status`

Reads `$AXL_RUNTIME_DIR` or `$XDG_RUNTIME_DIR/axltrainer/` or `/tmp/axltrainer-$UID/` (`state.json`). Reconciles a dead PID into `error`.

Params: `{}`

Result: the on-disk state plus `alive` (PID is running) and `log_path`.

`status` is one of: `idle`, `starting`, `encoding`, `training`, `sampling`, `pausing`, `paused`, `resuming`, `stopping`, `finished`, `error`.

Pause/resume is a GPU swap process. While `pausing` or `resuming`, `swap` is `{stage, detail, current, total}`.

### `train_start`

Spawns `bash start_train.sh` in a new session (`setsid`) so closing Ranko does not stop training. Stdout/stderr append to `train.log` in the runtime dir.

Params: `{}`

Fails if a live training PID already exists, including a process that has already marked `finished` but has not exited yet.

### `train_pause` / `train_resume` / `train_stop`

Writes `command.json` (`pause` | `resume` | `stop`). The trainer consumes it at the next swap-safe point.

Params: `{}`

Fails if no live training PID.

Pause offloads UNet / text encoders / optimizer state / VAE to CPU and `empty_cache`s. Resume reloads what the paused phase needs. Early-stop during encoding does not save a LoRA; during training it saves `{output_name}.safetensors` if that step has no checkpoint yet; during sampling it skips leftover repeats (the step checkpoint already exists).

### `train_reset`

Clears the on-disk trainer state back to `idle` and deletes this run's sample images and TensorBoard logs (same targets as `clean.py`). Optional weight deletion.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |
| `delete_weights` | bool | No | If true, also remove `{output_dir}/{output_name}_*` checkpoint dirs. Default false. |

Fails if the training PID is still alive. `clean.py` remains the CLI cleaner and uses the same helper.

## Example

```bash
printf '%s\n' '{"id":1,"method":"ping","params":{}}' | python -u api.py
```
