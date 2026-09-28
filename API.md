# Training Dashboard IPC

`api.py` is a local helper process. Ranko talks to it over a WebSocket using JSON-RPC (`{id, method, params}` → `{id, ok, result|error}`). There is no HTTP API, no stdin/stdout control channel, and no generation pipeline. Default bind is loopback; LAN is an IP allowlist, not a token.

Logs (TensorBoard, traceback, warnings) go to **stderr**.

## Launch

```bash
python -u api.py --host 127.0.0.1 --port 18765
# LAN:
python -u api.py --host 0.0.0.0 --port 18765 --allow-ip 192.168.1.20 --allow-ip 192.168.1.0/24
```

`--host` defaults to `127.0.0.1`. A non-loopback bind is allowed; clients are gated by `--allow-ip` / `AXL_WS_ALLOW` (IPv4/IPv6 or CIDR). Loopback (`127.0.0.1`, `::1`) is always admitted. An empty allowlist never means "allow the world".

Environment:

| Variable | Meaning |
|---|---|
| `AXL_PYTHON` | Optional. Ranko uses this interpreter instead of `python3`. |
| `AXL_WS_HOST` | WebSocket bind (default `127.0.0.1`). |
| `AXL_WS_PORT` | WebSocket port (default `18765`). |
| `AXL_WS_ALLOW` | Comma-separated client IPs/CIDRs. Loopback is always allowed. |
| `AXL_BLOB_WORKERS` | Encode process-pool size. Default `nproc`. `0` encodes in-process. |
| `AXL_BLOB_CACHE_DIR` | Processed-image cache (default `/tmp/axlranko/blob-cache`). |
| `AXL_BLOB_CACHE_BYTES` | Cache cap in bytes (default 1/4 of `MemTotal`). |
| `AXL_TRASH_DIR` | Drop destination (default `/tmp/axlranko/trash`). |

Working directory must be the repo root so `config.toml` resolves. Desktop Ranko locates `api.py` by walking up from the executable / `user.dir`, then connects to `ws://127.0.0.1:18765` (spawning the helper if nothing is listening). The Java client disables HTTP proxies so `http_proxy` cannot intercept localhost. The wasm UI does not spawn the helper; host/port live in Utils → Helper (`localStorage` + `?host=` / `?port=`).

`start_api.sh` is a debug wrapper that starts the same WebSocket helper.

## Framing

One JSON object per WebSocket text frame. UTF-8.

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

`id` is echoed back. Match replies on `id`. Control methods (`train_*`, dataset mutations, config/profile writes) are serialized on the server. `blob_stat` / `blob_batch` do not hold that lock while encoding. Blank frames/lines are ignored.

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
| `run_id` | string \| null | No | Run directory to read. Defaults to `state.json`'s `run_id`; with no run recorded and no `name` there is no run to read, and `run_id` comes back `null`. A run the trainer never recorded is reached by name, or by its `run_id` — every run directory of either root is listed by `list_runs`, which is what keeps a run whose `logging_dir` directory is gone (Reset used to delete it) readable. |
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

`run_id` alone is enough to read any run: every run-scoped method falls back to the name the run id was built from when `name` is omitted, which is how a client opens a run from `list_runs` that was created with a different `output_name` than the config now says.

`sample_sets` is `resolve_sample_sets` over that config: one entry per `[[validation.samples]]` block, or a single entry built from the flat `sample_*` scalars when the file has none. The flat `sample_prompts` / `sample_negative` / `sample_width` / `sample_height` / `sample_steps` / `sample_seed` / `sample_repeat` / `guidance_scale` keys in `config` mirror the first entry, so a client that only reads those keeps working. A block that fails validation is reported on stderr and yields `[]` rather than an IPC error, so the dashboard keeps rendering.

Training logs `Train/Loss` (per-step) and `Train/Avg_Loss` (Kohya-style epoch-window mean). If TensorBoard only has `Train/Loss` (older runs), `dashboard` synthesizes `Train/Avg_Loss` as a Kohya `LossRecorder` over a window of `min(n, 100)` points.

### `list_runs`

The dashboard's run history: every run directory under `output_dir` and `logging_dir`, whatever `output_name` it was created with, newest first.

Params: `{}`

Result:

```json
{
  "runs": [
    {
      "run_id": "Tsukuyomi_20260928_110928",
      "output_name": "Tsukuyomi",
      "output_dir": "/home/acite/LLM/axltrainer/outputs/Tsukuyomi_20260928_110928",
      "log_dir": "/home/acite/LLM/axltrainer/logs/Tsukuyomi_20260928_110928",
      "has_output": true,
      "has_log": false,
      "last_step": 4500,
      "samples": 12,
      "checkpoints": 46,
      "size_bytes": 11172201792,
      "modified": 1790587779.53,
      "current": true,
      "live": true
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `output_name` | The name the run id was built from, so a client that only carries the run id can still resolve that run's `{output_name}_samples` and weight directories. |
| `has_output` / `has_log` | Whether that run has a directory under each root. A run whose TensorBoard directory is gone is still listed. |
| `last_step` | The newest step any artifact of the run carries (a sample filename, a `{name}_sNNNNNN` or `{name}_eEEE_sNNNNNN` directory). No event file is read, so a run without logs reports how far it got. `null` when the run never wrote one. |
| `samples` | Sample PNGs directly in `{output_dir}/{run_id}/{output_name}_samples/`; `generated/` is not counted. |
| `checkpoints` | `.safetensors` files in the run's weight directories. |
| `size_bytes` / `modified` | Size of the output directory and its mtime (the log directory's when there is no output). |
| `current` | This is the run `state.json` is on — the only one the training controls act on. |
| `live` | That run's PID is alive and its status is a live one. `current` without `live` is a run that finished or died. |

The run `state.json` is on is prepended (with `has_output` / `has_log` false) when neither root holds a directory for it, so a client can always show what the trainer is doing.

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

Fails if a live training PID already exists, including a process that has already marked `finished` but has not exited yet. Also fails synchronously — before any GPU work — when `[training].resume_lora_path` is set but does not resolve to a `.safetensors` file, when that file's `ss_network_type` does not match `[network].network_type`, and when `[environment].amdfq` is `tail` or `vmm` but the corresponding `target/release/libamdfq_*_rs.so` is missing.

### `train_pause` / `train_resume` / `train_stop`

Writes `command.json` (`pause` | `resume` | `stop`). The trainer consumes it at the next swap-safe point.

Params: `{}`

Fails if no live training PID.

Pause offloads UNet / text encoders / optimizer state / VAE to CPU and `empty_cache`s. Resume reloads what the paused phase needs. Early-stop during encoding does not save a LoRA; during training it saves `{output_name}.safetensors` if that step has no checkpoint yet; during sampling it skips leftover repeats (the step checkpoint already exists).

### `train_reset`

Clears the on-disk trainer state back to `idle` so `train_start` can launch a new run. **Sample images and TensorBoard logs are kept** — they are what the dashboard's run history shows afterwards; only the LoRA weight directories are removable.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |
| `run_id` | string \| null | No | Run to clear. Resolved like `dashboard`. |
| `delete_weights` | bool | No | If true, also remove the run's `{output_name}_*` checkpoint dirs. Default false. |

The result carries `run_id` and `cleanup` (`run_dir`, `samples_dir`, `log_dir`, `weight_dirs`, `delete_samples`, `delete_logs`, `removed`, `skipped`, `errors`); `delete_samples` / `delete_logs` are always `false` here, so both paths land in `skipped`. When no run directory resolves, `run_id` is `null` and **nothing is deleted** — legacy flat artifacts are only reachable via `python clean.py --legacy-flat`.

Fails if the training PID is still alive. `clean.py` remains the CLI cleaner and uses the same helper — it is the tool that deletes a run's samples and logs.

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
    "spans": 4,
    "never_reuse": false
  }
}
```

`available` is false when nvtop is missing, times out, or returns no GPUs; `error` then has a short reason. CPU fields are still filled when possible. This method does not fail the IPC call — Ranko keeps the training UI up if hardware collection fails.

### Config, dataset, masks, blobs

After connect Ranko does not open trainer files. Paths in these methods are allowlisted (`config.toml`, `<repo>/configs/`, `[[environment.train_data]]` folders, `output_dir`, and — for images only — the `automation/` tree plus the automation `output_dir`).

- `config_get` `{}` → `{path, text}`. `config_save` `{text}` parse-checks then atomic-writes; a text whose `[environment].output_name` is not filename-safe (letters and digits, `-`, `_`, `.`) is refused, because the trainer would refuse to start with it.
- `profile_list` / `profile_get` `{name}` / `profile_save` `{name, text, overwrite}` / `profile_delete` `{name}`.
- `prompt_matrix` `{}` → `{path, text}` of repo-root `input_matrix.txt` (read-only). `prompt_profile_list` `{}` → `{profiles: [{name, version, modified, size, error}]}` for repo-root `prompt_profiles/*.json`; `version` is `null` when the file has no `version` key and `error` carries the reason an unreadable entry cannot be used. `prompt_profile_get` `{name}` → `{name, text}`; `prompt_profile_save` `{name, text, overwrite}` parse-checks that `text` is a JSON object with a `spec` object, then atomic-writes `<repo>/prompt_profiles/<name>.json`; `prompt_profile_delete` `{name}`. The version upgrades (v1 → v2 → v3) happen in the client, so the store never rewrites a profile.
- `dataset_list` `{directory}` → `{items: [{stem, image, txt, mask, width, height, tags, has_sidecar_mask, has_alpha}], orphans}`. Non-recursive. Orphan `.txt` names are listed; Statistics aborts when `orphans` is non-empty.
- `caption_write` `{directory, stem, text}`.
- `dataset_drop` `{directory, rate, seed?, stems?}`. Moves image+txt+mask to `/tmp/axlranko/trash`. `stems` limits the pool (the GUI passes the filtered set).
- `dataset_shuffle` `{directory, seed?}` → `{groups, renamed_files, first_stem, last_stem}`.
- `mask_get` / `mask_write` / `mask_delete` `{directory, stem, png_base64?}`. Lossless PNG only.
- `blob_stat` / `blob_batch` `{paths, max_edge, quality?, format?}`. `max_edge` is required (32–4096, contain, never upscale). Default `quality=80`, `format=jpeg`. JPEG/WebP flatten transparency onto black before encoding (dropping the alpha channel would leak leftover RGB in transparent pixels). Result items carry `hash` (SHA-256 of the processed bytes), `width`/`height`, `cache` (`hit`/`miss`), and for `blob_batch` `base64`. A bad path is a per-item `error`, not a failed RPC. Encode fans out across `AXL_BLOB_WORKERS` spawn processes (`trainer/blobcodec.py`, torch-free). Hits live under `/tmp/axlranko/blob-cache/` capped at 1/4 of host RAM. The cache key includes a codec version, so a flatten/resize change does not reuse bytes from an older encoder.
- `tag_lexicon` `{}` → `{text}` of `tagger/selected_tags.csv`.
- `checkpoint_export` `{source, dest}` server-local copy. `dest` must end `.safetensors`; refuse `source == dest`.
- `fs_listdir` `{path}` → `{path, parent, entries: [{name, path, is_dir, size, mtime_ms}]}`. Lists one directory after `Path.resolve()` (so `..` cannot escape). A file path errors. Unreadable children are skipped. No file bytes.
- `fs_roots` `{}` → `{roots: [{name, path}]}` with Home, Repo, each train-data folder, Output, Logs.

### Automation (prompt sets, ComfyUI workflows, generated-image jobs)

State lives under the repo's `automation/` tree (gitignored): `settings.json`, `workflows/*.json` (uploaded API-format workflows), `prompts/*.txt` (prompt sets, one prompt per line), `jobs/<job_id>/{job.json,log.txt,images/*.png}`. `AXL_AUTOMATION_DIR` overrides the root. Images of a job are servable through `blob_stat` / `blob_batch`: those paths are allowlisted in addition to the dataset and `output_dir` roots.

- `automation_config_get` `{}` → `{settings: {server, workflow, positive_node, count, poll, output_dir}, default_output_dir, paths: {root, workflows, prompts, jobs}}`. `automation_config_save` `{settings}` normalizes (a bare `host:port` becomes `http://…`, `count` 1–16, `poll` 0.1–10 s, a relative `output_dir` resolves against the repo) and writes atomically. A hand-edited file that fails validation falls back to the defaults instead of locking the page out.
- `automation_discover` `{server?}` → `{found, url, version, queue_running, queue_pending, checked: [{url, ok, reason}], probed_all}`. With `server` it probes exactly that address; otherwise it walks the machine's loopback listeners (`/proc/net/tcp{,6}`, 8188 first) and accepts only an answer carrying `system.comfyui_version`. `$AXL_COMFY_URL` is used when no address is given. Every request bypasses `http_proxy` — this machine's session exports one, and through it a loopback call answers `502` instead of reaching ComfyUI.
- `automation_workflow_list` `{}` → `{workflows: [{name, path, valid, error, node_count, save_image_nodes, batch_size_nodes, text_nodes: [{id, class_type, text}], positive_node, positive_node_guessed, missing_models}], default_workflow, model_check}`. `automation_workflow_validate` `{path, positive_node?}` reports the same shape for any file on the server. The model pre-check compares every literal enum input against the live `/object_info` and accepts both combo shapes ComfyUI 0.35 reports (`["COMBO", {"options": […]}]` and `[[names…], …]`). `automation_workflow_save` `{name, text}` refuses anything that is not API format (`{node_id: {class_type, inputs}}` — the editor format with `nodes`/`links` is rejected, as is a missing `SaveImage` or `CLIPTextEncode`); `automation_workflow_delete` `{name}`.
- `automation_prompt_list` `{}` → `{prompts: [{name, path, count, text}]}`; `automation_prompt_get` `{name}`; `automation_prompt_save` `{name, text}` (a text block or a list; blank lines dropped, ≤400 prompts, ≤4000 chars each); `automation_prompt_delete` `{name}`.
- `automation_job_start` `{prompts | prompt_set, server?, workflow?, positive_node?, count?, poll?, output_dir?}` → `{job, log_path}`. Writes `job.json` and spawns `trainer/run_automation.py --spec …` detached (`setsid`), so the batch survives Ranko closing; it returns immediately. Refuses while another job is running, without a workflow, or when `count > 1` and the workflow has no numeric `batch_size` input.
- `automation_job_list` `{}` → `{jobs: [{id, state, created_at, started_at, updated_at, finished_at, total, done, failed, images, preview_paths, workflow, positive_node, count, comfy_url, error}]}`, newest first, scoped to the configured `output_dir`. A `running` job whose PID is gone is rewritten to `error` (the log explains).
- `automation_job_get` `{id}` → the whole record plus `summary` and `log_tail` (last 40 lines): per-prompt `{index, text, state, seed, prompt_id, images, error}`.
- `automation_job_cancel` `{id}` → SIGTERMs the runner's process group, waits briefly, marks the job `cancelled`; a half-finished job keeps its images. `automation_job_retry_failed` `{id}` respawns the runner with `--only-failed` (the prompts that already produced images are not queued again). `automation_job_delete` `{id}` removes the job directory (refused while it runs).
- Every job writes one `.txt` next to each image with `seed`, `prompt_id`, `prompt` and ComfyUI's own file name; images are named by us (`p0003_01.png`) so the order never depends on the workflow's `filename_prefix`. Only `SaveImage` outputs are collected, `PreviewImage` nodes are ignored.

`vmm_va` is present only when `[environment].amdfq` is `"vmm"`. `used_bytes` is the GPU VA the VMM hook holds — what it has mapped right now, or, with `never_reuse`, everything it has ever mapped — read from `$AXL_RUNTIME_DIR/amdfq_vmm_va.<trainer-pid>.json` (0 if the trainer is not running or has not written yet). `never_reuse` is that file's mode when it is there (the running hook's own mode) and `[environment].amdfq_va_never_reuse` otherwise; Ranko titles the bar `GPU VA (live)` / `GPU VA (not returned)` from it. `total_bytes` is the GPU VM size: `journalctl -k` `vm size is N GB` first (no sudo), then `dmesg`, then `/sys/module/amdgpu/parameters/vm_size` when that value is positive, otherwise 256 TiB. `total_source` is `journal` / `dmesg` / `sysfs` / `default`. The module parameter is often `-1` (auto) and is not the live size.

## Example

```python
python -c "import api; print(api.dispatch('ping'))"
```
