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

`id` is echoed back. Match replies on `id`. Blank frames/lines are ignored.

**One client at a time.** The helper serves a single client session and does **no locking of its own**: ordering its own calls is the client's job (Ranko does it with one read/write lock per resource, `IpcResources.kt`). A connection is admitted by its first request:

* `hello {client, instance, lane}` — who is asking. The first instance to say hello **owns** the helper; the same instance may open as many connections as it likes (one per lane, see below), which is how the desktop app runs its control, poll, blob and long-running traffic on separate sockets. `client` is a display name (`axlranko-desktop`, `axlranko-web`), `instance` identifies this run of that client, `lane` is free text for logs. The reply is `{owner, client, instance, since, connections}`.
* **Any other instance is refused** and its connection closed:

  ```json
  {"id": 1, "ok": false, "code": "CLIENT_BUSY",
   "error": "axlranko-desktop owns the training helper (since 12:01:44); close it and retry",
   "holder": {"client": "axlranko-desktop", "instance": "axlranko-51e107b5", "since": 1790780504.1, "connections": 3}}
  ```

  There is no takeover request: the helper is released when the owner's last connection has been gone for five seconds, and then the next `hello` owns it.
* A request that arrives without a `hello` claims a *free* helper, exactly like an anonymous one — the path a script or an older client takes. While another client owns the helper, such a connection is refused the same way. A connection that never sends a request (the desktop's `helperListening` probe) claims nothing.

A connection is served **in order by one thread**, so a slow call delays what is queued behind it on *that* connection — which is why the client keeps separate ones: `control` (every write, so their order is the order they were made in), `poll` (reads), `blob`, and `long` (a tagger, an export: anything that can hold a resource for minutes). Two calls that must not overlap are kept apart by the client's resource table, not by the server.

## Methods

### `ping`

Params: `{}`

Result:

```json
{ "status": "ok" }
```

### `dashboard`

Reads the latest TensorBoard scalars under `{logging_dir}/{run_id}` and the current training config.

The reply carries the run's whole history (it is what the charts draw), and the helper keeps one event-file reader per run directory: a repeated poll reads only the events written since the last one (measured: 4.6 ms instead of 255 ms on a 24 k-point run; a live run's poll after 200 more steps took 20 ms). The reader is rebuilt when the newest event file changes or shrinks, so a restart in the same directory cannot serve stale numbers. What is left of a long run's poll is the size of the reply itself, not the disk read.

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

Refused with an error while a live trainer is using the GPU (`starting`, `encoding`, `training`,
`sampling`, `pausing`, `resuming`, `stopping`), when any generation is already `running` (the GPU is
single tenant, whatever run the other job belongs to), or when a value is out of range. A **paused**
run does not block it: pause has offloaded the UNet, both text encoders, the optimizers and the VAE
to CPU. `train_resume` is refused while a generation is running, so resuming cannot put a second SDXL
on the card.

Result:

```json
{
  "job": {
    "id": "rein_s000100_gen_20260915_161123",
    "state": "running",
    "mode": "single",
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
    "files": [],
    "images_done": 0,
    "total_images": 1,
    "error": null,
    "pid": 12345,
    "started_at": 1757500000.0
  },
  "log_path": "/out/rein_20260911_120000/rein_samples/generated/rein_s000100_gen_20260915_161123.log"
}
```

`mode` is `single` here. `files` (every image, in render order), `images_done` and `total_images`
matter for `generate_checkpoint_samples`, which renders one image per sample set and repeat.

### `generate_checkpoint_samples`

Renders the checkpoint's `[[validation.samples]]` sets as one detached job — the "sample this
checkpoint" action the Dashboard offers per checkpoint. Prompts, `width`/`height`, `steps`,
`guidance_scale`, `seed` and `repeat` are the ones that run samples with — the `config.toml` it saved
beside its logs, or the sets `sample_prompts_set` saved for it — resolved exactly as
`evaluate_checkpoint` resolves them (seed `0` = a fresh random seed per image); network type / dim /
alpha, `conv_dim` / `conv_alpha`, `clip_skip`, `max_token_length` and the base model come from the
checkpoint's own kohya metadata, as in `generate_sample`.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `checkpoint` | string | Yes | Path to a `.safetensors` LoRA file (any run's). |
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard`. |

Images land in `{output_dir}/{run_id}/{output_name}_samples/generated/` as
`{job_id}_p{set}_{repeat}.png` — sets counting from zero, like the run's own
`{output_name}_{step:06d}_p{set}_{repeat}.png` samples — so a training-produced sample is never
overwritten and a `Pn` badge can be shown. The job record is the same shape as `generate_sample`'s
with `mode: "sets"`, a `sample_sets` copy of what is being rendered (the plan the runner renders
from), `config_log_dir` = the run directory those prompts were resolved in, and `total_images` =
Σ `repeat`.

The job records the checkpoint's own `step` — its artifact directory name, or `ss_steps` from its
metadata — because that is what attaches the rendered images to that checkpoint in the Dashboard's
Checkpoints section. Counters (`current_step`, `total_steps`, `images_done`, `total_images`) are
always integers, never `null`: the client declares them as such, and an explicit `null` would fail
its decode and take the whole generated-samples list down with it.

Refused under the same GPU rules as `generate_sample`, plus when the run has no usable
`[[validation.samples]]` set (`this run has no sample prompts to render: …`).

### `generate_checkpoint_samples_batch`

The same pass for a whole **step range**: every checkpoint of the run whose step is inside
`from_step..to_step`, oldest first, rendered by one detached process.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `from_step` / `to_step` | integer | Yes | Inclusive bounds, `0 <= from_step <= to_step`. |
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard`. |

The plan is a `batch` job record — the ordered work list, `from_step`/`to_step`, how many
checkpoints and images, which entry is being rendered, and what failed — and the runner gives each
checkpoint its **own** `sets` job (with `batch_id`/`batch_index`/`batch_total`), so every image is
named, shown and followed exactly as a manual pass from that card would be. A checkpoint that fails
is recorded in the batch's `failed` list and on its own job, and the range carries on; the batch is
`done` if anything rendered and `error` if nothing did. The pipeline is built once for the range and
rebuilt only when a checkpoint's LoRA shape (base model, kind, rank/alpha, conv dim) differs.
The prompts are that run's own, resolved once per range: the batch record carries `config_log_dir`
and each checkpoint's own `sets` job records the `sample_sets` it rendered with, so a child spec
stands on its own.

Refused under the same GPU rules as `generate_checkpoint_samples`, on a malformed range, and when
the range covers no checkpoint of the run (`no checkpoints between step X and Y`).

Result: `{job, log_path}` with `mode: "batch"`.

### `evaluate_checkpoint`

Evaluates one checkpoint: its sample images are topped up to `depth` (a floor, not a target), the
Pixai tagger labels every one of them, and the labels are compared against the prompt each image was
rendered from. Detached like the generation entries: the reply is `{job, log_path}` and the job file
carries the progress and the result.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `checkpoint` | string | Yes | Path to a `.safetensors` LoRA file. |
| `depth` | integer | Yes | Lowest number of sample images to score, 1–512. |
| `threshold` | number | No | Tagger floor, 0–1 (default `0.35`; the model's own per-category calibrated threshold is the floor underneath it). |
| `categories` | string \| array | No | Tagger categories to score (default `general`). A prompt's tags in another category (`character`, `copyright`) count as misses unless it is selected. |
| `tags` | string \| array | No | **Narrow the scoring to these tags** (normalized the same way a caption's tags are). Every other prompt tag stops counting as a miss and every other label the tagger reports stops counting as an extra, so the scores answer "how well are these tags drawn" instead of "how well does the caption match the prompt". Omitted or empty scores every tag the prompt asks for. `evaluation_prompts` is what a client offers as the picker. |
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard`; the checkpoint's own run wins when its path sits under `output_dir`. |

The record is a job with `mode: "evaluate"`:

| Field | Description |
|---|---|
| `depth`, `threshold`, `categories` | The request, as accepted. |
| `tags` | The tags the scoring was narrowed to, normalized; `[]` means every tag a prompt asks for. The same selection is written to the run's own log directory (`{logging_dir}/{run_id}/evaluation_tags.json`, beside `sample_sets.json`), so `evaluation_prompts` offers it again the next time this run is evaluated. |
| `config_source` | The file the prompts came from: the sets saved for that run (`{logging_dir}/{run_id}/sample_sets.json`, `sample_prompts_set`), else the `config.toml` it saved beside its logs, else the hparams it recorded at startup, else today's repo `config.toml`. |
| `sample_sets` | The resolved prompt sets the top-up renders with. |
| `plan` | `{needed, depth, passes, per_pass, existing_images, render_total, sets: [{set_index, repeat, existing, render}]}`. `needed` is false when the checkpoint already holds `depth` images — then **nothing is rendered** and the pass goes straight to tagging. Otherwise `passes = ceil((depth - existing_images) / per_pass)` whole copies of the config's sample pass are rendered (`render_total = passes × per_pass` images), each set contributing its own `repeat × passes` and `render` listing the repeat indices that pass writes: the set's numbering continues after the highest index it already uses, so an earlier pass is never written over. The count overshoots `depth` by less than one pass — and the images already there count towards it, whatever produced them (the run's own sample point, an earlier `sets` pass, an earlier evaluation). |
| `images` | Every image to score: `{name, path, set_index, repeat_idx, source, prompt, tags, error}`. `source` is `run` (the trainer's own sample at the checkpoint's step) or `generated` (a recorded pass; its prompt is the one that pass drew with). `set_index` is `-1` for a `single` ad-hoc image. |
| `phase` | `rendering` → `tagging` → `scoring` → `done`; `images_done` / `total_images` are that phase's counters (nothing is rescored: a second evaluation re-tags and re-scores). |
| `files` | The images this job rendered itself (into `{output_dir}/{run_id}/{name}_samples/generated/` as `{job_id}_p{set}_{repeat}.png`), so they show on the checkpoint's card like any other pass. |
| `scores` | `null` until the pass ends, then `{tp, fp, fn, precision, recall, f1, union_tp, union_fp, union_fn, union_precision, union_recall, union_f1, images_scored, images_failed, images_skipped, tags, groups, top_false_positives, top_false_negatives}`. |

`f1` is the per-image scoreboard: the prompt is the ground truth of what was asked for, the tagger's
labels are the positive predictions, and the counts are summed over every image (a requested tag
found on 1 of 20 images is 1 true positive and 19 false negatives). `union_*` is the same over each
prompt's images pooled: a requested tag counts as found when any of them shows it. `groups` breaks
both boards down per prompt, and `top_false_positives` / `top_false_negatives` list the tags that
cost the most, most frequent first — every tag the tagger reports that the prompt did not ask for is
a false positive. Both are always floats, and every counter is an integer: `0.0` when a denominator
is empty. `scores.tags` echoes the `tags` the request narrowed to, and an image whose prompt asks for
none of them is counted in `images_skipped` rather than scored.

### `evaluation_prompts`

Read-only, no GPU: the prompts an evaluation of this checkpoint would be scored against, the tags
they ask for with their frequency, and the tags this run's last evaluation was narrowed to. This is
what the Dashboard's picker offers before a pass is started, so the selection and the scoring come
from the same config. It resolves the run and the config exactly as `evaluate_checkpoint` does (the
checkpoint's own run wins; the prompts saved for it, else its saved `config.toml`, else the hparams
it recorded, else today's file).

`selected_tags` is what that run's own evaluations recorded, in this order: the selection saved for
it (`{logging_dir}/{run_id}/evaluation_tags.json`, written when a pass is started), else the tags
the newest `evaluate` job of that run that is no longer running recorded (`tags`, or the
`scores.tags` the older records keep them in), else `[]` — a run whose evaluations never recorded a
selection, or one that scored every tag the prompts ask for. The selection is the run's, not the
checkpoint's, and it is offered for every checkpoint of that run.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `checkpoint` | string | Yes | Path to the `.safetensors` whose run's prompts are wanted. |
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard`, for a checkpoint outside `output_dir`. |

Result:

```json
{
  "run_id": "rein_20260911_120000",
  "output_name": "rein",
  "checkpoint": "/out/rein_20260911_120000/rein_s003050/rein.safetensors",
  "config_source": "/logs/rein_20260911_120000/config.toml",
  "sample_sets": [{"name": "set 1", "prompt": "1girl, anal", "repeat": 2}],
  "tags": [
    {"tag": "1girl", "count": 6, "frequency": 100.0},
    {"tag": "anal", "count": 2, "frequency": 33.3333}
  ],
  "selected_tags": ["anal"],
  "reason": ""
}
```

`tags` holds one row per tag that appears in at least one set's prompt, `count` = the sets asking for
it and `frequency` = `count / sets × 100`, most frequent first (`count` ties break on the tag name).
The list is computed over the sets, not the images, so a set that repeats 4 times counts once. A
config it cannot use is not an error: `tags` comes back empty with `reason` set, and the client scores
every tag the prompt asks for.

Refused under the same GPU rules as `generate_checkpoint_samples`, plus on a `depth` outside 1–512,
one whose whole-pass count would exceed 512 images, a `threshold` outside 0–1, and a run config with
no usable `[[validation.samples]]` set. Cancelled with `cancel_generation` (the SIGTERM lands between
rendered repeats and between tagged images; the images written so far stay in `files`).

### `cancel_generation`

Asks a running generation job to stop — the batch one included.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string \| null | No | The job to stop; null stops the running one. |

The generator handles `SIGTERM` (or `SIGINT`) by finishing the step it is in and stopping at the
next check point, so the job keeps its `running` state and its `cancelRequested` flag until the
process is really gone: the card says `cancelling…`, and no second generation may start on that GPU
meanwhile. It then closes as `cancelled` — not `error` — and keeps the images it had already
written in its `files`. Whatever was already rendered stays on the cards; the checkpoints the batch
had not reached are left alone.

A second signal leaves immediately. This method never takes the GPU gate: it has to work while the
trainer is using the card.

Result: the updated job record and `cancelled: true`.

Result: the same `{job, log_path}` shape, with `mode: "sets"`.

### `list_generated_samples`

Lists the generated samples of the resolved run, newest first (batch records among them). Read-only;
a `running` job whose process is gone is closed first — as `cancelled` when it had been asked to
stop, as `error` otherwise — so it never blocks the next one.

Params: `name` / `run_id` as in `list_samples`.

Result:

```json
{
  "run_id": "rein_20260911_120000",
  "jobs": [
    {
      "id": "rein_s000100_gen_20260915_161123",
      "state": "done",
      "mode": "single",
      "step": 100,
      "cfg": 5.0,
      "steps": 20,
      "seed": 12345,
      "current_step": 20,
      "total_steps": 20,
      "image_path": "/out/rein_20260911_120000/rein_samples/generated/rein_s000100_gen_20260915_161123.png",
      "files": [],
      "images_done": 0,
      "total_images": 1,
      "error": null
    }
  ]
}
```

`state` is `running`, `done`, `error` or `cancelled`; `jobs` is empty when the run has no
`generated/` directory.

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

### `checkpoint_pins`

Read-only. The checkpoints one run pinned in the Dashboard's Checkpoints section, read from that run's own `checkpoint_pins.json` inside its log directory.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |
| `run_id` | string \| null | No | The run whose pins to read. Defaults to the run `state.json` is on. |

Result:

```json
{
  "run_id": "rein_20260911_120000",
  "file": "/logs/rein_20260911_120000/checkpoint_pins.json",
  "pins": [
    {
      "path": "/out/rein_20260911_120000/rein_s000300/rein.safetensors",
      "dir": "rein_s000300",
      "step": 300,
      "pinned_at": 1757500000.5
    }
  ]
}
```

`run_id` and `file` are `null`, and `pins` empty, when no run resolves. Pins belong to one run — the file lives in the run's own log directory — so they survive Ranko closing and are never shared with another run. `pins` is in the order the checkpoints were pinned. A pin whose file is gone (Reset deleted the weights) stays in the file until it is unpinned; the Dashboard draws no card for it.

### `checkpoint_pin_set`

Pin or unpin one checkpoint of one run; the reply is that run's whole pin list, in the shape of `checkpoint_pins`. The file is replaced atomically, and its directory is created when the run has none (a run whose TensorBoard directory was deleted still belongs to the history list).

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | The checkpoint file — `list_checkpoints`' `path`. |
| `pinned` | bool | Yes | `true` pins, `false` unpins. |
| `name` | string \| null | No | Overrides `output_name`. |
| `run_id` | string \| null | No | Defaults to the run `state.json` is on. |
| `dir` | string \| null | No | Recorded with the pin, for a pin whose file is later gone. |
| `step` | int \| null | No | As above. |

Errors: no run resolves, `path` is empty, or a pin names a path that is not a file (`checkpoint not found: …`). Unpinning never checks the file, so a pin left behind by a Reset can still be removed.

### `train_status`

Reads `$AXL_RUNTIME_DIR` or `$XDG_RUNTIME_DIR/axltrainer/` or `/tmp/axltrainer-$UID/` (`state.json`). Reconciles a dead PID into `error`.

Params: `{}`

Result: the on-disk state plus `alive` (is that PID running) and `log_path`. A process that has exited but was never waited on counts as gone: `api.py` does not `wait()` the trainer it spawns, so a finished run's trainer is a zombie — its `/proc` entry stays and `kill(pid, 0)` still succeeds — and calling that alive kept the dashboard's GPU-busy state (and the checkpoint panel's `Generate sample`) until Ranko was restarted. Relevant state keys: `run_id` (run directory created for this run), `output_name`, and `resume` — `null` for a fresh run, otherwise

```json
{ "path": "/out/rein_…_final/rein.safetensors", "filename": "rein.safetensors", "step": 300, "epoch": 7, "loaded": 96, "skipped": 0 }
```

`status` is one of: `idle`, `starting`, `encoding`, `training`, `sampling`, `pausing`, `paused`, `resuming`, `stopping`, `finished`, `error`.

Pause/resume is a GPU swap process. While `pausing` or `resuming`, `swap` is `{stage, detail, current, total}`.

While sampling, `sampling` is `{active, repeat, repeats, denoise_step, denoise_steps, global_step, prompt_set, prompt_sets}`: `repeat`/`repeats` count the images of the whole pass (all `[[validation.samples]]` sets) and `prompt_set`/`prompt_sets` are 1-based (both `0` for a run with no sets).

`settings` is `{save_every_n_steps, sampling_enabled, next_save_step}`: the checkpoint cadence, the sampling switch and the step the next checkpoint is written at, as the trainer is actually running them. It starts from `config.toml` and follows `train_settings` (below); `next_save_step` moves whenever a checkpoint is written, and a cadence change restarts it from the step that adopted the change (`0` = no checkpoint is scheduled).

`requested` is `null`, or `{save_every_n_steps, sampling_enabled}`: a change `train_settings` accepted that the trainer has not adopted yet. It answers a click immediately — a switch flipped while a sample pass is running is accepted at once and lands when that pass ends — so a client shows the requested value and says when it applies instead of waiting for `settings` to catch up. It is `null` when nothing is outstanding, and also when no live PID is left to adopt a request (a run that ended keeps its `settings.json` until Reset).

### `train_start`

Spawns `bash start_train.sh` in a new session (`setsid`) so closing Ranko does not stop training. Stdout/stderr append to `train.log` in the runtime dir.

Params: `{}`

It also publishes `[training].save_every_n_steps` / `sampling_enabled` as this run's `settings` (and into `settings.json`), so the dashboard shows the run's cadence from the moment Start is pressed rather than the empty placeholder.

Fails if a live training PID already exists, including a process that has already marked `finished` but has not exited yet. Also fails synchronously — before any GPU work — when `[training].resume_lora_path` is set but does not resolve to a `.safetensors` file, when that file's `ss_network_type` does not match `[network].network_type`, and when `[environment].amdfq` is `tail` or `vmm` but the corresponding `target/release/libamdfq_*_rs.so` is missing.

### `train_pause` / `train_resume` / `train_stop`

Writes `command.json` (`pause` | `resume` | `stop`). The trainer consumes it at the next swap-safe point.

Params: `{}`

Fails if no live training PID. `train_resume` additionally fails while a `generate_sample` /
`generate_checkpoint_samples` job is running, because resuming would load a second SDXL next to the
generator.

Pause offloads UNet / text encoders / optimizer state / VAE to CPU and `empty_cache`s. Resume reloads what the paused phase needs. Early-stop during encoding does not save a LoRA; during training it saves `{output_name}.safetensors` if that step has no checkpoint yet; during sampling it skips leftover repeats (the step checkpoint already exists).

### `train_settings`

Retunes the run in progress: the checkpoint cadence, the sampling switch, or both. `train_start`
seeds the request from `config.toml`, so a change made here lasts for this run only — the file keeps
the value the next run starts from.

Params (at least one):

| Field | Type | Required | Description |
|---|---|---|---|
| `save_every_n_steps` | integer | No | `>= 1` (or `0` to stop writing checkpoints). The next checkpoint is written `N` steps after the step that adopts the change. |
| `sampling_enabled` | bool | No | Whether a checkpoint save also renders the `[[validation.samples]]` images. `false` = checkpoints only. |

A field the request leaves out keeps the run's value: the one in `settings.json` when it is there (`train_start` keeps that file complete), otherwise the values the run published in `settings` — so flipping the sampling switch alone can never change the cadence.

The request is written to `settings.json` in the runtime dir; the trainer adopts it at its next
optimizer step (after a pause, at the step the run resumes with) and publishes the effective values
back through `train_status.settings`. The reply is the `train_status` payload, so it carries the
change in `requested` right away and keeps the old values in `settings` until the trainer adopts
it — a sample pass already running finishes first, which is why the two differ for a while.

Fails if no live training PID, on an out-of-range value, and when neither field is given.

### `train_reset`

Clears the on-disk trainer state back to `idle` so `train_start` can launch a new run. **Nothing is deleted** — the LoRA checkpoint directories, the sample images and the TensorBoard logs all stay, which is what makes the cleared run browsable in the run history afterwards. Weights are removed only by `python clean.py`, which asks before it does.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string \| null | No | Overrides `output_name`. |
| `run_id` | string \| null | No | Run to clear. Resolved like `dashboard`. |

The result carries `run_id` and `cleanup` (`run_dir`, `samples_dir`, `log_dir`, `weight_dirs`, `delete_weights`, `delete_samples`, `delete_logs`, `removed`, `skipped`, `errors`); the three `delete_*` flags are always `false` here, so every path lands in `skipped` and `removed` is empty. A `delete_weights` param is not part of the method any more: Reset never removes a weight directory, and an older client still sending it changes nothing.

When no run directory resolves, `run_id` is `null` and **nothing is deleted** — legacy flat artifacts are only reachable via `python clean.py --legacy-flat`.

Fails if the training PID is still alive. `clean.py` remains the CLI cleaner and uses the same helper — it is the tool that deletes a run's samples, logs and weights.

### `sample_prompts`

Read-only: the prompts one run samples with, where they were resolved from, and whether that run is
the live one. This is what the Dashboard's Sampling Prompts section shows, and the same resolution
`evaluate_checkpoint` and the manual sample passes use, so what the section says is what those
render.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard` (a request that names nothing means the run `state.json` is on). |

Result:

```json
{
  "run_id": "rein_20260911_120000",
  "output_name": "rein",
  "file": "/logs/rein_20260911_120000/sample_sets.json",
  "edited": true,
  "config_source": "/logs/rein_20260911_120000/sample_sets.json",
  "sets": [
    {"name": "cowgirl", "prompt": "1girl, cowgirl position", "negative": "worst quality",
     "width": 1152, "height": 768, "steps": 35, "guidance_scale": 6.0,
     "guidance_rescale": 0.6, "seed": 0, "repeat": 2}
  ],
  "live": false,
  "reason": ""
}
```

`sets` is the whole resolved list — every key present, which is what the editor writes back. `edited`
is true once sets were saved for this run, and `file` is the JSON they live in (inside the run's own
log directory, beside the `config.toml` copy, which is never rewritten); otherwise the prompts are the
run's saved config, named by `config_source`. `live` is true while this run is the one the trainer is
running — an edit then lands on the run's next sample point. A config it cannot use is not an error:
`sets` comes back empty with `reason` set, and a request that resolves no run answers with
`run_id: null` and `reason: no run to read sampling prompts for`.

### `sample_prompts_set`

Saves the prompts a run should sample with, from then on.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `sets` | array \| null | Yes | The new `[[validation.samples]]` entries, or `null` to drop this run's own prompts so it uses its saved config again. |
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard`. |

The list is validated and stored whole in `{logging_dir}/{run_id}/sample_sets.json` (a directory that
does not exist yet is created, like `checkpoint_pin_set`'s). Nothing is written to `config.toml`, and
the run's `config.toml` snapshot is never rewritten: it stays the record of what the run trained with.
Every write path layers the file over that snapshot — the trainer's own sample points read it before
each pass, so an edit made while the run is live applies at its next checkpoint, and an evaluation or
a manual sample pass resolves it the same way.

An entry that omits a key is filled from **that run's** config, never from the repo file, so a partial
request cannot mix two runs' settings; the stored entries are complete. Ranges are api.py's
`resolve_sample_sets` ranges (`width`/`height` 64–4096, `steps` 1–150, `guidance_scale` 0–30,
`guidance_rescale` 0–1, `seed` 0–4294967295, `repeat` 1–32, a non-blank `prompt`), and the rejection
names the offending entry (`validation.samples[2]: steps must be between 1 and 150`).

The reply has the shape of `sample_prompts` — the run's whole prompt state after the change — so the
client refreshes from it without a second call.

Refused when the request names no run (`no run to set sampling prompts on`) or carries no `sets` at
all (`sets is required (null resets the run to its own config)`).

### `clear_checkpoint_samples`

Removes one checkpoint's sample images: the samples the run wrote at its step, the generated and
evaluated images of every pass that belongs to that card, and the job records (with their `.log`)
that produced them. The checkpoint file itself is kept, and nothing else is touched — another step's
samples, another checkpoint's passes and a range batch's own record stay where they are.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `checkpoint` | string | Yes | Path to the `.safetensors` whose card is being cleared. |
| `name` / `run_id` | string \| null | No | Resolve the run like `dashboard`; the checkpoint's own run wins when its path sits under `output_dir` (as in `evaluate_checkpoint`). |

A job belongs to the card when it names that checkpoint path, or — for a record that names none — when
its `step` is the checkpoint's step. That is the same rule the Checkpoints section groups by, so what
the card shows is exactly what goes.

Result:

```json
{
  "run_id": "rein_20260911_120000",
  "output_name": "rein",
  "checkpoint": "/out/rein_20260911_120000/rein_s003050/rein.safetensors",
  "step": 3050,
  "images": 7,
  "jobs": ["rein_s003050_sets_gen_20261002_101500"],
  "files": ["/out/rein_20260911_120000/rein_samples/rein_003050_p0_0.png"]
}
```

`step` is `null` for a checkpoint whose directory name and metadata both carry none (its pass records
are still cleared, by path). A file already gone is not an error and is not listed — a second press
answers with `images: 0` — and a file that could not be removed is warned about on the helper's
stderr.

Refused while a live trainer is using the GPU (`pause` the run, or stop it) and while another
generation is running, because both write the directory being cleaned.

### `dataset_tag`

Runs `tagger2/main.py` (the Pixai tagger v1: ViTDet, 30 877 Danbooru tags, PyTorch/ROCm) with the same interpreter as `api.py` (the `axl` env). Writes comma-separated captions next to every image in a folder (non-recursive). Overwrites existing `.txt` files. Ranko should reload Images / Statistics after a successful call. The legacy WD14 ONNX script (`tagger/main.py`) still runs by hand; its `selected_tags.csv` is what `tag_lexicon` reads for the Chinese tag names.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `directory` | string \| null | No | Dataset folder. Defaults to `[environment].train_data_dir`. |
| `threshold` | number | No | Lowest confidence a tag may keep, `0.0`–`1.0`. Default `0.35`. The model's own per-category calibrated threshold is the floor underneath it (`max(calibrated, threshold)`). |
| `categories` | list[string] \| string | No | Which of the model's six categories reach the caption: `general` (default), `character`, `copyright`, `style`, `meta`, `rating`. Unknown names are refused by the script. |
| `only_tags` | list[string] \| string | No | **Partial tagging**: a list of tags to *add* to the captions that show them. The categories are not used (a requested tag is looked up in all six), and `threshold` becomes the absolute confidence floor for these tags instead of a floor under the model's calibrated values. |
| `batch_size` | integer | No | Images per forward pass, `>= 1`. Omitting it leaves the script's own default (`1`; measured to make no difference on the author's GPU). |

Result:

```json
{
  "directory": "/abs/path",
  "engine": "pixai-tagger-v1.0",
  "mode": "full",
  "categories": ["general"],
  "threshold": 0.35,
  "thresholds": { "general": 0.35 },
  "provider": "pixai-tagger-v1.0 on cuda:0",
  "device": "cuda:0",
  "total": 100,
  "processed": 100,
  "failed": 0,
  "seconds": 12.3,
  "errors": [{ "file": "0001.png", "error": "…" }]
}
```

With `only_tags` the same call runs a **partial pass** instead, and the result says so:

```json
{
  "directory": "/abs/path",
  "engine": "pixai-tagger-v1.0",
  "mode": "partial",
  "only_tags": ["anal", "pussy"],
  "categories": [],
  "threshold": 0.6,
  "thresholds": {},
  "provider": "pixai-tagger-v1.0 on cuda:0",
  "device": "cuda:0",
  "total": 100,
  "processed": 15,
  "failed": 0,
  "seconds": 12.5,
  "errors": [],
  "added": { "anal": 12, "pussy": 3 },
  "unmatched": ["nonexistent tag"]
}
```

A partial pass is **add-only**: for each image it looks up the requested tags in the model's output and
appends the ones above `threshold` that the caption does not already hold (a weighted or underscored
form counts as already held), leaving every other tag exactly as written. `processed` counts the images
whose sidecar was actually rewritten, `added` says how many images gained each tag, and `unmatched`
lists requested tags that matched no image at all (usually a spelling the model does not use). A
caption that would gain nothing is not touched, and an image with no sidecar gains one only when
something matched.

Fails if the folder is missing, `threshold` is out of range, a training process is in a GPU-using status (`starting` / `encoding` / `training` / `sampling` / `pausing` / `resuming` / `stopping`), or the model is not in the local Hugging Face cache (the script never reaches the Hub on its own; `--download` is the one-time fetch). Pause (`paused`) is allowed because weights are offloaded. The tagger is a child process so GPU memory is released when it exits.

### `tagger_info`

Read-only, no GPU: what the tagger can write, straight from the model's own `config.json` (`tagger2/main.py --info`). Ranko's Auto-tag card uses it for the category switches and for the calibrated values it draws on the threshold slider.

Result:

```json
{
  "available": true,
  "engine": "pixai-tagger-v1.0",
  "model": "pixai-labs/pixai-tagger-v1.0",
  "model_path": "/home/…/snapshots/9fe10ad…",
  "cache_dir": "/repo/tagger2/miopen_cache",
  "categories": [
    { "key": "general", "count": 15043, "calibrated": 0.17 },
    { "key": "character", "count": 8308, "calibrated": 0.27 },
    { "key": "copyright", "count": 2460, "calibrated": 0.24 },
    { "key": "style", "count": 4917, "calibrated": 0.15 },
    { "key": "meta", "count": 145, "calibrated": 0.17 },
    { "key": "rating", "count": 4, "calibrated": 0.41 }
  ],
  "default_categories": ["general"],
  "reason": ""
}
```

Never fails: with the model not cached, `available` is `false`, `categories` is empty and `reason` says how to fetch it. The same shape is returned when the helper itself cannot run.


### `dataset_counts`

Read-only, no GPU: how many images each training folder holds, for the Utils → Training section's
step estimate. Counts the same way the trainer's dataset does — `jpg` / `jpeg` / `png` / `webp` /
`bmp`, `*.mask.png` excluded, recursively — and applies each folder's `repeat` to the per-epoch
figure.

Params:

| Field | Type | Required | Description |
|---|---|---|---|
| `dirs` | array of `{path, repeat}` | No | The folders the form currently holds, so the answer follows unsaved edits. Omitted (or empty), the config's own `[[environment.train_data]]` entries are used. |

Result:

```json
{
  "entries": [
    { "path": "/data/a", "repeat": 3, "images": 100, "error": null },
    { "path": "/data/gone", "repeat": 1, "images": 0, "error": "not a directory" }
  ],
  "images": 100,
  "samples": 300
}
```

`samples` is the per-epoch figure with repeats applied (`Σ images × repeat`); a folder that cannot be
read carries its reason and counts as zero rather than failing the call. The step arithmetic itself is
the client's (`model/StepEstimate.kt`), so epoch / batch / GA edits need no round trip.

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
- `dataset_counts` `{dirs?}` → `{entries: [{path, repeat, images, error}], images, samples}`. Read-only, no GPU; the Utils → Training step estimate's image counts (see the method above).
- `caption_write` `{directory, stem, text}`.
- `dataset_drop` `{directory, rate, seed?, stems?}`. Moves image+txt+mask to `/tmp/axlranko/trash`. `stems` limits the pool (the GUI passes the filtered set).
- `dataset_shuffle` `{directory, seed?}` → `{groups, renamed_files, first_stem, last_stem}`.
- `mask_get` / `mask_write` / `mask_delete` `{directory, stem, png_base64?}`. Lossless PNG only.
- `blob_stat` / `blob_batch` `{paths, max_edge, quality?, format?}`. `max_edge` is required (32–4096, contain, never upscale). Default `quality=80`, `format=jpeg`. JPEG/WebP flatten transparency onto black before encoding (dropping the alpha channel would leak leftover RGB in transparent pixels). Result items carry `hash` (SHA-256 of the processed bytes), `width`/`height`, `cache` (`hit`/`miss`), and for `blob_batch` `base64`. A bad path is a per-item `error`, not a failed RPC. Encode fans out across `AXL_BLOB_WORKERS` spawn processes (`trainer/blobcodec.py`, torch-free). Hits live under `/tmp/axlranko/blob-cache/` capped at 1/4 of host RAM. The cache key includes a codec version, so a flatten/resize change does not reuse bytes from an older encoder.
- `tag_lexicon` `{}` → `{text}` of `tagger/selected_tags.csv`.
- `tagger_info` `{}` → the tagger's `{available, engine, model, model_path, cache_dir, categories, default_categories, reason}` (see the method above). Read-only and never failing.
- `evaluation_prompts` `{checkpoint, name?, run_id?}` → the prompts and prompt-tag frequencies an evaluation of that checkpoint would score against (see the method above). Read-only and never failing.
- `sample_prompts` `{name?, run_id?}` → one run's effective prompt sets, the file they were resolved from (`edited` / `file` when the Dashboard saved sets for it), and whether it is the live run. Read-only and never failing. `sample_prompts_set` `{sets, name?, run_id?}` saves them for that run (`sets: null` drops them again) and answers with the same shape. `clear_checkpoint_samples` `{checkpoint, name?, run_id?}` removes one card's images and the pass records that produced them (see the three methods above — its params and reply are documented there).
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
- `automation_job_get` `{id}` → the whole record plus `summary` and `log_tail` (last 40 lines): per-prompt `{index, text, state, seed, prompt_id, images, image_seeds, error}`.
- `automation_job_cancel` `{id}` → SIGTERMs the runner's process group, waits briefly, marks the job `cancelled`; a half-finished job keeps its images. `automation_job_retry_failed` `{id}` respawns the runner with `--only-failed` (the prompts that already produced images are not queued again). `automation_job_delete` `{id}` removes the job directory (refused while it runs).
- `automation_image_regenerate` `{id, image}` → redraws **one** image with a new random seed, writing over that file (and its sidecar) and updating that image's entry in `image_seeds`; the prompt's other images are untouched. `automation_prompt_extend` `{id, prompt_index, count}` → adds `count` (1..16) images to one prompt, **each from its own random seed**, named with the next free numbers (`p0003_02.png`…). `automation_prompt_extend_all` `{id, count}` → the same for **every** prompt of the job, one entry after another, each continuing its own numbering. All three spawn `trainer/run_automation.py` detached and answer with the job's whole detail (the same payload `automation_job_get` returns); all three refuse while that job runs, on an unknown image/prompt, or a `count` out of range. A pass in flight is in the record's `pass` block as `mode` `image` / `append` / `append_all` with `prompt_index`, `images_done`, `total_images` and the last `image` written.
- `automation_image_delete` `{id, image}` → removes that image and its `.txt` sidecar, and drops the name from the record. Deleting a prompt's **last** image drops that prompt entry, and the remaining entries are renumbered so their `index` keeps matching their position (image files are never renamed). Refuses while the job runs, or for a name no prompt of that job holds.
- `automation_job_prompt_edit` `{id, prompt_index, text}` → rewrites one prompt's text in the record and nothing else. The `.txt` beside an already rendered image keeps what was actually sent; the next regeneration uses the new text. Refuses an empty or over-long (4000 char) text, but is allowed while the job runs (the runner never writes `text` back).
- Every job writes one `.txt` next to each image with `seed`, `prompt_id`, `prompt` and ComfyUI's own file name; images are named by us (`p0003_01.png`) so the order never depends on the workflow's `filename_prefix`. Only `SaveImage` outputs are collected, `PreviewImage` nodes are ignored. `image_seeds` is one seed per name in `images` (a job from before this exists has no such list — use `seed`, the last pass's).

`vmm_va` is present only when `[environment].amdfq` is `"vmm"`. `used_bytes` is the GPU VA the VMM hook holds — what it has mapped right now, or, with `never_reuse`, everything it has ever mapped — read from `$AXL_RUNTIME_DIR/amdfq_vmm_va.<trainer-pid>.json` (0 if the trainer is not running or has not written yet). `never_reuse` is that file's mode when it is there (the running hook's own mode) and `[environment].amdfq_va_never_reuse` otherwise; Ranko titles the bar `GPU VA (live)` / `GPU VA (not returned)` from it. `total_bytes` is the GPU VM size: `journalctl -k` `vm size is N GB` first (no sudo), then `dmesg`, then `/sys/module/amdgpu/parameters/vm_size` when that value is positive, otherwise 256 TiB. `total_source` is `journal` / `dmesg` / `sysfs` / `default`. The module parameter is often `-1` (auto) and is not the live size.

## Example

```python
python -c "import api; print(api.dispatch('ping'))"
```
