# API Reference Documentation

This document specifies the calling conventions, interface functionalities, parameter definitions, and response structures for the consolidated LoRA Training & Generation API.

---

## 1. Global Specifications

* **Base URL:** `http://<host>:<port>` (Default: `http://localhost:8000`)
* **Content-Type:** * `GET` requests: Standard query parameters.
* `POST` requests: `application/json` for the request body, query parameters appended to the URL where applicable.


* **Authentication:** None (Internal/Local network deployment).

---

## 2. Dashboard & Metrics Endpoints

### 2.1 Get Dashboard Data

Retrieves the active training configuration, real-time scalar metrics extracted from TensorBoard logs, and the most recent status snapshot.

* **HTTP Method:** `GET`
* **Endpoint:** `/api/dashboard`
* **Query Parameters:**

| Parameter    | Type    | Required | Description                                                                                                     |
|--------------|---------|----------|-----------------------------------------------------------------------------------------------------------------|
| `name`       | string  | No       | Overrides the target output subdirectory name. Defaults to the `output_name` found in the server configuration. |
| `start_step` | integer | No       | Filters and returns metrics containing a step counter greater than or equal to this value.                      |
| `end_step`   | integer | No       | Filters and returns metrics containing a step counter less than or equal to this value.                         |


* **Response Structure (`application/json`):**
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
    "Train/Loss": "float",
    "UNet/LR/Effective_Actual_LR": "float",
    "current_step": "integer"
  },
  "metrics": {
    "Metric/Tag/Name": [
      {
        "step": "integer",
        "value": "float",
        "wall_time": "float"
      }
    ]
  }
}

```



---

### 2.2 List Generated Samples

Scans the training output directory and returns a structural list of generated sample images grouped by their step intervals.

* **HTTP Method:** `GET`
* **Endpoint:** `/api/samples`
* **Query Parameters:**

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | string | No | Target output execution name used to locate the sample image directory folder. |


* **Response Structure (`application/json`):**
```json
{
  "samples": {
    "1000": [
      {
        "filename": "string (e.g., sample_1000_0.png)",
        "repeat_idx": "integer"
      }
    ],
    "-1": [
      {
        "filename": "string",
        "repeat_idx": "integer"
      }
    ]
  }
}

```


> *Note: A step key of `-1` indicates metadata parsing anomalies or unindexed diagnostic files.*



---

### 2.3 Fetch Sample Image Binary

Streams the raw file content of a specific image sample directly from disk.

* **HTTP Method:** `GET`
* **Endpoint:** `/api/samples/{filename}`
* **Path Parameters:**

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `filename` | string | Yes | The exact file name string obtained from the `/api/samples` list. |


* **Query Parameters:**

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | string | No | Output folder identifier matching the associated training instance run. |


* **Response Structure:**
* **Success (200 OK):** Binary stream (`media_type="image/png"`).
* **Failure (404 Not Found):** `{"detail": "Sample image not found"}`



---

## 3. Image Generation Endpoints

### 3.1 Shared JSON Body Data Models (`PromptOverrides`)

When dispatching `POST` requests, the options below can be wrapped inside a JSON payload object to temporarily modify module runtime constants:

```json
{
  "positive_prompt": "string or null",
  "negative_prompt": "string or null",
  "base_model_path": "string or null",
  "lora_path": "string or null",
  "lora_scale": "float or null",
  "realesrgan_model_path": "string or null",
  "max_token_length": "integer or null",
  "clip_skip": "integer or null",
  "output_filename_prefix": "string or null",
  "refinement_passes": [
    {
      "name": "string",
      "model": "string",
      "denoise": "float",
      "guide_size": "integer"
    }
  ]
}

```

---

### 3.2 Quick Mode Generation

Executes the primary base checkpoint pass only. This mirrors baseline debugging loops and bypasses complex upscaling chains.

* **HTTP Method:** `POST`
* **Endpoint:** `/api/quick`
* **Query Parameters:**

| Parameter   | Type    | Required | Description                                              |
|-------------|---------|----------|----------------------------------------------------------|
| `seed`      | integer | No       | Explicit pseudo-random generation seed initialization.   |
| `steps`     | integer | No       | Number of inference steps. Must be positive.             |
| `cfg_scale` | float   | No       | Classifier-free guidance scale threshold value.          |
| `width`     | integer | No       | Targeting horizontal image resolution. Must be positive. |
| `height`    | integer | No       | Targeting vertical image resolution. Must be positive.   |


* **Request Body:** `PromptOverrides` JSON object or empty body `{}`.
* **Response Structure:** Binary image payload (`media_type="image/png"`).

---

### 3.3 Full Pipeline Generation

Coordinates multi-stage inference workflows. It chains base rendering, structural super-resolution tile transformations, and deep inpainting detailing passes.

* **HTTP Method:** `POST`
* **Endpoint:** `/api/generate`
* **Query Parameters:**

| Parameter   | Type    | Required | Default | Description                             |
|-------------|---------|----------|---------|-----------------------------------------|
| `stages`    | integer | No       | `3`     | Execution pipeline cut-off point        |
| `seed`      | integer | No       | null    | Random generator anchor point index.    |
| `steps`     | integer | No       | null    | Main scheduler iteration interval.      |
| `cfg_scale` | float   | No       | null    | Prompt adherence intensity coefficient. |
| `width`     | integer | No       | null    | Initial sample width specification.     |
| `height`    | integer | No       | null    | Initial sample height specification.    |


* **Request Body:** `PromptOverrides` JSON object or empty body `{}`.
* **Response Structure:** Binary image payload (`media_type="image/png"`).

---

## 4. Curl Invocation Examples

### Example 1: Fetch filtered dashboard timeline step series

Extract metric steps recorded strictly inside the runtime interval boundaries between Step 500 and Step 1200.

```bash
curl -X 'GET' \
  'http://localhost:8000/api/dashboard?name=lora_run_01&start_step=500&end_step=1200' \
  -H 'accept: application/json'

```

### Example 2: Request execution of the full 3-Stage pipeline

Submit explicit model alterations, text conditions, and structural configurations to output a finalized high-fidelity PNG asset.

```bash
curl -X 'POST' \
  'http://localhost:8000/api/generate?stages=3&steps=30&cfg_scale=7.5' \
  -H 'accept: image/png' \
  -H 'Content-Type: application/json' \
  -d '{
    "positive_prompt": "masterpiece, 1girl, cyberpunk neon street atmosphere",
    "negative_prompt": "low quality, blurry, distorted hands",
    "lora_scale": 0.85
  }' \
  --output final_render.png

```