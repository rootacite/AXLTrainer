"""Settings, workflows and job records for the Automation page. Torch-free.

Layout under the repo root:

    automation/settings.json          what the page remembers (server, workflow, count, …)
    automation/workflows/<name>.json  API-format workflows uploaded from the client
    automation/prompts/<name>.txt     saved prompt sets, one prompt per line
    automation/jobs/<job_id>/job.json one run's record
    automation/jobs/<job_id>/log.txt  the runner's stdout/stderr
    automation/jobs/<job_id>/images/  the images plus a sidecar .txt each

The whole `automation/` tree is gitignored; training settings stay in `config.toml`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# `python trainer/run_automation.py` puts trainer/ on sys.path, `import api` does not;
# support both (see AGENT.md "Import dualism").
try:
    from control import is_pid_alive
except ImportError:
    from trainer.control import is_pid_alive

ROOT_NAME = "automation"
SETTINGS_NAME = "settings.json"
WORKFLOWS_DIR = "workflows"
PROMPTS_DIR = "prompts"
JOBS_DIR = "jobs"
IMAGES_DIR = "images"

STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"
STATES = (STATE_RUNNING, STATE_DONE, STATE_ERROR, STATE_CANCELLED)

PROMPT_STATE_PENDING = "pending"
PROMPT_STATE_RUNNING = "running"
PROMPT_STATE_DONE = "done"
PROMPT_STATE_ERROR = "error"
PROMPT_STATES = (PROMPT_STATE_PENDING, PROMPT_STATE_RUNNING, PROMPT_STATE_DONE, PROMPT_STATE_ERROR)

MIN_COUNT = 1
MAX_COUNT = 16
MIN_POLL = 0.1
MAX_POLL = 10.0
MAX_PROMPTS = 400
MAX_PROMPT_CHARS = 4000
MAX_NAME = 64
NAME_ILLEGAL = set('/\\:*?"<>|')

_STAMP = "%Y%m%d_%H%M%S"
_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
CLASS_SAVE_IMAGE = "SaveImage"
CLASS_TEXT_ENCODE = "CLIPTextEncode"
SAMPLER_CLASSES = ("KSampler", "KSamplerAdvanced", "KSampler (Efficient)")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def automation_root() -> Path:
    raw = os.environ.get("AXL_AUTOMATION_DIR")
    if raw:
        return Path(raw).expanduser()
    return repo_root() / ROOT_NAME


def settings_path() -> Path:
    return automation_root() / SETTINGS_NAME


def workflows_dir() -> Path:
    return automation_root() / WORKFLOWS_DIR


def prompts_dir() -> Path:
    return automation_root() / PROMPTS_DIR


def jobs_dir() -> Path:
    return automation_root() / JOBS_DIR


def default_output_dir() -> Path:
    return jobs_dir()


def default_settings() -> dict[str, Any]:
    return {
        "server": "",
        "workflow": "",
        "positive_node": "",
        "count": 1,
        "poll": 0.5,
        "output_dir": str(default_output_dir()),
    }


# --- json helpers ---


def atomic_write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    """Temp file + rename, the pattern `control.py` uses for `state.json`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_name(name: str, what: str = "name") -> str:
    trimmed = str(name or "").strip()
    if not trimmed:
        raise ValueError(f"a {what} is required")
    if len(trimmed) > MAX_NAME:
        raise ValueError(f"use at most {MAX_NAME} characters")
    if trimmed.startswith("."):
        raise ValueError(f"a {what} cannot start with a dot")
    if any(ch in NAME_ILLEGAL or ord(ch) < 32 for ch in trimmed):
        raise ValueError(f'a {what} cannot contain / \\ : * ? " < > |')
    return trimmed


def safe_name(name: str, fallback: str = "item") -> str:
    cleaned = _UNSAFE_NAME_RE.sub("_", str(name or "").strip()).strip().strip(".")
    return cleaned[:MAX_NAME] or fallback


# --- settings ---


def load_settings() -> dict[str, Any]:
    payload = _read_json(settings_path())
    merged = default_settings()
    if payload:
        for key in merged:
            if key in payload:
                merged[key] = payload[key]
    try:
        return normalize_settings(merged)
    except ValueError:
        # A hand-edited file must not lock the page out of its own settings.
        fallback = default_settings()
        fallback["server"] = str(merged.get("server") or "")
        return fallback


def normalize_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    defaults = default_settings()
    server = str(payload.get("server") or "").strip()
    if server and not server.startswith(("http://", "https://")):
        server = "http://" + server
    workflow = str(payload.get("workflow") or "").strip()
    positive = str(payload.get("positive_node") or "").strip()

    try:
        count = int(payload.get("count", defaults["count"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("count must be an integer") from exc
    if not MIN_COUNT <= count <= MAX_COUNT:
        raise ValueError(f"count must be between {MIN_COUNT} and {MAX_COUNT}")

    try:
        poll = float(payload.get("poll", defaults["poll"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("poll must be a number") from exc
    if not MIN_POLL <= poll <= MAX_POLL:
        raise ValueError(f"poll must be between {MIN_POLL:g} and {MAX_POLL:g} seconds")

    output_raw = str(payload.get("output_dir") or "").strip()
    output = Path(output_raw).expanduser() if output_raw else default_output_dir()
    if not output.is_absolute():
        output = (repo_root() / output).resolve()
    return {
        "server": server,
        "workflow": workflow,
        "positive_node": positive,
        "count": count,
        "poll": poll,
        "output_dir": str(output),
    }


def save_settings(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_settings(payload)
    atomic_write_json(settings_path(), normalized)
    return normalized


# --- workflows ---


def workflow_path(name: str) -> Path:
    return workflows_dir() / f"{validate_name(name, 'workflow name')}.json"


def workflow_list() -> list[dict[str, Any]]:
    directory = workflows_dir()
    if not directory.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.name.lower()):
        payload = _read_json(path)
        items.append(
            {
                "name": path.stem,
                "path": str(path),
                "valid": payload is not None,
                "error": None if payload is not None else "invalid JSON",
            }
        )
    return items


def workflow_save(name: str, text: str) -> dict[str, Any]:
    target = workflow_path(name)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the workflow text is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"workflow is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("a workflow must be a JSON object")
    report = validate_workflow(payload)
    if not report["valid"]:
        raise ValueError(f"workflow is not API format: {report['error']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return {"name": target.stem, "path": str(target), **report}


def workflow_delete(name: str) -> dict[str, Any]:
    target = workflow_path(name)
    if not target.is_file():
        raise ValueError(f"the workflow is already gone: {name}")
    if target.resolve().parent != workflows_dir().resolve():
        raise ValueError(f"not a workflow file: {target}")
    target.unlink()
    return {"name": target.stem}


def load_workflow(path: Union[str, Path]) -> dict[str, Any]:
    target = Path(path).expanduser()
    if not target.is_file():
        raise ValueError(f"no workflow file at {target}")
    payload = _read_json(target)
    if payload is None:
        raise ValueError(f"{target.name}: invalid JSON")
    return payload


# --- workflow validation ---


def _numeric_batch_nodes(workflow: Mapping[str, Any]) -> list[str]:
    changed: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get("batch_size"), (int, float)) and not isinstance(
            inputs.get("batch_size"), bool
        ):
            changed.append(str(node_id))
    return changed


def _text_nodes(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != CLASS_TEXT_ENCODE:
            continue
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        text = inputs.get("text")
        nodes.append(
            {
                "id": str(node_id),
                "class_type": CLASS_TEXT_ENCODE,
                "text": text if isinstance(text, str) else "",
            }
        )
    return nodes


def guess_positive_node(workflow: Mapping[str, Any]) -> str:
    """The CLIPTextEncode a sampler's `positive` link points at, else a lone one, else ""."""
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") not in SAMPLER_CLASSES:
            continue
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        link = inputs.get("positive")
        if isinstance(link, list) and link and isinstance(link[0], str):
            upstream = workflow.get(link[0])
            if isinstance(upstream, dict) and upstream.get("class_type") == CLASS_TEXT_ENCODE:
                return str(link[0])
    texts = _text_nodes(workflow)
    if len(texts) == 1:
        return texts[0]["id"]
    return ""


def combo_options(spec: Any) -> Optional[list[str]]:
    """The choices of a combo input, in either shape ComfyUI 0.35 reports.

    `["COMBO", {"options": [...]}]` (UpscaleModelLoader here) and `[[names…], {…}]`
    (DiffusersLoader / UltralyticsDetectorProvider) both appear on the same instance.
    """
    if isinstance(spec, list) and len(spec) == 2 and spec[0] == "COMBO" and isinstance(spec[1], dict):
        options = spec[1].get("options")
        return list(options) if isinstance(options, list) else None
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return list(spec[0])
    return None


def missing_models(workflow: Mapping[str, Any], object_info: Mapping[str, Any]) -> list[dict[str, str]]:
    """Literal enum inputs the instance does not offer (a model that is not installed)."""
    missing: list[dict[str, str]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        meta = object_info.get(str(node.get("class_type")))
        if not isinstance(meta, dict):
            continue
        declared = meta.get("input") if isinstance(meta.get("input"), dict) else {}
        specs: dict[str, Any] = {}
        for bucket in ("required", "optional"):
            entries = declared.get(bucket)
            if isinstance(entries, dict):
                specs.update(entries)
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        for name, value in inputs.items():
            if isinstance(value, list):  # a link, not a literal
                continue
            if name not in specs:
                continue
            options = combo_options(specs[name])
            if options is None or not isinstance(value, str):
                continue
            if value not in options:
                missing.append(
                    {"node": str(node_id), "class_type": str(node.get("class_type")), "input": name, "value": value}
                )
    return missing


def validate_workflow(
    workflow: Any,
    positive_node: str = "",
    object_info: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Structural checks, the positive-prompt candidates, and (given `object_info`) a model pre-check."""
    report: dict[str, Any] = {
        "valid": False,
        "error": None,
        "node_count": 0,
        "save_image_nodes": [],
        "batch_size_nodes": [],
        "text_nodes": [],
        "positive_node": "",
        "positive_node_guessed": False,
        "missing_models": [],
        "checked_models": False,
    }
    if not isinstance(workflow, dict) or not workflow:
        report["error"] = "the workflow is empty"
        return report
    if "nodes" in workflow or "links" in workflow:
        report["error"] = "this is the editor format; export the workflow as API format"
        return report
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            report["error"] = f"node {node_id!r} has no class_type"
            return report
        inputs = node.get("inputs")
        if inputs is not None and not isinstance(inputs, dict):
            report["error"] = f"node {node_id!r} has no inputs object"
            return report

    report["node_count"] = len(workflow)
    report["save_image_nodes"] = [
        str(node_id) for node_id, node in workflow.items() if node.get("class_type") == CLASS_SAVE_IMAGE
    ]
    if not report["save_image_nodes"]:
        report["error"] = "the workflow has no SaveImage node"
        return report
    report["batch_size_nodes"] = _numeric_batch_nodes(workflow)
    report["text_nodes"] = _text_nodes(workflow)
    if not report["text_nodes"]:
        report["error"] = "the workflow has no CLIPTextEncode node for the prompt"
        return report

    chosen = str(positive_node or "").strip()
    if chosen:
        node = workflow.get(chosen)
        if not isinstance(node, dict):
            report["error"] = f"node {chosen!r} does not exist"
            return report
        if node.get("class_type") != CLASS_TEXT_ENCODE:
            report["error"] = f"node {chosen!r} is {node.get('class_type')!r}, not a CLIPTextEncode"
            return report
        report["positive_node"] = chosen
    else:
        guessed = guess_positive_node(workflow)
        report["positive_node"] = guessed
        report["positive_node_guessed"] = bool(guessed)

    if object_info is not None:
        report["checked_models"] = True
        report["missing_models"] = missing_models(workflow, object_info)

    report["valid"] = True
    return report


# --- prompt sets ---


def normalize_prompts(prompts: Any) -> list[str]:
    """One prompt per line (or a list); blanks dropped, length and count bounded."""
    if isinstance(prompts, str):
        raw_lines = prompts.splitlines()
    elif isinstance(prompts, (list, tuple)):
        raw_lines = [str(item) for item in prompts]
    else:
        raise ValueError("prompts must be a list of strings or a text block")
    cleaned: list[str] = []
    for line in raw_lines:
        text = line.strip()
        if not text:
            continue
        if len(text) > MAX_PROMPT_CHARS:
            raise ValueError(f"a prompt is longer than {MAX_PROMPT_CHARS} characters")
        cleaned.append(text)
    if not cleaned:
        raise ValueError("no prompts to generate")
    if len(cleaned) > MAX_PROMPTS:
        raise ValueError(f"at most {MAX_PROMPTS} prompts per job")
    return cleaned


def prompt_set_path(name: str) -> Path:
    return prompts_dir() / f"{validate_name(name, 'prompt set name')}.txt"


def prompt_set_list() -> list[dict[str, Any]]:
    directory = prompts_dir()
    if not directory.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.txt"), key=lambda p: p.name.lower()):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        items.append(
            {
                "name": path.stem,
                "path": str(path),
                "count": len([line for line in text.splitlines() if line.strip()]),
                "text": text,
            }
        )
    return items


def prompt_set_get(name: str) -> dict[str, Any]:
    target = prompt_set_path(name)
    if not target.is_file():
        raise ValueError(f"no prompt set named {name}")
    return {"name": target.stem, "path": str(target), "text": target.read_text(encoding="utf-8")}


def prompt_set_save(name: str, text: Any) -> dict[str, Any]:
    prompts = normalize_prompts(text)
    target = prompt_set_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = "\n".join(prompts) + "\n"
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(blob, encoding="utf-8")
    os.replace(tmp, target)
    return {"name": target.stem, "path": str(target), "count": len(prompts)}


def prompt_set_delete(name: str) -> dict[str, Any]:
    target = prompt_set_path(name)
    if not target.is_file():
        raise ValueError(f"the prompt set is already gone: {name}")
    if target.resolve().parent != prompts_dir().resolve():
        raise ValueError(f"not a prompt set file: {target}")
    target.unlink()
    return {"name": target.stem}


# --- jobs ---


def new_job_id(stem: str, *, now: Optional[Union[datetime, float]] = None) -> str:
    if now is None:
        stamp = datetime.now()
    elif isinstance(now, (int, float)):
        stamp = datetime.fromtimestamp(now)
    else:
        stamp = now
    return f"{safe_name(stem, 'automation')}_{stamp.strftime(_STAMP)}"


def job_dir(job_id: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    root = Path(output_dir).expanduser() if output_dir else default_output_dir()
    return root / str(job_id)


def job_path(job_id: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    return job_dir(job_id, output_dir) / "job.json"


def log_path(job_id: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    return job_dir(job_id, output_dir) / "log.txt"


def images_dir(job_id: str, output_dir: Optional[Union[str, Path]] = None) -> Path:
    return job_dir(job_id, output_dir) / IMAGES_DIR


def image_name(prompt_index: int, image_index: int) -> str:
    """Our own name: `p0003_01.png`, so the order does not depend on ComfyUI's prefix."""
    return f"p{prompt_index + 1:04d}_{image_index:02d}.png"


def read_job(path: Union[str, Path]) -> Optional[dict[str, Any]]:
    return _read_json(Path(path))


def write_job(job: Mapping[str, Any], output_dir: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    payload = dict(job)
    target = job_path(str(payload.get("id") or "job"), payload.get("output_dir") or output_dir)
    atomic_write_json(target, payload)
    return payload


def update_job(job_id: str, output_dir: Optional[Union[str, Path]] = None, **fields: Any) -> dict[str, Any]:
    """Read-modify-write one job; a concurrent api.py call cannot lose fields it wrote."""
    path = job_path(job_id, output_dir)
    payload = read_job(path) or {"id": job_id}
    payload.update(fields)
    payload["updated_at"] = time.time()
    atomic_write_json(path, payload)
    return payload


def list_jobs(output_dir: Optional[Union[str, Path]] = None) -> list[dict[str, Any]]:
    """Every job under the output root, newest first. Malformed files are skipped, never raised."""
    root = Path(output_dir).expanduser() if output_dir else default_output_dir()
    if not root.is_dir():
        return []
    jobs: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        payload = read_job(entry / "job.json")
        if payload is None or not payload.get("id"):
            continue
        if payload.get("state") not in STATES:
            payload["state"] = STATE_ERROR
            payload.setdefault("error", "job file has no valid state")
        jobs.append(payload)
    jobs.sort(key=lambda job: (float(job.get("started_at") or job.get("created_at") or 0.0), str(job.get("id"))), reverse=True)
    return jobs


def job_summary(job: Mapping[str, Any]) -> dict[str, Any]:
    prompts = job.get("prompts") if isinstance(job.get("prompts"), list) else []
    images: list[str] = []
    for prompt in prompts:
        if isinstance(prompt, dict) and isinstance(prompt.get("images"), list):
            images.extend(str(name) for name in prompt["images"])
    preview = images[:8]
    done = sum(1 for p in prompts if isinstance(p, dict) and p.get("state") == PROMPT_STATE_DONE)
    failed = sum(1 for p in prompts if isinstance(p, dict) and p.get("state") == PROMPT_STATE_ERROR)
    output_dir = job.get("output_dir")
    root = images_dir(str(job.get("id")), output_dir) if job.get("id") else None
    return {
        "id": job.get("id"),
        "state": job.get("state"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
        "total": len(prompts),
        "done": done,
        "failed": failed,
        "images": len(images),
        "preview_paths": [str(root / name) for name in preview] if root is not None else [],
        "workflow": job.get("workflow"),
        "output_dir": job.get("output_dir"),
        "positive_node": job.get("positive_node"),
        "count": job.get("count"),
        "comfy_url": job.get("comfy_url"),
        "error": job.get("error"),
    }


def reconcile_jobs(output_dir: Optional[Union[str, Path]] = None) -> list[dict[str, Any]]:
    """A `running` job whose process is gone is an error; the record is rewritten once."""
    jobs = list_jobs(output_dir)
    for job in jobs:
        if job.get("state") != STATE_RUNNING:
            continue
        if is_pid_alive(job.get("pid")):
            continue
        updated = update_job(
            str(job["id"]),
            job.get("output_dir") or output_dir,
            state=STATE_ERROR,
            error=job.get("error") or "the runner exited before finishing (see the job's log.txt)",
            finished_at=job.get("finished_at") or time.time(),
        )
        job.clear()
        job.update(updated)
    return jobs
