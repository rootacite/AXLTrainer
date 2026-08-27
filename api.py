import json
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from trainer.config import TrainConfig, _load_toml_config
from trainer.loss_log import synthesize_avg_loss

_IPC_STDOUT = sys.stdout


def _write(payload: dict[str, Any]) -> None:
    _IPC_STDOUT.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    _IPC_STDOUT.flush()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _train_config_dict() -> dict[str, Any]:
    cfg = TrainConfig()
    data = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    data.update(_load_toml_config())
    return _json_safe(data)


def _get_tensorboard_metrics(
    log_dir: str,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
) -> dict:
    if not os.path.exists(log_dir):
        return {}

    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        return {}

    latest_log_dir = str(max(event_files, key=os.path.getmtime).parent)
    ea = EventAccumulator(latest_log_dir, size_guidance={"scalars": 0})
    ea.Reload()

    metrics: dict = {}
    if "scalars" in ea.Tags():
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            filtered = [
                {"step": e.step, "value": float(e.value), "wall_time": float(e.wall_time)}
                for e in events
                if (start_step is None or e.step >= start_step)
                and (end_step is None or e.step <= end_step)
            ]
            metrics[tag] = filtered

    return metrics


def handle_ping(_params: dict[str, Any]) -> dict[str, str]:
    return {"status": "ok"}


def handle_dashboard(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    output_name = params.get("name") or cfg.get("output_name", "default")
    logging_dir = cfg.get("logging_dir", "./logs")
    target_log_dir = os.path.join(str(logging_dir), str(output_name))

    start_step = params.get("start_step")
    end_step = params.get("end_step")
    metrics = _get_tensorboard_metrics(target_log_dir, start_step, end_step)
    if not metrics.get("Train/Avg_Loss") and metrics.get("Train/Loss"):
        metrics["Train/Avg_Loss"] = synthesize_avg_loss(metrics["Train/Loss"])

    latest_stats: dict[str, Any] = {}
    for tag, data in metrics.items():
        if data:
            latest_stats[tag] = data[-1]["value"]
            latest_stats["current_step"] = data[-1]["step"]

    return {"config": cfg, "latest_stats": latest_stats, "metrics": metrics}


_SAMPLE_NAME = re.compile(r"_(\d+)_(\d+)\.png$")


def scan_samples(sample_dir: Path) -> dict[str, list]:
    if not sample_dir.exists():
        return {}

    grouped: dict[int, list] = defaultdict(list)
    for img_path in sample_dir.glob("*.png"):
        match = _SAMPLE_NAME.search(img_path.name)
        if match:
            step, repeat_idx = int(match.group(1)), int(match.group(2))
        else:
            step, repeat_idx = -1, 0
        grouped[step].append(
            {
                "filename": img_path.name,
                "repeat_idx": repeat_idx,
                "path": str(img_path.resolve()),
            }
        )

    for step in grouped:
        grouped[step] = sorted(grouped[step], key=lambda x: x["repeat_idx"])

    return {str(k): grouped[k] for k in sorted(grouped.keys(), reverse=True)}


def handle_list_samples(params: dict[str, Any]) -> dict[str, Any]:
    cfg = _train_config_dict()
    output_dir = cfg.get("output_dir", "./output")
    output_name = params.get("name") or cfg.get("output_name", "default")
    sample_dir = Path(str(output_dir)) / f"{output_name}_samples"
    return {"samples": scan_samples(sample_dir)}


_HANDLERS = {
    "ping": handle_ping,
    "dashboard": handle_dashboard,
    "list_samples": handle_list_samples,
}


def dispatch(method: str, params: Optional[dict[str, Any]] = None) -> Any:
    handler = _HANDLERS.get(method)
    if handler is None:
        raise ValueError(f"unknown method: {method}")
    return handler(params or {})


def run_ipc_loop() -> None:
    # Keep stdout exclusive for NDJSON IPC. All logs go to stderr.
    sys.stdout = sys.stderr
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            if not isinstance(method, str) or not method:
                raise ValueError("missing method")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("params must be an object")
            result = dispatch(method, params)
            _write({"id": req_id, "ok": True, "result": _json_safe(result)})
        except Exception as exc:
            traceback.print_exc()
            _write({"id": req_id, "ok": False, "error": str(exc)})


if __name__ == "__main__":
    run_ipc_loop()
