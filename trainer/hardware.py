"""Host hardware snapshot for Ranko: nvtop JSON + sysfs temps + /proc CPU.

nvtop 3.3.2 has no Unix socket; `-s/--snapshot` prints a JSON array of GPUs.
Process lists are dropped (cmdlines are huge). Edge/junction come from DRM
hwmon; CPU util is a /proc/stat delta, temp from thermal zones, RAM from
/proc/meminfo. CPU package power is not collected (RAPL/turbostat need root).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

NVTOP_TIMEOUT_S = 2.0
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_SKIP_THERMAL_TYPES = frozenset(
    {
        "acpitz",
        "iwlwifi",
        "iwlwifi_1",
        "pch",
        "pch_cannonlake",
        "pch_skylake",
        "pch_cometlake",
        "nvme",
        "amdgpu",
    }
)

# (idle, total) from the previous /proc/stat sample.
_cpu_prev: tuple[int, int] | None = None


def reset_cpu_tracker() -> None:
    global _cpu_prev
    _cpu_prev = None


def parse_metric_number(raw: Any) -> float | None:
    """Turn nvtop fields like '92%', '303W', '72C', '2165MHz' into floats."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    text = str(raw).strip()
    if not text:
        return None
    match = _NUM_RE.search(text)
    if not match:
        return None
    return float(match.group(0))


def parse_bytes(raw: Any) -> int | None:
    number = parse_metric_number(raw)
    if number is None:
        return None
    return int(number)


def collect_hardware_status(
    *,
    nvtop_bin: Optional[str] = None,
    nvtop_runner: Optional[Callable[[], list[Any]]] = None,
    drm_root: str | Path = "/sys/class/drm",
    proc_stat: str | Path = "/proc/stat",
    proc_cpuinfo: str | Path = "/proc/cpuinfo",
    proc_meminfo: str | Path = "/proc/meminfo",
    thermal_root: str | Path = "/sys/class/thermal",
    now: Optional[float] = None,
) -> dict[str, Any]:
    ts = time.time() if now is None else float(now)
    error: str | None = None
    gpus: list[dict[str, Any]] = []
    try:
        raw_list = nvtop_runner() if nvtop_runner is not None else _run_nvtop_snapshot(nvtop_bin)
        if not isinstance(raw_list, list):
            raise RuntimeError("nvtop -s did not return a JSON array")
        for index, item in enumerate(raw_list):
            if isinstance(item, dict):
                gpus.append(_parse_gpu(item, index))
        if not gpus:
            error = "nvtop returned no GPUs"
    except FileNotFoundError:
        error = "nvtop not found on PATH"
    except subprocess.TimeoutExpired:
        error = "nvtop snapshot timed out"
    except json.JSONDecodeError:
        error = "nvtop snapshot was not valid JSON"
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__

    hwmon_temps = read_gpu_hwmon_temps(drm_root)
    for gpu, temps in zip(gpus, hwmon_temps):
        _apply_hwmon_temps(gpu, temps)

    cpu = read_cpu_snapshot(
        proc_stat=proc_stat,
        proc_cpuinfo=proc_cpuinfo,
        proc_meminfo=proc_meminfo,
        thermal_root=thermal_root,
    )
    return {
        "available": error is None and bool(gpus),
        "error": error,
        "ts": ts,
        "gpus": gpus,
        "cpu": cpu,
    }


def _run_nvtop_snapshot(nvtop_bin: Optional[str] = None) -> list[Any]:
    binary = nvtop_bin or shutil.which("nvtop")
    if not binary:
        raise FileNotFoundError("nvtop not found on PATH")
    proc = subprocess.run(
        [binary, "-s"],
        capture_output=True,
        text=True,
        timeout=NVTOP_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"nvtop exited {proc.returncode}"
        raise RuntimeError(detail)
    return json.loads(proc.stdout)


def _parse_gpu(raw: dict[str, Any], index: int) -> dict[str, Any]:
    temp = parse_metric_number(raw.get("temp"))
    name = str(raw.get("device_name") or "").strip() or f"GPU {index}"
    return {
        "index": index,
        "name": name,
        "gpu_clock_mhz": parse_metric_number(raw.get("gpu_clock")),
        "mem_clock_mhz": parse_metric_number(raw.get("mem_clock")),
        "fan_pct": parse_metric_number(raw.get("fan_speed")),
        "gpu_util_pct": parse_metric_number(raw.get("gpu_util")),
        "mem_util_pct": parse_metric_number(raw.get("mem_util")),
        "power_w": parse_metric_number(raw.get("power_draw")),
        "temp_c": temp,
        "temp_edge_c": temp,
        "temp_junction_c": None,
        "temp_mem_c": None,
        "mem_total_bytes": parse_bytes(raw.get("mem_total")),
        "mem_used_bytes": parse_bytes(raw.get("mem_used")),
        "mem_free_bytes": parse_bytes(raw.get("mem_free")),
    }


def read_gpu_hwmon_temps(drm_root: str | Path = "/sys/class/drm") -> list[dict[str, float]]:
    root = Path(drm_root)
    if not root.is_dir():
        return []
    cards: list[tuple[int, Path]] = []
    for child in root.iterdir():
        name = child.name
        if not name.startswith("card"):
            continue
        suffix = name[4:]
        if not suffix.isdigit():
            continue
        cards.append((int(suffix), child))
    cards.sort()

    found: list[dict[str, float]] = []
    for _, card in cards:
        hwmon_dir = card / "device" / "hwmon"
        if not hwmon_dir.is_dir():
            continue
        for hwmon in sorted(hwmon_dir.glob("hwmon*")):
            temps = _read_hwmon_temps(hwmon)
            if temps:
                found.append(temps)
                break
    return found


def _read_hwmon_temps(hwmon: Path) -> dict[str, float]:
    temps: dict[str, float] = {}
    for label_path in hwmon.glob("temp*_label"):
        stem = label_path.name  # temp1_label
        idx = stem[len("temp") :].split("_", 1)[0]
        input_path = hwmon / f"temp{idx}_input"
        try:
            label = label_path.read_text(encoding="utf-8").strip().lower()
            millideg = float(input_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if label:
            temps[label] = millideg / 1000.0
    return temps


def _apply_hwmon_temps(gpu: dict[str, Any], temps: dict[str, float]) -> None:
    if "edge" in temps:
        gpu["temp_edge_c"] = temps["edge"]
        gpu["temp_c"] = temps["edge"]
    if "junction" in temps:
        gpu["temp_junction_c"] = temps["junction"]
    if "mem" in temps:
        gpu["temp_mem_c"] = temps["mem"]


def read_cpu_snapshot(
    *,
    proc_stat: str | Path = "/proc/stat",
    proc_cpuinfo: str | Path = "/proc/cpuinfo",
    proc_meminfo: str | Path = "/proc/meminfo",
    thermal_root: str | Path = "/sys/class/thermal",
) -> dict[str, Any]:
    idle, total = _read_proc_stat(proc_stat)
    util = _cpu_util_from_delta(idle, total)
    n_logical = os.cpu_count() or 0
    mem_total, mem_used = _read_meminfo(proc_meminfo)
    return {
        "name": _cpu_model_name(proc_cpuinfo),
        "n_logical": int(n_logical),
        "util_pct": util,
        "temp_c": _cpu_temp_c(thermal_root),
        "mem_total_bytes": mem_total,
        "mem_used_bytes": mem_used,
    }


def _cpu_util_from_delta(idle: int, total: int) -> float | None:
    global _cpu_prev
    util: float | None = None
    if _cpu_prev is not None:
        prev_idle, prev_total = _cpu_prev
        d_idle = idle - prev_idle
        d_total = total - prev_total
        if d_total > 0:
            util = max(0.0, min(100.0, (1.0 - d_idle / d_total) * 100.0))
    _cpu_prev = (idle, total)
    return util


def _read_proc_stat(path: str | Path) -> tuple[int, int]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return 0, 0
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        parts = line.split()
        # cpu user nice system idle iowait irq softirq steal guest guest_nice
        values = [int(x) for x in parts[1:] if x.isdigit() or (x.startswith("-") and x[1:].isdigit())]
        if len(values) < 4:
            break
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values[:8]) if len(values) >= 8 else sum(values)
        return idle, total
    return 0, 0


def _read_meminfo(path: str | Path) -> tuple[int | None, int | None]:
    """Return (mem_total_bytes, mem_used_bytes) from /proc/meminfo."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None, None
    fields: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        number = parse_metric_number(rest)
        if number is None:
            continue
        fields[key.strip()] = int(number) * 1024
    total = fields.get("MemTotal")
    if total is None or total <= 0:
        return None, None
    available = fields.get("MemAvailable")
    if available is not None:
        used = max(0, total - available)
        return total, used
    free = fields.get("MemFree", 0)
    buffers = fields.get("Buffers", 0)
    cached = fields.get("Cached", 0)
    used = max(0, total - free - buffers - cached)
    return total, used


def _cpu_model_name(path: str | Path) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            return value.strip()
    return ""


def _cpu_temp_c(thermal_root: str | Path) -> float | None:
    root = Path(thermal_root)
    if not root.is_dir():
        return None
    zones: list[tuple[int, str, Path]] = []
    for zone in sorted(root.glob("thermal_zone*")):
        type_path = zone / "type"
        temp_path = zone / "temp"
        try:
            kind = type_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        rank = _thermal_rank(kind)
        if rank is None:
            continue
        zones.append((rank, kind, temp_path))
    zones.sort()
    for _, _, temp_path in zones:
        try:
            millideg = float(temp_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        # Some drivers report already-Celsius tenths; millidegree is the sysfs contract.
        if millideg > 1000:
            return millideg / 1000.0
        return millideg
    return None


def _thermal_rank(kind: str) -> int | None:
    lowered = kind.strip().lower()
    if lowered == "x86_pkg_temp":
        return 0
    if lowered == "k10temp":
        return 1
    if "pkg" in lowered or lowered in {"tctl", "tdie"}:
        return 2
    if lowered in _SKIP_THERMAL_TYPES or lowered.startswith("iwlwifi") or lowered.startswith("pch_"):
        return None
    if lowered.startswith("acpi"):
        return None
    return 10
