"""Runtime, hardware, and resource monitoring helpers."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from experiment.system import load_system_settings

from .formatting import format_runtime_float, format_seconds
from .settings import now_text


PROCESS_STARTED_AT = time.time()
PROCESS_CPU_STARTED_AT = time.process_time()


def process_max_rss_mb() -> Optional[float]:
    """Return process high-water memory in MB when the platform exposes it."""
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _psutil_metrics() -> Dict[str, object]:
    """Read optional system memory metrics when psutil is installed."""
    try:
        import psutil
    except Exception:
        return {"psutil_available": False}

    memory = psutil.virtual_memory()
    return {
        "psutil_available": True,
        "system_ram_total_mb": float(memory.total) / (1024.0 * 1024.0),
        "system_ram_available_mb": float(memory.available) / (1024.0 * 1024.0),
        "system_ram_used_pct": float(memory.percent),
    }


def _disk_metrics(results_root: Path | None = None) -> Dict[str, object]:
    """Capture free disk space near the results directory."""
    root = Path(".") if results_root is None else Path(results_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root)
    except Exception:
        return {}
    total = float(usage.total)
    free = float(usage.free)
    return {
        "disk_path": str(root.resolve()),
        "disk_total_gb": total / (1024.0**3),
        "disk_free_gb": free / (1024.0**3),
        "disk_free_pct": 100.0 * free / total if total else 0.0,
    }


def _nvidia_smi_metrics() -> Dict[str, object]:
    """Read NVIDIA GPU utilization when nvidia-smi is available."""
    query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=1.5,
        )
    except Exception:
        return {"nvidia_smi_available": False}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"nvidia_smi_available": False}

    gpus = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        name, util, used, total, temp = parts
        gpus.append(
            {
                "name": name,
                "utilization_pct": _float_or_none(util),
                "memory_used_mb": _float_or_none(used),
                "memory_total_mb": _float_or_none(total),
                "temperature_c": _float_or_none(temp),
            }
        )
    return {"nvidia_smi_available": True, "nvidia_gpus": gpus}


def _float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def runtime_metrics_payload(results_root: Path | None = None) -> Dict[str, object]:
    """Capture lightweight live runtime and CPU/GPU/resource metrics."""
    cpu_count = int(os.cpu_count() or 1)
    wall_s = max(1e-9, time.time() - PROCESS_STARTED_AT)
    cpu_s = max(0.0, time.process_time() - PROCESS_CPU_STARTED_AT)
    payload: Dict[str, object] = {
        "captured_at": now_text(),
        "cpu_count": cpu_count,
        "process_wall_s": wall_s,
        "process_wall_hms": format_seconds(wall_s),
        "process_cpu_time_s": cpu_s,
        "process_cpu_time_hms": format_seconds(cpu_s),
        "process_avg_cpu_pct": 100.0 * cpu_s / wall_s,
    }
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        payload.update(
            {
                "load_1m": float(load_1m),
                "load_5m": float(load_5m),
                "load_15m": float(load_15m),
                "load_1m_per_cpu_pct": 100.0 * float(load_1m) / float(cpu_count),
            }
        )
    except Exception:
        pass

    max_rss_mb = process_max_rss_mb()
    if max_rss_mb is not None:
        payload["process_max_rss_mb"] = float(max_rss_mb)

    payload.update(_psutil_metrics())
    payload.update(_disk_metrics(results_root))
    payload.update(_nvidia_smi_metrics())
    payload.update(_torch_accelerator_metrics())
    return payload


def _torch_accelerator_metrics() -> Dict[str, object]:
    """Capture torch CUDA or Apple MPS memory metrics when torch is present."""
    payload: Dict[str, object] = {}
    try:
        import torch

        payload["torch_version"] = str(torch.__version__)
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            device_index = int(torch.cuda.current_device())
            payload["cuda_device"] = device_index
            payload["cuda_device_name"] = str(torch.cuda.get_device_name(device_index))
            payload["cuda_memory_allocated_mb"] = float(torch.cuda.memory_allocated(device_index)) / (1024.0 * 1024.0)
            payload["cuda_memory_reserved_mb"] = float(torch.cuda.memory_reserved(device_index)) / (1024.0 * 1024.0)
            payload["cuda_max_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved(device_index)) / (1024.0 * 1024.0)

        mps_backend = getattr(torch.backends, "mps", None)
        payload["torch_mps_available"] = bool(mps_backend and mps_backend.is_available())
        torch_mps = getattr(torch, "mps", None)
        if payload["torch_mps_available"] and torch_mps is not None:
            current_allocated = getattr(torch_mps, "current_allocated_memory", None)
            driver_allocated = getattr(torch_mps, "driver_allocated_memory", None)
            if current_allocated is not None:
                payload["mps_memory_allocated_mb"] = float(current_allocated()) / (1024.0 * 1024.0)
            if driver_allocated is not None:
                payload["mps_driver_memory_mb"] = float(driver_allocated()) / (1024.0 * 1024.0)
    except Exception as exc:
        payload["accelerator_metrics_error"] = str(exc)
    return payload


def hardware_snapshot_payload(config_path: Path | None = None) -> Dict[str, object]:
    """Capture reproducibility-relevant host details without shell scripts."""
    settings = load_system_settings(config_path)
    payload: Dict[str, object] = {
        "captured_at": now_text(),
        "configured_device": settings.get("device", "cpu"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy as np

        payload["numpy_version"] = np.__version__
    except Exception:
        pass
    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        payload["torch_mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass
    return payload


def resource_rows(runtime: Mapping[str, object]) -> List[List[str]]:
    """Render CPU/GPU/resource metrics into rows for the dashboard."""
    rows = [
        ["Process wall time", str(runtime.get("process_wall_hms", "n/a"))],
        ["Process CPU time", str(runtime.get("process_cpu_time_hms", "n/a"))],
        ["Average process CPU", f"{format_runtime_float(runtime.get('process_avg_cpu_pct'))}%"],
        ["CPU count", str(runtime.get("cpu_count", "n/a"))],
    ]
    if "load_1m" in runtime:
        rows.append(
            [
                "System load 1/5/15m",
                (
                    f"{format_runtime_float(runtime.get('load_1m'))} / "
                    f"{format_runtime_float(runtime.get('load_5m'))} / "
                    f"{format_runtime_float(runtime.get('load_15m'))}"
                ),
            ]
        )
        rows.append(["1m load per CPU", f"{format_runtime_float(runtime.get('load_1m_per_cpu_pct'))}%"])
    if "process_max_rss_mb" in runtime:
        rows.append(["Max process memory", f"{format_runtime_float(runtime.get('process_max_rss_mb'))} MB"])
    if runtime.get("psutil_available"):
        rows.append(["System RAM used", f"{format_runtime_float(runtime.get('system_ram_used_pct'))}%"])
        rows.append(["System RAM available", f"{format_runtime_float(runtime.get('system_ram_available_mb'))} MB"])
    if "disk_free_gb" in runtime:
        rows.append(["Disk free", f"{format_runtime_float(runtime.get('disk_free_gb'))} GB"])
    if runtime.get("cuda_device_name"):
        rows.extend(
            [
                ["CUDA device", str(runtime.get("cuda_device_name"))],
                ["CUDA memory allocated", f"{format_runtime_float(runtime.get('cuda_memory_allocated_mb'))} MB"],
                ["CUDA memory reserved", f"{format_runtime_float(runtime.get('cuda_memory_reserved_mb'))} MB"],
                ["CUDA max reserved", f"{format_runtime_float(runtime.get('cuda_max_memory_reserved_mb'))} MB"],
            ]
        )
    elif runtime.get("torch_mps_available"):
        rows.append(["Apple MPS", "available"])
        if "mps_memory_allocated_mb" in runtime:
            rows.append(["MPS memory allocated", f"{format_runtime_float(runtime.get('mps_memory_allocated_mb'))} MB"])
        if "mps_driver_memory_mb" in runtime:
            rows.append(["MPS driver memory", f"{format_runtime_float(runtime.get('mps_driver_memory_mb'))} MB"])
    else:
        rows.append(["Accelerator", "CPU or no live accelerator metrics"])

    for index, gpu in enumerate(runtime.get("nvidia_gpus", []) or []):
        if not isinstance(gpu, Mapping):
            continue
        rows.append(
            [
                f"NVIDIA GPU {index}",
                (
                    f"{gpu.get('name', 'unknown')} | util {format_runtime_float(gpu.get('utilization_pct'))}% | "
                    f"VRAM {format_runtime_float(gpu.get('memory_used_mb'))}/"
                    f"{format_runtime_float(gpu.get('memory_total_mb'))} MB | "
                    f"{format_runtime_float(gpu.get('temperature_c'))} C"
                ),
            ]
        )
    return rows


def resource_warning_rows(runtime: Mapping[str, object]) -> List[List[str]]:
    """Return warning rows for resource conditions that can break long runs."""
    rows: List[List[str]] = []
    if float(runtime.get("disk_free_gb", 999.0) or 999.0) < 5.0:
        rows.append(["Disk space", "Less than 5 GB free near the results folder."])
    if float(runtime.get("disk_free_pct", 100.0) or 100.0) < 10.0:
        rows.append(["Disk space", "Less than 10% free near the results folder."])
    if float(runtime.get("system_ram_used_pct", 0.0) or 0.0) > 90.0:
        rows.append(["System RAM", "More than 90% of RAM is in use."])
    if float(runtime.get("load_1m_per_cpu_pct", 0.0) or 0.0) > 150.0:
        rows.append(["CPU load", "Load is high relative to the number of CPU cores."])
    for gpu in runtime.get("nvidia_gpus", []) or []:
        if not isinstance(gpu, Mapping):
            continue
        used = _float_or_none(gpu.get("memory_used_mb"))
        total = _float_or_none(gpu.get("memory_total_mb"))
        if used is not None and total and total > 0 and used / total > 0.9:
            rows.append(["GPU memory", f"{gpu.get('name', 'GPU')} is using more than 90% of VRAM."])
        temp = _float_or_none(gpu.get("temperature_c"))
        if temp is not None and temp >= 85.0:
            rows.append(["GPU temperature", f"{gpu.get('name', 'GPU')} is at {temp:.0f} C."])
    return rows
