"""Progress snapshot helpers for plots.py commands."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

from experiment.config import load_config
from experiment.io import atomic_write_text

from .render import write_dashboard_html
from .resources import runtime_metrics_payload
from .settings import dashboard_enabled, dashboard_path, now_text


def selection_summary(selection: object) -> str:
    """Convert a PlotSelection object into one readable line."""
    fields = [
        ("kpis", getattr(selection, "kpis", ())),
        ("families", getattr(selection, "families", ())),
        ("seeds", getattr(selection, "seeds", ())),
        ("patterns", getattr(selection, "patterns", ())),
        ("blocked_beams", getattr(selection, "blocked_beams", ())),
        ("train_fracs", getattr(selection, "train_fracs", ())),
        ("blockages", getattr(selection, "blockages", ())),
        ("epochs", getattr(selection, "epochs", ())),
    ]
    parts = []
    for label, values in fields:
        if values:
            parts.append(f"{label}={','.join(str(value) for value in values)}")
    if getattr(selection, "organize", False):
        parts.append("organize=true")
    return "; ".join(parts) if parts else "default plotting selection"


def write_plotting_progress(
    *,
    config_path: str | Path,
    results_root: str | Path,
    command: str,
    selection: object,
    status: str,
    message: str = "",
    error: Optional[BaseException] = None,
) -> None:
    """Write plotting progress JSON and refresh the optional dashboard."""
    config = load_config(config_path)
    package_root = Path(str(config.config_path)).parent
    results_root_path = Path(results_root).expanduser()
    if not results_root_path.is_absolute():
        results_root_path = package_root / results_root_path
    results_root_path.mkdir(parents=True, exist_ok=True)
    plots_root = package_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "updated_at": now_text(),
        "command": str(command),
        "status": str(status),
        "selection": selection_summary(selection),
        "message": str(message),
    }
    if error is not None:
        payload["error"] = str(error)
        payload["traceback"] = traceback.format_exc()

    progress_path = plots_root / "plotting_live_progress.json"
    atomic_write_text(progress_path, json.dumps(payload, indent=2, sort_keys=False) + "\n")

    if not dashboard_enabled(config):
        return

    active = {
        "status": str(status),
        "current_stage": f"plotting:{command}",
        "plotting_progress": payload,
        "error": payload.get("error", ""),
        "traceback": payload.get("traceback", ""),
    }
    snapshot = {
        "updated_at": now_text(),
        "total_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 1 if status == "failed" else 0,
        "running_jobs": 1 if status == "running" else 0,
        "pending_jobs": 0,
        "seed_count": 0,
        "active_job_ordinal": 0,
        "batch_progress_pct": 0.0,
        "average_completed_job_hms": "",
        "current_job_eta_hms": "",
        "current_job_eta_finish_at": "",
        "batch_eta_hms": "",
        "batch_eta_finish_at": "",
        "seed_summaries": [],
        "seed_overview": {},
        "runtime_metrics": runtime_metrics_payload(results_root_path),
        "remaining_jobs": [],
        "active_run": active,
        "rows": [],
    }
    write_dashboard_html(dashboard_path(config), config=config, snapshot=snapshot)
