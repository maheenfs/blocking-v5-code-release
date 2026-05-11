"""Dashboard settings and path helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Mapping, Optional

from experiment.system import load_system_settings, package_root_from_config


DEFAULT_DASHBOARD_SETTINGS: Dict[str, object] = {
    "enabled": True,
    "path": "auto",
    "refresh_seconds": 5,
    "show_resource_monitor": True,
    "show_latest_epochs": 5,
    "show_plotting_plan": True,
}


def now_text() -> str:
    """Return a local timestamp for dashboard snapshots."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def dashboard_settings(config: object) -> Dict[str, object]:
    """Load dashboard settings from system_settings.json with safe defaults."""
    config_path = getattr(config, "config_path", None)
    settings = load_system_settings(config_path)
    result = dict(DEFAULT_DASHBOARD_SETTINGS)
    raw = settings.get("dashboard")
    if isinstance(raw, Mapping):
        result.update({str(key): value for key, value in raw.items()})
    return result


def dashboard_enabled(config: object, override: Optional[bool] = None) -> bool:
    """Return whether HTML dashboard writing is enabled for this run."""
    if override is not None:
        return bool(override)
    return bool(dashboard_settings(config).get("enabled", True))


def dashboard_path(config: object) -> Path:
    """Resolve the dashboard HTML path relative to the release package root."""
    config_path = getattr(config, "config_path", None)
    root = package_root_from_config(config_path)
    raw = dashboard_settings(config).get("path", "auto")
    if raw in (None, "", "auto"):
        return Path(str(getattr(config, "outdir"))) / "dashboard.html"
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else root / path
