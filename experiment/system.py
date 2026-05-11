"""Local runtime settings that do not change the experiment protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SYSTEM_SETTINGS: dict[str, object] = {
    "device": "cpu",
    "torch_num_threads": None,
    "torch_num_interop_threads": None,
    "matplotlib_cache_dir": "./.cache/matplotlib",
    "xdg_cache_dir": "./.cache",
    "dashboard": {
        "enabled": True,
        "path": "auto",
        "refresh_seconds": 5,
        "show_resource_monitor": True,
        "show_latest_epochs": 5,
        "show_plotting_plan": True,
    },
}


def package_root_from_config(config_path: Path | None = None) -> Path:
    """Resolve the release root used for system-local settings."""
    if config_path is None:
        return Path(__file__).resolve().parents[1]
    return Path(config_path).expanduser().resolve().parent


def _resolve_setting_path(root: Path, raw: object) -> Path:
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else root / path


def load_system_settings(config_path: Path | None = None) -> dict[str, object]:
    """Load system-only runtime settings, falling back to safe CPU defaults."""
    root = package_root_from_config(config_path)
    settings = dict(DEFAULT_SYSTEM_SETTINGS)
    path = root / "system_settings.json"
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            settings.update({str(key): value for key, value in loaded.items()})
            default_dashboard = DEFAULT_SYSTEM_SETTINGS.get("dashboard")
            loaded_dashboard = loaded.get("dashboard")
            if isinstance(default_dashboard, Mapping) and isinstance(loaded_dashboard, Mapping):
                dashboard_settings = dict(default_dashboard)
                dashboard_settings.update({str(key): value for key, value in loaded_dashboard.items()})
                settings["dashboard"] = dashboard_settings
    return settings


def configure_plot_environment(config_path: Path | None = None) -> None:
    """Set Matplotlib cache locations before Matplotlib is imported."""
    root = package_root_from_config(config_path)
    settings = load_system_settings(config_path)
    mpl_cache = _resolve_setting_path(root, settings.get("matplotlib_cache_dir", "./.cache/matplotlib"))
    xdg_cache = _resolve_setting_path(root, settings.get("xdg_cache_dir", "./.cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _maybe_set_torch_threads(torch_module: Any, settings: Mapping[str, object]) -> None:
    for key, setter_name in (
        ("torch_num_threads", "set_num_threads"),
        ("torch_num_interop_threads", "set_num_interop_threads"),
    ):
        raw = settings.get(key)
        if raw is None:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{key} must be a positive integer or null.")
        setter = getattr(torch_module, setter_name, None)
        if setter is not None:
            setter(value)


def select_torch_device(torch_module: Any, requested: object) -> Any:
    """Choose a torch.device from cpu, auto, cuda, cuda:N, or mps."""
    name = str(requested or "cpu").strip().lower()
    if name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if name.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError("system_settings.json requested CUDA, but torch.cuda.is_available() is false.")
        return torch_module.device(name)
    if name == "mps":
        if not (getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available()):
            raise RuntimeError("system_settings.json requested MPS, but torch.backends.mps.is_available() is false.")
        return torch_module.device("mps")
    if name == "cpu":
        return torch_module.device("cpu")
    raise ValueError(f"Unsupported device in system_settings.json: {requested!r}")


def configure_torch_runtime(config_path: Path | None = None) -> Any:
    """Apply local PyTorch runtime settings and return the selected device."""
    import torch

    settings = load_system_settings(config_path)
    _maybe_set_torch_threads(torch, settings)
    return select_torch_device(torch, settings.get("device", "cpu"))
