"""Compatibility imports for the dashboard package.

Runner code imports from runner.dashboard, while the implementation now lives
in the top-level dashboard package so each concern stays in a small file.
"""

from dashboard import (
    DEFAULT_DASHBOARD_SETTINGS,
    dashboard_enabled,
    dashboard_path,
    dashboard_settings,
    format_runtime_float,
    hardware_snapshot_payload,
    plotting_focus_text,
    process_max_rss_mb,
    render_html_table,
    runtime_metrics_payload,
    write_dashboard_html,
)

__all__ = [
    "DEFAULT_DASHBOARD_SETTINGS",
    "dashboard_enabled",
    "dashboard_path",
    "dashboard_settings",
    "format_runtime_float",
    "hardware_snapshot_payload",
    "plotting_focus_text",
    "process_max_rss_mb",
    "render_html_table",
    "runtime_metrics_payload",
    "write_dashboard_html",
]
