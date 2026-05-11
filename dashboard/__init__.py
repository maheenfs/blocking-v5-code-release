"""Small dashboard package used by the runner and plotting CLI.

The runner imports only the functions exported here.  The implementation is
split across focused modules so status analysis, resource sampling, and HTML
rendering can be edited independently.
"""

from .artifacts import plotting_focus_text
from .formatting import format_runtime_float, render_html_table
from .render import write_dashboard_html
from .resources import hardware_snapshot_payload, process_max_rss_mb, runtime_metrics_payload
from .settings import DEFAULT_DASHBOARD_SETTINGS, dashboard_enabled, dashboard_path, dashboard_settings

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
