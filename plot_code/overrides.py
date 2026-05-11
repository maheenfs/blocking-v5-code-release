"""Single-plot override registry.

Edit the ``overrides`` section in ``plot_config.json`` for plot-specific fixes
such as a tighter legend, a custom y-axis range, or a higher save DPI. Keys can
be full paths, selected_plots relative paths, or the PNG filename.
"""

from __future__ import annotations

from .style import PLOT_OVERRIDES, plot_override

__all__ = ["PLOT_OVERRIDES", "plot_override"]
