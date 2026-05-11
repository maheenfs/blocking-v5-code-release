"""Full plotting workflow."""

from __future__ import annotations

from pathlib import Path

from ..config import DEFAULT_CONFIG_PATH, DEFAULT_RESULTS_ROOT, PlotSelection
from ..data_loading import discover_runs
from ..organization import PlotOutput, organize_results
from .compare_plots import build_compare_plots
from .provenance_plots import build_provenance_plots
from .run_plots import build_run_plots
from .selected_plots import build_selected_plots


def build_full_plots(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    config: str | Path = DEFAULT_CONFIG_PATH,
    selection: PlotSelection | None = None,
    organize: bool = True,
) -> list[PlotOutput]:
    selection = selection or PlotSelection()
    outputs: list[PlotOutput] = []
    for run in discover_runs(results_root):
        outputs.extend(build_run_plots(run.path, selection=selection, organize=False))
    outputs.extend(build_compare_plots(results_root, config=config, selection=selection, organize=False))
    outputs.extend(build_provenance_plots(config=config, selection=selection, organize=False))
    outputs.extend(build_selected_plots(selection=selection, organize=False, results_root=results_root, config=config))
    if organize or selection.organize:
        organize_results(results_root, selection=selection, copy_mode=selection.copy_mode, clean=selection.clean_organized, plot_outputs=outputs)
    return outputs
