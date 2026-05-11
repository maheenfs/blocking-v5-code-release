"""Post-run plotting adapters used by the experiment runner."""

from __future__ import annotations

from pathlib import Path

from plot_code.config import DEFAULT_CONFIG_PATH
from plot_code.families.compare_plots import build_compare_plots
from plot_code.families.provenance_plots import build_provenance_plots
from plot_code.families.run_plots import build_run_plots
from plot_code.families.selected_plots import build_selected_plots


def generate_run_plots(run_dir: str | Path) -> None:
    """Generate per-run plots after one experiment job finishes."""
    build_run_plots(run_dir)


def generate_batch_comparisons(results_root: str | Path, *, config: str | Path | None = None) -> None:
    """Generate batch-wide comparison plots after all configured jobs finish."""
    build_compare_plots(results_root, config=config or DEFAULT_CONFIG_PATH)


def generate_provenance_outputs(config: str | Path | None = None, *, out_dir: str | Path | None = None) -> list[object]:
    """Generate provenance outputs that document geometry and protocol choices."""
    return build_provenance_plots(config=config or DEFAULT_CONFIG_PATH, out_dir=out_dir)


def generate_selected_outputs(results_root: str | Path, *, config: str | Path | None = None) -> list[object]:
    """Regenerate only the curated selected paper plot bundle."""
    return build_selected_plots(results_root=results_root, config=config or DEFAULT_CONFIG_PATH, organize=False)


__all__ = [
    "generate_batch_comparisons",
    "generate_provenance_outputs",
    "generate_run_plots",
    "generate_selected_outputs",
]
