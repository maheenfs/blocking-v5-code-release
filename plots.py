"""Publication-facing plotting entrypoint.

All plotting is implemented in the `plot_code/` package. This file exists only
so existing plotting commands can keep using `plots.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from plot_code.cli import main
from plot_code.config import DEFAULT_CONFIG_PATH, DEFAULT_RESULTS_ROOT, PlotSelection
from plot_code.families.compare_plots import build_compare_plots
from plot_code.families.full_plots import build_full_plots
from plot_code.families.provenance_plots import build_provenance_plots
from plot_code.families.run_plots import build_run_plots
from plot_code.families.selected_plots import build_selected_plots
from plot_code.organization import organize_results


def generate_run_plots(run_dir: str | Path) -> None:
    build_run_plots(run_dir)


def generate_batch_comparisons(results_root: str | Path, *, config: str | Path | None = None) -> None:
    build_compare_plots(results_root, config=config or DEFAULT_CONFIG_PATH)


def generate_provenance_outputs(config: str | Path | None = None, *, out_dir: str | Path | None = None) -> list[object]:
    return build_provenance_plots(config=config or DEFAULT_CONFIG_PATH, out_dir=out_dir)


def regenerate_paper_figures() -> None:
    build_selected_plots(PlotSelection())


def generate_full_plot_outputs(results_root: str | Path = DEFAULT_RESULTS_ROOT, *, config: str | Path | None = None) -> None:
    build_full_plots(results_root, config=config or DEFAULT_CONFIG_PATH)


__all__ = [
    "build_compare_plots",
    "build_full_plots",
    "build_provenance_plots",
    "build_run_plots",
    "build_selected_plots",
    "generate_batch_comparisons",
    "generate_full_plot_outputs",
    "generate_provenance_outputs",
    "generate_run_plots",
    "organize_results",
    "regenerate_paper_figures",
]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Plot command failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
