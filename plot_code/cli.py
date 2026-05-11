"""Command-line interface for the new plot_code package."""

from __future__ import annotations

import argparse
from pathlib import Path

from dashboard.plotting_progress import write_plotting_progress

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SELECTED_ROOT,
    PlotSelection,
    parse_csv_text,
    parse_float_csv,
    parse_int_csv,
)
from .families.compare_plots import build_compare_plots
from .families.full_plots import build_full_plots
from .families.provenance_plots import build_provenance_plots
from .families.run_plots import build_run_plots
from .families.selected_plots import build_selected_plots
from .organization import organize_results


def _add_selection_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kpi", "--kpis", dest="kpis", default="", help="Comma-separated KPI keys, e.g. primary,top1,top3_m1db.")
    parser.add_argument("--families", default="", help="Comma-separated plot families, e.g. blockage,epochs,train-fraction,freeze.")
    parser.add_argument("--seeds", default="", help="Comma-separated seeds, e.g. 123,456,789.")
    parser.add_argument("--patterns", default="", help="Comma-separated Set-B patterns, e.g. 1,2.")
    parser.add_argument("--blocked-beams", default="", help="Comma-separated blocked beams, e.g. B4,B0.")
    parser.add_argument("--train-fracs", default="", help="Comma-separated adaptation fractions, e.g. 0.01,0.1,1.0.")
    parser.add_argument("--blockages", default="", help="Comma-separated blockage percentages, e.g. 20,80,100.")
    parser.add_argument("--epochs", default="", help="Comma-separated FT epoch budgets, e.g. 1,3,10.")
    parser.add_argument("--organize", action="store_true", help="Copy generated plots into plots/organized_plots navigation folders.")
    parser.add_argument("--clean-organized", action="store_true", help="Clean navigation folders before organizing.")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How organized_plots entries are materialized. Use copy for release/researcher folders.",
    )


def _selection_from_args(args: argparse.Namespace) -> PlotSelection:
    return PlotSelection(
        kpis=parse_csv_text(getattr(args, "kpis", "")),
        families=parse_csv_text(getattr(args, "families", "")),
        seeds=parse_int_csv(getattr(args, "seeds", "")),
        patterns=parse_int_csv(getattr(args, "patterns", "")),
        blocked_beams=parse_csv_text(getattr(args, "blocked_beams", "")),
        train_fracs=parse_float_csv(getattr(args, "train_fracs", "")),
        blockages=parse_int_csv(getattr(args, "blockages", "")),
        epochs=parse_int_csv(getattr(args, "epochs", "")),
        organize=bool(getattr(args, "organize", False)),
        clean_organized=bool(getattr(args, "clean_organized", False)),
        copy_mode=str(getattr(args, "copy_mode", "copy")),
    )


def _progress_config_path(args: argparse.Namespace) -> str:
    return str(getattr(args, "config", DEFAULT_CONFIG_PATH))


def _progress_results_root(args: argparse.Namespace) -> str:
    if hasattr(args, "results_root"):
        return str(getattr(args, "results_root"))
    if hasattr(args, "run_dir"):
        return str(Path(str(getattr(args, "run_dir"))).expanduser().resolve().parent)
    return str(DEFAULT_RESULTS_ROOT)


def _write_progress(
    args: argparse.Namespace,
    *,
    command: str,
    selection: PlotSelection,
    status: str,
    message: str = "",
    error: BaseException | None = None,
) -> None:
    """Refresh plotting progress without making dashboard I/O a hard dependency."""
    try:
        write_plotting_progress(
            config_path=_progress_config_path(args),
            results_root=_progress_results_root(args),
            command=command,
            selection=selection,
            status=status,
            message=message,
            error=error,
        )
    except Exception:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Blocking V5 results. With no subcommand, rebuild the selected paper plot bundle."
    )
    sub = parser.add_subparsers(dest="command")
    parser.set_defaults(command="selected")

    selected = sub.add_parser(
        "selected",
        aliases=["paper-figures"],
        help="Regenerate plots/selected_plots atomically; existing PNGs are preserved if required inputs are missing.",
    )
    selected.add_argument("--selected-root", default="")
    selected.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    selected.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    _add_selection_flags(selected)

    full = sub.add_parser("full", help="Regenerate selected, per-run, comparison, provenance, and organized plots.")
    full.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    full.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    _add_selection_flags(full)

    compare = sub.add_parser("compare", aliases=["comparisons"], help="Generate comparison plots.")
    compare.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    compare.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    _add_selection_flags(compare)

    provenance = sub.add_parser("provenance", help="Generate provenance and beam-location plots.")
    provenance.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    provenance.add_argument("--out-dir", default="")
    _add_selection_flags(provenance)

    run_plots = sub.add_parser("run-plots", help="Regenerate plots for one results/<run>/ folder.")
    run_plots.add_argument("--run-dir", required=True, help="Path to one run directory.")
    _add_selection_flags(run_plots)

    organize = sub.add_parser("organize-only", help="Organize existing PNGs without generating new plots.")
    organize.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    organize.add_argument("--selected-root", default="")
    _add_selection_flags(organize)
    organize.set_defaults(organize=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    selection = _selection_from_args(args)
    command = args.command

    _write_progress(args, command=command, selection=selection, status="running", message="Plot command started.")
    try:
        if command in {"selected", "paper-figures"}:
            build_selected_plots(
                selection,
                organize=True,
                selected_root=getattr(args, "selected_root", "") or DEFAULT_SELECTED_ROOT,
                results_root=getattr(args, "results_root", DEFAULT_RESULTS_ROOT),
                config=getattr(args, "config", DEFAULT_CONFIG_PATH),
            )
        elif command == "full":
            build_full_plots(args.results_root, config=args.config, selection=selection, organize=True)
        elif command in {"compare", "comparisons"}:
            build_compare_plots(args.results_root, config=args.config, selection=selection, organize=selection.organize)
        elif command == "provenance":
            build_provenance_plots(config=args.config, out_dir=args.out_dir or None, selection=selection, organize=selection.organize)
        elif command == "run-plots":
            build_run_plots(args.run_dir, selection=selection, organize=selection.organize)
        elif command == "organize-only":
            organize_results(
                args.results_root,
                selected_root=args.selected_root or None,
                selection=selection,
                copy_mode=selection.copy_mode,
                clean=selection.clean_organized,
            )
        else:
            parser.error(f"Unknown command: {command}")
    except BaseException as exc:
        _write_progress(args, command=command, selection=selection, status="failed", message="Plot command failed.", error=exc)
        raise
    else:
        _write_progress(args, command=command, selection=selection, status="completed", message="Plot command completed.")


if __name__ == "__main__":
    main()
