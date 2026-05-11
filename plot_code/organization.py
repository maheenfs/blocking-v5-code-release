"""Organize generated plots into browsable folders.

The canonical plot files stay where each plot family creates them. This module
copies those PNGs into lookup folders such as by_kpi, by_seed, by_family, and
by_blocked_beam so a researcher can find a plot from more than one direction.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Iterable

from .config import (
    DEFAULT_PLOTS_ROOT,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SELECTED_ROOT,
    PACKAGE_ROOT,
    PlotSelection,
    normalize_beam_token,
)
from .filters import matches_selection


@dataclass(frozen=True)
class PlotOutput:
    source_path: Path
    title: str
    kpi: str = "primary"
    family: str = "other"
    scope: str = "results"
    seeds: tuple[str, ...] = ()
    pattern: int | None = None
    blocked_beams: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    figure: str = ""
    role: str = ""
    data_source: str = ""


KNOWN_ORGANIZED_DIRS = (
    "by_kpi",
    "by_seed",
    "by_family",
    "by_pattern",
    "by_blocked_beam",
    "by_method",
    "selected_paper_plots",
    "all_plots_flat",
)


def _slug(value: object, default: str = "unknown") -> str:
    text = str(value or default).strip()
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def _relative(path: Path, *extra_bases: Path) -> str:
    for base in (*extra_bases, DEFAULT_PLOTS_ROOT, PACKAGE_ROOT, DEFAULT_RESULTS_ROOT):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            pass
    return path.name


def _organized_filename(path: Path, *extra_bases: Path) -> str:
    return _slug(_relative(path, *extra_bases).replace("/", "__"))


def _method_dirname(value: object) -> str:
    text = str(value or "").strip()
    token = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    aliases = {
        "bl_ag": "BL_AG",
        "blag": "BL_AG",
        "bl_aw": "BL_AW",
        "blaw": "BL_AW",
        "ft": "FT",
        "ft_freeze_cnn": "FT_freeze_cnn",
        "ft_no_freeze": "FT_no_freeze",
        "ft_freeze_head": "FT_freeze_head",
        "ft_freeze_all": "FT_freeze_all",
        "no_freeze": "FT_no_freeze",
        "freeze_cnn": "FT_freeze_cnn",
        "freeze_head": "FT_freeze_head",
        "freeze_all": "FT_freeze_all",
        "max_setb": "MAX_SETB",
        "nn_angle": "NN_ANGLE",
        "random_setb": "RANDOM_SETB",
    }
    return aliases.get(token, _slug(text))


def _copy_or_link(source: Path, destination: Path, copy_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_mode == "symlink":
        destination.symlink_to(source.resolve())
    elif copy_mode == "hardlink":
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)


def _infer_kpi(path: Path, *extra_bases: Path) -> str:
    parts = [part.lower() for part in Path(_relative(path, *extra_bases)).parts]
    known = {"primary", "top1", "top2", "top3", "top4", "top1_m1db", "top2_m1db", "top3_m0db", "top3_m1db", "top3_m2db"}
    for part in parts:
        if part in known:
            return part
    if "multi_kpi" in parts or "multi-kpi" in path.name.lower():
        return "multi_kpi"
    return "primary"


def _infer_family(path: Path, *extra_bases: Path) -> str:
    text = _relative(path, *extra_bases).lower()
    if "selected_plots" in text or text.startswith("fig") or "/fig" in text:
        if "procedure" in text:
            return "procedure"
        if "setup" in text or "beam_locations" in text:
            return "setup"
        if "heatmap" in text:
            return "heatmaps"
        if "non_ml" in text:
            return "non_ml_baselines"
        if "multi_kpi" in text:
            return "multi_kpi"
        if "convergence" in text:
            return "convergence"
        if "epoch" in text:
            return "epoch_budget_vs_accuracy"
        if "train_fraction" in text or "train-frac" in text:
            return "train_fraction_vs_accuracy"
        if "freeze" in text:
            return "freeze_mode_ablation"
        if "cross_pattern" in text or "compared" in text:
            return "cross_pattern_comparison"
        if "all_methods" in text:
            return "method_comparison"
        if "blockage" in text:
            return "blockage_vs_accuracy"
        return "selected_paper"
    if "provenance" in text or "beam-locations" in text:
        return "provenance"
    if "train_loss" in text or "/loss/" in text or "_loss" in text:
        return "train_loss"
    if "history" in text or "convergence" in text:
        return "convergence"
    if "heatmap" in text:
        return "heatmaps"
    if "non_ml" in text or "non-ml" in text:
        return "non_ml_baselines"
    if "epoch" in text:
        return "epoch_budget_vs_accuracy"
    if "train_frac" in text or "train_fraction" in text:
        return "train_fraction_vs_accuracy"
    if "freeze" in text:
        return "freeze_mode_ablation"
    if "compare_patterns" in text or "cross" in text or "compared" in text:
        return "cross_pattern_comparison"
    if "all_methods" in text or "method" in text:
        return "method_comparison"
    if "blockage" in text:
        return "blockage_vs_accuracy"
    if "multi_kpi" in text:
        return "multi_kpi"
    return "other"


def _infer_seeds(path: Path, *extra_bases: Path) -> tuple[str, ...]:
    text = _relative(path, *extra_bases).lower()
    seeds = tuple(sorted(set(re.findall(r"seed(\d+)", text)), key=int))
    if seeds:
        return seeds
    if "multiseed" in text or "multi_seed" in text:
        return ("123", "456", "789")
    return ()


def _infer_pattern_and_beams(path: Path, *extra_bases: Path) -> tuple[int | None, tuple[str, ...]]:
    text = _relative(path, *extra_bases)
    pattern: int | None = None
    beams: list[str] = []
    for match in re.finditer(r"[Pp]([12])[_-]?[Bb](\d+)", text):
        pattern = int(match.group(1))
        beam = normalize_beam_token(match.group(2))
        if beam not in beams:
            beams.append(beam)
    if pattern is None:
        match = re.search(r"(?:pattern|P)([12])", text, re.IGNORECASE)
        if match:
            pattern = int(match.group(1))
    return pattern, tuple(beams)


def _infer_methods(path: Path, *extra_bases: Path) -> tuple[str, ...]:
    text = _relative(path, *extra_bases).lower()
    methods = []
    checks = [
        ("BL_AG", ("bl_ag", "bl-ag", "baseline")),
        ("BL_AW", ("bl_aw", "bl-aw")),
        ("FT_freeze_cnn", ("freeze_cnn", "ft")),
        ("FT_no_freeze", ("no_freeze",)),
        ("FT_freeze_head", ("freeze_head",)),
        ("MAX_SETB", ("max-setb", "max_setb")),
        ("NN_ANGLE", ("nn-angle", "nn_angle")),
    ]
    for label, needles in checks:
        if any(needle in text for needle in needles):
            methods.append(label)
    return tuple(dict.fromkeys(methods))


def infer_plot_output(
    path: Path,
    *,
    selected_root: Path = DEFAULT_SELECTED_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    plots_root: Path = DEFAULT_PLOTS_ROOT,
) -> PlotOutput:
    pattern, beams = _infer_pattern_and_beams(path, selected_root, plots_root, results_root)
    return PlotOutput(
        source_path=path,
        title=path.stem.replace("_", " "),
        kpi=_infer_kpi(path, selected_root, plots_root, results_root),
        family=_infer_family(path, selected_root, plots_root, results_root),
        scope="selected" if selected_root in path.parents else "results",
        seeds=_infer_seeds(path, selected_root, plots_root, results_root),
        pattern=pattern,
        blocked_beams=beams,
        methods=_infer_methods(path, selected_root, plots_root, results_root),
    )


def iter_plot_pngs(results_root: Path, selected_root: Path, plots_root: Path = DEFAULT_PLOTS_ROOT) -> Iterable[Path]:
    roots = []
    if plots_root.exists():
        roots.append(plots_root)
    if selected_root.exists():
        roots.append(selected_root)
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.rglob("*.png")):
            if "organized_plots" in path.parts:
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def destination_dirs(plot: PlotOutput, organized_root: Path) -> list[Path]:
    dirs = [
        organized_root / "by_kpi" / _slug(plot.kpi),
        organized_root / "by_family" / _slug(plot.family),
        organized_root / "all_plots_flat",
    ]
    if plot.seeds:
        seed_label = "multiseed_" + "_".join(plot.seeds) if len(plot.seeds) > 1 else f"seed_{plot.seeds[0]}"
        dirs.append(organized_root / "by_seed" / seed_label)
    if plot.pattern is not None:
        dirs.append(organized_root / "by_pattern" / f"pattern_{plot.pattern}")
    else:
        dirs.append(organized_root / "by_pattern" / "cross_or_unknown")
    for beam in plot.blocked_beams:
        prefix = f"P{plot.pattern}_" if plot.pattern is not None else ""
        dirs.append(organized_root / "by_blocked_beam" / _slug(prefix + beam))
    for method in plot.methods:
        dirs.append(organized_root / "by_method" / _method_dirname(method))
    if plot.scope == "selected":
        dirs.append(organized_root / "selected_paper_plots")
    return list(dict.fromkeys(dirs))


def clean_navigation_tree(organized_root: Path) -> None:
    for name in KNOWN_ORGANIZED_DIRS:
        shutil.rmtree(organized_root / name, ignore_errors=True)
    (organized_root / "organized_plot_manifest.csv").unlink(missing_ok=True)


def organize_results(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    selected_root: str | Path | None = DEFAULT_SELECTED_ROOT,
    plots_root: str | Path = DEFAULT_PLOTS_ROOT,
    selection: PlotSelection | None = None,
    copy_mode: str = "copy",
    clean: bool = False,
    plot_outputs: Iterable[PlotOutput] | None = None,
) -> Path:
    """Copy generated plots into a searchable organized tree and write a manifest.

    Builders pass explicit PlotOutput records so organization uses plot metadata
    from the code that created the figure. When called by organize-only, the
    filename inference remains as a fallback for existing PNGs.
    """

    results_root = Path(results_root)
    selected_root = Path(selected_root) if selected_root is not None else DEFAULT_SELECTED_ROOT
    plots_root = Path(plots_root)
    selection = selection or PlotSelection()
    organized_root = plots_root / "organized_plots"
    organized_root.mkdir(parents=True, exist_ok=True)
    if clean:
        clean_navigation_tree(organized_root)

    manifest_rows: list[dict[str, str]] = []
    if plot_outputs is None:
        plots = [
            infer_plot_output(source, selected_root=selected_root, results_root=results_root, plots_root=plots_root)
            for source in iter_plot_pngs(results_root, selected_root, plots_root)
        ]
    else:
        plots = [plot for plot in plot_outputs if plot.source_path.exists()]

    for plot in plots:
        source = plot.source_path
        if not matches_selection(
            {
                "kpi": plot.kpi,
                "family": plot.family,
                "seeds": plot.seeds,
                "pattern": plot.pattern,
                "blocked_beams": plot.blocked_beams,
            },
            selection,
        ):
            continue
        filename = _organized_filename(source, selected_root, plots_root, results_root)
        for destination_dir in destination_dirs(plot, organized_root):
            destination = destination_dir / filename
            _copy_or_link(source, destination, copy_mode)
            manifest_rows.append(
                {
                    "organized_path": destination.relative_to(organized_root).as_posix(),
                    "source_path": _relative(source, selected_root, plots_root, results_root),
                    "title": plot.title,
                    "kpi": plot.kpi,
                    "family": plot.family,
                    "scope": plot.scope,
                    "seeds": "+".join(plot.seeds),
                    "pattern": "" if plot.pattern is None else str(plot.pattern),
                    "blocked_beams": "+".join(plot.blocked_beams),
                    "methods": "+".join(plot.methods),
                    "figure": plot.figure,
                    "role": plot.role,
                    "data_source": plot.data_source,
                }
            )

    manifest_path = organized_root / "organized_plot_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "organized_path",
            "source_path",
            "title",
            "kpi",
            "family",
            "scope",
            "seeds",
            "pattern",
            "blocked_beams",
            "methods",
            "figure",
            "role",
            "data_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path
