"""Selected main-paper plot bundle.

The selected bundle is manifest-driven. The curated
``selected_plots_manifest.csv`` is the source of truth and is never rebuilt
from files on disk.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import shutil
import tempfile

from matplotlib.lines import Line2D

from experiment.geometry import build_setb_tx_indices, seta_grid_angles, setb_pattern_angles
from ..charts import categorical_bar, grouped_bar
from ..config import DEFAULT_CONFIG_PATH, DEFAULT_RESULTS_ROOT, DEFAULT_SELECTED_ROOT, PlotSelection, load_config, normalize_beam_token
from ..data_loading import as_float, as_int, discover_runs, filtered_rows, mean_std, mean_value, read_csv, write_csv
from ..filters import matches_selection
from ..metrics import multi_kpi_specs
from ..organization import PlotOutput, organize_results
from ..style import color_for, figure_size, save_figure, plt
from .selected_source_data import refresh_selected_source_data

STATUS_FIELDS = ["selected_file", "status", "reason", "family", "figure", "role", "source_path", "data_source"]
CROSS_SOURCES = ("best", "mid", "worst", "same_b3", "same_b7")
FREEZE_MODES = ("no_freeze", "freeze_cnn", "freeze_head", "freeze_all")
METHOD_LABELS = (("BL-AG", "blag"), ("FT freeze_cnn", "ft"), ("BL-AW", "blaw"))


def _selected_plan(selected_root: Path) -> list[dict[str, str]]:
    return read_csv(selected_root / "selected_plots_manifest.csv")


def _selected_plan_root(selected_root: Path) -> Path:
    if (selected_root / "selected_plots_manifest.csv").exists():
        return selected_root
    if (DEFAULT_SELECTED_ROOT / "selected_plots_manifest.csv").exists():
        return DEFAULT_SELECTED_ROOT
    raise FileNotFoundError(
        "selected_plots_manifest.csv is required in the selected output root "
        f"or in {DEFAULT_SELECTED_ROOT}"
    )


def _seed_selected_root(plan_root: Path, selected_root: Path) -> None:
    if plan_root == selected_root:
        return
    manifest = plan_root / "selected_plots_manifest.csv"
    selected_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, selected_root / manifest.name)

    source_root = plan_root / "source_data"
    if not source_root.exists():
        return
    for source_file in source_root.rglob("*"):
        if not source_file.is_file():
            continue
        target = selected_root / source_file.relative_to(plan_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)


def _source_data_root(selected_root: Path) -> Path:
    return selected_root / "source_data"


def _family_for_file(selected_file: str) -> str:
    folder = selected_file.split("/", 1)[0]
    mapping = {
        "fig00_procedure": "procedure",
        "fig01_setup": "setup",
        "fig02_aggregate": "method_comparison",
        "fig03_blag_degradation": "blockage_vs_accuracy",
        "fig04_ft_recovery": "blockage_vs_accuracy",
        "fig05_multiseed_representative": "blockage_vs_accuracy",
        "fig06_freeze_ablation": "freeze_mode_ablation",
        "fig07_data_efficiency": "train_fraction_vs_accuracy",
        "fig08_epoch_budget": "epoch_budget_vs_accuracy",
        "fig09_cross_pattern_methods": "cross_pattern_comparison",
        "fig10_cross_pattern_train_fraction": "cross_pattern_train_fraction",
        "fig11_multi_kpi": "multi_kpi",
        "fig12_non_ml": "non_ml_baselines",
        "fig13_intermediate_blockage": "method_comparison",
        "fig14_blaw_ceiling": "blockage_vs_accuracy",
        "fig15_cross_pattern_baseline": "cross_pattern_comparison",
        "fig16_cross_pattern_ft": "cross_pattern_comparison",
        "fig17_heatmap": "heatmaps",
        "fig18_convergence": "convergence",
    }
    return mapping.get(folder, "selected_paper")


def _pattern_from_text(*values: object) -> int | None:
    text = " ".join(str(value) for value in values)
    match = re.search(r"\b[Pp]([12])\b|[_/-][Pp]([12])[_/-]", text)
    if not match:
        return None
    return int(next(group for group in match.groups() if group))


def _beams_from_text(*values: object) -> tuple[str, ...]:
    text = " ".join(str(value) for value in values)
    beams: list[str] = []
    for match in re.finditer(r"\b[Bb](\d+)\b|block[Bb](\d+)", text):
        beam = normalize_beam_token(next(group for group in match.groups() if group))
        if beam not in beams:
            beams.append(beam)
    return tuple(beams)


def _blockage_from_file(selected_file: str) -> int | None:
    match = re.search(r"blockage[_-](\d+)", selected_file)
    return int(match.group(1)) if match else None


def _seeds_from_row(row: dict[str, str]) -> tuple[str, ...]:
    text = f"{row.get('role', '')} {row.get('data_source', '')}"
    seeds = sorted(
        {value for pair in re.findall(r"seed(\d+)|single-seed\s+(\d+)", text) for value in pair if value},
        key=int,
    )
    if seeds:
        return tuple(seeds)
    if "3 seeds" in text or "123+456+789" in text:
        return ("123", "456", "789")
    if "results/" in text or "created from" in text:
        return ("123",)
    return ()


def _source_name_from_file(selected_file: str) -> str | None:
    name = Path(selected_file).name
    for source_name in CROSS_SOURCES:
        if name.startswith(f"{source_name}_"):
            return source_name
    return None


def _side_from_file(selected_file: str) -> str | None:
    name = Path(selected_file).name.lower()
    if "_p1_" in name:
        return "p1"
    if "_p2_" in name:
        return "p2"
    return None


def _plot_output(
    row: dict[str, str],
    path: Path,
    *,
    title: str,
    family: str,
    kpi: str = "primary",
    pattern: int | None = None,
    blocked_beams: tuple[str, ...] = (),
    methods: tuple[str, ...] = (),
) -> PlotOutput:
    return PlotOutput(
        source_path=path,
        title=title,
        kpi=kpi,
        family=family,
        scope="selected",
        seeds=_seeds_from_row(row),
        pattern=pattern,
        blocked_beams=blocked_beams,
        methods=methods,
        figure=row.get("figure", ""),
        role=row.get("role", ""),
        data_source=row.get("data_source", ""),
    )


def _plan_metadata(row: dict[str, str]) -> dict[str, object]:
    selected_file = row.get("selected_file", "")
    return {
        "kpi": "primary",
        "family": _family_for_file(selected_file),
        "seeds": _seeds_from_row(row),
        "pattern": _pattern_from_text(selected_file, row.get("role", "")),
        "blocked_beams": _beams_from_text(selected_file, row.get("role", "")),
    }


def _summary_rows(source_data: Path, pattern: int) -> list[dict[str, str]]:
    return read_csv(source_data / f"p{pattern}_summary_ft_freeze_cnn_config.csv")


def _summary_value(row: dict[str, str], prefix: str) -> float:
    return as_float(row.get(f"{prefix}_primary_%"))


def _summary_error(row: dict[str, str], prefix: str) -> float:
    return as_float(row.get(f"{prefix}_primary_seed_std"), 0.0)


def _sorted_beams(rows: list[dict[str, str]]) -> list[str]:
    return sorted({row.get("blocked_beam", "") for row in rows if row.get("blocked_beam")}, key=lambda beam: int(beam[1:]))


def _sorted_blockages(rows: list[dict[str, str]]) -> list[int]:
    return sorted({as_int(row.get("blockage_%")) for row in rows})


def _summary_row(rows: list[dict[str, str]], beam: str, blockage: int) -> dict[str, str]:
    return next((row for row in rows if row.get("blocked_beam") == beam and as_int(row.get("blockage_%")) == blockage), {})


def _procedure_plot(row: dict[str, str], path: Path, *, source_selected_root: Path) -> PlotOutput:
    source = source_selected_root / "fig00_procedure" / "procedure_new_protocol.png"
    if source.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)
        return _plot_output(row, path, title="Experimental procedure overview", family="procedure", methods=("protocol",))

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.axis("off")
    labels = ["Clean Set-A data", "Build Set-B codebook", "Block Set-A beams", "Fine-tune with small Set-B subset", "Evaluate KPI accuracy"]
    x_positions = [0.08, 0.29, 0.50, 0.71, 0.92]
    for idx, (x_pos, label) in enumerate(zip(x_positions, labels)):
        ax.text(
            x_pos,
            0.54,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": color_for(label, idx), "edgecolor": "none"},
            transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate("", xy=(x_positions[idx + 1] - 0.08, 0.54), xytext=(x_pos + 0.08, 0.54), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.5})
    saved = save_figure(fig, path, plot_id=path)
    return _plot_output(row, saved, title="Experimental procedure overview", family="procedure", methods=("protocol",))


def _tx_index_to_angles(tx_index: int) -> tuple[float, float]:
    azimuths, elevations = seta_grid_angles()
    tx = int(tx_index)
    return float(azimuths[tx % len(azimuths)]), float(elevations[tx // len(azimuths)])


def _all_grid_points() -> list[tuple[int, float, float]]:
    azimuths, elevations = seta_grid_angles()
    points: list[tuple[int, float, float]] = []
    for row_index, elevation in enumerate(reversed(elevations)):
        seta_row = len(elevations) - 1 - row_index
        for az_index, azimuth in enumerate(azimuths):
            tx_index = seta_row * len(azimuths) + az_index
            points.append((tx_index, float(azimuth), float(elevation)))
    return points


def _pattern_beam_points(pattern: int) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for beam_index, (angles, tx_index) in enumerate(zip(setb_pattern_angles(int(pattern)), build_setb_tx_indices(int(pattern)))):
        points.append(
            {
                "beam": f"B{beam_index}",
                "seta_number": int(tx_index),
                "azimuth": float(angles[0]),
                "elevation": float(angles[1]),
            }
        )
    return points


def _setup_plot(row: dict[str, str], path: Path, *, pattern: int, config: dict, results_root: Path) -> PlotOutput:
    style = {
        1: {"fill": "#1E9798", "edge": "#0E6D70", "line": "#B7E3E4", "subtitle": "Diagonal Set-B beam pattern"},
        2: {"fill": "#DE5B4D", "edge": "#A8362C", "line": "#F4C2BC", "subtitle": "Anti-diagonal Set-B beam pattern"},
    }[int(pattern)]
    remaining_face = "#F5FAFD"
    remaining_edge = "#C6D9EA"
    remaining_text = "#88A2BC"
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.patch.set_facecolor("white")
    azimuths, elevations = seta_grid_angles()
    setb_points = _pattern_beam_points(int(pattern))
    setb_numbers = {int(point["seta_number"]) for point in setb_points}
    ax.set_title(f"Pattern {int(pattern)} beam locations on the 32-beam Set-A grid\n{style['subtitle']}", fontsize=13, pad=18)
    ax.grid(True, color="#D8E4F0", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Azimuth (deg)", fontsize=11)
    ax.set_ylabel("Elevation (deg)", fontsize=11)
    ax.set_xticks(list(azimuths))
    ax.set_yticks(sorted(elevations))
    ax.tick_params(labelsize=9)
    ax.set_xlim(min(azimuths) - 8, max(azimuths) + 8)
    ax.set_ylim(min(elevations) - 6, max(elevations) + 6)
    for spine in ax.spines.values():
        spine.set_color("#C9D7E6")
        spine.set_linewidth(1.4)
    for seta_number, azimuth, elevation in _all_grid_points():
        if seta_number in setb_numbers:
            continue
        ax.scatter([azimuth], [elevation], s=1800, facecolors=remaining_face, edgecolors=remaining_edge, linewidths=1.7, zorder=2)
        ax.text(azimuth, elevation, str(seta_number), ha="center", va="center", fontsize=13, color=remaining_text, fontweight="bold", zorder=3)
    ax.plot([float(point["azimuth"]) for point in setb_points], [float(point["elevation"]) for point in setb_points], color=str(style["line"]), linewidth=3.2, zorder=1)
    for point in setb_points:
        azimuth = float(point["azimuth"])
        elevation = float(point["elevation"])
        ax.scatter([azimuth], [elevation], s=2100, c=str(style["fill"]), edgecolors=str(style["edge"]), linewidths=2.0, zorder=4)
        ax.text(azimuth, elevation + 0.15, str(point["beam"]), ha="center", va="center", fontsize=20, color="white", fontweight="bold", zorder=5)
        ax.text(azimuth, elevation - 1.6, str(point["seta_number"]), ha="center", va="center", fontsize=13, color="white", fontweight="bold", zorder=5)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=str(style["fill"]), markeredgecolor=str(style["edge"]), markeredgewidth=1.5, markersize=16, label=f"Pattern {int(pattern)} Set-B beams"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=remaining_face, markeredgecolor=remaining_edge, markeredgewidth=1.3, markersize=16, label="Remaining Set-A beams"),
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2, fontsize=12.5, frameon=True)
    fig.text(
        0.015,
        0.025,
        "Large text is the pattern-local beam ID B0..B7; the smaller number below it is the zero-based Set-A tx index.",
        ha="left",
        va="bottom",
        fontsize=12,
        color="#7B8BA3",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return _plot_output(row, path, title=f"Pattern {pattern}: Set-B beam locations", family="setup", pattern=pattern, blocked_beams=tuple(str(point["beam"]) for point in setb_points), methods=("setup",))


def _summary_method_plot(row: dict[str, str], path: Path, *, pattern: int, blockage: int, source_data: Path, family: str) -> PlotOutput | None:
    rows = _summary_rows(source_data, pattern)
    beams = _sorted_beams(rows)
    if not beams:
        return None
    series = []
    for label, prefix in METHOD_LABELS:
        values = [_summary_value(_summary_row(rows, beam, blockage), prefix) for beam in beams]
        errors = [_summary_error(_summary_row(rows, beam, blockage), prefix) for beam in beams]
        if any(value == value for value in values):
            series.append((label, values, errors))
    if not series:
        return None
    title = f"Pattern {pattern}: method comparison at {blockage}% blockage"
    saved = grouped_bar(labels=beams, series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family=family, pattern=pattern, blocked_beams=tuple(beams), methods=tuple(item[0] for item in series))


def _beam_multiseed_plot(row: dict[str, str], path: Path, *, pattern: int, beam: str, source_data: Path) -> PlotOutput | None:
    rows = [item for item in _summary_rows(source_data, pattern) if item.get("blocked_beam") == beam]
    blockages = _sorted_blockages(rows)
    if not rows or not blockages:
        return None
    series = []
    for label, prefix in METHOD_LABELS:
        values = [_summary_value(_summary_row(rows, beam, blockage), prefix) for blockage in blockages]
        errors = [_summary_error(_summary_row(rows, beam, blockage), prefix) for blockage in blockages]
        if any(value == value for value in values):
            series.append((label, values, errors))
    if not series:
        return None
    title = f"Pattern {pattern} {beam}: multi-seed accuracy vs blockage"
    saved = grouped_bar(labels=[str(bp) for bp in blockages], series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="blockage_vs_accuracy", pattern=pattern, blocked_beams=(beam,), methods=tuple(item[0] for item in series))


def _summary_bar_plot(row: dict[str, str], path: Path, *, pattern: int, prefix: str, title_label: str, source_data: Path) -> PlotOutput | None:
    rows = _summary_rows(source_data, pattern)
    beams = _sorted_beams(rows)
    blockages = _sorted_blockages(rows)
    if not beams or not blockages:
        return None
    series = []
    for beam in beams:
        values = [_summary_value(_summary_row(rows, beam, blockage), prefix) for blockage in blockages]
        if any(value == value for value in values):
            series.append((beam, values))
    if not series:
        return None
    title = f"Pattern {pattern}: {title_label}"
    saved = categorical_bar(labels=[str(bp) for bp in blockages], series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="blockage_vs_accuracy", pattern=pattern, blocked_beams=tuple(beams), methods=(prefix,))


def _cross_all_methods(row: dict[str, str], path: Path, *, source_name: str, side: str, source_data: Path) -> PlotOutput | None:
    rows = read_csv(source_data / f"compare_patterns_{source_name}_summary_table.csv")
    blockages = sorted({row.get("blockage_%", "") for row in rows}, key=lambda value: int(float(value or 0)))
    if not rows or not blockages:
        return None
    value_key = f"{side}_value"
    err_key = f"{side}_seed_std"
    series = []
    for method in ("BL-AG", "FT:freeze_cnn", "BL-AW"):
        method_rows = [item for item in rows if item.get("method") == method]
        values = []
        errors = []
        for blockage in blockages:
            candidate = next((item for item in method_rows if item.get("blockage_%") == blockage), {})
            values.append(as_float(candidate.get(value_key)))
            errors.append(as_float(candidate.get(err_key), 0.0))
        if any(value == value for value in values):
            series.append((method.replace(":", " "), values, errors))
    if not series:
        return None
    side_label = side.upper()
    title = f"Cross-pattern {source_name.replace('_', ' ')}: {side_label} all methods"
    saved = grouped_bar(labels=blockages, series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="cross_pattern_comparison", methods=tuple(item[0] for item in series))


def _cross_compared(row: dict[str, str], path: Path, *, source_name: str, method: str, title_method: str, source_data: Path) -> PlotOutput | None:
    rows = [item for item in read_csv(source_data / f"compare_patterns_{source_name}_summary_table.csv") if item.get("method") == method]
    blockages = sorted({row.get("blockage_%", "") for row in rows}, key=lambda value: int(float(value or 0)))
    if not rows or not blockages:
        return None
    series = []
    for label, value_key, err_key in (("P1", "p1_value", "p1_seed_std"), ("P2", "p2_value", "p2_seed_std")):
        values = []
        errors = []
        for blockage in blockages:
            candidate = next((item for item in rows if item.get("blockage_%") == blockage), {})
            values.append(as_float(candidate.get(value_key)))
            errors.append(as_float(candidate.get(err_key), 0.0))
        series.append((label, values, errors))
    title = f"Cross-pattern {source_name.replace('_', ' ')}: {title_method}"
    saved = grouped_bar(labels=blockages, series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="cross_pattern_comparison", methods=(method.replace(":", "_"),))


def _non_ml_plot(row: dict[str, str], path: Path, *, source_data: Path, scope: str) -> PlotOutput | None:
    rows = [item for item in read_csv(source_data / "non_ml_baseline_summary.csv") if item.get("scope") == scope]
    if not rows:
        return None
    labels = list(dict.fromkeys(item.get("label", "") for item in rows if item.get("label")))
    methods = list(dict.fromkeys(item.get("method", "") for item in rows if item.get("method")))
    series = []
    for method in methods:
        values = []
        errors = []
        for label in labels:
            candidate = next((item for item in rows if item.get("method") == method and item.get("label") == label), {})
            values.append(as_float(candidate.get("mean_test_primary_%")))
            errors.append(as_float(candidate.get("seed_std_pp"), 0.0))
        if any(value == value for value in values):
            series.append((method, values, errors))
    if not series:
        return None
    title = "Non-learning baseline comparison" if scope == "representative" else "Non-learning baselines across all beams"
    saved = grouped_bar(labels=labels, series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="non_ml_baselines", methods=tuple(methods))


def _runs_for(results_root: Path, *, pattern: int, beam: str | None = None) -> list:
    wanted_beam = normalize_beam_token(beam) if beam else None
    runs = [run for run in discover_runs(results_root) if run.pattern == pattern and (wanted_beam is None or run.beam == wanted_beam)]
    seed_123 = [run for run in runs if run.seed == 123]
    return seed_123 or runs


def _mean_for_runs(runs: list, *, method: str, metric: str, freeze: str | None = None, train_frac: float | None = None, blockage: int | None = None, epochs: int | None = None) -> tuple[float, float]:
    values = []
    for run in runs:
        value = mean_value(filtered_rows(run.rows, method=method, freeze=freeze, train_frac=train_frac, blockage=blockage, epochs=epochs), metric)
        if value == value:
            values.append(value)
    return mean_std(values)


def _freeze_ablation(row: dict[str, str], path: Path, *, pattern: int, beam: str, results_root: Path, config: dict) -> PlotOutput | None:
    runs = _runs_for(results_root, pattern=pattern, beam=beam)
    if not runs:
        return None
    train_frac = float(dict(config.get("plotting", {})).get("comparison_primary_train_frac", 0.01))
    epoch = int(dict(config.get("plotting", {})).get("baseline_ft_epoch_for_plots", 10))
    blockage = 100
    values = []
    errors = []
    for freeze in FREEZE_MODES:
        avg, std = _mean_for_runs(runs, method="FT", freeze=freeze, train_frac=train_frac, blockage=blockage, epochs=epoch, metric="test_top3_m1db_%")
        values.append(avg)
        errors.append(std)
    if not any(value == value for value in values):
        return None
    title = f"Pattern {pattern} {beam}: freeze ablation"
    saved = grouped_bar(labels=list(FREEZE_MODES), series=[("FT", values, errors)], title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="freeze_mode_ablation", pattern=pattern, blocked_beams=(beam,), methods=FREEZE_MODES)


def _train_fraction(row: dict[str, str], path: Path, *, pattern: int, blockage: int, results_root: Path, config: dict) -> PlotOutput | None:
    runs = _runs_for(results_root, pattern=pattern)
    if not runs:
        return None
    plotting = dict(config.get("plotting", {}))
    train_fracs = [float(value) for value in plotting.get("comparison_fraction_sweep_train_fracs", [0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0])]
    epoch = int(plotting.get("baseline_ft_epoch_for_plots", 10))
    beams = sorted({run.beam for run in runs}, key=lambda beam: int(beam[1:]))
    series = []
    for beam in beams:
        values = []
        for train_frac in train_fracs:
            avg, _ = _mean_for_runs([run for run in runs if run.beam == beam], method="FT", freeze="freeze_cnn", train_frac=train_frac, blockage=blockage, epochs=epoch, metric="test_top3_m1db_%")
            values.append(avg)
        if any(value == value for value in values):
            series.append((beam, values))
    if not series:
        return None
    title = f"Pattern {pattern}: train fraction sweep at {blockage}% blockage"
    saved = categorical_bar(labels=[f"{value:g}" for value in train_fracs], series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="train_fraction_vs_accuracy", pattern=pattern, blocked_beams=tuple(beams), methods=("FT_freeze_cnn",))


def _epoch_budget(row: dict[str, str], path: Path, *, pattern: int, results_root: Path, config: dict) -> PlotOutput | None:
    runs = _runs_for(results_root, pattern=pattern)
    if not runs:
        return None
    epochs = [int(value) for value in config.get("training", {}).get("ft_epoch_sweep", [1, 2, 3, 5, 8, 10, 12])]
    train_frac = float(dict(config.get("plotting", {})).get("comparison_primary_train_frac", 0.01))
    beams = sorted({run.beam for run in runs}, key=lambda beam: int(beam[1:]))
    series = []
    for beam in beams:
        values = []
        for epoch in epochs:
            avg, _ = _mean_for_runs([run for run in runs if run.beam == beam], method="FT", freeze="freeze_cnn", train_frac=train_frac, epochs=epoch, metric="test_top3_m1db_%")
            values.append(avg)
        if any(value == value for value in values):
            series.append((beam, values))
    if not series:
        return None
    title = f"Pattern {pattern}: epoch-budget sweep"
    saved = categorical_bar(labels=[str(epoch) for epoch in epochs], series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="epoch_budget_vs_accuracy", pattern=pattern, blocked_beams=tuple(beams), methods=("FT_freeze_cnn",))


def _cross_train_fraction(row: dict[str, str], path: Path, *, source_name: str, results_root: Path, config: dict) -> PlotOutput | None:
    pairs = {"best": ((1, "B4"), (2, "B0")), "mid": ((1, "B6"), (2, "B2")), "worst": ((1, "B7"), (2, "B7"))}
    if source_name not in pairs:
        return None
    plotting = dict(config.get("plotting", {}))
    train_fracs = [float(value) for value in plotting.get("comparison_fraction_sweep_train_fracs", [0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0])]
    epoch = int(plotting.get("baseline_ft_epoch_for_plots", 10))
    series = []
    beams: list[str] = []
    for pattern, beam in pairs[source_name]:
        runs = _runs_for(results_root, pattern=pattern, beam=beam)
        if not runs:
            continue
        beams.append(f"P{pattern}_{beam}")
        values = []
        for train_frac in train_fracs:
            avg, _ = _mean_for_runs(runs, method="FT", freeze="freeze_cnn", train_frac=train_frac, blockage=100, epochs=epoch, metric="test_top3_m1db_%")
            values.append(avg)
        series.append((f"P{pattern} {beam}", values))
    if not series:
        return None
    title = f"Cross-pattern {source_name}: train-fraction sweep"
    saved = categorical_bar(labels=[f"{value:g}" for value in train_fracs], series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="cross_pattern_train_fraction", blocked_beams=tuple(beams), methods=("FT_freeze_cnn",))


def _multi_kpi(row: dict[str, str], path: Path, *, pattern: int, beam: str, results_root: Path, config: dict) -> PlotOutput | None:
    runs = _runs_for(results_root, pattern=pattern, beam=beam)
    if not runs:
        return None
    metrics = multi_kpi_specs(config)
    plotting = dict(config.get("plotting", {}))
    train_frac = float(plotting.get("comparison_primary_train_frac", 0.01))
    epoch = int(plotting.get("baseline_ft_epoch_for_plots", 10))
    blockage = _blockage_from_file(path.name) or 100
    series = []
    for label, method, freeze in (("BL-AG", "BL-AG", None), ("FT freeze_cnn", "FT", "freeze_cnn"), ("BL-AW", "BL-AW", None)):
        values = []
        for metric in metrics:
            avg, _ = _mean_for_runs(runs, method=method, freeze=freeze, train_frac=train_frac if method == "FT" else None, blockage=blockage, epochs=epoch if method == "FT" else None, metric=metric.column)
            values.append(avg)
        if any(value == value for value in values):
            series.append((label, values, None))
    if not series:
        return None
    title = f"Pattern {pattern} {beam}: multi-KPI view at {blockage}% blockage"
    saved = grouped_bar(labels=[metric.key for metric in metrics], series=series, title=title, ylabel="Accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="multi_kpi", pattern=pattern, blocked_beams=(beam,), methods=tuple(item[0] for item in series))


def _matrix_for_prefix(rows: list[dict[str, str]], beams: list[str], blockages: list[int], prefix: str) -> list[list[float]]:
    return [[_summary_value(_summary_row(rows, beam, blockage), prefix) for blockage in blockages] for beam in beams]


HEATMAP_VMIN = 40.0
HEATMAP_VMAX = 100.0


def _text_color(value: float, *, vmin: float = HEATMAP_VMIN, vmax: float = HEATMAP_VMAX) -> str:
    if value != value:
        return "black"
    scale = max(1e-9, float(vmax) - float(vmin))
    normalized = min(1.0, max(0.0, (float(value) - float(vmin)) / scale))
    red, green, blue, _ = plt.get_cmap("RdYlGn")(normalized)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance >= 0.58 else "white"


def _draw_method_heatmap(ax, matrix: list[list[float]], *, beams: list[str], blockages: list[int], title: str):
    data = [[float(value) for value in row] for row in matrix]
    image = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    ax.set_xticks(range(len(blockages)))
    ax.set_xticklabels([f"{value}%" for value in blockages])
    ax.set_yticks(range(len(beams)))
    ax.set_yticklabels(beams)
    ax.set_xlabel("Blockage level")
    ax.set_ylabel("Blocked beam")
    ax.set_title(title, fontsize=10.5, pad=10)
    for row_idx, values in enumerate(data):
        for col_idx, value in enumerate(values):
            if value == value:
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=8, color=_text_color(value))
    return image


def _seed_label(rows_by_pattern: dict[int, list[dict[str, str]]]) -> str:
    seed_values = sorted({row.get("seed_ids", "") for rows in rows_by_pattern.values() for row in rows if row.get("seed_ids")})
    return seed_values[0] if len(seed_values) == 1 else "selected seeds"


def _combined_heatmap(row: dict[str, str], path: Path, *, source_data: Path) -> PlotOutput | None:
    rows_by_pattern = {pattern: _summary_rows(source_data, pattern) for pattern in (1, 2)}
    grids: dict[int, tuple[list[str], list[int], list[list[float]], list[list[float]]]] = {}
    for pattern, rows in rows_by_pattern.items():
        beams = _sorted_beams(rows)
        blockages = _sorted_blockages(rows)
        if rows and beams and blockages:
            grids[pattern] = (
                beams,
                blockages,
                _matrix_for_prefix(rows, beams, blockages, "blag"),
                _matrix_for_prefix(rows, beams, blockages, "ft"),
            )
    if not grids:
        return None

    fig, axes = plt.subplots(2, 2, figsize=figure_size("heatmap"), constrained_layout=True)
    image = None
    panel_labels = {1: ("a", "b"), 2: ("c", "d")}
    for row_idx, pattern in enumerate((1, 2)):
        if pattern not in grids:
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")
            continue
        beams, blockages, blag, ft = grids[pattern]
        image = _draw_method_heatmap(
            axes[row_idx, 0],
            blag,
            beams=beams,
            blockages=blockages,
            title=f"({panel_labels[pattern][0]}) Pattern {pattern} - BL-AG (clean-trained, no adaptation)",
        )
        _draw_method_heatmap(
            axes[row_idx, 1],
            ft,
            beams=beams,
            blockages=blockages,
            title=f"({panel_labels[pattern][1]}) Pattern {pattern} - FT freeze-CNN (1% blocked, 10 epochs)",
        )
    if image is None:
        return None
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02, location="right")
    colorbar.set_label("Top-3 within 1 dB accuracy (%)")
    seed_label = _seed_label(rows_by_pattern)
    title = f"Beam-by-blockage accuracy: BL-AG (left) versus 1% FT freeze-CNN (right) - {seed_label}"
    fig.suptitle(title, fontsize=12)
    saved = save_figure(fig, path, plot_id=path)
    blocked_beams = tuple(f"P{pattern} {beam}" for pattern, (beams, _blockages, _blag, _ft) in grids.items() for beam in beams)
    return _plot_output(row, saved, title=title, family="heatmaps", blocked_beams=blocked_beams, methods=("BL-AG", "FT_freeze_cnn"))


def _convergence(row: dict[str, str], path: Path, *, pattern: int, beam: str, results_root: Path) -> PlotOutput | None:
    runs = _runs_for(results_root, pattern=pattern, beam=beam)
    histories = []
    for run in runs:
        histories.extend(sorted((run.path / "jobs").glob("finetune/trainfrac_0.010000/blockage_100/freeze_freeze_cnn/history.csv")))
        histories.extend(sorted((run.path / "jobs").glob("**/freeze_freeze_cnn/history.csv")))
    history_path = next((item for item in histories if item.exists()), None)
    if history_path is None:
        return None
    rows = read_csv(history_path)
    if not rows:
        return None
    labels = [row.get("epoch", str(idx + 1)) for idx, row in enumerate(rows)]
    series = []
    for label, column in (("validation", "val_top3_m1db_%"), ("test", "test_top3_m1db_%")):
        values = [as_float(item.get(column)) for item in rows]
        if any(value == value for value in values):
            series.append((label, values))
    if not series:
        return None
    title = f"Pattern {pattern} {beam}: convergence diagnostic"
    saved = categorical_bar(labels=labels, series=series, title=title, ylabel="Top-3 within 1 dB accuracy (%)", path=path, plot_id=path)
    return _plot_output(row, saved, title=title, family="convergence", pattern=pattern, blocked_beams=(beam,), methods=("FT_freeze_cnn",))


def _generate_plan_row(
    row: dict[str, str],
    *,
    output_root: Path,
    source_selected_root: Path,
    source_data: Path,
    results_root: Path,
    config: dict,
) -> tuple[PlotOutput | None, str]:
    selected_file = row.get("selected_file", "")
    if not selected_file:
        return None, "missing selected_file"
    path = output_root / selected_file
    folder = selected_file.split("/", 1)[0]
    pattern = _pattern_from_text(selected_file, row.get("role", ""))
    beams = _beams_from_text(selected_file, row.get("role", ""))
    beam = beams[0] if beams else None
    blockage = _blockage_from_file(selected_file)

    if folder == "fig00_procedure":
        return _procedure_plot(row, path, source_selected_root=source_selected_root), ""
    if folder == "fig01_setup" and pattern is not None:
        return _setup_plot(row, path, pattern=pattern, config=config, results_root=results_root), ""
    if folder in {"fig02_aggregate", "fig13_intermediate_blockage"} and pattern is not None and blockage is not None:
        output = _summary_method_plot(row, path, pattern=pattern, blockage=blockage, source_data=source_data, family="method_comparison")
        return output, "" if output else "summary source data missing"
    if folder == "fig05_multiseed_representative" and pattern is not None and "aggregate_methods" in selected_file and blockage is not None:
        output = _summary_method_plot(row, path, pattern=pattern, blockage=blockage, source_data=source_data, family="blockage_vs_accuracy")
        return output, "" if output else "summary source data missing"
    if folder == "fig05_multiseed_representative" and pattern is not None and beam is not None:
        output = _beam_multiseed_plot(row, path, pattern=pattern, beam=beam, source_data=source_data)
        return output, "" if output else "summary source data missing"
    if folder == "fig03_blag_degradation" and pattern is not None:
        output = _summary_bar_plot(row, path, pattern=pattern, prefix="blag", title_label="BL-AG degradation", source_data=source_data)
        return output, "" if output else "summary source data missing"
    if folder == "fig04_ft_recovery" and pattern is not None:
        output = _summary_bar_plot(row, path, pattern=pattern, prefix="ft", title_label="FT freeze_cnn recovery", source_data=source_data)
        return output, "" if output else "summary source data missing"
    if folder == "fig14_blaw_ceiling" and pattern is not None:
        output = _summary_bar_plot(row, path, pattern=pattern, prefix="blaw", title_label="BL-AW retraining ceiling", source_data=source_data)
        return output, "" if output else "summary source data missing"
    if folder == "fig06_freeze_ablation" and pattern is not None and beam is not None:
        output = _freeze_ablation(row, path, pattern=pattern, beam=beam, results_root=results_root, config=config)
        return output, "" if output else "matching run results missing"
    if folder == "fig07_data_efficiency" and pattern is not None and blockage is not None:
        output = _train_fraction(row, path, pattern=pattern, blockage=blockage, results_root=results_root, config=config)
        return output, "" if output else "matching run results missing"
    if folder == "fig08_epoch_budget" and pattern is not None:
        output = _epoch_budget(row, path, pattern=pattern, results_root=results_root, config=config)
        return output, "" if output else "matching run results missing"
    if folder == "fig09_cross_pattern_methods":
        source_name = _source_name_from_file(selected_file)
        side = _side_from_file(selected_file)
        if source_name and side:
            output = _cross_all_methods(row, path, source_name=source_name, side=side, source_data=source_data)
            return output, "" if output else "cross-pattern source data missing"
    if folder == "fig10_cross_pattern_train_fraction":
        source_name = _source_name_from_file(selected_file)
        if source_name:
            output = _cross_train_fraction(row, path, source_name=source_name, results_root=results_root, config=config)
            return output, "" if output else "matching run results missing"
    if folder == "fig11_multi_kpi" and pattern is not None and beam is not None:
        output = _multi_kpi(row, path, pattern=pattern, beam=beam, results_root=results_root, config=config)
        return output, "" if output else "matching run results missing"
    if folder == "fig12_non_ml":
        scope = "all_blocked_beams" if "all_beams" in selected_file else "representative"
        output = _non_ml_plot(row, path, source_data=source_data, scope=scope)
        return output, "" if output else "non-ML source data missing"
    if folder == "fig15_cross_pattern_baseline":
        source_name = _source_name_from_file(selected_file)
        if source_name:
            output = _cross_compared(row, path, source_name=source_name, method="BL-AG", title_method="clean-trained baseline", source_data=source_data)
            return output, "" if output else "cross-pattern source data missing"
    if folder == "fig16_cross_pattern_ft":
        source_name = _source_name_from_file(selected_file)
        if source_name:
            output = _cross_compared(row, path, source_name=source_name, method="FT:freeze_cnn", title_method="proposed FT", source_data=source_data)
            return output, "" if output else "cross-pattern source data missing"
    if folder == "fig17_heatmap":
        output = _combined_heatmap(row, path, source_data=source_data)
        return output, "" if output else "summary source data missing"
    if folder == "fig18_convergence" and pattern is not None and beam is not None:
        output = _convergence(row, path, pattern=pattern, beam=beam, results_root=results_root)
        return output, "" if output else "matching history.csv missing"
    return None, f"no generator for {selected_file}"


def build_selected_plots(
    selection: PlotSelection | None = None,
    *,
    organize: bool = True,
    selected_root: str | Path = DEFAULT_SELECTED_ROOT,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    config: str | Path = DEFAULT_CONFIG_PATH,
) -> list[PlotOutput]:
    selection = selection or PlotSelection()
    selected_root = Path(selected_root)
    results_root = Path(results_root)
    config_data = load_config(config)
    plan_root = _selected_plan_root(selected_root)
    _seed_selected_root(plan_root, selected_root)
    source_data = _source_data_root(selected_root)
    refresh_selected_source_data(results_root, selected_root, config_data)
    staged_outputs: list[PlotOutput] = []
    status_rows: list[dict[str, object]] = []
    skipped_selected = 0
    included_count = 0

    with tempfile.TemporaryDirectory(prefix="blocking_v5_selected_stage_") as temp_dir:
        stage_root = Path(temp_dir) / "selected_plots"
        for row in _selected_plan(plan_root):
            metadata = _plan_metadata(row)
            family = str(metadata["family"])
            selected_file = row.get("selected_file", "")
            if not matches_selection(metadata, selection):
                status_rows.append(
                    {
                        "selected_file": selected_file,
                        "status": "filtered",
                        "reason": "excluded by CLI selection",
                        "family": family,
                        "figure": row.get("figure", ""),
                        "role": row.get("role", ""),
                        "source_path": "",
                        "data_source": row.get("data_source", ""),
                    }
                )
                continue
            included_count += 1
            output, reason = _generate_plan_row(
                row,
                output_root=stage_root,
                source_selected_root=plan_root,
                source_data=source_data,
                results_root=results_root,
                config=config_data,
            )
            status = "generated" if output else "skipped"
            if output:
                staged_outputs.append(output)
            else:
                skipped_selected += 1
            status_rows.append(
                {
                    "selected_file": selected_file,
                    "status": status,
                    "reason": reason,
                    "family": family,
                    "figure": row.get("figure", ""),
                    "role": row.get("role", ""),
                    "source_path": selected_file if output else "",
                    "data_source": row.get("data_source", ""),
                }
            )

        committed_outputs: list[PlotOutput] = []
        if included_count and skipped_selected == 0:
            for output in staged_outputs:
                rel_path = output.source_path.relative_to(stage_root)
                final_path = selected_root / rel_path
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output.source_path, final_path)
                committed_outputs.append(replace(output, source_path=final_path))
        elif staged_outputs:
            reason = f"not overwritten because selected generation was incomplete: {skipped_selected} included plot(s) skipped"
            for status_row in status_rows:
                if status_row["status"] == "generated":
                    status_row["status"] = "preserved"
                    status_row["reason"] = reason
                    status_row["source_path"] = ""

    write_csv(selected_root / "selected_plots_generation_status.csv", status_rows, STATUS_FIELDS)
    if committed_outputs and (organize or selection.organize):
        organize_results(results_root, selected_root=selected_root, selection=selection, copy_mode=selection.copy_mode, clean=selection.clean_organized, plot_outputs=committed_outputs)
    return committed_outputs
