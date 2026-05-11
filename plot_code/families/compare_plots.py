"""Comparison plots across completed runs and seeds."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..charts import categorical_bar, grouped_bar
from ..config import DEFAULT_CONFIG_PATH, DEFAULT_PLOTS_ROOT, DEFAULT_RESULTS_ROOT, DEFAULT_SELECTED_ROOT, PACKAGE_ROOT, PlotSelection, load_config, normalize_beam_token
from ..data_loading import as_float, discover_runs, filtered_rows, mean_std, mean_value, read_csv
from ..filters import family_enabled
from ..metrics import metric_specs_from_keys
from ..organization import PlotOutput, organize_results
from .selected_source_data import refresh_selected_source_data


def _has_numeric_value(values: list[float]) -> bool:
    """Return true when a plotted series has at least one real numeric value."""
    return any(value == value for value in values)


def _run_matches(run, selection: PlotSelection) -> bool:
    if selection.seeds and run.seed not in selection.seeds:
        return False
    if selection.patterns and run.pattern not in selection.patterns:
        return False
    if selection.blocked_beams and run.beam not in {normalize_beam_token(b) for b in selection.blocked_beams}:
        return False
    return True


def _seed_tuple(runs) -> tuple[str, ...]:
    return tuple(str(seed) for seed in sorted({run.seed for run in runs if run.seed is not None}))


def _build_non_ml_compare_plots(out_root: Path, selection: PlotSelection, source_data_root: Path) -> list[PlotOutput]:
    if selection.kpis and not ({"primary", "top3_m1db"} & set(selection.kpis)):
        return []
    source = source_data_root / "non_ml_baseline_summary.csv"
    rows = read_csv(source)
    outputs: list[PlotOutput] = []
    if not rows:
        return outputs

    for scope, title, filename in [
        ("representative", "Non-learning baseline comparison", "representative_non_ml_baselines.png"),
        ("all_blocked_beams", "Non-learning baselines across all blocked beams", "all_beams_non_ml_baselines.png"),
    ]:
        scoped_rows = [row for row in rows if row.get("scope") == scope]
        if not scoped_rows:
            continue
        labels = list(dict.fromkeys(row.get("label", "") for row in scoped_rows if row.get("label")))
        methods = list(dict.fromkeys(row.get("method", "") for row in scoped_rows if row.get("method")))
        series = []
        for method in methods:
            values = []
            errors = []
            for label in labels:
                candidate = next((row for row in scoped_rows if row.get("label") == label and row.get("method") == method), {})
                values.append(as_float(candidate.get("mean_test_primary_%")))
                errors.append(as_float(candidate.get("seed_std_pp"), 0.0))
            if any(value == value for value in values):
                series.append((method, values, errors))
        if not series:
            continue
        path = grouped_bar(
            labels=labels,
            series=series,
            title=title,
            ylabel="Top-3 within 1 dB accuracy (%)",
            path=out_root / "non_ml_baselines" / "primary" / filename,
        )
        outputs.append(
            PlotOutput(
                source_path=path,
                title=title,
                kpi="primary",
                family="non_ml_baselines",
                scope="results",
                seeds=("123", "456", "789"),
                methods=tuple(methods),
                data_source=source.relative_to(PACKAGE_ROOT).as_posix(),
            )
        )
    return outputs


def build_compare_plots(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    *,
    config: str | Path = DEFAULT_CONFIG_PATH,
    selection: PlotSelection | None = None,
    organize: bool = False,
) -> list[PlotOutput]:
    selection = selection or PlotSelection()
    cfg = load_config(config)
    selected_root = DEFAULT_SELECTED_ROOT
    source_data_root = selected_root / "source_data"
    refresh_selected_source_data(results_root, selected_root, cfg)
    metrics = metric_specs_from_keys(selection.kpis or tuple(dict(cfg.get("plotting", {})).get("metric_plot_keys", ["primary"])), cfg)
    blockages = list(selection.blockages or dict(cfg.get("plotting", {})).get("comparison_focus_blockages", [20, 80, 100]))
    blockages = [int(bp) for bp in blockages]
    train_frac = float((selection.train_fracs or [dict(cfg.get("plotting", {})).get("comparison_primary_train_frac", 0.01)])[0])
    baseline_epoch = int(dict(cfg.get("plotting", {})).get("baseline_ft_epoch_for_plots", 10))
    runs = [run for run in discover_runs(results_root) if _run_matches(run, selection)]
    out_root = DEFAULT_PLOTS_ROOT / "compare_plots"
    outputs: list[PlotOutput] = []

    by_pattern: dict[int, list] = defaultdict(list)
    for run in runs:
        if run.pattern is not None:
            by_pattern[int(run.pattern)].append(run)

    for metric in metrics:
        for pattern, pattern_runs in sorted(by_pattern.items()):
            beams = sorted({run.beam for run in pattern_runs})
            if family_enabled(selection, "method_comparison"):
                for blockage in blockages:
                    labels = beams
                    series = []
                    for name, method, freeze in [("BL-AG", "BL-AG", None), ("FT freeze_cnn", "FT", "freeze_cnn"), ("BL-AW", "BL-AW", None)]:
                        values = []
                        errors = []
                        for beam in beams:
                            seed_values = []
                            for run in pattern_runs:
                                if run.beam != beam:
                                    continue
                                rows = filtered_rows(
                                    run.rows,
                                    method=method,
                                    freeze=freeze,
                                    train_frac=train_frac if method == "FT" else None,
                                    blockage=blockage,
                                    epochs=baseline_epoch if method == "FT" else None,
                                )
                                value = mean_value(rows, metric.column)
                                if value == value:
                                    seed_values.append(value)
                            avg, std = mean_std(seed_values)
                            values.append(avg)
                            errors.append(std)
                        if _has_numeric_value(values):
                            series.append((name, values, errors))
                    if not series:
                        continue
                    title = f"Pattern {pattern}: method comparison at {blockage}% blockage"
                    path = grouped_bar(
                        labels=labels,
                        series=series,
                        title=title,
                        ylabel=metric.label,
                        path=out_root / f"pattern_{pattern}" / metric.key / f"all_methods_blockage_{blockage:03d}.png",
                    )
                    outputs.append(
                        PlotOutput(
                            source_path=path,
                            title=title,
                            kpi=metric.key,
                            family="method_comparison",
                            scope="results",
                            seeds=_seed_tuple(pattern_runs),
                            pattern=pattern,
                            blocked_beams=tuple(beams),
                            methods=tuple(name.replace(" ", "_") for name, _values, _errors in series),
                        )
                    )

            if family_enabled(selection, "blockage_vs_accuracy"):
                labels = [str(bp) for bp in blockages]
                series_bars = []
                for beam in beams:
                    values = []
                    for blockage in blockages:
                        seed_values = []
                        for run in pattern_runs:
                            if run.beam != beam:
                                continue
                            rows = filtered_rows(run.rows, method="FT", freeze="freeze_cnn", train_frac=train_frac, blockage=blockage, epochs=baseline_epoch)
                            value = mean_value(rows, metric.column)
                            if value == value:
                                seed_values.append(value)
                        values.append(mean_std(seed_values)[0])
                    if _has_numeric_value(values):
                        series_bars.append((beam, values))
                if not series_bars:
                    continue
                title = f"Pattern {pattern}: FT freeze_cnn across blocked beams"
                path = categorical_bar(
                    labels=labels,
                    series=series_bars,
                    title=title,
                    ylabel=metric.label,
                    path=out_root / f"pattern_{pattern}" / metric.key / "ft_freeze_cnn_vs_blockage.png",
                )
                outputs.append(
                    PlotOutput(
                        source_path=path,
                        title=title,
                        kpi=metric.key,
                        family="blockage_vs_accuracy",
                        scope="results",
                        seeds=_seed_tuple(pattern_runs),
                        pattern=pattern,
                        blocked_beams=tuple(beams),
                        methods=("FT_freeze_cnn",),
                    )
                )

    if family_enabled(selection, "cross_pattern_comparison"):
        for source_name in ["best", "mid", "worst", "same_b3", "same_b7"]:
            source = source_data_root / f"compare_patterns_{source_name}_summary_table.csv"
            rows = read_csv(source)
            if not rows:
                continue
            labels = sorted({row.get("blockage_%", "") for row in rows}, key=lambda value: int(float(value or 0)))
            if not labels:
                continue
            for metric in metrics:
                for method in ["BL-AG", "FT:freeze_cnn", "BL-AW"]:
                    method_rows = [row for row in rows if row.get("method") == method]
                    if not method_rows:
                        continue
                    series = []
                    for label, value_key, err_key in [("P1", "p1_value", "p1_seed_std"), ("P2", "p2_value", "p2_seed_std")]:
                        values = []
                        errors = []
                        for blockage in labels:
                            candidates = [row for row in method_rows if row.get("blockage_%") == blockage]
                            values.append(as_float(candidates[0].get(value_key)) if candidates else float("nan"))
                            errors.append(as_float(candidates[0].get(err_key), 0.0) if candidates else 0.0)
                        if _has_numeric_value(values):
                            series.append((label, values, errors))
                    if not series:
                        continue
                    title = f"Cross-pattern {source_name.replace('_', ' ')}: {method.replace('FT:freeze_cnn', 'FT freeze_cnn')}"
                    path = grouped_bar(
                        labels=labels,
                        series=series,
                        title=title,
                        ylabel=metric.label,
                        path=out_root / "cross_pattern" / metric.key / f"{source_name}_{method.replace(':', '_').lower()}_comparison.png",
                    )
                    outputs.append(
                        PlotOutput(
                            source_path=path,
                            title=title,
                            kpi=metric.key,
                            family="cross_pattern_comparison",
                            scope="results",
                            seeds=("123", "456", "789"),
                            methods=(method.replace(":", "_"),),
                            data_source=source.relative_to(PACKAGE_ROOT).as_posix(),
                        )
                    )

    if family_enabled(selection, "non_ml_baselines"):
        outputs.extend(_build_non_ml_compare_plots(out_root, selection, source_data_root))

    if organize or selection.organize:
        organize_results(results_root, selection=selection, copy_mode=selection.copy_mode, clean=selection.clean_organized, plot_outputs=outputs)
    return outputs
