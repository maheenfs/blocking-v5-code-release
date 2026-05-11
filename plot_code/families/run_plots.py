"""Per-run plotting from one results/<run_name>/ folder."""

from __future__ import annotations

from pathlib import Path

from ..charts import categorical_bar, grouped_bar
from ..config import DEFAULT_CONFIG_PATH, DEFAULT_PLOTS_ROOT, PlotSelection, load_config
from ..data_loading import as_float, as_int, filtered_rows, mean_value, parse_run_name, read_csv
from ..filters import family_enabled
from ..metrics import metric_specs_from_keys, multi_kpi_specs
from ..organization import PlotOutput, organize_results
from ..style import slugify


def _selected_values(config: dict, selection: PlotSelection) -> tuple[list[float], list[int], list[int]]:
    plotting = dict(config.get("plotting", {}))
    train_fracs = list(selection.train_fracs or plotting.get("run_fraction_sweep_train_fracs", [0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0]))
    blockages = list(selection.blockages or plotting.get("run_plot_blockages", [20, 80, 100]))
    epochs = list(selection.epochs or config.get("training", {}).get("ft_epoch_sweep", [1, 2, 3, 5, 8, 10, 12]))
    return [float(v) for v in train_fracs], [int(v) for v in blockages], [int(v) for v in epochs]


def _run_context(run_dir: Path) -> str:
    return run_dir.name.replace("blocking_v5_", "").replace("_", " ")


def _run_metadata(run_dir: Path) -> tuple[tuple[str, ...], int | None, tuple[str, ...]]:
    seed, pattern, beam_index = parse_run_name(run_dir)
    seeds = () if seed is None else (str(seed),)
    beams = () if beam_index is None else (f"B{beam_index}",)
    return seeds, pattern, beams


def _record(
    path: Path,
    *,
    title: str,
    kpi: str,
    family: str,
    run_dir: Path,
    methods: tuple[str, ...],
) -> PlotOutput:
    seeds, pattern, beams = _run_metadata(run_dir)
    return PlotOutput(
        source_path=path,
        title=title,
        kpi=kpi,
        family=family,
        scope="results",
        seeds=seeds,
        pattern=pattern,
        blocked_beams=beams,
        methods=methods,
    )


def _loss_ylim(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if value == value]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    span = max(1e-6, high - low)
    pad = max(0.03, 0.12 * span)
    ymin = max(0.0, low - pad) if low >= 0.0 else low - pad
    ymax = high + pad
    if ymax <= ymin:
        ymax = ymin + 1.0
    return ymin, ymax


def _loss_series(rows: list[dict[str, str]]) -> list[tuple[str, list[float]]]:
    total_values = [as_float(row.get("train_loss_total")) for row in rows]
    ce_values = [as_float(row.get("train_loss_ce")) for row in rows]
    kpi_values = [as_float(row.get("train_loss_kpi")) for row in rows]
    series: list[tuple[str, list[float]]] = [("Train total", total_values)]

    has_distinct_ce = any(
        total == total and ce == ce and abs(float(total) - float(ce)) > 1e-10
        for total, ce in zip(total_values, ce_values)
    )
    if has_distinct_ce:
        series.append(("Cross-entropy", ce_values))

    has_kpi_component = any(value == value and abs(float(value)) > 1e-12 for value in kpi_values)
    if has_kpi_component:
        series.append(("KPI surrogate", kpi_values))
    return series


def _history_context(run_dir: Path, history_path: Path) -> dict[str, object] | None:
    try:
        parts = history_path.relative_to(run_dir).parts
    except ValueError:
        return None

    if parts[:3] == ("jobs", "baseline", "bl_ag_train"):
        return {
            "stem": "bl_ag_train_loss",
            "title": f"{_run_context(run_dir)}: BL-AG training loss by epoch",
            "blockage": None,
            "train_frac": None,
            "methods": ("BL-AG",),
        }

    if len(parts) >= 4 and parts[:2] == ("jobs", "baseline") and parts[2].startswith("bl_aw_blockage_"):
        blockage = as_int(parts[2].split("_")[-1])
        return {
            "stem": f"bl_aw_blockage_{blockage:03d}_loss",
            "title": f"{_run_context(run_dir)}: BL-AW training loss at {blockage}% blockage",
            "blockage": blockage,
            "train_frac": None,
            "methods": ("BL-AW",),
        }

    if len(parts) >= 6 and parts[:2] == ("jobs", "finetune"):
        train_frac = as_float(parts[2].replace("trainfrac_", "", 1))
        blockage = as_int(parts[3].replace("blockage_", "", 1))
        freeze = parts[4].replace("freeze_", "", 1)
        stem = f"ft_{train_frac:.6f}_blockage_{blockage:03d}_{freeze}_loss"
        return {
            "stem": stem,
            "title": f"{_run_context(run_dir)}: FT loss, tf={train_frac:g}, blockage={blockage}%, {freeze}",
            "blockage": blockage,
            "train_frac": train_frac,
            "methods": (f"FT {freeze}",),
        }
    return None


def _history_selected(context: dict[str, object], selection: PlotSelection) -> bool:
    blockage = context.get("blockage")
    if selection.blockages and blockage is not None and int(blockage) not in set(selection.blockages):
        return False
    if selection.blockages and blockage is None:
        return False
    train_frac = context.get("train_frac")
    if selection.train_fracs and train_frac is not None and all(abs(float(train_frac) - tf) > 1e-8 for tf in selection.train_fracs):
        return False
    if selection.train_fracs and train_frac is None:
        return False
    return True


def _build_training_loss_plots(run_dir: Path, out_root: Path, selection: PlotSelection) -> list[PlotOutput]:
    histories = sorted(run_dir.glob("jobs/**/history.csv"))
    outputs: list[PlotOutput] = []
    if not histories:
        return outputs

    loss_dir = out_root / "train_loss"
    for history_path in histories:
        context = _history_context(run_dir, history_path)
        if context is None or not _history_selected(context, selection):
            continue
        rows = read_csv(history_path)
        if not rows or "train_loss_total" not in rows[0]:
            continue
        labels = [str(as_int(row.get("epoch"), idx + 1)) for idx, row in enumerate(rows)]
        series = _loss_series(rows)
        values = [value for _, items in series for value in items]
        if not any(value == value for value in values):
            continue
        title = str(context["title"])
        stem = str(context["stem"])
        methods = tuple(str(item) for item in context["methods"])
        ylim = _loss_ylim(values)
        path = categorical_bar(
            labels=labels,
            series=series,
            title=title,
            ylabel="Loss",
            path=loss_dir / f"{stem}.png",
            ylim=ylim,
            annotate=False,
            value_format="{:.4f}",
        )
        outputs.append(_record(path, title=title, kpi="train_loss", family="train_loss", run_dir=run_dir, methods=methods))
    return outputs


def build_run_plots(
    run_dir: str | Path,
    *,
    selection: PlotSelection | None = None,
    organize: bool = False,
) -> list[PlotOutput]:
    selection = selection or PlotSelection()
    run_dir = Path(run_dir)
    rows = read_csv(run_dir / "results.csv")
    if not rows:
        return []
    config = load_config(run_dir / "config.json") if (run_dir / "config.json").exists() else load_config(DEFAULT_CONFIG_PATH)
    metrics = metric_specs_from_keys(selection.kpis or tuple(dict(config.get("plotting", {})).get("metric_plot_keys", ["primary"])), config)
    train_fracs, blockages, epochs = _selected_values(config, selection)
    baseline_epoch = int(dict(config.get("plotting", {})).get("baseline_ft_epoch_for_plots", 10))
    out_root = DEFAULT_PLOTS_ROOT / "run_plots" / run_dir.name
    context = _run_context(run_dir)
    outputs: list[PlotOutput] = []

    for metric in metrics:
        metric_dir = out_root / slugify(metric.key)
        if family_enabled(selection, "blockage_vs_accuracy"):
            for train_frac in train_fracs:
                labels = [str(bp) for bp in blockages]
                series = []
                for method, freeze in [("BL-AG", None), ("BL-AW", None), ("FT", "no_freeze"), ("FT", "freeze_cnn"), ("FT", "freeze_head"), ("FT", "freeze_all")]:
                    name = method if freeze is None else f"FT {freeze}"
                    values = [
                        mean_value(
                            filtered_rows(rows, method=method, freeze=freeze, train_frac=train_frac if method == "FT" else None, blockage=bp, epochs=baseline_epoch if method == "FT" else None),
                            metric.column,
                        )
                        for bp in blockages
                    ]
                    if any(v == v for v in values):
                        series.append((name, values, None))
                title = f"{context}: accuracy vs blockage, train fraction {train_frac:g}"
                path = grouped_bar(
                    labels=labels,
                    series=series,
                    title=title,
                    ylabel=metric.label,
                    path=metric_dir / "blockage_vs_accuracy" / f"tf{train_frac:g}_ep{baseline_epoch}.png",
                )
                outputs.append(
                    _record(
                        path,
                        title=title,
                        kpi=metric.key,
                        family="blockage_vs_accuracy",
                        run_dir=run_dir,
                        methods=tuple(item[0] for item in series),
                    )
                )

        if family_enabled(selection, "train_fraction_vs_accuracy"):
            for blockage in blockages:
                labels = [f"{tf:g}" for tf in train_fracs]
                series = []
                for freeze in ["no_freeze", "freeze_cnn", "freeze_head", "freeze_all"]:
                    values = [
                        mean_value(filtered_rows(rows, method="FT", freeze=freeze, train_frac=tf, blockage=blockage, epochs=baseline_epoch), metric.column)
                        for tf in train_fracs
                    ]
                    if any(v == v for v in values):
                        series.append((freeze, values, None))
                title = f"{context}: accuracy vs train fraction at {blockage}% blockage"
                path = grouped_bar(
                    labels=labels,
                    series=series,
                    title=title,
                    ylabel=metric.label,
                    path=metric_dir / "train_fraction_vs_accuracy" / f"blockage_{blockage:03d}.png",
                )
                outputs.append(
                    _record(
                        path,
                        title=title,
                        kpi=metric.key,
                        family="train_fraction_vs_accuracy",
                        run_dir=run_dir,
                        methods=tuple(item[0] for item in series),
                    )
                )

        if family_enabled(selection, "epoch_budget_vs_accuracy"):
            labels = [str(epoch) for epoch in epochs]
            series = []
            for freeze in ["no_freeze", "freeze_cnn", "freeze_head", "freeze_all"]:
                values = []
                for epoch in epochs:
                    epoch_rows = filtered_rows(rows, method="FT", freeze=freeze, train_frac=train_fracs[0] if train_fracs else 0.01, epochs=epoch)
                    values.append(mean_value(epoch_rows, metric.column))
                if any(v == v for v in values):
                    series.append((freeze, values))
            title = f"{context}: epoch budget sweep"
            path = categorical_bar(
                labels=labels,
                series=series,
                title=title,
                ylabel=metric.label,
                path=metric_dir / "epoch_budget_vs_accuracy" / "epoch_budget.png",
            )
            outputs.append(
                _record(
                    path,
                    title=title,
                    kpi=metric.key,
                    family="epoch_budget_vs_accuracy",
                    run_dir=run_dir,
                    methods=tuple(item[0] for item in series),
                )
            )

    if family_enabled(selection, "multi_kpi"):
        kpi_metrics = metric_specs_from_keys(selection.kpis, config) if selection.kpis else multi_kpi_specs(config)
        train_frac = train_fracs[0] if train_fracs else float(dict(config.get("plotting", {})).get("comparison_primary_train_frac", 0.01))
        for blockage in blockages:
            labels = [metric.key for metric in kpi_metrics]
            series = []
            for name, method, freeze in [("BL-AG", "BL-AG", None), ("FT freeze_cnn", "FT", "freeze_cnn"), ("BL-AW", "BL-AW", None)]:
                values = [
                    mean_value(
                        filtered_rows(
                            rows,
                            method=method,
                            freeze=freeze,
                            train_frac=train_frac if method == "FT" else None,
                            blockage=blockage,
                            epochs=baseline_epoch if method == "FT" else None,
                        ),
                        metric.column,
                    )
                    for metric in kpi_metrics
                ]
                if any(value == value for value in values):
                    series.append((name, values, None))
            if series:
                title = f"{context}: multi-KPI view at {blockage}% blockage"
                path = grouped_bar(
                    labels=labels,
                    series=series,
                    title=title,
                    ylabel="Accuracy (%)",
                    path=out_root / "multi_kpi" / f"blockage_{blockage:03d}_tf{train_frac:g}_ep{baseline_epoch}.png",
                )
                outputs.append(
                    _record(
                        path,
                        title=title,
                        kpi="multi_kpi",
                        family="multi_kpi",
                        run_dir=run_dir,
                        methods=tuple(item[0] for item in series),
                    )
                )

    if bool(dict(config.get("plotting", {})).get("plot_loss_curves", False)) and family_enabled(selection, "train_loss"):
        outputs.extend(_build_training_loss_plots(run_dir, out_root, selection))

    if organize or selection.organize:
        organize_results(run_dir.parent, selection=selection, copy_mode=selection.copy_mode, clean=selection.clean_organized, plot_outputs=outputs)
    return outputs
