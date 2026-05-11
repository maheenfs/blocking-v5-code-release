"""Refresh selected-plot source tables from completed run results."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from statistics import mean

from ..config import normalize_beam_token
from ..data_loading import RunInfo, discover_runs, filtered_rows, mean_std, mean_value, write_csv
from ..metrics import metric_column


SUMMARY_FIELDS = [
    "blocked_beam",
    "seed_values",
    "seed_count",
    "blockage_%",
    "ft_freeze",
    "ft_train_frac",
    "ft_epochs",
    "ft_primary_%",
    "ft_primary_seed_std",
    "blag_primary_%",
    "blag_primary_seed_std",
    "blaw_primary_%",
    "blaw_primary_seed_std",
]

CROSS_FIELDS = [
    "method",
    "metric",
    "p1_seed_values",
    "p2_seed_values",
    "blockage_%",
    "p1_value",
    "p1_seed_std",
    "p2_value",
    "p2_seed_std",
    "delta_pp",
    "delta_seed_std",
]

NON_ML_FIELDS = [
    "scope",
    "label",
    "pattern",
    "blocked_beam_index",
    "method",
    "metric",
    "mean_test_primary_%",
    "seed_std_pp",
    "seed_n",
    "blocked_levels",
]

NON_ML_METHODS = ("MAX-SETB", "NN-ANGLE", "RANDOM-SETB")


def refresh_selected_source_data(results_root: str | Path, selected_root: str | Path, config: dict) -> list[Path]:
    """Regenerate publication source CSVs when a complete run set exists.

    The release ships source CSVs so selected plots can be inspected before a
    new run. Once the configured batch finishes, this function replaces those
    tables with values computed from the current ``results/`` tree. Partial
    result trees are left untouched to avoid mixing old and new provenance.
    """
    results_root = Path(results_root)
    selected_root = Path(selected_root)
    runs = discover_runs(results_root)
    if not runs or not _configured_runs_are_complete(runs, config):
        return []

    source_root = selected_root / "source_data"
    with tempfile.TemporaryDirectory(prefix="blocking_v5_source_data_") as temp_dir:
        temp_root = Path(temp_dir) / "source_data"
        generated = _write_source_tables(temp_root, runs, config)
        if not generated:
            return []
        written: list[Path] = []
        for temp_path in generated:
            final_path = source_root / temp_path.name
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_path, final_path)
            written.append(final_path)
    return written


def _configured_seeds(config: dict) -> tuple[int, ...]:
    training = dict(config.get("training", {}) or {})
    seeds: list[int] = []
    if "seed" in training:
        seeds.append(int(training["seed"]))
    seeds.extend(int(seed) for seed in training.get("seeds", []) or [])
    return tuple(dict.fromkeys(seeds))


def _configured_jobs(config: dict) -> tuple[tuple[int, int], ...]:
    jobs = []
    for job in config.get("batch_jobs", []) or []:
        jobs.append((int(job["pattern"]), int(job["blocked_beam_index"])))
    if jobs:
        return tuple(jobs)
    pattern = int(config.get("setb_pattern", 1))
    beams = config.get("blocked_setb_indices", []) or []
    return tuple((pattern, int(beam)) for beam in beams)


def _configured_runs_are_complete(runs: list[RunInfo], config: dict) -> bool:
    expected = {
        (seed, pattern, beam)
        for seed in _configured_seeds(config)
        for pattern, beam in _configured_jobs(config)
    }
    if not expected:
        return False
    present = {(run.seed, run.pattern, run.beam_index) for run in runs}
    return expected.issubset(present)


def _plotting_value(config: dict, key: str, default: object) -> object:
    return dict(config.get("plotting", {}) or {}).get(key, default)


def _primary_column(config: dict) -> str:
    return metric_column("primary", config)


def _blockages(config: dict) -> tuple[int, ...]:
    return tuple(int(value) for value in config.get("blockage_levels", []) or [])


def _selected_methods(config: dict) -> tuple[str, ...]:
    return tuple(str(method) for method in config.get("experiments", []) or [])


def _format_float(value: float, digits: int = 6) -> str:
    return "" if value != value else f"{float(value):.{digits}f}"


def _seed_text(seeds: list[int]) -> str:
    return "+".join(str(seed) for seed in sorted(dict.fromkeys(seeds)))


def _joined_ints(values: tuple[int, ...]) -> str:
    return "+".join(str(value) for value in values)


def _runs_for(runs: list[RunInfo], *, pattern: int, beam: int | str) -> list[RunInfo]:
    wanted_beam = normalize_beam_token(beam)
    return [run for run in runs if run.pattern == int(pattern) and run.beam == wanted_beam]


def _seed_values(
    runs: list[RunInfo],
    *,
    method: str,
    column: str,
    freeze: str | None = None,
    train_frac: float | None = None,
    blockage: int | None = None,
    epochs: int | None = None,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for run in runs:
        if run.seed is None:
            continue
        value = mean_value(
            filtered_rows(
                run.rows,
                method=method,
                freeze=freeze,
                train_frac=train_frac,
                blockage=blockage,
                epochs=epochs,
            ),
            column,
        )
        if value == value:
            values[int(run.seed)] = value
    return values


def _aggregate(values_by_seed: dict[int, float]) -> tuple[float, float, list[int]]:
    seeds = sorted(values_by_seed)
    avg, std = mean_std(values_by_seed[seed] for seed in seeds)
    return avg, std, seeds


def _summary_rows_for_pattern(runs: list[RunInfo], config: dict, pattern: int) -> list[dict[str, object]]:
    column = _primary_column(config)
    train_frac = float(_plotting_value(config, "comparison_primary_train_frac", 0.01))
    epoch = int(_plotting_value(config, "baseline_ft_epoch_for_plots", 10))
    methods = set(_selected_methods(config))
    rows: list[dict[str, object]] = []
    beams = sorted(
        {run.beam_index for run in runs if run.pattern == int(pattern) and run.beam_index is not None}
    )
    for beam in beams:
        beam_runs = _runs_for(runs, pattern=pattern, beam=int(beam))
        for blockage in _blockages(config):
            ft = _method_summary(
                beam_runs,
                selected="FT" in methods,
                method="FT",
                column=column,
                freeze="freeze_cnn",
                train_frac=train_frac,
                blockage=blockage,
                epochs=epoch,
            )
            blag = _method_summary(
                beam_runs,
                selected="BL-AG" in methods,
                method="BL-AG",
                column=column,
                blockage=blockage,
            )
            blaw = _method_summary(
                beam_runs,
                selected="BL-AW" in methods,
                method="BL-AW",
                column=column,
                blockage=blockage,
            )
            seed_values = sorted(set(ft[2]) | set(blag[2]) | set(blaw[2]))
            if not seed_values:
                continue
            rows.append(
                {
                    "blocked_beam": normalize_beam_token(beam),
                    "seed_values": _seed_text(seed_values),
                    "seed_count": len(seed_values),
                    "blockage_%": int(blockage),
                    "ft_freeze": "freeze_cnn",
                    "ft_train_frac": _format_float(train_frac),
                    "ft_epochs": int(epoch),
                    "ft_primary_%": _format_float(ft[0]),
                    "ft_primary_seed_std": _format_float(ft[1]),
                    "blag_primary_%": _format_float(blag[0]),
                    "blag_primary_seed_std": _format_float(blag[1]),
                    "blaw_primary_%": _format_float(blaw[0]),
                    "blaw_primary_seed_std": _format_float(blaw[1]),
                }
            )
    return rows


def _method_summary(
    runs: list[RunInfo],
    *,
    selected: bool,
    method: str,
    column: str,
    freeze: str | None = None,
    train_frac: float | None = None,
    blockage: int | None = None,
    epochs: int | None = None,
) -> tuple[float, float, list[int]]:
    if not selected:
        return float("nan"), float("nan"), []
    values = _seed_values(
        runs,
        method=method,
        freeze=freeze,
        train_frac=train_frac,
        blockage=blockage,
        epochs=epochs,
        column=column,
    )
    return _aggregate(values)


def _comparison_pairs(config: dict) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    provenance = dict(config.get("scenario_provenance", {}) or {})
    raw = dict(provenance.get("pattern_comparisons", {}) or {})
    pairs: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for name, item in raw.items():
        pairs[str(name)] = ((1, int(item["p1_beam"])), (2, int(item["p2_beam"])))
    pairs.setdefault("same_b3", ((1, 3), (2, 3)))
    pairs.setdefault("same_b7", ((1, 7), (2, 7)))
    return pairs


def _method_specs(config: dict) -> tuple[tuple[str, str, str | None], ...]:
    methods = set(_selected_methods(config))
    specs: list[tuple[str, str, str | None]] = []
    if "BL-AG" in methods:
        specs.append(("BL-AG", "BL-AG", None))
    if "BL-AW" in methods:
        specs.append(("BL-AW", "BL-AW", None))
    if "FT" in methods:
        specs.append(("FT:freeze_cnn", "FT", "freeze_cnn"))
    return tuple(specs)


def _cross_rows(runs: list[RunInfo], config: dict, pair: tuple[tuple[int, int], tuple[int, int]]) -> list[dict[str, object]]:
    column = _primary_column(config)
    train_frac = float(_plotting_value(config, "comparison_primary_train_frac", 0.01))
    epoch = int(_plotting_value(config, "baseline_ft_epoch_for_plots", 10))
    (p1_pattern, p1_beam), (p2_pattern, p2_beam) = pair
    p1_runs = _runs_for(runs, pattern=p1_pattern, beam=p1_beam)
    p2_runs = _runs_for(runs, pattern=p2_pattern, beam=p2_beam)
    rows: list[dict[str, object]] = []
    for method_label, method, freeze in _method_specs(config):
        for blockage in _blockages(config):
            common_filters = {"method": method, "freeze": freeze, "blockage": blockage, "column": column}
            if method == "FT":
                common_filters.update({"train_frac": train_frac, "epochs": epoch})
            p1_values = _seed_values(p1_runs, **common_filters)
            p2_values = _seed_values(p2_runs, **common_filters)
            p1_avg, p1_std, p1_seeds = _aggregate(p1_values)
            p2_avg, p2_std, p2_seeds = _aggregate(p2_values)
            common_seeds = sorted(set(p1_values) & set(p2_values))
            deltas = [p1_values[seed] - p2_values[seed] for seed in common_seeds]
            delta_avg = mean(deltas) if deltas else float("nan")
            delta_std = mean_std(deltas)[1]
            if p1_avg != p1_avg and p2_avg != p2_avg:
                continue
            rows.append(
                {
                    "method": method_label,
                    "metric": "primary",
                    "p1_seed_values": _seed_text(p1_seeds),
                    "p2_seed_values": _seed_text(p2_seeds),
                    "blockage_%": int(blockage),
                    "p1_value": _format_float(p1_avg),
                    "p1_seed_std": _format_float(p1_std),
                    "p2_value": _format_float(p2_avg),
                    "p2_seed_std": _format_float(p2_std),
                    "delta_pp": _format_float(delta_avg),
                    "delta_seed_std": _format_float(delta_std),
                }
            )
    return rows


def _seed_level_average(run: RunInfo, *, method: str, column: str, blockages: tuple[int, ...]) -> float:
    values = []
    for blockage in blockages:
        value = mean_value(filtered_rows(run.rows, method=method, blockage=blockage), column)
        if value == value:
            values.append(value)
    return mean(values) if values else float("nan")


def _non_ml_rows(runs: list[RunInfo], config: dict) -> list[dict[str, object]]:
    column = _primary_column(config)
    selected = set(_selected_methods(config))
    methods = [method for method in NON_ML_METHODS if method in selected]
    blockages = tuple(value for value in _blockages(config) if value > 0)
    blocked_text = _joined_ints(blockages)
    rows: list[dict[str, object]] = []
    if not methods or not blockages:
        return rows

    representative = [(1, 6), (2, 2)]
    for pattern, beam in representative:
        scope_runs = _runs_for(runs, pattern=pattern, beam=beam)
        _append_non_ml_scope_rows(
            rows,
            scope_runs,
            methods=methods,
            column=column,
            blockages=blockages,
            scope="representative",
            label=f"P{pattern} B{beam}",
            pattern=pattern,
            beam=beam,
            blocked_text=blocked_text,
        )

    for pattern in sorted({run.pattern for run in runs if run.pattern is not None}):
        scope_runs = [run for run in runs if run.pattern == int(pattern)]
        _append_non_ml_scope_rows(
            rows,
            scope_runs,
            methods=methods,
            column=column,
            blockages=blockages,
            scope="all_blocked_beams",
            label=f"P{pattern} all beams",
            pattern=int(pattern),
            beam="",
            blocked_text=blocked_text,
        )
    return rows


def _append_non_ml_scope_rows(
    rows: list[dict[str, object]],
    scope_runs: list[RunInfo],
    *,
    methods: list[str],
    column: str,
    blockages: tuple[int, ...],
    scope: str,
    label: str,
    pattern: int,
    beam: int | str,
    blocked_text: str,
) -> None:
    if not scope_runs:
        return
    for method in methods:
        per_seed_values: dict[int, list[float]] = {}
        for run in scope_runs:
            if run.seed is None:
                continue
            value = _seed_level_average(run, method=method, column=column, blockages=blockages)
            if value == value:
                per_seed_values.setdefault(int(run.seed), []).append(value)
        seed_means = {seed: mean(values) for seed, values in per_seed_values.items() if values}
        avg, std, seeds = _aggregate(seed_means)
        if not seeds:
            continue
        rows.append(
            {
                "scope": scope,
                "label": label,
                "pattern": int(pattern),
                "blocked_beam_index": beam,
                "method": method,
                "metric": "primary",
                "mean_test_primary_%": _format_float(avg),
                "seed_std_pp": _format_float(std),
                "seed_n": len(seeds),
                "blocked_levels": blocked_text,
            }
        )


def _write_source_tables(source_root: Path, runs: list[RunInfo], config: dict) -> list[Path]:
    written: list[Path] = []
    for pattern in sorted({run.pattern for run in runs if run.pattern is not None}):
        rows = _summary_rows_for_pattern(runs, config, int(pattern))
        if rows:
            written.append(
                write_csv(
                    source_root / f"p{int(pattern)}_summary_ft_freeze_cnn_config.csv",
                    rows,
                    SUMMARY_FIELDS,
                )
            )

    for name, pair in _comparison_pairs(config).items():
        rows = _cross_rows(runs, config, pair)
        if rows:
            written.append(
                write_csv(
                    source_root / f"compare_patterns_{name}_summary_table.csv",
                    rows,
                    CROSS_FIELDS,
                )
            )

    non_ml_rows = _non_ml_rows(runs, config)
    if non_ml_rows:
        written.append(write_csv(source_root / "non_ml_baseline_summary.csv", non_ml_rows, NON_ML_FIELDS))
    return written
