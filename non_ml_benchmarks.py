"""Evaluate non-ML Set-B baselines for Blocking V5.

This script is intentionally separate from the ML result pipeline. It checks
whether simple non-ML rules are worth mentioning:

- MAX-SETB: choose the strongest measured Set-B beam for each sample.
- NN-ANGLE: choose the strongest measured Set-B beam, then rank all Set-A beams
  by angular distance from that Set-B beam.
- RANDOM-SETB: choose random Set-B beams; useful only as a lower-bound sanity check.
- STAT-NORMAL-24: fit a normal distribution to observed Set-B RSRP values and
  sample 24 synthetic values. This is RSRP-only exploratory analysis because
  synthetic values do not correspond to real beam identities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiment.config import BatchJob, load_config
from experiment.data import build_blocked_dataset_views
from experiment.geometry import build_setb_tx_indices
from experiment.io import write_csv
from experiment.metrics import build_metric_keys, evaluate_ranked_beams
from experiment.non_ml import max_setb_rankings, nn_angle_space_rankings


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: float) -> str:
    return f"{float(value):.6f}" if np.isfinite(float(value)) else "nan"


def load_test_split(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    data_dir = run_dir / "data"
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    test_idx = np.load(data_dir / "test_idx.npy")
    features = np.load(data_dir / "X_setb.npy")[test_idx]
    labels = np.load(data_dir / "y_tx.npy")[test_idx]
    rsrp_tx = np.load(data_dir / "rsrp_tx.npy")[test_idx]
    return features, labels, rsrp_tx, float(meta["global_min_db"]), int(meta.get("seed", -1))


def random_b_metric_mean(
    features_b: np.ndarray,
    rsrp_tx: np.ndarray,
    *,
    setb_tx: Sequence[int],
    kpi: object,
    trials: int,
    rng: np.random.Generator,
    primary_key: str,
) -> Tuple[Dict[str, float], float]:
    setb_tx_array = np.asarray(list(setb_tx), dtype=np.int64)
    metric_rows: List[Dict[str, float]] = []
    for _ in range(int(trials)):
        scores = rng.random((int(features_b.shape[0]), len(setb_tx_array)))
        order = np.argsort(scores, axis=1)
        rankings = setb_tx_array[order]
        metric_rows.append(evaluate_ranked_beams(rankings, rsrp_tx, kpi=kpi))

    keys = sorted({key for row in metric_rows for key in row})
    mean_metrics = {
        key: float(np.mean([finite_float(row.get(key)) for row in metric_rows]))
        for key in keys
    }
    primary_values = np.asarray([finite_float(row.get(primary_key)) for row in metric_rows], dtype=float)
    primary_std = float(np.nanstd(primary_values, ddof=1)) if primary_values.size > 1 else 0.0
    return mean_metrics, primary_std


def statistical_sampled24_summary(features_b: np.ndarray, *, trials: int, rng: np.random.Generator) -> Dict[str, float]:
    values = np.asarray(features_b, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    max_b = np.asarray(features_b, dtype=np.float64).max(axis=1)
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    p_exceeds: List[float] = []
    mean_sampled_max: List[float] = []
    positive_gain: List[float] = []
    for _ in range(int(trials)):
        sampled_max = rng.normal(loc=mu, scale=max(sigma, 1e-9), size=(int(features_b.shape[0]), 24)).max(axis=1)
        p_exceeds.append(float(np.mean(sampled_max > max_b)))
        mean_sampled_max.append(float(np.mean(sampled_max)))
        positive_gain.append(float(np.mean(np.maximum(0.0, sampled_max - max_b))))

    return {
        "mu_db": mu,
        "sigma_db": sigma,
        "mean_max_b_db": float(np.mean(max_b)),
        "mean_sampled24_max_db": float(np.mean(mean_sampled_max)),
        "p_sampled24_exceeds_max_b_%": 100.0 * float(np.mean(p_exceeds)),
        "expected_positive_gain_db": float(np.mean(positive_gain)),
    }


def metric_row(
    *,
    seed: int,
    method: str,
    job: BatchJob,
    blockage_pct: int,
    trials: int,
    metrics: Dict[str, float],
    primary_std_pp: float = 0.0,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "seed": int(seed),
        "method": str(method),
        "pattern": int(job.pattern),
        "blocked_beam_index": int(job.blocked_beam_index),
        "blockage_%": int(blockage_pct),
        "trials": int(trials),
        "primary_std_pp": fmt(primary_std_pp),
    }
    for key, value in metrics.items():
        row[f"test_{key}"] = fmt(value)
    return row


def summarize_primary(rows: Sequence[Dict[str, object]], *, primary_col: str) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    scopes = [
        ("blocked_only", lambda blockage: int(blockage) > 0),
        ("all_levels", lambda blockage: True),
    ]
    for scope_name, include_row in scopes:
        grouped: Dict[Tuple[str, str], List[float]] = {}
        for row in rows:
            blockage = int(row.get("blockage_%", 0))
            if not include_row(blockage):
                continue
            key = (str(row.get("method", "")), str(row.get("pattern", "")))
            grouped.setdefault(key, []).append(finite_float(row.get(primary_col)))

        for (method, pattern), values in sorted(grouped.items()):
            vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
            if vals.size == 0:
                continue
            summary.append(
                {
                    "scenario_scope": scope_name,
                    "method": method,
                    "pattern": pattern,
                    "scenarios": int(vals.size),
                    primary_col: fmt(float(np.mean(vals))),
                    f"{primary_col}_std": fmt(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0),
                }
            )
    return summary


def summarize_primary_by_beam(rows: Sequence[Dict[str, object]], *, primary_col: str) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    scopes = [
        ("blocked_only", lambda blockage: int(blockage) > 0),
        ("all_levels", lambda blockage: True),
    ]
    for scope_name, include_row in scopes:
        grouped: Dict[Tuple[str, str, str], List[float]] = {}
        for row in rows:
            blockage = int(row.get("blockage_%", 0))
            if not include_row(blockage):
                continue
            key = (
                str(row.get("method", "")),
                str(row.get("pattern", "")),
                str(row.get("blocked_beam_index", "")),
            )
            grouped.setdefault(key, []).append(finite_float(row.get(primary_col)))

        for (method, pattern, blocked_beam_index), values in sorted(grouped.items()):
            vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
            if vals.size == 0:
                continue
            summary.append(
                {
                    "scenario_scope": scope_name,
                    "method": method,
                    "pattern": pattern,
                    "blocked_beam_index": blocked_beam_index,
                    "scenarios": int(vals.size),
                    primary_col: fmt(float(np.mean(vals))),
                    f"{primary_col}_std": fmt(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0),
                }
            )
    return summary


def accuracy_ylim(values: Sequence[float]) -> Tuple[float, float]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return 0.0, 100.0
    low = 5.0 * np.floor((float(vals.min()) - 5.0) / 5.0)
    high = 5.0 * np.ceil((float(vals.max()) + 5.0) / 5.0)
    if high - low < 20.0:
        pad = (20.0 - (high - low)) / 2.0
        low -= pad
        high += pad
    return max(0.0, low), min(100.0, high)


def plot_non_ml_baselines(
    outdir: Path,
    summary: Sequence[Dict[str, object]],
    summary_by_beam: Sequence[Dict[str, object]],
    *,
    primary_col: str,
) -> None:
    cache_dir = outdir / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping non-ML baseline plots because matplotlib is unavailable: {exc}")
        return

    methods = ["MAX-SETB", "NN-ANGLE"]
    colors = {"MAX-SETB": "#64748b", "NN-ANGLE": "#2563eb"}
    legend_labels = {method: f"Non-ML: {method}" for method in methods}

    representative = []
    for pattern, beam in [(1, 6), (2, 2)]:
        label = f"P{pattern} B{beam}"
        for method in methods:
            matches = [
                row
                for row in summary_by_beam
                if row.get("scenario_scope") == "blocked_only"
                and row.get("method") == method
                and int(row.get("pattern", -1)) == int(pattern)
                and int(row.get("blocked_beam_index", -1)) == int(beam)
            ]
            if matches:
                representative.append((label, method, finite_float(matches[0].get(primary_col))))

    aggregate = []
    for pattern in [1, 2]:
        label = f"P{pattern} all beams"
        for method in methods:
            matches = [
                row
                for row in summary
                if row.get("scenario_scope") == "blocked_only"
                and row.get("method") == method
                and int(row.get("pattern", -1)) == int(pattern)
            ]
            if matches:
                aggregate.append((label, method, finite_float(matches[0].get(primary_col))))

    def make_plot(rows: Sequence[Tuple[str, str, float]], filename: str, title: str) -> None:
        if not rows:
            return
        labels = list(dict.fromkeys(label for label, _, _ in rows))
        x = np.arange(len(labels), dtype=float)
        width = 0.34
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        values_for_ylim: List[float] = []
        for offset, method in [(-width / 2.0, "MAX-SETB"), (width / 2.0, "NN-ANGLE")]:
            vals = []
            for label in labels:
                match = [value for row_label, row_method, value in rows if row_label == label and row_method == method]
                vals.append(float(match[0]) if match else float("nan"))
            values_for_ylim.extend(vals)
            bars = ax.bar(x + offset, vals, width=width, label=legend_labels[method], color=colors[method])
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Top-3 within 1 dB accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(*accuracy_ylim(values_for_ylim))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=300)
        plt.close(fig)

    make_plot(
        representative,
        "non_ml_baselines_representative.png",
        "Non-ML baselines on representative blocked beams",
    )
    make_plot(
        aggregate,
        "non_ml_baselines_all_beams.png",
        "Non-ML baselines across all blocked beams",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-ML Blocking V5 baselines.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")), help="Blocking V5 config path")
    parser.add_argument("--outdir", default="", help="Output directory; defaults to results/non_ml_benchmarks")
    parser.add_argument("--random-trials", type=int, default=100, help="Random-B Monte Carlo trials")
    parser.add_argument("--stat-trials", type=int, default=100, help="STAT-NORMAL-24 Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=20260422, help="Analysis RNG seed")
    args = parser.parse_args()

    config = load_config(args.config)
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (config.outdir / "non_ml_benchmarks")
    outdir.mkdir(parents=True, exist_ok=True)

    metric_keys = build_metric_keys(config.kpi)
    primary_key = f"top{int(config.kpi.primary_topk)}_m{int(config.kpi.primary_margin_db)}db_%"
    primary_col = f"test_{primary_key}"

    accuracy_rows: List[Dict[str, object]] = []
    stat_rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(int(args.seed))

    for job in config.batch_jobs:
        run_dir = config.run_dir(job)
        if not (run_dir / "data").is_dir():
            print(f"Skipping missing run data: {run_dir}")
            continue

        clean_features, clean_labels, clean_rsrp, global_min_db, run_seed = load_test_split(run_dir)
        setb_tx = build_setb_tx_indices(int(job.pattern))

        for blockage_pct in config.blockage_levels:
            features_b, _, rsrp_b = build_blocked_dataset_views(
                clean_features=clean_features,
                clean_labels=clean_labels,
                clean_rsrp=clean_rsrp,
                setb_pattern=int(job.pattern),
                blocked_setb_col=int(job.blocked_beam_index),
                global_min_db=float(global_min_db),
                blockage_pct=int(blockage_pct),
            )

            max_metrics = evaluate_ranked_beams(
                max_setb_rankings(features_b, int(job.pattern)),
                rsrp_b,
                kpi=config.kpi,
            )
            accuracy_rows.append(
                metric_row(
                    seed=run_seed,
                    method="MAX-SETB",
                    job=job,
                    blockage_pct=int(blockage_pct),
                    trials=1,
                    metrics=max_metrics,
                )
            )

            nn_metrics = evaluate_ranked_beams(
                nn_angle_space_rankings(features_b, int(job.pattern)),
                rsrp_b,
                kpi=config.kpi,
            )
            accuracy_rows.append(
                metric_row(
                    seed=run_seed,
                    method="NN-ANGLE",
                    job=job,
                    blockage_pct=int(blockage_pct),
                    trials=1,
                    metrics=nn_metrics,
                )
            )

            random_metrics, random_primary_std = random_b_metric_mean(
                features_b,
                rsrp_b,
                setb_tx=setb_tx,
                kpi=config.kpi,
                trials=int(args.random_trials),
                rng=rng,
                primary_key=primary_key,
            )
            accuracy_rows.append(
                metric_row(
                    seed=run_seed,
                    method="RANDOM-SETB",
                    job=job,
                    blockage_pct=int(blockage_pct),
                    trials=int(args.random_trials),
                    metrics=random_metrics,
                    primary_std_pp=random_primary_std,
                )
            )

            stat_summary = statistical_sampled24_summary(features_b, trials=int(args.stat_trials), rng=rng)
            stat_row: Dict[str, object] = {
                "seed": int(run_seed),
                "method": "STAT-NORMAL-24",
                "pattern": int(job.pattern),
                "blocked_beam_index": int(job.blocked_beam_index),
                "blockage_%": int(blockage_pct),
                "trials": int(args.stat_trials),
            }
            stat_row.update({key: fmt(value) for key, value in stat_summary.items()})
            stat_rows.append(stat_row)

    accuracy_fields = [
        "seed",
        "method",
        "pattern",
        "blocked_beam_index",
        "blockage_%",
        "trials",
        "primary_std_pp",
    ]
    accuracy_fields.extend(f"test_{key}" for key in metric_keys)
    write_csv(outdir / "non_ml_accuracy.csv", accuracy_rows, accuracy_fields)

    stat_fields = [
        "seed",
        "method",
        "pattern",
        "blocked_beam_index",
        "blockage_%",
        "trials",
        "mu_db",
        "sigma_db",
        "mean_max_b_db",
        "mean_sampled24_max_db",
        "p_sampled24_exceeds_max_b_%",
        "expected_positive_gain_db",
    ]
    write_csv(outdir / "statistical_sampled24.csv", stat_rows, stat_fields)

    summary = summarize_primary(accuracy_rows, primary_col=primary_col)
    write_csv(
        outdir / "non_ml_accuracy_summary.csv",
        summary,
        ["scenario_scope", "method", "pattern", "scenarios", primary_col, f"{primary_col}_std"],
    )
    summary_by_beam = summarize_primary_by_beam(accuracy_rows, primary_col=primary_col)
    write_csv(
        outdir / "non_ml_accuracy_summary_by_beam.csv",
        summary_by_beam,
        ["scenario_scope", "method", "pattern", "blocked_beam_index", "scenarios", primary_col, f"{primary_col}_std"],
    )
    plot_non_ml_baselines(outdir, summary, summary_by_beam, primary_col=primary_col)

    print(f"Wrote non-ML benchmark outputs to: {outdir}")
    for row in summary:
        print(
            f"  {row['scenario_scope']} {row['method']} P{row['pattern']}: "
            f"{row[primary_col]} +/- {row[f'{primary_col}_std']} ({primary_col})"
        )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
