"""Artifact-aware dashboard rows.

These helpers inspect the same files produced by the release runner.  They do
not mutate results and they avoid importing the runner, which keeps the
dashboard usable from both training and plotting commands.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from experiment.checkpoints import bl_ag_checkpoint_candidates, find_bl_ag_checkpoint
from experiment.names import slugify
from experiment.run_plan import postprocessing_plan

from .formatting import format_list, format_seconds, format_train_fraction_list


PREPARED_DATA_FILES = (
    "X_setb.npy",
    "y_tx.npy",
    "rsrp_tx.npy",
    "train_idx.npy",
    "val_idx.npy",
    "test_idx.npy",
    "meta.json",
)

NON_ML_METHODS = ("MAX-SETB", "NN-ANGLE", "RANDOM-SETB")


def active_run_dir(config: object, active_run: Mapping[str, object]) -> Optional[Path]:
    """Resolve the active run directory from the dashboard payload."""
    run_dir = active_run.get("run_dir")
    if run_dir:
        return Path(str(run_dir))
    run_name = str(active_run.get("run_name", "")).strip()
    if not run_name or run_name == "idle":
        return None
    return Path(str(getattr(config, "outdir"))) / run_name


def _job_specs(config: object) -> List[tuple[int, int, str]]:
    """Return configured pattern/beam jobs without importing runner.pipeline."""
    jobs = []
    for item in getattr(config, "batch_jobs", ()) or ():
        jobs.append((int(getattr(item, "pattern")), int(getattr(item, "blocked_beam_index")), str(getattr(item, "description"))))
    if jobs:
        return jobs
    pattern = int(getattr(config, "setb_pattern", 1))
    return [(pattern, int(beam), f"P{pattern} block B{int(beam)}") for beam in getattr(config, "blocked_setb_indices", ())]


def all_config_run_dirs(config: object) -> List[Path]:
    """Return every run directory expected from the selected seeds and jobs."""
    dirs: List[Path] = []
    seed_values = config.seed_values() if hasattr(config, "seed_values") else (getattr(getattr(config, "training", None), "seed", 0),)
    for seed in seed_values:
        seed_config = config.for_seed(int(seed)) if hasattr(config, "for_seed") else config
        for pattern, blocked_beam_index, _description in _job_specs(seed_config):
            if hasattr(seed_config, "run_name"):
                run_name = seed_config.run_name(pattern=pattern, blocked_beam_index=blocked_beam_index)
            else:
                run_name = f"run_P{pattern}_blockB{blocked_beam_index}"
            dirs.append(Path(str(getattr(seed_config, "outdir"))) / run_name)
    return dirs


def _files_present(directory: Path, names: Sequence[str]) -> bool:
    return all((directory / name).exists() for name in names)


def _state(done: int, total: int, *, selected: bool = True, blocked: bool = False) -> str:
    if not selected:
        return "not selected"
    if blocked:
        return "blocked"
    if total <= 0:
        return "not needed"
    if done == total:
        return "done"
    if done > 0:
        return "partial"
    return "pending"


def _count_existing(paths: Iterable[Path]) -> int:
    return sum(1 for path in paths if path.exists())


def _ft_branch_dirs(config: object, run_dir: Path) -> List[Path]:
    training = getattr(config, "training", None)
    roots: List[Path] = []
    for train_frac in getattr(training, "train_fracs_sweep", ()) or ():
        train_frac_tag = slugify(f"trainfrac_{float(train_frac):.6f}")
        for blockage_pct in getattr(config, "blockage_levels", ()) or ():
            for freeze_mode in getattr(training, "ft_freeze_modes", ()) or ():
                roots.append(
                    run_dir
                    / "jobs"
                    / "finetune"
                    / train_frac_tag
                    / f"blockage_{int(blockage_pct):03d}"
                    / f"freeze_{slugify(str(freeze_mode))}"
                )
    return roots


def _history_paths(config: object, run_dir: Optional[Path]) -> List[Path]:
    if run_dir is None:
        return []
    paths = [
        run_dir / "jobs" / "baseline" / "bl_ag_train" / "history.csv",
        *[
            run_dir / "jobs" / "baseline" / f"bl_aw_blockage_{int(blockage):03d}" / "history.csv"
            for blockage in getattr(config, "blockage_levels", ()) or ()
        ],
    ]
    paths.extend(branch / "history.csv" for branch in _ft_branch_dirs(config, run_dir))
    return paths


def _average_epoch_seconds(paths: Iterable[Path]) -> Optional[float]:
    values: List[float] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    raw = row.get("epoch_time_s", "")
                    if str(raw).strip():
                        values.append(float(raw))
        except Exception:
            continue
    if not values:
        return None
    return sum(values) / float(len(values))


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _short_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def plotting_focus_text(config: object) -> Dict[str, str]:
    """Summarize which plot families and fixed-blockage levels are configured."""
    plotting = dict(getattr(config, "plotting", {}) or {})
    plan = postprocessing_plan(config)
    blockage_levels = getattr(config, "blockage_levels", ())
    return {
        "runner_after_each_run": format_list(plan.after_each_run),
        "runner_after_batch": format_list(plan.after_batch),
        "run_plot_blockages": format_list(plotting.get("run_plot_blockages"), suffix="%"),
        "multi_kpi_blockages": format_list(plotting.get("multi_kpi_blockages"), suffix="%"),
        "comparison_focus_blockages": format_list(plotting.get("comparison_focus_blockages"), suffix="%"),
        "all_blockage_levels": format_list(blockage_levels, suffix="%"),
        "metric_plot_keys": format_list(plotting.get("metric_plot_keys")),
        "run_plot_families": format_list(plotting.get("run_plot_families")),
        "comparison_plot_families": format_list(plotting.get("comparison_plot_families")),
        "multiseed_run_plot_families": format_list(plotting.get("multiseed_run_plot_families")),
    }


def run_selection_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Build the run/protocol rows shown at the top of the dashboard."""
    training = getattr(config, "training", None)
    kpi = getattr(config, "kpi", None)
    stages = list(getattr(config, "stages", ()))
    experiments = list(getattr(config, "experiments", ()))
    run_dir = active_run_dir(config, active_run)
    weights_path = find_bl_ag_checkpoint(run_dir) if run_dir is not None else None
    candidate_text = format_list(bl_ag_checkpoint_candidates(run_dir)) if run_dir is not None else "n/a"

    if "FT" not in experiments or "finetune" not in stages:
        ft_weight_status = "FT not selected"
    elif "baseline" not in stages:
        exists_text = "found" if weights_path is not None else "missing"
        ft_weight_status = f"FT-only; stored BL-AG weights {exists_text}"
    else:
        ft_weight_status = "full run; BL-AG is trained or reused before FT"

    active_beam = "n/a"
    if active_run.get("pattern") is not None and active_run.get("blocked_beam_index") is not None:
        active_beam = f"P{int(active_run.get('pattern', 0))} / B{int(active_run.get('blocked_beam_index', 0))}"

    primary_kpi = kpi.primary_key() if kpi is not None and hasattr(kpi, "primary_key") else "n/a"
    seeds = getattr(training, "seeds", ()) if training is not None else ()
    seed = getattr(training, "seed", "") if training is not None else ""
    seed_values = [seed]
    seed_values.extend(int(value) for value in seeds if str(value) != str(seed))

    return [
        ["Run", str(active_run.get("run_name", "idle") or "idle")],
        ["Active beam", active_beam],
        ["Seeds", format_list(seed_values)],
        ["Stages", format_list(stages)],
        ["Experiments", format_list(experiments)],
        ["Overwrite mode", "enabled" if active_run.get("overwrite", getattr(config, "overwrite", False)) else "off"],
        ["Blockage levels", format_list(getattr(config, "blockage_levels", ()), suffix="%")],
        ["Primary KPI", primary_kpi],
        ["FT train fractions", format_train_fraction_list(getattr(training, "train_fracs_sweep", ()))],
        ["FT freeze modes", format_list(getattr(training, "ft_freeze_modes", ()))],
        ["FT epoch budgets", format_list(getattr(training, "ft_epoch_sweep", ()))],
        ["FT sampling", str(getattr(training, "ft_sampling", "n/a"))],
        ["BL-AG checkpoint path", str(weights_path) if weights_path is not None else candidate_text],
        ["FT weight reuse", ft_weight_status],
    ]


def stage_plan_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Build a simple selected-stage table."""
    stages = [str(value) for value in getattr(config, "stages", ())]
    plan = postprocessing_plan(config)
    active_stage = str(active_run.get("current_stage", "") or "")
    active_status = str(active_run.get("status", "") or "")
    rows: List[List[str]] = []
    for stage in stages:
        if active_status == "completed":
            state = "completed"
        elif active_status in {"failed", "interrupted"}:
            state = "stopped"
        elif stage == active_stage or active_stage.startswith(stage):
            state = "running"
        else:
            state = "selected"
        rows.append([stage, state])
    rows.append(["after each run", format_list(plan.after_each_run) or "none"])
    rows.append(["after batch", format_list(plan.after_batch) or "none"])
    return rows


def artifact_plan_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Inspect expected artifacts for the current active run."""
    run_dir = active_run_dir(config, active_run)
    if run_dir is None:
        return [["Run directory", "idle", "No active run yet.", "Start a run or plotting command."]]

    stages = set(str(value) for value in getattr(config, "stages", ()))
    experiments = set(str(value) for value in getattr(config, "experiments", ()))
    plan = postprocessing_plan(config)
    rows: List[List[str]] = []

    prepared_done = _files_present(run_dir / "data", PREPARED_DATA_FILES)
    rows.append(
        [
            "Prepare data",
            _state(1 if prepared_done else 0, 1, selected="prepare" in stages),
            f"{run_dir / 'data' / 'meta.json'}",
            "Creates clean Set-B arrays, labels, and splits.",
        ]
    )

    bl_ag_train = run_dir / "jobs" / "baseline" / "bl_ag_train"
    bl_ag_eval = run_dir / "jobs" / "baseline" / "bl_ag_eval"
    need_bl_ag = "BL-AG" in experiments or "FT" in experiments
    bl_ag_done = _files_present(bl_ag_train, ("weights_best.pt", "history.csv", "train_result.json"))
    bl_ag_eval_done = _files_present(bl_ag_eval, ("rows.csv",))
    rows.append(
        [
            "BL-AG weights",
            _state(1 if bl_ag_done else 0, 1, selected=need_bl_ag),
            str(bl_ag_train / "weights_best.pt"),
            "Needed by FT-only runs and BL-AG evaluation.",
        ]
    )
    rows.append(
        [
            "BL-AG evaluation",
            _state(1 if bl_ag_eval_done else 0, 1, selected=need_bl_ag),
            str(bl_ag_eval / "rows.csv"),
            "Stores BL-AG metrics for every blockage level.",
        ]
    )

    stored_manifest = run_dir / "stored_weights" / "baseline_pretrained" / "manifest.csv"
    rows.append(
        [
            "Stored baseline weights",
            "done" if stored_manifest.exists() else "pending",
            str(stored_manifest),
            "Researcher-facing BL-AG/BL-AW checkpoint index.",
        ]
    )

    bl_aw_paths = [
        run_dir / "jobs" / "baseline" / f"bl_aw_blockage_{int(blockage):03d}" / "rows.csv"
        for blockage in getattr(config, "blockage_levels", ()) or ()
    ]
    bl_aw_done = _count_existing(bl_aw_paths)
    rows.append(
        [
            "BL-AW blockage sweep",
            _state(bl_aw_done, len(bl_aw_paths), selected="BL-AW" in experiments),
            f"{bl_aw_done}/{len(bl_aw_paths)} rows.csv files",
            "One trained aware baseline per blockage level.",
        ]
    )

    non_ml_done = (run_dir / "jobs" / "baseline" / "non_ml_baselines" / "rows.csv").exists()
    non_ml_selected = any(method in experiments for method in NON_ML_METHODS)
    rows.append(
        [
            "Non-ML baselines",
            _state(1 if non_ml_done else 0, 1, selected=non_ml_selected),
            str(run_dir / "jobs" / "baseline" / "non_ml_baselines" / "rows.csv"),
            "MAX-SETB, NN-ANGLE, or RANDOM-SETB rows when selected.",
        ]
    )

    ft_branches = _ft_branch_dirs(config, run_dir)
    ft_done = _count_existing(branch / "rows.csv" for branch in ft_branches)
    ft_blocked = "FT" in experiments and "finetune" in stages and find_bl_ag_checkpoint(run_dir) is None
    rows.append(
        [
            "FT branches",
            _state(ft_done, len(ft_branches), selected="FT" in experiments and "finetune" in stages, blocked=ft_blocked),
            f"{ft_done}/{len(ft_branches)} branch rows.csv files",
            "Requires BL-AG weights; reuses shortcuts for 0% data, 0% blockage, or freeze_all.",
        ]
    )

    aggregate_done = (run_dir / "results.csv").exists() and (run_dir / "aggregate_done.json").exists()
    rows.append(
        [
            "Aggregate results",
            _state(1 if aggregate_done else 0, 1, selected="aggregate" in stages),
            str(run_dir / "results.csv"),
            "Merges all stage rows into the run-level result table.",
        ]
    )

    package_root = Path(str(getattr(config, "config_path"))).parent
    run_plot_root = package_root / "plots" / "run_plots" / run_dir.name
    run_plot_count = len(list(run_plot_root.rglob("*.png"))) if run_plot_root.exists() else 0
    run_plots_selected = "run_plots" in plan.after_each_run
    rows.append(
        [
            "Run plots",
            _state(1 if run_plot_count else 0, 1, selected=run_plots_selected),
            f"{run_plot_count} PNG files",
            "Written under plots/run_plots after one run completes when selected.",
        ]
    )
    return rows


def ft_only_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Show fine-tuning dependency and branch status."""
    run_dir = active_run_dir(config, active_run)
    training = getattr(config, "training", None)
    stages = set(str(value) for value in getattr(config, "stages", ()))
    experiments = set(str(value) for value in getattr(config, "experiments", ()))
    if "FT" not in experiments or run_dir is None:
        return [["FT mode", "not selected", "", ""]]

    weights = find_bl_ag_checkpoint(run_dir)
    candidates = bl_ag_checkpoint_candidates(run_dir)
    branches = _ft_branch_dirs(config, run_dir)
    done = _count_existing(branch / "rows.csv" for branch in branches)
    mode = "FT-only" if "finetune" in stages and "baseline" not in stages else "full run"
    missing = max(0, len(branches) - done)
    return [
        ["Mode", mode, "Uses stored BL-AG weights" if mode == "FT-only" else "Runs baseline before FT", ""],
        ["BL-AG checkpoint", "found" if weights is not None else "missing", str(weights or candidates[0]), "FT can use the original stage checkpoint or the exported stored checkpoint."],
        ["Train fractions", format_train_fraction_list(getattr(training, "train_fracs_sweep", ())), "", ""],
        ["Freeze modes", format_list(getattr(training, "ft_freeze_modes", ())), "", ""],
        ["Blockage levels", format_list(getattr(config, "blockage_levels", ()), suffix="%"), "", ""],
        ["Branches", f"{done} done / {missing} missing / {len(branches)} expected", str(run_dir / "jobs" / "finetune"), ""],
    ]


def _kpi_keys(config: object) -> List[str]:
    kpi = getattr(config, "kpi", None)
    if kpi is None:
        return []
    keys = ["top1_%"]
    for topk in getattr(kpi, "topks", ()) or ():
        if int(topk) != 1:
            keys.append(f"top{int(topk)}_incl_%")
        for margin in getattr(kpi, "margins_db", ()) or ():
            keys.append(f"top{int(topk)}_m{int(margin)}db_%")
    keys.extend(["avg_top1_rsrp_db", "p95_gap_db"])
    return list(dict.fromkeys(keys))


def kpi_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Show KPI selection and current training values."""
    kpi = getattr(config, "kpi", None)
    primary = kpi.primary_key() if kpi is not None and hasattr(kpi, "primary_key") else "n/a"
    plotting = dict(getattr(config, "plotting", {}) or {})
    epoch_progress = active_run.get("epoch_progress")
    current = epoch_progress if isinstance(epoch_progress, Mapping) else {}
    rows = [
        ["Primary KPI", primary, "Optimization and budget selection target"],
        ["All KPI fields", format_list(_kpi_keys(config)), "Columns expected in result CSVs"],
        ["Plot KPI keys", format_list(plotting.get("metric_plot_keys")), "Default KPI lines for plots"],
        ["Multi-KPI keys", format_list(plotting.get("multi_kpi_metric_keys")), "KPI panel or multi-metric plots"],
        ["Current validation primary", str(current.get("val_primary_pct", "n/a")), "From the live epoch callback"],
        ["Current test primary", str(current.get("test_primary_pct", "n/a")), "From the live epoch callback"],
        ["Current epoch", f"{current.get('epoch', 'n/a')} / {current.get('epochs', 'n/a')}", str(current.get("job_name", ""))],
    ]
    run_dir = active_run_dir(config, active_run)
    if run_dir is not None:
        results_rows = _read_csv_rows(run_dir / "results.csv")
        rows.append(["Run result rows", str(len(results_rows)) if results_rows else "not aggregated", str(run_dir / "results.csv")])
    return rows


def plotting_status_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Show plotting command, generated plot counts, and manifest status."""
    outdir = Path(str(getattr(config, "outdir")))
    package_root = Path(str(getattr(config, "config_path"))).parent
    plots_root = package_root / "plots"
    selected_root = plots_root / "selected_plots"
    progress = read_plotting_progress(plots_root)
    selected_pngs = len(list(selected_root.rglob("*.png"))) if selected_root.exists() else 0
    selected_manifest = selected_root / "selected_plots_manifest.csv"
    status_rows = _read_csv_rows(selected_root / "selected_plots_generation_status.csv")
    skipped = [row for row in status_rows if str(row.get("status", "")).lower() not in {"generated", "copied", "ok"}]
    organized_root = plots_root / "organized_plots"
    organized_pngs = len(list(organized_root.rglob("*.png"))) if organized_root.exists() else 0
    run_pngs = 0
    for run_dir in all_config_run_dirs(config):
        run_plot_root = plots_root / "run_plots" / run_dir.name
        if run_plot_root.exists():
            run_pngs += len(list(run_plot_root.rglob("*.png")))

    rows = [
        ["Plot command", str(progress.get("command", "none")), str(progress.get("status", "not running"))],
        ["Plot selection", str(progress.get("selection", "default")), str(progress.get("message", ""))],
        ["Selected bundle", f"{selected_pngs} PNG files", "plots/selected_plots"],
        ["Selected manifest", "present" if selected_manifest.exists() else "missing", str(selected_manifest)],
        ["Selected status rows", f"{len(status_rows)} rows", f"{len(skipped)} skipped or missing"],
        ["Run plots", f"{run_pngs} PNG files", "plots/run_plots/<run>"],
        ["Organized navigation", f"{organized_pngs} PNG files", str(organized_root)],
    ]
    if active_run.get("current_stage") == "plotting":
        rows.insert(0, ["Runner plot step", "active", str(active_run.get("run_dir", ""))])
    return rows


def read_plotting_progress(plots_root: Path) -> Dict[str, object]:
    """Load the optional plotting progress JSON written by plots.py."""
    payload = _read_json(Path(plots_root) / "plotting_live_progress.json")
    return payload or {}


def eta_rows(config: object, snapshot: Mapping[str, object], active_run: Mapping[str, object]) -> List[List[str]]:
    """Show runtime and ETA at batch, job, branch, and plotting granularity."""
    run_dir = active_run_dir(config, active_run)
    avg_epoch_s = _average_epoch_seconds(_history_paths(config, run_dir))
    ft_branches = _ft_branch_dirs(config, run_dir) if run_dir is not None else []
    ft_done = _count_existing(branch / "rows.csv" for branch in ft_branches)
    remaining_ft = max(0, len(ft_branches) - ft_done)
    rows = [
        ["Current job ETA", str(snapshot.get("current_job_eta_hms") or "n/a"), str(snapshot.get("current_job_eta_finish_at") or "")],
        ["Batch ETA", str(snapshot.get("batch_eta_hms") or "n/a"), str(snapshot.get("batch_eta_finish_at") or "")],
        [
            "Representative job average",
            str(snapshot.get("average_completed_job_hms") or "n/a"),
            f"{snapshot.get('job_duration_basis', 'Estimated from completed rows.')} Sample count: {snapshot.get('job_duration_sample_count', 'n/a')}.",
        ],
        ["Historical epoch average", format_seconds(avg_epoch_s) if avg_epoch_s is not None else "n/a", "Read from existing history.csv files."],
        ["FT branch status", f"{ft_done} done / {remaining_ft} remaining", f"{len(ft_branches)} expected branches."],
    ]
    package_root = Path(str(getattr(config, "config_path"))).parent
    plotting = read_plotting_progress(package_root / "plots")
    if plotting:
        rows.append(["Plotting status", str(plotting.get("status", "n/a")), str(plotting.get("updated_at", ""))])
    return rows


def resume_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Explain what will be reused, overwritten, or blocked on resume."""
    run_dir = active_run_dir(config, active_run)
    if run_dir is None:
        return [["Resume state", "idle", "No run directory is active."]]
    overwrite = bool(active_run.get("overwrite", getattr(config, "overwrite", False)))
    weights = find_bl_ag_checkpoint(run_dir)
    stages = set(str(value) for value in getattr(config, "stages", ()))
    experiments = set(str(value) for value in getattr(config, "experiments", ()))
    rows = [
        ["Overwrite", "enabled" if overwrite else "off", "Existing selected-stage outputs are rebuilt." if overwrite else "Existing complete artifacts are reused."],
        ["Run directory", "present" if run_dir.exists() else "missing", str(run_dir)],
    ]
    if "FT" in experiments and "finetune" in stages:
        if weights is not None:
            rows.append(["FT dependency", "ready", "Stored BL-AG weights are present."])
        else:
            rows.append(["FT dependency", "blocked", "Run baseline first or copy an accepted BL-AG checkpoint into this run directory."])
    rows.append(["Safe next command", "resume", "Use the same run.py command without --overwrite to reuse complete artifacts."])
    return rows


def error_rows(active_run: Mapping[str, object], artifact_rows: Sequence[Sequence[str]]) -> List[List[str]]:
    """Show active failures and likely causes."""
    rows: List[List[str]] = []
    status = str(active_run.get("status", ""))
    if status in {"failed", "interrupted"} or active_run.get("error"):
        error = str(active_run.get("error", ""))
        trace = str(active_run.get("traceback", ""))
        likely_fix = "Inspect the traceback and rerun after fixing the missing dependency or bad config."
        if "BL-AG weights" in error or "weights_best.pt" in error:
            likely_fix = "Run baseline first or place the stored BL-AG checkpoint in stored_weights/baseline_pretrained/."
        rows.append([status or "error", error[:260], likely_fix])
        if trace:
            rows.append(["traceback", " | ".join(trace.strip().splitlines()[-3:])[:260], "Last traceback lines"])
    for row in artifact_rows:
        if len(row) >= 2 and str(row[1]) == "blocked":
            rows.append([str(row[0]), str(row[2])[:260], str(row[3])[:260] if len(row) > 3 else ""])
    return rows or [["No active errors", "ok", "No failed state is recorded in active_run.json."]]


def config_identity_rows(config: object, active_run: Mapping[str, object]) -> List[List[str]]:
    """Compare current config settings with the active run snapshot."""
    config_path = Path(str(getattr(config, "config_path")))
    run_dir = active_run_dir(config, active_run)
    run_config_path = run_dir / "config.json" if run_dir is not None else None
    run_config = _read_json(run_config_path) if run_config_path is not None else None
    rows = [["Current config hash", _short_hash(config_path), str(config_path)]]
    if run_config_path is None:
        rows.append(["Run snapshot hash", "n/a", "No active run."])
        return rows
    rows.append(["Run snapshot hash", _short_hash(run_config_path), str(run_config_path)])
    if not run_config:
        rows.append(["Config diff", "pending", "Run config.json has not been written yet."])
        return rows

    current_training = getattr(config, "training", None)
    current_seed = int(active_run.get("seed", getattr(current_training, "seed", -1)) or -1)
    comparisons = [
        ("Stages changed", list(getattr(config, "stages", ())), run_config.get("stages")),
        ("Experiments changed", list(getattr(config, "experiments", ())), run_config.get("experiments")),
        ("Train fractions changed", list(getattr(current_training, "train_fracs_sweep", ())), _nested(run_config, "training", "train_fracs_sweep")),
        ("Freeze modes changed", list(getattr(current_training, "ft_freeze_modes", ())), _nested(run_config, "training", "ft_freeze_modes")),
        ("FT epochs changed", list(getattr(current_training, "ft_epoch_sweep", ())), _nested(run_config, "training", "ft_epoch_sweep")),
        ("Seed changed", current_seed, _nested(run_config, "training", "seed")),
        ("KPI changed", config.kpi.primary_key() if hasattr(config, "kpi") else "n/a", _primary_from_snapshot(run_config)),
    ]
    for label, current, previous in comparisons:
        changed = _normalized(current) != _normalized(previous)
        rows.append([label, "yes" if changed else "no", f"current={current}; run={previous}"])
    return rows


def _nested(payload: Mapping[str, object], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _primary_from_snapshot(payload: Mapping[str, object]) -> str:
    kpi = payload.get("kpi")
    if not isinstance(kpi, Mapping):
        return "n/a"
    return f"top{int(kpi.get('primary_topk', 0))}_m{int(kpi.get('primary_margin_db', 0))}db_%"


def _normalized(value: object) -> object:
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, float):
        return round(value, 10)
    return value
