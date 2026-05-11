"""Command-line runner for Blocking V5.

This file runs one job or the full batch, keeps the live progress files up to
date, and triggers the configured postprocessing steps around the core
experiment code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

from experiment.config import BatchJob, ExperimentConfig, load_config
from experiment.io import atomic_write_json, atomic_write_text, format_seconds, write_csv
from experiment.preflight import ConfigSelectionError, MissingDependencyError
from experiment.run_plan import postprocessing_plan
from experiment.stages import run_single_experiment

from .dashboard import (
    dashboard_enabled,
    dashboard_path,
    hardware_snapshot_payload,
    runtime_metrics_payload,
    write_dashboard_html,
)

PROGRESS_FIELDS = [
    "run_name",
    "seed",
    "pattern",
    "blocked_beam_index",
    "description",
    "status",
    "current_stage",
    "started_at",
    "completed_at",
    "elapsed_s",
    "elapsed_hms",
    "eta_s",
    "eta_hms",
    "run_dir",
    "error",
]

EPOCH_TRAIL_SIZE = 5
ETA_RECENT_JOB_COUNT = 5
ETA_LONG_JOB_SECONDS = 600.0


def build_jobs(config: ExperimentConfig) -> List[BatchJob]:
    """Batch job list, falling back to one job per blocked beam."""
    if config.batch_jobs:
        return list(config.batch_jobs)
    if config.blocked_setb_indices:
        return [
            make_batch_job(int(config.setb_pattern), int(blocked_beam_index))
            for blocked_beam_index in config.blocked_setb_indices
        ]
    raise ValueError("Config does not define batch_jobs or blocked_setb_indices.")


# ---------------------------------------------------------------------------
# Progress files and ETA helpers
# ---------------------------------------------------------------------------
def progress_root(config: ExperimentConfig) -> Path:
    """Ensure the results root exists before writing progress files into it."""
    config.outdir.mkdir(parents=True, exist_ok=True)
    return config.outdir


def progress_paths(config: ExperimentConfig) -> Dict[str, Path]:
    """Keep every progress artifact in one predictable place under the results root."""
    root = progress_root(config)
    return {
        "active_run": root / "active_run.json",
        "progress_json": root / "batch_live_progress.json",
        "progress_csv": root / "batch_live_progress.csv",
        "dashboard": dashboard_path(config),
        "dashboard_error": root / "dashboard_error.json",
        "hardware_snapshot": root / "hardware_snapshot.json",
    }


def write_hardware_snapshot(config: ExperimentConfig) -> None:
    atomic_write_json(progress_paths(config)["hardware_snapshot"], hardware_snapshot_payload(config.config_path))


def now_text() -> str:
    """Format timestamps once in a consistent local time format."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_batch_job(pattern: int, blocked_beam_index: int) -> BatchJob:
    """Standard job label used across config, progress, and plots."""
    return BatchJob(
        pattern=int(pattern),
        blocked_beam_index=int(blocked_beam_index),
        description=f"P{int(pattern)} block B{int(blocked_beam_index)}",
    )


def job_identity(config: ExperimentConfig, job: BatchJob) -> Dict[str, object]:
    """Collect the fields that identify one job everywhere in the runner."""
    return {
        "run_name": config.run_name(pattern=job.pattern, blocked_beam_index=job.blocked_beam_index),
        "seed": int(config.training.seed),
        "pattern": int(job.pattern),
        "blocked_beam_index": int(job.blocked_beam_index),
        "description": str(job.description),
    }


def job_row(config: ExperimentConfig, job: BatchJob) -> Dict[str, object]:
    """Default batch-progress row before the job starts running."""
    return {
        **job_identity(config, job),
        "status": "pending",
        "current_stage": "",
        "started_at": "",
        "completed_at": "",
        "elapsed_s": "",
        "elapsed_hms": "",
        "eta_s": "",
        "eta_hms": "",
        "run_dir": str(config.run_dir(job)),
        "error": "",
    }


def progress_float(row: Dict[str, object], key: str) -> Optional[float]:
    """Read numeric values out of mutable progress rows without raising on blanks."""
    value = row.get(key)
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def row_seed(row: Dict[str, object]) -> str:
    """Stable string form for comparing seed values in progress rows."""
    return str(row.get("seed", "")).strip()


def active_seed_value(rows: List[Dict[str, object]], active_payload: Optional[Dict[str, object]]) -> str:
    """Seed currently driving the live ETA, if one can be inferred."""
    running_row = running_progress_row(rows)
    if running_row is not None:
        return row_seed(running_row)
    if active_payload:
        return str(active_payload.get("seed", "")).strip()
    return ""


def completed_job_durations(rows: List[Dict[str, object]], *, seed: Optional[str] = None) -> List[float]:
    """Completed-job durations in row order, optionally narrowed to one seed."""
    durations: List[float] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        if seed is not None and row_seed(row) != seed:
            continue
        value = progress_float(row, "elapsed_s")
        if value is not None and value >= 0.0:
            durations.append(float(value))
    return durations


def representative_duration_values(values: List[float], *, running_elapsed_s: Optional[float]) -> List[float]:
    """Filter resume-skip durations out of the job-duration sample.

    A safe resume can mark already-complete jobs as completed in seconds. Those
    rows are useful for progress counts, but they are not useful for predicting
    how long an unfinished training job will take.
    """
    finite = [float(value) for value in values if value >= 0.0]
    if not finite:
        if running_elapsed_s is not None and running_elapsed_s >= ETA_LONG_JOB_SECONDS:
            return [float(running_elapsed_s)]
        return []

    long_values = [value for value in finite if value >= ETA_LONG_JOB_SECONDS]
    if long_values:
        return long_values[-ETA_RECENT_JOB_COUNT:]
    if running_elapsed_s is not None and running_elapsed_s >= ETA_LONG_JOB_SECONDS:
        return [float(running_elapsed_s)]
    return finite[-ETA_RECENT_JOB_COUNT:]


def job_duration_samples(
    rows: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]],
    seed: Optional[str] = None,
) -> List[float]:
    """Representative completed-job durations for ETA estimates."""
    running_row = running_progress_row(rows)
    running_elapsed_s = progress_float(running_row or {}, "elapsed_s")
    running_seed = row_seed(running_row or {})
    requested_seed = str(seed).strip() if seed is not None else active_seed_value(rows, active_payload)

    if requested_seed:
        seed_running_elapsed = running_elapsed_s if requested_seed == running_seed else None
        seed_values = representative_duration_values(
            completed_job_durations(rows, seed=requested_seed),
            running_elapsed_s=seed_running_elapsed,
        )
        if seed_values:
            return seed_values

    return representative_duration_values(
        completed_job_durations(rows),
        running_elapsed_s=running_elapsed_s,
    )


def average_completed_job_seconds(
    rows: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]] = None,
    seed: Optional[str] = None,
) -> Optional[float]:
    """Estimate one real job duration while ignoring resume-skip outliers."""
    values = job_duration_samples(rows, active_payload=active_payload, seed=seed)
    if not values:
        return None
    return float(sum(values) / len(values))


def job_duration_basis_text(rows: List[Dict[str, object]], *, active_payload: Optional[Dict[str, object]]) -> str:
    """Human-readable explanation for the ETA average shown in the dashboard."""
    active_seed = active_seed_value(rows, active_payload)
    samples = job_duration_samples(rows, active_payload=active_payload)
    if not samples:
        return "No representative completed job is available yet."
    if active_seed and representative_duration_values(
        completed_job_durations(rows, seed=active_seed),
        running_elapsed_s=progress_float(running_progress_row(rows) or {}, "elapsed_s"),
    ):
        return f"Recent real jobs from active seed {active_seed}; resume-skip rows are ignored."
    return "Recent real completed jobs; resume-skip rows are ignored when longer jobs are available."


def running_progress_row(rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Currently running row when the batch has an active job."""
    for row in rows:
        if row.get("status") == "running":
            return row
    return None


def epoch_eta_seconds(active_payload: Optional[Dict[str, object]]) -> Optional[float]:
    """Estimate remaining training time from the latest epoch callback when available."""
    if not active_payload:
        return None
    epoch_progress = active_payload.get("epoch_progress")
    if not isinstance(epoch_progress, dict):
        return None
    try:
        current_epoch = int(epoch_progress.get("epoch", 0))
        total_epochs = int(epoch_progress.get("epochs", 0))
        epoch_time_s = float(epoch_progress.get("epoch_time_s", 0.0))
    except Exception:
        return None
    if current_epoch < 1 or total_epochs < 1 or epoch_time_s <= 0.0:
        return None
    remaining_epochs = max(0, total_epochs - current_epoch)
    return float(remaining_epochs * epoch_time_s)


def running_job_eta_seconds(
    rows: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]],
) -> Optional[float]:
    """Estimate the remaining time for the currently running job."""
    running_row = running_progress_row(rows)
    average_job_s = average_completed_job_seconds(rows, active_payload=active_payload)
    elapsed_s = progress_float(running_row or {}, "elapsed_s")
    job_remaining_s: Optional[float] = None
    if running_row is not None and average_job_s is not None and elapsed_s is not None:
        job_remaining_s = float(max(0.0, average_job_s - elapsed_s))

    epoch_eta = epoch_eta_seconds(active_payload)
    if epoch_eta is not None and job_remaining_s is not None:
        return float(max(epoch_eta, job_remaining_s))
    if epoch_eta is not None:
        return float(epoch_eta)

    if job_remaining_s is None:
        return None
    return job_remaining_s


def batch_eta_seconds(
    rows: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]],
) -> Optional[float]:
    """Estimate the remaining time for the whole batch using completed jobs and the active epoch timing."""
    pending_count = sum(1 for row in rows if row.get("status") == "pending")
    average_job_s = average_completed_job_seconds(rows, active_payload=active_payload)
    current_job_eta_s = running_job_eta_seconds(rows, active_payload=active_payload)

    if current_job_eta_s is not None and average_job_s is not None:
        return float(current_job_eta_s + pending_count * average_job_s)
    if current_job_eta_s is not None:
        return float(current_job_eta_s)
    if average_job_s is not None and pending_count > 0:
        return float(pending_count * average_job_s)
    return None


def finish_text_from_seconds(seconds: Optional[float]) -> str:
    """Convert a remaining-seconds estimate into a local completion timestamp."""
    if seconds is None:
        return ""
    finish_time = time.time() + max(0.0, float(seconds))
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finish_time))


def remaining_jobs(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """List the jobs that have not completed yet in display order."""
    pending: List[Dict[str, object]] = []
    for row in rows:
        if row.get("status") in {"pending", "running"}:
            pending.append(
                {
                    "run_name": str(row.get("run_name", "")),
                    "pattern": int(row.get("pattern", 0) or 0),
                    "blocked_beam_index": int(row.get("blocked_beam_index", 0) or 0),
                    "description": str(row.get("description", "")),
                    "status": str(row.get("status", "")),
                    "current_stage": str(row.get("current_stage", "")),
                }
            )
    return pending


def trim_epoch_trail(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Keep the latest five epoch callbacks for dashboard inspection."""
    return entries[-EPOCH_TRAIL_SIZE:]


def format_exception_text(exc: BaseException) -> str:
    """Convert interruptions and ordinary exceptions into a dashboard-friendly message."""
    if isinstance(exc, KeyboardInterrupt):
        return "Interrupted"
    text = str(exc).strip()
    return text or exc.__class__.__name__


def seed_progress_summaries(
    rows: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Summarize batch progress per seed for multi-seed runs."""
    ordered_seeds: List[str] = []
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        seed = str(row.get("seed", "")).strip() or "n/a"
        if seed not in grouped:
            ordered_seeds.append(seed)
            grouped[seed] = []
        grouped[seed].append(row)

    average_job_s = average_completed_job_seconds(rows, active_payload=active_payload)
    running_row = running_progress_row(rows)
    running_seed = str(running_row.get("seed", "")).strip() if running_row is not None else ""
    current_eta_s = running_job_eta_seconds(rows, active_payload=active_payload)

    summaries: List[Dict[str, object]] = []
    for seed in ordered_seeds:
        seed_rows = grouped[seed]
        total = len(seed_rows)
        completed = sum(1 for row in seed_rows if row.get("status") == "completed")
        failed = sum(1 for row in seed_rows if row.get("status") == "failed")
        running = sum(1 for row in seed_rows if row.get("status") == "running")
        pending = sum(1 for row in seed_rows if row.get("status") == "pending")
        jobs_left = int(pending + running)
        state = "running" if running else "failed" if failed else "pending" if pending else "completed"
        active_index = next((index for index, row in enumerate(seed_rows, start=1) if row.get("status") == "running"), 0)
        seed_average_s = average_completed_job_seconds(rows, active_payload=active_payload, seed=seed) or average_job_s
        eta_s: Optional[float] = None
        if seed_average_s is not None:
            eta_s = float(pending * seed_average_s)
            if running and seed == running_seed:
                eta_s += float(current_eta_s if current_eta_s is not None else seed_average_s)
            elif running:
                eta_s += float(seed_average_s)
        elif running and seed == running_seed and current_eta_s is not None:
            eta_s = float(current_eta_s)

        summaries.append(
            {
                "seed": seed,
                "total_jobs": total,
                "completed_jobs": completed,
                "running_jobs": running,
                "failed_jobs": failed,
                "pending_jobs": pending,
                "jobs_left": jobs_left,
                "state": state,
                "active_job_ordinal": active_index,
                "active_job_label": f"{active_index} / {total}" if active_index else "",
                "progress_pct": (100.0 * float(completed) / float(total)) if total else 0.0,
                "eta_s": eta_s,
                "eta_hms": format_seconds(eta_s) if eta_s is not None else "",
                "eta_finish_at": finish_text_from_seconds(eta_s),
            }
        )
    return summaries


def seed_progress_overview(
    rows: List[Dict[str, object]],
    seed_summaries: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Return the dashboard headline state for the active seed."""
    running_row = running_progress_row(rows)
    active_seed = ""
    if running_row is not None:
        active_seed = str(running_row.get("seed", "")).strip()
    elif active_payload:
        active_seed = str(active_payload.get("seed", "")).strip()

    active_summary = next((item for item in seed_summaries if str(item.get("seed", "")) == active_seed), None)
    remaining_summaries = [item for item in seed_summaries if int(item.get("jobs_left", 0) or 0) > 0]
    completed_seeds = sum(1 for item in seed_summaries if str(item.get("state", "")) == "completed")
    failed_seeds = sum(1 for item in seed_summaries if int(item.get("failed_jobs", 0) or 0) > 0)
    seed_eta_values = [
        float(item["eta_s"])
        for item in remaining_summaries
        if item.get("eta_s") not in ("", None)
    ]
    seed_eta_total_s = sum(seed_eta_values) if seed_eta_values and len(seed_eta_values) == len(remaining_summaries) else None
    next_seed = next(
        (
            str(item.get("seed", ""))
            for item in remaining_summaries
            if str(item.get("seed", "")) != active_seed
        ),
        "",
    )

    return {
        "active_seed": active_seed or "n/a",
        "active_seed_state": str(active_summary.get("state", "n/a")) if active_summary else "n/a",
        "active_seed_job": str(active_summary.get("active_job_label", "")) if active_summary else "",
        "active_seed_jobs_left": int(active_summary.get("jobs_left", 0) or 0) if active_summary else 0,
        "active_seed_eta_s": active_summary.get("eta_s") if active_summary else None,
        "active_seed_eta_hms": str(active_summary.get("eta_hms", "")) if active_summary else "",
        "active_seed_eta_finish_at": str(active_summary.get("eta_finish_at", "")) if active_summary else "",
        "remaining_seed_count": len(remaining_summaries),
        "completed_seed_count": completed_seeds,
        "failed_seed_count": failed_seeds,
        "next_seed": next_seed,
        "seed_eta_total_s": seed_eta_total_s,
        "seed_eta_total_hms": format_seconds(seed_eta_total_s) if seed_eta_total_s is not None else "",
        "seed_eta_total_finish_at": finish_text_from_seconds(seed_eta_total_s),
    }


def active_job_ordinal(rows: List[Dict[str, object]]) -> int:
    """Return the 1-based active job number in batch order."""
    for index, row in enumerate(rows, start=1):
        if row.get("status") == "running":
            return int(index)
    completed = sum(1 for row in rows if row.get("status") == "completed")
    if completed:
        return min(int(completed), len(rows))
    return 0


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------


def dashboard_snapshot(rows: List[Dict[str, object]], *, active_payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Assemble the dashboard state shared by JSON snapshots and HTML."""
    total_jobs = len(rows)
    completed = sum(1 for row in rows if row.get("status") == "completed")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    running = sum(1 for row in rows if row.get("status") == "running")
    pending = sum(1 for row in rows if row.get("status") == "pending")

    # JSON snapshots and the HTML dashboard read from the same assembled state so
    # the different progress views cannot drift out of sync.
    current_job_eta_s = running_job_eta_seconds(rows, active_payload=active_payload)
    batch_eta_s = batch_eta_seconds(rows, active_payload=active_payload)
    average_job_s = average_completed_job_seconds(rows, active_payload=active_payload)
    seed_summaries = seed_progress_summaries(rows, active_payload=active_payload)
    seed_overview = seed_progress_overview(rows, seed_summaries, active_payload=active_payload)
    active_ordinal = active_job_ordinal(rows)
    duration_samples = job_duration_samples(rows, active_payload=active_payload)

    return {
        "updated_at": now_text(),
        "total_jobs": total_jobs,
        "completed_jobs": completed,
        "failed_jobs": failed,
        "running_jobs": running,
        "pending_jobs": pending,
        "seed_count": len(seed_summaries),
        "active_job_ordinal": active_ordinal,
        "batch_progress_pct": (100.0 * float(completed) / float(total_jobs)) if total_jobs else 0.0,
        "average_completed_job_s": average_job_s,
        "average_completed_job_hms": format_seconds(average_job_s) if average_job_s is not None else "",
        "job_duration_sample_count": len(duration_samples),
        "job_duration_basis": job_duration_basis_text(rows, active_payload=active_payload),
        "current_job_eta_s": current_job_eta_s,
        "current_job_eta_hms": format_seconds(current_job_eta_s) if current_job_eta_s is not None else "",
        "current_job_eta_finish_at": finish_text_from_seconds(current_job_eta_s),
        "batch_eta_s": batch_eta_s,
        "batch_eta_hms": format_seconds(batch_eta_s) if batch_eta_s is not None else "",
        "batch_eta_finish_at": finish_text_from_seconds(batch_eta_s),
        "seed_summaries": seed_summaries,
        "seed_overview": seed_overview,
        "runtime_metrics": runtime_metrics_payload(),
        "remaining_jobs": remaining_jobs(rows),
        "active_run": active_payload or {"status": "idle"},
        "rows": rows,
    }


def write_dashboard(
    config: ExperimentConfig,
    rows: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]],
    dashboard: Optional[bool] = None,
    snapshot: Optional[Dict[str, object]] = None,
) -> None:
    """Write the optional HTML dashboard snapshot when it is enabled.

    Dashboard HTML is a monitor, not part of the scientific outputs. A browser
    or filesystem issue should not stop training after the JSON/CSV progress
    files have already been written.
    """
    if not dashboard_enabled(config, dashboard):
        return
    html_snapshot = snapshot or dashboard_snapshot(rows, active_payload=active_payload)
    paths = progress_paths(config)
    try:
        write_dashboard_html(paths["dashboard"], config=config, snapshot=html_snapshot)
        try:
            paths["dashboard_error"].unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as exc:
        error_payload = {
            "updated_at": now_text(),
            "error": format_exception_text(exc),
            "dashboard": str(paths["dashboard"]),
            "traceback": traceback.format_exc(),
        }
        try:
            atomic_write_json(paths["dashboard_error"], error_payload)
        except Exception:
            pass


def write_progress_snapshot(
    config: ExperimentConfig,
    rows: List[Dict[str, object]],
    *,
    active_payload: Optional[Dict[str, object]] = None,
    dashboard: Optional[bool] = None,
) -> None:
    """Refresh the JSON, CSV, and optional HTML progress views together."""
    paths = progress_paths(config)
    snapshot = dashboard_snapshot(rows, active_payload=active_payload)
    snapshot["runtime_metrics"] = runtime_metrics_payload(config.outdir)
    atomic_write_json(paths["progress_json"], snapshot)
    write_csv(paths["progress_csv"], rows, PROGRESS_FIELDS)
    write_dashboard(config, rows, active_payload=active_payload, dashboard=dashboard, snapshot=snapshot)


# ---------------------------------------------------------------------------
# Active run state
# ---------------------------------------------------------------------------


def write_active_run(config: ExperimentConfig, payload: Dict[str, object]) -> None:
    """Persist the current run state so a live process can be inspected externally."""
    atomic_write_json(progress_paths(config)["active_run"], payload)


def set_active_run(
    config: ExperimentConfig,
    *,
    status: str,
    job: Optional[BatchJob] = None,
    elapsed_s: Optional[float] = None,
    **extra: object,
) -> Dict[str, object]:
    """Update the active-run file with shared job identity and timing fields."""
    overwrite_value = extra.pop("overwrite", getattr(config, "overwrite", False))
    payload: Dict[str, object] = {
        "status": str(status),
        "updated_at": now_text(),
        "overwrite": bool(overwrite_value),
    }
    if job is not None:
        payload.update(job_identity(config, job))
    if elapsed_s is not None:
        payload["elapsed_s"] = float(elapsed_s)
        payload["elapsed_hms"] = format_seconds(float(elapsed_s))
    payload.update(extra)
    write_active_run(config, payload)
    return payload


def clear_active_run(config: ExperimentConfig) -> None:
    """Mark the runner idle after the current job or batch finishes."""
    set_active_run(config, status="idle")


def stage_name_from_epoch_payload(payload: Dict[str, object]) -> str:
    """Turn an epoch callback payload into the stage label shown in progress files."""
    return str(payload.get("job_name", "training"))


def epoch_trail_entry(payload: Dict[str, object]) -> Dict[str, object]:
    """Keep the epoch fields the dashboard needs without copying the whole payload."""
    return {
        "job_name": str(payload.get("job_name", "")),
        "epoch": int(payload.get("epoch", 0) or 0),
        "epochs": int(payload.get("epochs", 0) or 0),
        "train_loss_total": payload.get("train_loss_total"),
        "train_loss_ce": payload.get("train_loss_ce"),
        "train_loss_kpi": payload.get("train_loss_kpi"),
        "val_primary_pct": payload.get("val_primary_pct"),
        "test_primary_pct": payload.get("test_primary_pct"),
        "epoch_time_s": payload.get("epoch_time_s"),
    }


def parse_run_name_parts(config: ExperimentConfig, run_name: str) -> tuple[Optional[int], int, int]:
    """Parse one run name into optional seed, pattern, and blocked-beam identifiers."""
    prefix = re.escape(str(config.run_name_prefix))
    match = re.fullmatch(rf"{prefix}(?:(?:_seed|_s)(\d+))?_P(\d+)_blockB(\d+)", str(run_name).strip())
    if match is None:
        raise ValueError(
            f"Run name must match {config.run_name_prefix}[_seed<seed>]_P<pattern>_blockB<beam>. "
            f"Got {run_name!r}."
        )
    seed = int(match.group(1)) if match.group(1) is not None else None
    pattern = int(match.group(2))
    blocked_beam_index = int(match.group(3))
    return seed, pattern, blocked_beam_index


def parse_run_name(config: ExperimentConfig, run_name: str) -> BatchJob:
    """Parse one run name back into the pattern and blocked-beam identifiers."""
    _seed, pattern, blocked_beam_index = parse_run_name_parts(config, run_name)

    for job in build_jobs(config):
        if int(job.pattern) == pattern and int(job.blocked_beam_index) == blocked_beam_index:
            return job
    return make_batch_job(pattern, blocked_beam_index)


def find_row(rows: List[Dict[str, object]], *, run_name: str) -> Dict[str, object]:
    """Look up the mutable progress row for one job by its run name."""
    for row in rows:
        if row["run_name"] == run_name:
            return row
    raise KeyError(f"Unknown run_name: {run_name}")


def update_row_elapsed(row: Dict[str, object], started_at: float) -> float:
    """Refresh the elapsed time fields stored in one batch-progress row."""
    elapsed_s = time.time() - started_at
    row["elapsed_s"] = f"{elapsed_s:.3f}"
    row["elapsed_hms"] = format_seconds(elapsed_s)
    return float(elapsed_s)


def finish_row(
    row: Dict[str, object],
    *,
    status: str,
    started_at: float,
    finished_at: Optional[float] = None,
    error: str = "",
) -> float:
    """Mark one batch-progress row as completed or failed and freeze its timing fields."""
    completed_at = time.time() if finished_at is None else float(finished_at)
    elapsed_s = max(0.0, completed_at - started_at)
    row["status"] = str(status)
    row["current_stage"] = str(status)
    row["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completed_at))
    row["elapsed_s"] = f"{elapsed_s:.3f}"
    row["elapsed_hms"] = format_seconds(elapsed_s)
    row["eta_s"] = ""
    row["eta_hms"] = ""
    row["error"] = str(error)
    return float(elapsed_s)


def run_batch_postprocessing(
    config: ExperimentConfig,
    *,
    config_path: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """Run the batch-wide outputs that only make sense after all jobs finish."""
    from .postprocess import generate_batch_comparisons, generate_provenance_outputs, generate_selected_outputs

    config_source = config_path or str(config.config_path)
    steps = postprocessing_plan(config).after_batch
    if not steps:
        active_payload = set_active_run(config, status="idle")
        if progress_callback is not None:
            progress_callback(active_payload)
        return active_payload

    labels = {
        "compare": "comparisons",
        "provenance": "provenance",
        "selected": "selected_plots",
    }
    try:
        for step in steps:
            active_payload = set_active_run(config, status="running", current_stage=labels.get(step, step))
            if progress_callback is not None:
                progress_callback(active_payload)
            if step == "compare":
                generate_batch_comparisons(config.outdir, config=config_source)
            elif step == "provenance":
                generate_provenance_outputs(config_source, out_dir=config.outdir / "provenance")
            elif step == "selected":
                generate_selected_outputs(config.outdir, config=config_source)
            else:
                raise ValueError(f"Unknown batch postprocessing step: {step}")
    except BaseException as exc:
        error_text = format_exception_text(exc)
        # Post-processing failures should still leave the batch in an explicit
        # failed/interrupted state instead of looking like it is still running.
        active_payload = set_active_run(
            config,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            current_stage="postprocessing_failed",
            error=error_text,
            traceback=traceback.format_exc(),
        )
        if progress_callback is not None:
            progress_callback(active_payload)
        raise
    else:
        active_payload = set_active_run(config, status="idle")
        if progress_callback is not None:
            progress_callback(active_payload)
        return active_payload


# ---------------------------------------------------------------------------
# Main runner paths
# ---------------------------------------------------------------------------
def run_batch(
    config: ExperimentConfig,
    *,
    config_path: Optional[str] = None,
    overwrite: Optional[bool] = None,
    dashboard: Optional[bool] = None,
) -> None:
    """Run every configured job and keep the batch progress files current."""
    overwrite_flag = bool(config.overwrite if overwrite is None else overwrite)
    postprocess_steps = postprocessing_plan(config)
    write_hardware_snapshot(config)
    work_items = []
    for seed in config.seed_values():
        seed_config = config.for_seed(seed)
        work_items.extend((seed_config, job) for job in build_jobs(seed_config))
    rows = [job_row(seed_config, job) for seed_config, job in work_items]
    active_payload: Optional[Dict[str, object]] = {
        "status": "idle",
        "updated_at": now_text(),
        "overwrite": overwrite_flag,
    }
    write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

    for seed_config, job in work_items:
        run_name = seed_config.run_name(pattern=job.pattern, blocked_beam_index=job.blocked_beam_index)
        row = find_row(rows, run_name=run_name)
        started_at = time.time()
        recent_epochs: List[Dict[str, object]] = []
        row["status"] = "running"
        row["current_stage"] = "starting"
        row["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at))
        row["error"] = ""
        row["eta_s"] = ""
        row["eta_hms"] = ""
        active_payload = set_active_run(
            seed_config,
            status="running",
            job=job,
            current_stage="starting",
            overwrite=overwrite_flag,
        )
        write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

        def stage_callback(stage: str, payload: Dict[str, object]) -> None:
            nonlocal active_payload
            # Stage callbacks update the coarse-grained status shown in the live
            # batch table even for steps that do not emit per-epoch progress.
            row["current_stage"] = str(stage)
            elapsed_s = update_row_elapsed(row, started_at)
            row["eta_s"] = ""
            row["eta_hms"] = ""
            active_payload = set_active_run(
                seed_config,
                status="running",
                job=job,
                current_stage=str(stage),
                stage_payload=payload,
                recent_epochs=recent_epochs,
                elapsed_s=elapsed_s,
                overwrite=overwrite_flag,
            )
            write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

        def epoch_callback(payload: Dict[str, object]) -> None:
            nonlocal active_payload
            # Epoch callbacks drive the detailed training monitor: ETA, current
            # epoch, and the rolling trail of the last few epochs.
            row["current_stage"] = stage_name_from_epoch_payload(payload)
            elapsed_s = update_row_elapsed(row, started_at)
            recent_epochs.append(epoch_trail_entry(payload))
            recent_epochs[:] = trim_epoch_trail(recent_epochs)
            epoch_eta_s = epoch_eta_seconds({"epoch_progress": payload})
            row["eta_s"] = f"{epoch_eta_s:.3f}" if epoch_eta_s is not None else ""
            row["eta_hms"] = format_seconds(epoch_eta_s) if epoch_eta_s is not None else ""
            active_payload = set_active_run(
                seed_config,
                status="running",
                job=job,
                current_stage=row["current_stage"],
                epoch_progress=payload,
                recent_epochs=recent_epochs,
                current_job_eta_s=epoch_eta_s,
                current_job_eta_hms=format_seconds(epoch_eta_s) if epoch_eta_s is not None else "",
                elapsed_s=elapsed_s,
                overwrite=overwrite_flag,
            )
            write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

        try:
            run_dir = run_single_experiment(
                seed_config,
                job,
                overwrite=overwrite,
                stage_callback=stage_callback,
                epoch_callback=epoch_callback,
            )
            if "run_plots" in postprocess_steps.after_each_run:
                row["current_stage"] = "plotting"
                row["eta_s"] = ""
                row["eta_hms"] = ""
                active_payload = set_active_run(
                    seed_config,
                    status="running",
                    job=job,
                    current_stage="plotting",
                    run_dir=str(run_dir),
                    recent_epochs=recent_epochs,
                    overwrite=overwrite_flag,
                )
                write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)
                from .postprocess import generate_run_plots

                generate_run_plots(run_dir)
            row["run_dir"] = str(run_dir)
            elapsed_s = finish_row(row, status="completed", started_at=started_at)
            active_payload = set_active_run(
                seed_config,
                status="completed",
                job=job,
                run_dir=str(run_dir),
                recent_epochs=recent_epochs,
                elapsed_s=elapsed_s,
                overwrite=overwrite_flag,
            )
        except BaseException as exc:
            error_text = format_exception_text(exc)
            elapsed_s = finish_row(row, status="failed", started_at=started_at, error=error_text)
            active_payload = set_active_run(
                seed_config,
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                job=job,
                error=error_text,
                traceback=traceback.format_exc(),
                recent_epochs=recent_epochs,
                elapsed_s=elapsed_s,
                overwrite=overwrite_flag,
            )
            write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)
            raise

        write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

    active_payload = run_batch_postprocessing(
        config,
        config_path=config_path,
        progress_callback=lambda payload: write_progress_snapshot(config, rows, active_payload=payload, dashboard=dashboard),
    )
    write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)


def run_one(
    config: ExperimentConfig,
    *,
    pattern: int,
    blocked_beam_index: int,
    overwrite: Optional[bool] = None,
    dashboard: Optional[bool] = None,
) -> None:
    """Run one job directly from the CLI using the same pipeline as batch mode."""
    overwrite_flag = bool(config.overwrite if overwrite is None else overwrite)
    postprocess_steps = postprocessing_plan(config)
    write_hardware_snapshot(config)
    job = make_batch_job(pattern, blocked_beam_index)
    started_at = time.time()
    recent_epochs: List[Dict[str, object]] = []
    row = job_row(config, job)
    rows = [row]
    row["status"] = "running"
    row["current_stage"] = "starting"
    row["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at))
    active_payload = set_active_run(
        config,
        status="running",
        job=job,
        current_stage="starting",
        overwrite=overwrite_flag,
    )
    write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

    def stage_callback(stage: str, payload: Dict[str, object]) -> None:
        nonlocal active_payload
        row["current_stage"] = str(stage)
        elapsed_s = update_row_elapsed(row, started_at)
        row["eta_s"] = ""
        row["eta_hms"] = ""
        active_payload = set_active_run(
            config,
            status="running",
            job=job,
            current_stage=str(stage),
            stage_payload=payload,
            recent_epochs=recent_epochs,
            elapsed_s=elapsed_s,
            overwrite=overwrite_flag,
        )
        write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

    def epoch_callback(payload: Dict[str, object]) -> None:
        nonlocal active_payload
        row["current_stage"] = stage_name_from_epoch_payload(payload)
        elapsed_s = update_row_elapsed(row, started_at)
        recent_epochs.append(epoch_trail_entry(payload))
        recent_epochs[:] = trim_epoch_trail(recent_epochs)
        epoch_eta_s = epoch_eta_seconds({"epoch_progress": payload})
        row["eta_s"] = f"{epoch_eta_s:.3f}" if epoch_eta_s is not None else ""
        row["eta_hms"] = format_seconds(epoch_eta_s) if epoch_eta_s is not None else ""
        active_payload = set_active_run(
            config,
            status="running",
            job=job,
            current_stage=row["current_stage"],
            epoch_progress=payload,
            recent_epochs=recent_epochs,
            current_job_eta_s=epoch_eta_s,
            current_job_eta_hms=format_seconds(epoch_eta_s) if epoch_eta_s is not None else "",
            elapsed_s=elapsed_s,
            overwrite=overwrite_flag,
        )
        write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)

    try:
        run_dir = run_single_experiment(
            config,
            job,
            overwrite=overwrite,
            stage_callback=stage_callback,
            epoch_callback=epoch_callback,
        )
        if "run_plots" in postprocess_steps.after_each_run:
            row["current_stage"] = "plotting"
            row["eta_s"] = ""
            row["eta_hms"] = ""
            active_payload = set_active_run(
                config,
                status="running",
                job=job,
                current_stage="plotting",
                run_dir=str(run_dir),
                recent_epochs=recent_epochs,
                overwrite=overwrite_flag,
            )
            write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)
            from .postprocess import generate_run_plots

            generate_run_plots(run_dir)
        row["run_dir"] = str(run_dir)
        elapsed_s = finish_row(row, status="completed", started_at=started_at)
        active_payload = set_active_run(
            config,
            status="completed",
            job=job,
            run_dir=str(run_dir),
            recent_epochs=recent_epochs,
            elapsed_s=elapsed_s,
            overwrite=overwrite_flag,
        )
        write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)
        clear_active_run(config)
        print(f"Finished: {run_dir}")
    except BaseException as exc:
        error_text = format_exception_text(exc)
        elapsed_s = finish_row(row, status="failed", started_at=started_at, error=error_text)
        active_payload = set_active_run(
            config,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            job=job,
            error=error_text,
            traceback=traceback.format_exc(),
            recent_epochs=recent_epochs,
            elapsed_s=elapsed_s,
            overwrite=overwrite_flag,
        )
        write_progress_snapshot(config, rows, active_payload=active_payload, dashboard=dashboard)
        raise


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the runner."""
    parser = argparse.ArgumentParser(description="Run Blocking V5 experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "config.json"),
        help="Path to the Blocking V5 config.json file.",
    )
    parser.add_argument("--pattern", type=int, default=None, help="Run one job for this Set-B pattern.")
    parser.add_argument("--blocked-beam", type=int, default=None, help="Run one job for this blocked Set-B beam.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run or resume one packaged run by name, for example blocking_v5_seed123_P1_blockB1.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs even when files already exist.")
    parser.add_argument(
        "--dashboard",
        dest="dashboard",
        action="store_true",
        default=None,
        help="Write the optional live HTML dashboard during the run.",
    )
    parser.add_argument(
        "--no-dashboard",
        dest="dashboard",
        action="store_false",
        help="Do not write the optional live HTML dashboard during the run.",
    )
    return parser.parse_args()


def main() -> None:
    """Dispatch to single-run or batch mode based on the CLI arguments."""
    args = parse_args()
    config = load_config(args.config)
    has_pattern_pair = not (args.pattern is None and args.blocked_beam is None)
    if (args.pattern is None) != (args.blocked_beam is None):
        raise ValueError("Use --pattern and --blocked-beam together for a single run.")
    if args.run_name is not None and has_pattern_pair:
        raise ValueError("Use either --run-name or --pattern/--blocked-beam, not both.")

    if args.run_name is None and not has_pattern_pair:
        run_batch(config, config_path=args.config, overwrite=True if args.overwrite else None, dashboard=args.dashboard)
        return

    if args.run_name is not None:
        seed, _pattern, _blocked_beam_index = parse_run_name_parts(config, args.run_name)
        run_config = config.for_seed(seed) if seed is not None else config
        job = parse_run_name(run_config, args.run_name)
        run_one(
            run_config,
            pattern=int(job.pattern),
            blocked_beam_index=int(job.blocked_beam_index),
            overwrite=True if args.overwrite else None,
            dashboard=args.dashboard,
        )
        return

    run_one(
        config,
        pattern=int(args.pattern),
        blocked_beam_index=int(args.blocked_beam),
        overwrite=True if args.overwrite else None,
        dashboard=args.dashboard,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except (ConfigSelectionError, MissingDependencyError, FileNotFoundError, ValueError) as exc:
        print(f"Run setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
