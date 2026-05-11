"""Re-render dashboard HTML from live progress JSON.

This is a monitor-only helper. It does not train, restart, overwrite model
weights, or edit scientific result tables. Its main use is keeping dashboard
HTML aligned with the current source code when a long run is already active.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Mapping

from experiment.config import load_config
from runner.pipeline import dashboard_snapshot, progress_paths

from .render import write_dashboard_html
from .resources import runtime_metrics_payload


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _status_counts(rows: object) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(rows, list):
        return counts
    for row in rows:
        if isinstance(row, Mapping):
            status = str(row.get("status", "unknown") or "unknown")
            counts[status] = counts.get(status, 0) + 1
    return counts


def refresh_dashboard_once(release_root: Path) -> Dict[str, object]:
    """Render dashboard.html from batch_live_progress.json once."""
    config = load_config(release_root / "config.json")
    paths = progress_paths(config)
    progress = _read_json(paths["progress_json"])
    if not progress:
        raise FileNotFoundError(f"No readable progress JSON at {paths['progress_json']}")

    active = progress.get("active_run") if isinstance(progress.get("active_run"), dict) else {}
    if not active:
        active = _read_json(paths["active_run"])
    rows = progress.get("rows") if isinstance(progress.get("rows"), list) else []

    snapshot = dashboard_snapshot(rows, active_payload=active)
    # Keep the training process timestamp. A refresher must not make a stopped
    # run look fresh by stamping the dashboard with its own loop time.
    snapshot["updated_at"] = progress.get("updated_at", snapshot.get("updated_at"))
    snapshot["runtime_metrics"] = runtime_metrics_payload(config.outdir)
    write_dashboard_html(paths["dashboard"], config=config, snapshot=snapshot)
    return snapshot


def run_refresher(release_root: Path, *, interval_s: float) -> None:
    """Refresh dashboard.html until the batch is no longer active."""
    while True:
        snapshot = refresh_dashboard_once(release_root)
        counts = _status_counts(snapshot.get("rows"))
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"completed={counts.get('completed', 0)} "
            f"running={counts.get('running', 0)} "
            f"pending={counts.get('pending', 0)} "
            f"failed={counts.get('failed', 0)} "
            f"batch_eta={snapshot.get('batch_eta_hms') or 'n/a'}",
            flush=True,
        )
        if counts.get("running", 0) == 0 and (counts.get("pending", 0) == 0 or counts.get("failed", 0) > 0):
            return
        time.sleep(max(1.0, float(interval_s)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh dashboard.html from live progress JSON.")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path containing config.json and results/.",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between refreshes.")
    parser.add_argument("--once", action="store_true", help="Render once and exit.")
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    if args.once:
        snapshot = refresh_dashboard_once(release_root)
        print(
            f"dashboard refreshed; batch_eta={snapshot.get('batch_eta_hms') or 'n/a'}; "
            f"average_job={snapshot.get('average_completed_job_hms') or 'n/a'}"
        )
        return
    run_refresher(release_root, interval_s=args.interval)


if __name__ == "__main__":
    main()
