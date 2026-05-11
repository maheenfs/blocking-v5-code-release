"""Static HTML renderer for the optional dashboard."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import List, Mapping

from .artifacts import (
    artifact_plan_rows,
    config_identity_rows,
    error_rows,
    eta_rows,
    ft_only_rows,
    kpi_rows,
    plotting_focus_text,
    plotting_status_rows,
    resume_rows,
    run_selection_rows,
    stage_plan_rows,
)
from .formatting import format_optional_float, render_html_table
from .resources import resource_rows, resource_warning_rows
from .settings import dashboard_settings


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _batch_rows(snapshot: Mapping[str, object], headers: List[str]) -> List[List[str]]:
    rows = []
    for row in snapshot.get("rows", []) or []:
        if isinstance(row, Mapping):
            rows.append([str(row.get(header, "")) for header in headers])
    return rows


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _initial_state_marker(snapshot: Mapping[str, object], active: Mapping[str, object]) -> Mapping[str, str]:
    """Return the dashboard state marker at render time.

    The browser script below refines this marker as the HTML ages. That keeps a
    stale file useful after a run crashes, when Python is no longer available to
    rewrite the dashboard.
    """
    status = str(active.get("status", "idle") or "idle").lower()
    total = _safe_int(snapshot.get("total_jobs", 0))
    completed = _safe_int(snapshot.get("completed_jobs", 0))
    failed = _safe_int(snapshot.get("failed_jobs", 0))
    running = _safe_int(snapshot.get("running_jobs", 0))

    if failed > 0 or status in {"failed", "interrupted"}:
        return {
            "key": "crashed",
            "label": "CRASHED / STOPPED",
            "detail": "A failure or interruption is recorded. Check the Errors panel and latest log before restarting.",
        }
    if total > 0 and completed >= total:
        return {
            "key": "complete",
            "label": "COMPLETE",
            "detail": "All configured jobs are marked complete. Review selected plots and manifests before packaging.",
        }
    if running > 0 or status == "running":
        return {
            "key": "running",
            "label": "RUNNING",
            "detail": "Dashboard updates are fresh. This marker turns stale if updates stop.",
        }
    return {
        "key": "old",
        "label": "OLD / IDLE",
        "detail": "No live run is recorded in this snapshot. Confirm process state before assuming training is active.",
    }


def _remaining_jobs_html(snapshot: Mapping[str, object]) -> str:
    items = []
    for item in snapshot.get("remaining_jobs", []) or []:
        if not isinstance(item, Mapping):
            continue
        items.append(
            "<li>"
            f"{html.escape(str(item.get('run_name', '')))}"
            f" | {html.escape(str(item.get('status', '')))}"
            f" | {html.escape(str(item.get('current_stage', '')))}"
            "</li>"
        )
    return "".join(items) if items else "<li>None</li>"


def _epoch_rows(active: Mapping[str, object], latest_epoch_count: int) -> List[List[str]]:
    recent_epochs = active.get("recent_epochs", [])
    if not isinstance(recent_epochs, list):
        return []
    rows: List[List[str]] = []
    for item in recent_epochs[-latest_epoch_count:] if latest_epoch_count else []:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            [
                f"{int(item.get('epoch', 0))} / {int(item.get('epochs', 0))}",
                str(item.get("job_name", "")),
                format_optional_float(item.get("train_loss_total"), ndigits=4),
                format_optional_float(item.get("train_loss_ce"), ndigits=4),
                format_optional_float(item.get("train_loss_kpi"), ndigits=4),
                format_optional_float(item.get("val_primary_pct"), ndigits=3),
                format_optional_float(item.get("test_primary_pct"), ndigits=3),
                format_optional_float(item.get("epoch_time_s"), ndigits=2, suffix="s"),
            ]
        )
    return rows


def _seed_rows(snapshot: Mapping[str, object]) -> List[List[str]]:
    overview = _mapping(snapshot.get("seed_overview"))
    active_seed = str(overview.get("active_seed", ""))
    rows: List[List[str]] = []
    for item in snapshot.get("seed_summaries", []) or []:
        if not isinstance(item, Mapping):
            continue
        seed = str(item.get("seed", ""))
        state = str(item.get("state", ""))
        if seed and seed == active_seed and state != "completed":
            state = f"{state} active"
        rows.append(
            [
                seed,
                state,
                str(item.get("total_jobs", "")),
                str(item.get("completed_jobs", "")),
                str(item.get("jobs_left", "")),
                str(item.get("active_job_label", "") or "n/a"),
                str(item.get("running_jobs", "")),
                str(item.get("failed_jobs", "")),
                str(item.get("pending_jobs", "")),
                f"{float(item.get('progress_pct', 0.0)):.1f}%",
                str(item.get("eta_hms", "") or "n/a"),
                str(item.get("eta_finish_at", "") or "n/a"),
            ]
        )
    return rows


def _plotting_selection_table(config: object) -> str:
    plot_focus = plotting_focus_text(config)
    return render_html_table(
        ["Plot setting", "Selected"],
        [
            ["Runner after each run", plot_focus["runner_after_each_run"]],
            ["Runner after batch", plot_focus["runner_after_batch"]],
            ["KPI keys", plot_focus["metric_plot_keys"]],
            ["Run plot families", plot_focus["run_plot_families"]],
            ["Comparison families", plot_focus["comparison_plot_families"]],
            ["Multi-seed run families", plot_focus["multiseed_run_plot_families"]],
            ["Run fixed-blockage focus", plot_focus["run_plot_blockages"]],
            ["Multi-KPI fixed-blockage focus", plot_focus["multi_kpi_blockages"]],
            ["Comparison fixed-blockage focus", plot_focus["comparison_focus_blockages"]],
            ["All evaluated blockage levels", plot_focus["all_blockage_levels"]],
        ],
    )


def write_dashboard_html(path: Path, *, config: object, snapshot: Mapping[str, object]) -> None:
    """Write the optional static HTML dashboard snapshot."""
    settings = dashboard_settings(config)
    refresh_seconds = max(1, int(settings.get("refresh_seconds", 5) or 5))
    active = _mapping(snapshot.get("active_run")) or {"status": "idle"}
    epoch_progress = _mapping(active.get("epoch_progress"))
    seed_overview = _mapping(snapshot.get("seed_overview"))
    latest_epoch_count = max(0, int(settings.get("show_latest_epochs", 5) or 0))
    runtime = _mapping(snapshot.get("runtime_metrics"))
    artifact_rows = artifact_plan_rows(config, active)

    progress_headers = [
        "run_name",
        "seed",
        "status",
        "current_stage",
        "pattern",
        "blocked_beam_index",
        "elapsed_hms",
        "eta_hms",
        "error",
    ]
    epoch_headers = ["Epoch", "Training branch", "Train loss", "CE loss", "KPI loss", "Val primary", "Test primary", "Epoch time"]
    seed_headers = ["Seed", "State", "Jobs", "Completed", "Jobs left", "Current job", "Running", "Failed", "Pending", "Progress", "ETA", "ETA finish"]
    active_job_ordinal = int(snapshot.get("active_job_ordinal", 0) or 0)
    total_jobs = int(snapshot.get("total_jobs", 0) or 0)
    active_job_text = f"{active_job_ordinal} / {total_jobs}" if active_job_ordinal > 0 else f"0 / {total_jobs}"
    epoch_label = ""
    if epoch_progress:
        epoch_label = f"{int(epoch_progress.get('epoch', 0))} / {int(epoch_progress.get('epochs', 0))}"
    state_marker = _initial_state_marker(snapshot, active)
    stale_after_seconds = max(60, refresh_seconds * 6)
    crash_after_seconds = max(600, refresh_seconds * 60)
    old_after_seconds = max(1800, refresh_seconds * 120)

    resource_section = ""
    if bool(settings.get("show_resource_monitor", True)):
        warning_rows = resource_warning_rows(runtime)
        warnings = (
            render_html_table(["Resource", "Warning"], warning_rows)
            if warning_rows
            else "<p class=\"ok\">No resource warnings detected.</p>"
        )
        resource_section = f"""
  <section class="panel wide">
    <h2>Resources</h2>
    {render_html_table(["Metric", "Value"], resource_rows(runtime))}
  </section>
  <section class="panel wide">
    <h2>Resource Warnings</h2>
    {warnings}
  </section>
"""

    plot_section = ""
    if bool(settings.get("show_plotting_plan", True)):
        plot_section = f"""
  <section class="panel wide">
    <h2>Plotting Selection</h2>
    {_plotting_selection_table(config)}
  </section>
  <section class="panel wide">
    <h2>Plotting Monitor</h2>
    {render_html_table(["Area", "State", "Details"], plotting_status_rows(config, active))}
  </section>
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Blocking V5 Run Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d6dae1;
      --header: #e9eef6;
      --accent: #2457a6;
      --ok: #1b6f3a;
      --warn: #9b5b00;
      --bad: #a52626;
      --old: #6b7280;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: var(--ink);
      background: var(--bg);
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; }}
    p {{ margin: 6px 0; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .ok {{ color: var(--ok); }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin: 8px 0 18px 0;
      flex-wrap: wrap;
    }}
    .toolbar button {{
      border: 1px solid #aab4c2;
      background: white;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
    }}
    .state-banner {{
      display: flex;
      gap: 14px;
      align-items: flex-start;
      justify-content: space-between;
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 8px solid var(--accent);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 0 0 18px 0;
    }}
    .state-banner.state-running {{ border-left-color: var(--accent); }}
    .state-banner.state-stale {{ border-left-color: var(--warn); }}
    .state-banner.state-crashed {{ border-left-color: var(--bad); }}
    .state-banner.state-complete {{ border-left-color: var(--ok); }}
    .state-banner.state-old {{ border-left-color: var(--old); }}
    .state-pill {{
      display: inline-block;
      min-width: 142px;
      text-align: center;
      border-radius: 6px;
      padding: 7px 10px;
      color: white;
      background: var(--accent);
      font-weight: 700;
      font-size: 13px;
      letter-spacing: 0;
    }}
    .state-running .state-pill {{ background: var(--accent); }}
    .state-stale .state-pill {{ background: var(--warn); }}
    .state-crashed .state-pill {{ background: var(--bad); }}
    .state-complete .state-pill {{ background: var(--ok); }}
    .state-old .state-pill {{ background: var(--old); }}
    .state-detail {{
      flex: 1;
      min-width: 220px;
    }}
    .state-detail strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .state-legend {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      max-width: 520px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
      gap: 12px;
      margin: 18px 0 20px 0;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .card strong {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .card .value {{
      font-size: 24px;
      font-weight: 650;
    }}
    .metric-line {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }}
    .sections {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
      gap: 16px;
      margin-bottom: 16px;
    }}
    .wide {{ margin-bottom: 16px; }}
    ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid var(--line);
    }}
    th, td {{
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid #edf0f4;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      background: var(--header);
      color: #26364d;
      font-weight: 650;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    @media (max-width: 850px) {{
      body {{ margin: 14px; }}
      .sections {{ grid-template-columns: 1fr; }}
      .state-banner {{ flex-direction: column; }}
      .state-pill {{ min-width: 0; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
  <script>
    (function () {{
      const key = "blocking-v5-dashboard-scroll";
      const staleAfterSeconds = {stale_after_seconds};
      const crashAfterSeconds = {crash_after_seconds};
      const oldAfterSeconds = {old_after_seconds};
      function dashboardAgeSeconds(value) {{
        const parsed = Date.parse(String(value || "").replace(" ", "T"));
        if (Number.isNaN(parsed)) {{
          return null;
        }}
        return Math.max(0, Math.round((Date.now() - parsed) / 1000));
      }}
      function ageText(seconds) {{
        if (seconds === null) {{
          return "unknown age";
        }}
        if (seconds < 60) {{
          return seconds + "s old";
        }}
        const minutes = Math.floor(seconds / 60);
        const rem = seconds % 60;
        if (minutes < 60) {{
          return minutes + "m " + rem + "s old";
        }}
        const hours = Math.floor(minutes / 60);
        return hours + "h " + (minutes % 60) + "m old";
      }}
      function setMarker(marker, state, label, detail) {{
        marker.className = "state-banner state-" + state;
        const pill = document.getElementById("run-state-pill");
        const body = document.getElementById("run-state-detail");
        if (pill) {{
          pill.textContent = label;
        }}
        if (body) {{
          body.textContent = detail;
        }}
      }}
      function updateStateMarker() {{
        const marker = document.getElementById("run-state-marker");
        if (!marker) {{
          return;
        }}
        const status = String(marker.dataset.activeStatus || "").toLowerCase();
        const running = parseInt(marker.dataset.runningJobs || "0", 10) || 0;
        const failed = parseInt(marker.dataset.failedJobs || "0", 10) || 0;
        const completed = parseInt(marker.dataset.completedJobs || "0", 10) || 0;
        const total = parseInt(marker.dataset.totalJobs || "0", 10) || 0;
        const age = dashboardAgeSeconds(marker.dataset.updatedAt);
        const ageLabel = ageText(age);
        if (failed > 0 || status === "failed" || status === "interrupted") {{
          setMarker(marker, "crashed", "CRASHED / STOPPED", "A failure or interruption is recorded. Check the Errors panel and latest log before restarting. Snapshot is " + ageLabel + ".");
        }} else if (total > 0 && completed >= total) {{
          const oldSuffix = age !== null && age >= oldAfterSeconds ? " OLD" : "";
          setMarker(marker, "complete", "COMPLETE" + oldSuffix, "All configured jobs are marked complete. Snapshot is " + ageLabel + ".");
        }} else if (running > 0 || status === "running") {{
          if (age !== null && age >= crashAfterSeconds) {{
            setMarker(marker, "crashed", "CRASHED / STOPPED", "The snapshot still says running, but the dashboard has not updated for " + ageLabel + ". Confirm process state before restarting.");
          }} else if (age !== null && age >= staleAfterSeconds) {{
            setMarker(marker, "stale", "STALE", "The snapshot says running, but updates have paused. Snapshot is " + ageLabel + ".");
          }} else {{
            setMarker(marker, "running", "RUNNING", "Dashboard updates are fresh. Snapshot is " + ageLabel + ".");
          }}
        }} else {{
          setMarker(marker, "old", "OLD / IDLE", "No live run is recorded in this snapshot. Snapshot is " + ageLabel + ".");
        }}
      }}
      window.addEventListener("load", function () {{
        const y = sessionStorage.getItem(key);
        if (y !== null) {{
          window.scrollTo(0, parseInt(y, 10) || 0);
        }}
        updateStateMarker();
        window.setInterval(updateStateMarker, 1000);
        window.setTimeout(function () {{
          sessionStorage.setItem(key, String(window.scrollY || 0));
          window.location.reload();
        }}, {refresh_seconds * 1000});
      }});
      window.addEventListener("beforeunload", function () {{
        sessionStorage.setItem(key, String(window.scrollY || 0));
      }});
    }})();
  </script>
</head>
<body>
  <h1>Blocking V5 Run Dashboard</h1>
  <p class="muted">Results root: {html.escape(str(getattr(config, "outdir", "")))}</p>
  <div class="toolbar">
    <button type="button" onclick="window.location.reload()">Refresh now</button>
    <span class="muted">Auto-refresh every {refresh_seconds} seconds. Updated {html.escape(str(snapshot.get("updated_at", "")))}.</span>
  </div>
  <section
    id="run-state-marker"
    class="state-banner state-{html.escape(state_marker["key"])}"
    data-updated-at="{html.escape(str(snapshot.get("updated_at", "")))}"
    data-active-status="{html.escape(str(active.get("status", "idle") or "idle"))}"
    data-running-jobs="{html.escape(str(snapshot.get("running_jobs", 0)))}"
    data-failed-jobs="{html.escape(str(snapshot.get("failed_jobs", 0)))}"
    data-completed-jobs="{html.escape(str(snapshot.get("completed_jobs", 0)))}"
    data-total-jobs="{html.escape(str(snapshot.get("total_jobs", 0)))}">
    <span class="state-pill" id="run-state-pill">{html.escape(state_marker["label"])}</span>
    <div class="state-detail">
      <strong>Run State Marker</strong>
      <span id="run-state-detail">{html.escape(state_marker["detail"])}</span>
    </div>
    <div class="state-legend">
      Running = fresh updates; Stale = running snapshot has stopped updating;
      Crashed / stopped = failed/interrupted or stale too long; Complete = all
      configured jobs done; Old = idle or previous snapshot.
    </div>
  </section>
  <div class="summary">
    <div class="card"><strong>Total jobs</strong><div class="value">{snapshot.get("total_jobs", 0)}</div></div>
    <div class="card"><strong>Seeds</strong><div class="value">{snapshot.get("seed_count", 0)}</div></div>
    <div class="card"><strong>Active job</strong><div class="value">{html.escape(active_job_text)}</div><div class="metric-line">{float(snapshot.get("batch_progress_pct", 0.0)):.1f}% complete</div></div>
    <div class="card"><strong>Active seed</strong><div class="value">{html.escape(str(seed_overview.get("active_seed", "n/a")))}</div><div class="metric-line">{html.escape(str(seed_overview.get("active_seed_state", "n/a")))}</div></div>
    <div class="card"><strong>Seed job</strong><div class="value">{html.escape(str(seed_overview.get("active_seed_job", "") or "n/a"))}</div><div class="metric-line">{html.escape(str(seed_overview.get("active_seed_jobs_left", 0)))} left in seed</div></div>
    <div class="card"><strong>Seed ETA</strong><div class="value">{html.escape(str(seed_overview.get("active_seed_eta_hms", "") or "n/a"))}</div><div class="metric-line">{html.escape(str(seed_overview.get("active_seed_eta_finish_at", "") or "finish unknown"))}</div></div>
    <div class="card"><strong>Remaining seeds</strong><div class="value">{html.escape(str(seed_overview.get("remaining_seed_count", 0)))}</div><div class="metric-line">next {html.escape(str(seed_overview.get("next_seed", "") or "n/a"))}</div></div>
    <div class="card"><strong>Seed ETA total</strong><div class="value">{html.escape(str(seed_overview.get("seed_eta_total_hms", "") or "n/a"))}</div><div class="metric-line">{html.escape(str(seed_overview.get("seed_eta_total_finish_at", "") or "finish unknown"))}</div></div>
    <div class="card"><strong>Completed</strong><div class="value">{snapshot.get("completed_jobs", 0)}</div></div>
    <div class="card"><strong>Failed</strong><div class="value">{snapshot.get("failed_jobs", 0)}</div></div>
    <div class="card"><strong>Current ETA</strong><div class="value">{html.escape(str(snapshot.get("current_job_eta_hms") or "n/a"))}</div><div class="metric-line">{html.escape(str(snapshot.get("current_job_eta_finish_at") or "finish unknown"))}</div></div>
    <div class="card"><strong>Full batch ETA</strong><div class="value">{html.escape(str(snapshot.get("batch_eta_hms") or "n/a"))}</div><div class="metric-line">{html.escape(str(snapshot.get("batch_eta_finish_at") or "finish unknown"))}</div></div>
    <div class="card"><strong>Epoch</strong><div class="value">{html.escape(epoch_label or "n/a")}</div><div class="metric-line">{html.escape(str(active.get("current_stage", "idle") or "idle"))}</div></div>
  </div>
  <div class="sections">
    <section class="panel">
      <h2>Run Selection</h2>
      {render_html_table(["Field", "Value"], run_selection_rows(config, active))}
    </section>
    <section class="panel">
      <h2>Work Left</h2>
      <p><strong>Remaining jobs:</strong> {len(snapshot.get("remaining_jobs", []) or [])}</p>
      <ul>{_remaining_jobs_html(snapshot)}</ul>
    </section>
  </div>
  <section class="panel wide">
    <h2>Dynamic Run Plan</h2>
    {render_html_table(["Stage", "State"], stage_plan_rows(config, active))}
  </section>
  <section class="panel wide">
    <h2>Artifact-Aware Status</h2>
    {render_html_table(["Artifact", "State", "Location", "Meaning"], artifact_rows)}
  </section>
  <section class="panel wide">
    <h2>FT-Only and Branch Status</h2>
    {render_html_table(["Field", "State", "Details", "Note"], ft_only_rows(config, active))}
  </section>
  <section class="panel wide">
    <h2>KPI Monitor</h2>
    {render_html_table(["Field", "Value", "Meaning"], kpi_rows(config, active))}
  </section>
  <section class="panel wide">
    <h2>ETA and Timing</h2>
    {render_html_table(["Area", "Estimate", "Details"], eta_rows(config, snapshot, active))}
  </section>
  <section class="panel wide">
    <h2>Resume and Overwrite</h2>
    {render_html_table(["Area", "State", "Guidance"], resume_rows(config, active))}
  </section>
  <section class="panel wide">
    <h2>Errors</h2>
    {render_html_table(["Area", "Message", "Likely fix"], error_rows(active, artifact_rows))}
  </section>
  <section class="panel wide">
    <h2>Config Identity</h2>
    {render_html_table(["Field", "Value", "Details"], config_identity_rows(config, active))}
  </section>
  <section class="panel wide">
    <h2>Seed Progress</h2>
    {render_html_table(seed_headers, _seed_rows(snapshot)) if _seed_rows(snapshot) else "<p>No seed data yet.</p>"}
  </section>
{plot_section}
{resource_section}
  <section class="panel wide">
    <h2>Latest {latest_epoch_count} Epochs</h2>
    {render_html_table(epoch_headers, _epoch_rows(active, latest_epoch_count)) if _epoch_rows(active, latest_epoch_count) else "<p>No epoch data yet.</p>"}
  </section>
  <section class="panel wide">
    <h2>Batch Progress</h2>
    {render_html_table(progress_headers, _batch_rows(snapshot, progress_headers))}
  </section>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(html_text, encoding="utf-8")
    tmp.replace(path)
