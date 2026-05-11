"""Dashboard helper checks that do not start training."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from dashboard.artifacts import (
    artifact_plan_rows,
    config_identity_rows,
    ft_only_rows,
    kpi_rows,
    plotting_status_rows,
    resume_rows,
)
from dashboard.plotting_progress import selection_summary
from dashboard.render import write_dashboard_html
from dashboard.resources import runtime_metrics_payload
from experiment.config import load_config
from plot_code.config import PlotSelection
from runner.pipeline import dashboard_snapshot


RELEASE_ROOT = Path(__file__).resolve().parents[1]


class DashboardHelperTests(unittest.TestCase):
    def test_artifact_and_ft_rows_are_readable_without_results(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        seed_config = config.for_seed(123)
        active = {
            "status": "running",
            "run_name": seed_config.run_name(pattern=1, blocked_beam_index=1),
            "seed": 123,
            "pattern": 1,
            "blocked_beam_index": 1,
            "current_stage": "starting",
        }

        artifact_rows = artifact_plan_rows(config, active)
        ft_rows = ft_only_rows(config, active)
        resume = resume_rows(config, active)

        self.assertTrue(any(row[0] == "FT branches" for row in artifact_rows))
        self.assertTrue(any(row[0] == "BL-AG checkpoint" for row in ft_rows))
        self.assertTrue(any(row[0] == "Safe next command" for row in resume))

    def test_kpi_plotting_and_config_rows_have_expected_sections(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        active = {"status": "idle"}

        self.assertTrue(any(row[0] == "Primary KPI" for row in kpi_rows(config, active)))
        self.assertTrue(any(row[0] == "Selected bundle" for row in plotting_status_rows(config, active)))
        self.assertTrue(any(row[0] == "Current config hash" for row in config_identity_rows(config, active)))

    def test_selection_summary_only_lists_active_filters(self) -> None:
        selection = PlotSelection(kpis=("primary",), families=("blockage", "epochs"), seeds=(123,), organize=True)

        summary = selection_summary(selection)

        self.assertIn("kpis=primary", summary)
        self.assertIn("families=blockage,epochs", summary)
        self.assertIn("organize=true", summary)

    def test_dashboard_render_writes_all_dynamic_panels(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        snapshot = {
            "updated_at": "test",
            "total_jobs": 1,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "running_jobs": 1,
            "pending_jobs": 0,
            "seed_count": 1,
            "active_job_ordinal": 1,
            "batch_progress_pct": 0.0,
            "average_completed_job_hms": "",
            "job_duration_sample_count": 0,
            "job_duration_basis": "",
            "current_job_eta_hms": "",
            "current_job_eta_finish_at": "",
            "batch_eta_hms": "",
            "batch_eta_finish_at": "",
            "seed_summaries": [],
            "seed_overview": {},
            "runtime_metrics": runtime_metrics_payload(RELEASE_ROOT),
            "remaining_jobs": [],
            "active_run": {"status": "running", "current_stage": "starting"},
            "rows": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dashboard.html"
            write_dashboard_html(path, config=config, snapshot=snapshot)
            text = path.read_text(encoding="utf-8")

        for heading in (
            "Dynamic Run Plan",
            "Artifact-Aware Status",
            "FT-Only and Branch Status",
            "KPI Monitor",
            "Plotting Monitor",
            "ETA and Timing",
            "Resume and Overwrite",
            "Errors",
            "Resource Warnings",
            "Config Identity",
        ):
            self.assertIn(heading, text)
        self.assertIn("Active seed", text)
        self.assertIn("Seed ETA total", text)
        self.assertIn("Run State Marker", text)
        self.assertIn("data-active-status=\"running\"", text)

    def test_dashboard_snapshot_has_active_seed_details(self) -> None:
        rows = [
            _progress_row("seed123_P1", seed=123, status="completed", elapsed_s="100.0"),
            _progress_row("seed456_P1", seed=456, status="running", elapsed_s="25.0"),
            _progress_row("seed456_P2", seed=456, status="pending"),
            _progress_row("seed789_P1", seed=789, status="pending"),
        ]
        active_payload = {
            "status": "running",
            "seed": 456,
            "epoch_progress": {"epoch": 1, "epochs": 2, "epoch_time_s": 10.0},
        }

        snapshot = dashboard_snapshot(rows, active_payload=active_payload)
        overview = snapshot["seed_overview"]
        seed_456 = next(row for row in snapshot["seed_summaries"] if row["seed"] == "456")

        self.assertEqual(overview["active_seed"], "456")
        self.assertEqual(overview["active_seed_job"], "1 / 2")
        self.assertEqual(overview["active_seed_jobs_left"], 2)
        self.assertEqual(overview["remaining_seed_count"], 2)
        self.assertEqual(seed_456["jobs_left"], 2)
        self.assertEqual(seed_456["active_job_label"], "1 / 2")

    def test_dashboard_eta_ignores_resume_skip_outliers(self) -> None:
        rows = [
            _progress_row("seed123_P1", seed=123, status="completed", elapsed_s="0.7"),
            _progress_row("seed123_P2", seed=123, status="completed", elapsed_s="0.8"),
            _progress_row("seed789_P1", seed=789, status="completed", elapsed_s="5800.0"),
            _progress_row("seed789_P2", seed=789, status="completed", elapsed_s="5900.0"),
            _progress_row("seed789_P3", seed=789, status="running", elapsed_s="3000.0"),
            _progress_row("seed789_P4", seed=789, status="pending"),
            _progress_row("seed789_P5", seed=789, status="pending"),
        ]
        active_payload = {
            "status": "running",
            "seed": 789,
            "epoch_progress": {"epoch": 1, "epochs": 2, "epoch_time_s": 10.0},
        }

        snapshot = dashboard_snapshot(rows, active_payload=active_payload)

        self.assertAlmostEqual(float(snapshot["average_completed_job_s"]), 5850.0)
        self.assertEqual(snapshot["job_duration_sample_count"], 2)
        self.assertGreater(float(snapshot["current_job_eta_s"]), 2800.0)
        self.assertGreater(float(snapshot["batch_eta_s"]), 14500.0)
        self.assertIn("resume-skip rows are ignored", str(snapshot["job_duration_basis"]))


def _progress_row(name: str, *, seed: int, status: str, elapsed_s: str = "") -> dict[str, object]:
    return {
        "run_name": name,
        "seed": seed,
        "pattern": 1,
        "blocked_beam_index": 1,
        "description": name,
        "status": status,
        "current_stage": "training" if status == "running" else "",
        "started_at": "",
        "completed_at": "",
        "elapsed_s": elapsed_s,
        "elapsed_hms": "",
        "eta_s": "",
        "eta_hms": "",
        "run_dir": "",
        "error": "",
    }


if __name__ == "__main__":
    unittest.main()
