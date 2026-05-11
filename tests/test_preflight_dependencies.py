"""Preflight tests for stage selections and missing dependencies."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiment.config import load_config
from experiment.preflight import ConfigSelectionError, MissingDependencyError, validate_run_dependencies
from experiment.stages import export_baseline_checkpoints


RELEASE_ROOT = Path(__file__).resolve().parents[1]


class PreflightDependencyTests(unittest.TestCase):
    def test_default_full_run_selection_passes_static_and_dependency_checks(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = replace(config, outdir=Path(tmpdir) / "results")
            validate_run_dependencies(isolated, isolated.batch_jobs[0], overwrite=False)

    def test_prepare_only_selection_is_allowed(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = replace(config, outdir=Path(tmpdir) / "results", stages=("prepare",))
            validate_run_dependencies(isolated, isolated.batch_jobs[0], overwrite=False)

    def test_aggregate_only_can_use_existing_rows_without_prepared_data(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = replace(
                config,
                outdir=Path(tmpdir) / "results",
                stages=("aggregate",),
                experiments=("BL-AG",),
            )
            run_dir = isolated.run_dir(isolated.batch_jobs[0])
            rows_path = run_dir / "jobs" / "baseline" / "bl_ag_eval" / "rows.csv"
            rows_path.parent.mkdir(parents=True)
            rows_path.write_text("method\nBL-AG\n", encoding="utf-8")

            validate_run_dependencies(isolated, isolated.batch_jobs[0], overwrite=False)

    def test_ft_only_requires_prepared_data_and_bl_ag_weights(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = replace(
                config,
                outdir=Path(tmpdir) / "results",
                stages=("finetune", "aggregate"),
                experiments=("FT",),
            )
            with self.assertRaises(MissingDependencyError) as context:
                validate_run_dependencies(isolated, isolated.batch_jobs[0], overwrite=False)

        message = str(context.exception)
        self.assertIn("prepared data is missing", message)
        self.assertIn("FT-only runs need stored BL-AG weights", message)

    def test_ft_only_accepts_exported_bl_ag_checkpoint_path(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = replace(
                config,
                outdir=Path(tmpdir) / "results",
                stages=("finetune", "aggregate"),
                experiments=("FT",),
            )
            run_dir = isolated.run_dir(isolated.batch_jobs[0])
            stored = run_dir / "stored_weights" / "baseline_pretrained" / "BL-AG_clean_pretrain_weights.pt"
            stored.parent.mkdir(parents=True)
            stored.write_bytes(b"bl-ag")
            with self.assertRaises(MissingDependencyError) as context:
                validate_run_dependencies(isolated, isolated.batch_jobs[0], overwrite=False)

        message = str(context.exception)
        self.assertIn("prepared data is missing", message)
        self.assertNotIn("FT-only runs need stored BL-AG weights", message)

    def test_aggregate_only_requires_existing_stage_rows(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = replace(
                config,
                outdir=Path(tmpdir) / "results",
                stages=("aggregate",),
                experiments=("BL-AG",),
            )
            with self.assertRaises(MissingDependencyError) as context:
                validate_run_dependencies(isolated, isolated.batch_jobs[0], overwrite=False)

        message = str(context.exception)
        self.assertIn("aggregate-only runs need existing jobs/**/rows.csv", message)
        self.assertIn("selected baseline-family results are missing", message)

    def test_invalid_config_selection_fails_before_runtime(self) -> None:
        payload = json.loads((RELEASE_ROOT / "config.json").read_text(encoding="utf-8"))
        payload["training"]["ft_freeze_modes"] = ["freeze_cnn", "freeze_backbone"]
        payload["kpi"]["primary_topk"] = 8

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_config.json"
            config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(ConfigSelectionError) as context:
                load_config(config_path)

        message = str(context.exception)
        self.assertIn("unknown FT freeze modes", message)
        self.assertIn("kpi.primary_topk must be included", message)

    def test_baseline_checkpoint_export_labels_ft_start_weights(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = replace(config, outdir=Path(tmpdir) / "results")
            job = isolated.batch_jobs[0]
            run_dir = isolated.run_dir(job)
            bl_ag = run_dir / "jobs" / "baseline" / "bl_ag_train" / "weights_best.pt"
            bl_aw = run_dir / "jobs" / "baseline" / "bl_aw_blockage_020" / "weights_best.pt"
            bl_ag.parent.mkdir(parents=True)
            bl_aw.parent.mkdir(parents=True)
            bl_ag.write_bytes(b"bl-ag")
            bl_aw.write_bytes(b"bl-aw")

            export_baseline_checkpoints(run_dir, isolated, job, overwrite=False)

            export_root = run_dir / "stored_weights" / "baseline_pretrained"
            manifest = (export_root / "manifest.csv").read_text(encoding="utf-8")
            guide = (export_root / "STORED_WEIGHTS_GUIDE.md").read_text(encoding="utf-8")
            self.assertTrue((export_root / "BL-AG_clean_pretrain_weights.pt").exists())
            self.assertTrue((export_root / "BL-AW_blockage_020_aware_weights.pt").exists())
            self.assertIn("clean_pretrained_ft_start", manifest)
            self.assertIn("use_for_ft_start", manifest)
            self.assertIn("Use this file as the starting checkpoint for FT-only sweeps", guide)


if __name__ == "__main__":
    unittest.main()
