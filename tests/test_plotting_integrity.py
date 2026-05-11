"""Release plotting bundle integrity checks that do not regenerate plots."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from plot_code.families.selected_source_data import refresh_selected_source_data


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SELECTED_ROOT = RELEASE_ROOT / "plots" / "selected_plots"


class ReleasePlottingIntegrityTests(unittest.TestCase):
    def test_selected_manifest_points_to_packaged_pngs(self) -> None:
        manifest_path = SELECTED_ROOT / "selected_plots_manifest.csv"
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 61)
        heatmaps = [row["selected_file"] for row in rows if row["selected_file"].startswith("fig17_heatmap/")]
        self.assertEqual(heatmaps, ["fig17_heatmap/beam_blockage_heatmap.png"])
        missing: list[str] = []
        for row in rows:
            path = SELECTED_ROOT / row["selected_file"]
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(row["selected_file"])

        self.assertEqual(missing, [])

    def test_selected_source_data_files_are_packaged(self) -> None:
        source_root = SELECTED_ROOT / "source_data"
        expected = {
            "compare_patterns_best_summary_table.csv",
            "compare_patterns_mid_summary_table.csv",
            "compare_patterns_same_b3_summary_table.csv",
            "compare_patterns_same_b7_summary_table.csv",
            "compare_patterns_worst_summary_table.csv",
            "multi_seed_run_plots_manifest.csv",
            "non_ml_baseline_summary.csv",
            "p1_summary_ft_freeze_cnn_config.csv",
            "p2_summary_ft_freeze_cnn_config.csv",
            "seed_aggregation_manifest.csv",
        }
        found = {path.name for path in source_root.glob("*.csv")}

        self.assertEqual(found, expected)

    def test_selected_source_data_refresh_waits_for_complete_run_set(self) -> None:
        config = _minimal_refresh_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected_root = root / "plots" / "selected_plots"
            source_path = selected_root / "source_data" / "p1_summary_ft_freeze_cnn_config.csv"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("sentinel\nold\n", encoding="utf-8")

            _write_result_rows(root / "results" / "blocking_v5_seed123_P1_blockB1" / "results.csv", pattern_offset=0)
            written = refresh_selected_source_data(root / "results", selected_root, config)

            self.assertEqual(written, [])
            self.assertEqual(source_path.read_text(encoding="utf-8"), "sentinel\nold\n")

    def test_selected_source_data_refresh_uses_current_results_when_complete(self) -> None:
        config = _minimal_refresh_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected_root = root / "plots" / "selected_plots"
            _write_result_rows(root / "results" / "blocking_v5_seed123_P1_blockB1" / "results.csv", pattern_offset=0)
            _write_result_rows(root / "results" / "blocking_v5_seed123_P2_blockB0" / "results.csv", pattern_offset=10)

            written = refresh_selected_source_data(root / "results", selected_root, config)

            self.assertTrue(written)
            summary_rows = _read_rows(selected_root / "source_data" / "p1_summary_ft_freeze_cnn_config.csv")
            row_20 = next(row for row in summary_rows if row["blocked_beam"] == "B1" and row["blockage_%"] == "20")
            self.assertEqual(row_20["ft_primary_%"], "90.000000")
            self.assertEqual(row_20["blag_primary_%"], "80.000000")
            self.assertEqual(row_20["blaw_primary_%"], "95.000000")

            cross_rows = _read_rows(selected_root / "source_data" / "compare_patterns_best_summary_table.csv")
            ft_20 = next(row for row in cross_rows if row["method"] == "FT:freeze_cnn" and row["blockage_%"] == "20")
            self.assertEqual(ft_20["p1_value"], "90.000000")
            self.assertEqual(ft_20["p2_value"], "100.000000")

def _minimal_refresh_config() -> dict:
    return {
        "training": {"seed": 123, "seeds": []},
        "batch_jobs": [
            {"pattern": 1, "blocked_beam_index": 1},
            {"pattern": 2, "blocked_beam_index": 0},
        ],
        "experiments": ["BL-AG", "BL-AW", "FT", "MAX-SETB", "NN-ANGLE"],
        "blockage_levels": [0, 20],
        "plotting": {"comparison_primary_train_frac": 0.01, "baseline_ft_epoch_for_plots": 10},
        "kpi": {"primary_topk": 3, "primary_margin_db": 1},
        "scenario_provenance": {"pattern_comparisons": {"best": {"p1_beam": 1, "p2_beam": 0}}},
    }


def _write_result_rows(path: Path, *, pattern_offset: int) -> None:
    path.parent.mkdir(parents=True)
    fieldnames = ["method", "freeze", "blockage_%", "train_frac", "ft_epochs", "test_top3_m1db_%"]
    rows = []
    for blockage, base in [(0, 98.0), (20, 80.0)]:
        rows.extend(
            [
                {"method": "BL-AG", "freeze": "", "blockage_%": blockage, "train_frac": "0.000000", "ft_epochs": 0, "test_top3_m1db_%": base + pattern_offset},
                {"method": "BL-AW", "freeze": "", "blockage_%": blockage, "train_frac": "0.000000", "ft_epochs": 0, "test_top3_m1db_%": base + 15.0 + pattern_offset},
                {"method": "FT", "freeze": "freeze_cnn", "blockage_%": blockage, "train_frac": "0.010000", "ft_epochs": 10, "test_top3_m1db_%": base + 10.0 + pattern_offset},
                {"method": "MAX-SETB", "freeze": "", "blockage_%": blockage, "train_frac": "0.000000", "ft_epochs": 0, "test_top3_m1db_%": 35.0 + pattern_offset},
                {"method": "NN-ANGLE", "freeze": "", "blockage_%": blockage, "train_frac": "0.000000", "ft_epochs": 0, "test_top3_m1db_%": 65.0 + pattern_offset},
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
