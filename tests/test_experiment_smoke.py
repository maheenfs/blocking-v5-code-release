"""Fast release smoke tests that do not start training."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from experiment.config import KpiConfig, load_config
from experiment.data import build_blocked_dataset_views, build_setb_features_from_rsrp
from experiment.geometry import build_setb_tx_indices
from experiment.metrics import evaluate_ranked_beams
from experiment.model import CNNBeamPredictor, parameter_counts_for_freeze
from experiment.run_plan import postprocessing_plan
from experiment.system import load_system_settings, select_torch_device
from runner.dashboard import dashboard_enabled, dashboard_path, dashboard_settings
from runner.jobs import make_batch_job, parse_run_name_parts


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def full_ranking(*front: int) -> np.ndarray:
    """Build one valid 32-beam ranking with selected beams at the front."""
    seen = set()
    ordered: list[int] = []
    for beam in front:
        beam_i = int(beam)
        if beam_i not in seen:
            ordered.append(beam_i)
            seen.add(beam_i)
    ordered.extend(beam for beam in range(32) if beam not in seen)
    return np.asarray(ordered, dtype=np.int64)


class ReleaseExperimentSmokeTests(unittest.TestCase):
    def test_config_paths_and_seed_expansion_are_release_local(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")

        self.assertEqual(config.seed_values(), (123, 456, 789))
        self.assertEqual(config.outdir, RELEASE_ROOT / "results")
        self.assertEqual(config.values_cube_path, RELEASE_ROOT / "VIV0R4_116_00.npy")

        seed_config = config.for_seed(456)
        self.assertEqual(seed_config.training.seed, 456)
        self.assertEqual(seed_config.run_name(pattern=1, blocked_beam_index=4), "blocking_v5_seed456_P1_blockB4")

    def test_dataset_is_packaged_for_standalone_release(self) -> None:
        dataset_path = RELEASE_ROOT / "VIV0R4_116_00.npy"

        self.assertTrue(dataset_path.exists(), "Release dataset is missing.")
        values = np.load(dataset_path, mmap_mode="r", allow_pickle=False)

        self.assertEqual(values.shape, (120000, 32, 8))
        self.assertEqual(str(values.dtype), "float32")

    def test_default_config_matches_paper_protocol(self) -> None:
        config = load_config(RELEASE_ROOT / "config.json")

        self.assertEqual(config.blockage_levels, (0, 20, 40, 60, 80, 100))
        self.assertEqual(config.training.pretrain_epochs, 20)
        self.assertEqual(config.training.aware_epochs, 20)
        self.assertEqual(config.training.ft_epoch_sweep, (1, 2, 3, 5, 8, 10, 12))
        self.assertEqual(config.training.ft_freeze_modes, ("no_freeze", "freeze_cnn", "freeze_head", "freeze_all"))
        self.assertEqual(config.training.train_fracs_sweep, (0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0))
        self.assertEqual(config.kpi.primary_key(), "top3_m1db_%")

        postprocessing = postprocessing_plan(config)
        self.assertEqual(postprocessing.after_each_run, ())
        self.assertEqual(postprocessing.after_batch, ("selected",))

    def test_system_settings_default_to_paper_cpu_runtime(self) -> None:
        settings = load_system_settings(RELEASE_ROOT / "config.json")

        self.assertEqual(settings["device"], "cpu")
        self.assertEqual(settings["matplotlib_cache_dir"], "./.cache/matplotlib")
        self.assertTrue(settings["dashboard"]["enabled"])
        self.assertEqual(settings["dashboard"]["show_latest_epochs"], 5)

        try:
            import torch
        except Exception as exc:  # pragma: no cover - local dependency guard
            self.skipTest(f"torch is not importable: {exc}")

        self.assertEqual(str(select_torch_device(torch, settings["device"])), "cpu")

        config = load_config(RELEASE_ROOT / "config.json")
        self.assertTrue(dashboard_enabled(config))
        self.assertEqual(dashboard_path(config), RELEASE_ROOT / "results" / "dashboard.html")
        self.assertEqual(dashboard_settings(config)["refresh_seconds"], 5)

    def test_architecture_parameter_counts_match_paper(self) -> None:
        total, trainable = parameter_counts_for_freeze(setb_size=8, freeze_mode="no_freeze")
        _, head_only = parameter_counts_for_freeze(setb_size=8, freeze_mode="freeze_cnn")
        _, cnn_only = parameter_counts_for_freeze(setb_size=8, freeze_mode="freeze_head")
        _, none_trainable = parameter_counts_for_freeze(setb_size=8, freeze_mode="freeze_all")

        self.assertEqual(total, 82096)
        self.assertEqual(trainable, 82096)
        self.assertEqual(cnn_only, 4784)
        self.assertEqual(head_only, 77312)
        self.assertEqual(none_trainable, 0)

    def test_blockage_rebuilds_setb_features_and_labels(self) -> None:
        setb_tx = build_setb_tx_indices(1)
        blocked_tx = setb_tx[0]
        replacement_tx = setb_tx[1]
        clean_rsrp = np.full((2, 32), -40.0, dtype=np.float32)
        clean_rsrp[:, blocked_tx] = 10.0
        clean_rsrp[:, replacement_tx] = 5.0

        clean_features = build_setb_features_from_rsrp(clean_rsrp, setb_tx)
        clean_labels = clean_rsrp.argmax(axis=1).astype(np.int64)
        features, labels, blocked_rsrp = build_blocked_dataset_views(
            clean_features=clean_features,
            clean_labels=clean_labels,
            clean_rsrp=clean_rsrp,
            setb_pattern=1,
            blocked_setb_col=0,
            global_min_db=-100.0,
            blockage_pct=100,
        )

        np.testing.assert_array_equal(clean_labels, np.asarray([blocked_tx, blocked_tx]))
        np.testing.assert_array_equal(labels, np.asarray([replacement_tx, replacement_tx]))
        np.testing.assert_allclose(blocked_rsrp[:, blocked_tx], -100.0)
        np.testing.assert_allclose(features[:, 0], -100.0)
        np.testing.assert_allclose(features[:, 1], 5.0)

    def test_ranked_beam_kpis_use_real_rsrp_values(self) -> None:
        kpi = KpiConfig(topks=(1, 2), margins_db=(0,), primary_topk=2, primary_margin_db=0)
        rsrp = np.zeros((2, 32), dtype=np.float32)
        rsrp[0, 3] = 10.0
        rsrp[1, 5] = 10.0
        rankings = np.vstack([full_ranking(3), full_ranking(0, 5)])

        metrics = evaluate_ranked_beams(rankings, rsrp, kpi=kpi)

        self.assertAlmostEqual(metrics["top1_%"], 50.0)
        self.assertAlmostEqual(metrics["top2_m0db_%"], 100.0)

    def test_model_forward_shape_and_run_name_parsing(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - local dependency guard
            self.skipTest(f"torch is not importable: {exc}")

        model = CNNBeamPredictor(setb_size=8)
        logits = model(torch.randn(4, 8))
        self.assertEqual(tuple(logits.shape), (4, 32))

        config = load_config(RELEASE_ROOT / "config.json")
        seed, pattern, blocked_beam = parse_run_name_parts(config, "blocking_v5_seed123_P1_blockB1")
        self.assertEqual((seed, pattern, blocked_beam), (123, 1, 1))

        job = make_batch_job(2, 7)
        self.assertEqual((job.pattern, job.blocked_beam_index, job.description), (2, 7, "P2 block B7"))


if __name__ == "__main__":
    unittest.main()
