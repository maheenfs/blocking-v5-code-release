"""Checkpoint path helpers shared by training, preflight, and dashboard code."""

from __future__ import annotations

from pathlib import Path


BL_AG_STAGE_CHECKPOINT = Path("jobs") / "baseline" / "bl_ag_train" / "weights_best.pt"
BL_AG_STORED_CHECKPOINT = Path("stored_weights") / "baseline_pretrained" / "BL-AG_clean_pretrain_weights.pt"


def bl_ag_checkpoint_candidates(run_dir: Path) -> tuple[Path, Path]:
    """Return the accepted BL-AG checkpoint paths for FT-only runs."""
    return (run_dir / BL_AG_STAGE_CHECKPOINT, run_dir / BL_AG_STORED_CHECKPOINT)


def find_bl_ag_checkpoint(run_dir: Path) -> Path | None:
    """Find the BL-AG checkpoint, preferring the original training artifact."""
    for path in bl_ag_checkpoint_candidates(run_dir):
        if path.exists():
            return path
    return None
