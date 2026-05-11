"""Experiment-stage orchestration for prepare, baselines, fine-tuning, and aggregation."""

from .pipeline import (
    aggregate_stage,
    artifact_dir_for_result_row,
    enrich_result_row_metadata,
    export_baseline_checkpoints,
    freeze_mode_for_result_row,
    make_results_row,
    report_stage,
    results_row_fieldnames,
    run_baseline_stage,
    run_finetune_stage,
    run_single_experiment,
    write_run_config_snapshot,
    write_setb_mapping,
)

__all__ = [
    "aggregate_stage",
    "artifact_dir_for_result_row",
    "enrich_result_row_metadata",
    "export_baseline_checkpoints",
    "freeze_mode_for_result_row",
    "make_results_row",
    "report_stage",
    "results_row_fieldnames",
    "run_baseline_stage",
    "run_finetune_stage",
    "run_single_experiment",
    "write_run_config_snapshot",
    "write_setb_mapping",
]
