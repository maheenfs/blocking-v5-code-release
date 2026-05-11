"""Training loop, epoch history, and model-result persistence helpers."""

from .pipeline import (
    EpochRecord,
    TrainResult,
    compute_budget_metrics,
    evaluate_model_on_val_test,
    optional_float,
    optional_int,
    read_train_metadata,
    save_epoch_history,
    save_train_result,
    sum_history_epoch_time,
    train_model,
)

__all__ = [
    "EpochRecord",
    "TrainResult",
    "compute_budget_metrics",
    "evaluate_model_on_val_test",
    "optional_float",
    "optional_int",
    "read_train_metadata",
    "save_epoch_history",
    "save_train_result",
    "sum_history_epoch_time",
    "train_model",
]
