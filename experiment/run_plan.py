"""Helpers for optional runner postprocessing choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RUN_POSTPROCESS_ALIASES = {
    "run-plots": "run_plots",
    "run_plots": "run_plots",
}

BATCH_POSTPROCESS_ALIASES = {
    "compare": "compare",
    "comparisons": "compare",
    "comparison": "compare",
    "provenance": "provenance",
    "selected": "selected",
    "paper-figures": "selected",
    "paper_figures": "selected",
}

DISABLED_TOKENS = {"", "none", "off", "false", "no", "disabled"}
DEFAULT_AFTER_EACH_RUN = ("run_plots",)
DEFAULT_AFTER_BATCH = ("compare", "provenance")


@dataclass(frozen=True)
class PostprocessingPlan:
    after_each_run: tuple[str, ...]
    after_batch: tuple[str, ...]


def _raw_steps(raw: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Iterable):
        return tuple(str(value) for value in raw)
    return (str(raw),)


def _normalize_steps(raw: object, aliases: dict[str, str], default: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for value in _raw_steps(raw, default):
        token = str(value).strip().lower().replace(" ", "-")
        if token in DISABLED_TOKENS:
            continue
        normalized = aliases.get(token, token.replace("-", "_"))
        if normalized not in values:
            values.append(normalized)
    return tuple(values)


def postprocessing_plan(config: object) -> PostprocessingPlan:
    """Read runner postprocessing choices from config.plotting."""
    plotting = dict(getattr(config, "plotting", {}) or {})
    runner_cfg = dict(plotting.get("runner_postprocessing", {}) or {})
    return PostprocessingPlan(
        after_each_run=_normalize_steps(runner_cfg.get("after_each_run"), RUN_POSTPROCESS_ALIASES, DEFAULT_AFTER_EACH_RUN),
        after_batch=_normalize_steps(runner_cfg.get("after_batch"), BATCH_POSTPROCESS_ALIASES, DEFAULT_AFTER_BATCH),
    )


def validate_postprocessing_plan(config: object) -> list[str]:
    """Return validation issues for runner postprocessing choices."""
    plan = postprocessing_plan(config)
    issues: list[str] = []
    unknown_run = sorted(set(plan.after_each_run) - set(RUN_POSTPROCESS_ALIASES.values()))
    if unknown_run:
        issues.append(f"unknown after_each_run postprocessing steps: {', '.join(unknown_run)}.")
    unknown_batch = sorted(set(plan.after_batch) - set(BATCH_POSTPROCESS_ALIASES.values()))
    if unknown_batch:
        issues.append(f"unknown after_batch postprocessing steps: {', '.join(unknown_batch)}.")
    return issues
