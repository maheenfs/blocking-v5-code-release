"""Filter helpers shared by CLI commands and plot organization."""

from __future__ import annotations

from .config import PlotSelection, normalize_beam_token, normalize_family_token


FAMILY_ALIASES = {
    "blockage": "blockage_vs_accuracy",
    "blockage-vs-accuracy": "blockage_vs_accuracy",
    "accuracy-vs-blockage": "blockage_vs_accuracy",
    "blockage-by-freeze": "blockage_vs_accuracy",
    "blag": "blockage_vs_accuracy",
    "ft-recovery": "blockage_vs_accuracy",
    "blaw": "blockage_vs_accuracy",
    "epoch": "epoch_budget_vs_accuracy",
    "epochs": "epoch_budget_vs_accuracy",
    "epoch-budget": "epoch_budget_vs_accuracy",
    "epoch-budget-vs-accuracy": "epoch_budget_vs_accuracy",
    "train-fraction": "train_fraction_vs_accuracy",
    "train-frac": "train_fraction_vs_accuracy",
    "data-efficiency": "train_fraction_vs_accuracy",
    "train-fraction-vs-accuracy": "train_fraction_vs_accuracy",
    "freeze": "freeze_mode_ablation",
    "freeze-mode": "freeze_mode_ablation",
    "freeze-mode-ablation": "freeze_mode_ablation",
    "method": "method_comparison",
    "methods": "method_comparison",
    "method-comparison": "method_comparison",
    "aggregate": "method_comparison",
    "cross-pattern": "cross_pattern_comparison",
    "cross-pattern-comparison": "cross_pattern_comparison",
    "cross-pattern-methods": "cross_pattern_comparison",
    "cross-train-fraction": "cross_pattern_train_fraction",
    "cross-pattern-train-fraction": "cross_pattern_train_fraction",
    "multi-kpi": "multi_kpi",
    "kpi": "multi_kpi",
    "non-ml": "non_ml_baselines",
    "nonml": "non_ml_baselines",
    "non-ml-baselines": "non_ml_baselines",
    "loss": "train_loss",
    "loss-curves": "train_loss",
    "train-loss": "train_loss",
    "training-loss": "train_loss",
    "train_loss": "train_loss",
    "heatmap": "heatmaps",
    "heatmaps": "heatmaps",
    "convergence": "convergence",
    "history": "convergence",
    "provenance": "provenance",
    "setup": "setup",
    "procedure": "procedure",
    "selected": "selected_paper",
    "selected-paper": "selected_paper",
}


def canonical_family(value: object) -> str:
    token = normalize_family_token(value)
    return FAMILY_ALIASES.get(token, token.replace("-", "_"))


def family_enabled(selection: PlotSelection, *families: object) -> bool:
    """Return True when a plot family should be generated for this selection."""

    if not selection.families:
        return True
    wanted = {canonical_family(item) for item in selection.families}
    return any(canonical_family(family) in wanted for family in families)


def selected_or_all(requested: tuple, available: list):
    return list(requested) if requested else list(available)


def matches_selection(metadata: dict, selection: PlotSelection) -> bool:
    """Return True when inferred plot metadata passes the requested filters."""

    if selection.kpis and metadata.get("kpi") not in set(selection.kpis):
        return False
    if selection.families:
        family = canonical_family(metadata.get("family", ""))
        wanted = {canonical_family(item) for item in selection.families}
        if family not in wanted:
            return False
    if selection.seeds:
        wanted_seeds = {str(seed) for seed in selection.seeds}
        plot_seeds = {str(seed) for seed in metadata.get("seeds", [])}
        if plot_seeds and not (plot_seeds & wanted_seeds):
            return False
    if selection.patterns:
        pattern = metadata.get("pattern")
        if pattern not in selection.patterns:
            return False
    if selection.blocked_beams:
        wanted_beams = {normalize_beam_token(beam) for beam in selection.blocked_beams}
        plot_beams = {normalize_beam_token(beam) for beam in metadata.get("blocked_beams", [])}
        if plot_beams and not (plot_beams & wanted_beams):
            return False
    return True
