"""Pre-run checks for Blocking V5 release configurations.

The checks here run before any training starts.  They make configuration
mistakes and missing stage dependencies fail with clear, actionable messages
instead of surfacing later as obscure file or empty-result errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .checkpoints import bl_ag_checkpoint_candidates, find_bl_ag_checkpoint
from .names import slugify
from .run_plan import postprocessing_plan, validate_postprocessing_plan


ALLOWED_STAGES = {"prepare", "baseline", "finetune", "aggregate"}
ALLOWED_EXPERIMENTS = {"BL-AG", "BL-AW", "FT", "MAX-SETB", "NN-ANGLE", "RANDOM-SETB"}
ALLOWED_FREEZE_MODES = {"no_freeze", "freeze_cnn", "freeze_head", "freeze_all"}
ALLOWED_FT_SAMPLING = {"with_replacement", "without_replacement"}
BASELINE_METHODS = {"BL-AG", "BL-AW", "MAX-SETB", "NN-ANGLE", "RANDOM-SETB"}
SELECTED_PLOT_REQUIRED_EXPERIMENTS = {"BL-AG", "BL-AW", "FT", "MAX-SETB", "NN-ANGLE"}
SELECTED_PLOT_REQUIRED_BLOCKAGES = {0, 20, 40, 60, 80, 100}

PREPARED_DATA_FILES = (
    "X_setb.npy",
    "y_tx.npy",
    "rsrp_tx.npy",
    "train_idx.npy",
    "val_idx.npy",
    "test_idx.npy",
    "meta.json",
)


class ConfigSelectionError(ValueError):
    """Raised when a config asks for an unsupported or inconsistent selection."""


class MissingDependencyError(FileNotFoundError):
    """Raised when selected stages need files that are not present yet."""


def _missing_files(directory: Path, names: Iterable[str]) -> list[Path]:
    return [directory / name for name in names if not (directory / name).exists()]


def _has_prepared_data(run_dir: Path) -> bool:
    return not _missing_files(run_dir / "data", PREPARED_DATA_FILES)


def _ft_branch_rows(config: object, run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    training = getattr(config, "training")
    for train_frac in getattr(training, "train_fracs_sweep", ()) or ():
        train_frac_tag = slugify(f"trainfrac_{float(train_frac):.6f}")
        for blockage_pct in getattr(config, "blockage_levels", ()) or ():
            for freeze_mode in getattr(training, "ft_freeze_modes", ()) or ():
                paths.append(
                    run_dir
                    / "jobs"
                    / "finetune"
                    / train_frac_tag
                    / f"blockage_{int(blockage_pct):03d}"
                    / f"freeze_{slugify(str(freeze_mode))}"
                    / "rows.csv"
                )
    return paths


def _selected_plot_required_jobs(config: object) -> set[tuple[int, int]]:
    """Return pattern/beam jobs needed for the publication selected bundle."""
    jobs: set[tuple[int, int]] = set()
    provenance = dict(getattr(config, "scenario_provenance", {}) or {})

    within_pattern = dict(provenance.get("within_pattern", {}) or {})
    for key, pattern in (("P1", 1), ("P2", 2)):
        jobs.update((pattern, int(beam)) for beam in within_pattern.get(key, []) or [])

    comparisons = dict(provenance.get("pattern_comparisons", {}) or {})
    for item in comparisons.values():
        jobs.add((1, int(item["p1_beam"])))
        jobs.add((2, int(item["p2_beam"])))

    representatives = dict(getattr(config, "representative_beams", {}) or {})
    if "P1" in representatives:
        jobs.add((1, int(representatives["P1"])))
    if "P2" in representatives:
        jobs.add((2, int(representatives["P2"])))
    return jobs


def _validate_selected_plot_config(
    config: object,
    *,
    experiments: set[str],
    blockages: set[int],
    train_fracs: tuple[float, ...],
    ft_epochs: tuple[int, ...],
    freeze_modes: tuple[str, ...],
) -> list[str]:
    """Check that runner selected-only plotting has the data it needs."""
    if "selected" not in postprocessing_plan(config).after_batch:
        return []

    issues: list[str] = []
    missing_experiments = sorted(SELECTED_PLOT_REQUIRED_EXPERIMENTS - experiments)
    if missing_experiments:
        issues.append(
            "selected paper plots need these experiments for the full publication bundle: "
            f"{', '.join(missing_experiments)}."
        )

    missing_blockages = sorted(SELECTED_PLOT_REQUIRED_BLOCKAGES - blockages)
    if missing_blockages:
        issues.append(
            "selected paper plots need blockage levels "
            f"{', '.join(str(value) for value in sorted(SELECTED_PLOT_REQUIRED_BLOCKAGES))}; "
            f"missing {', '.join(str(value) for value in missing_blockages)}."
        )

    plotting = dict(getattr(config, "plotting", {}) or {})
    primary_train_frac = float(plotting.get("comparison_primary_train_frac", 0.01))
    if not any(abs(value - primary_train_frac) <= 1e-8 for value in train_fracs):
        issues.append(
            "selected paper plots need training.train_fracs_sweep to include "
            f"comparison_primary_train_frac={primary_train_frac:g}."
        )

    baseline_epoch = int(plotting.get("baseline_ft_epoch_for_plots", 10))
    if baseline_epoch not in ft_epochs:
        issues.append(
            "selected paper plots need training.ft_epoch_sweep to include "
            f"baseline_ft_epoch_for_plots={baseline_epoch}."
        )
    if "freeze_cnn" not in freeze_modes:
        issues.append("selected paper plots need training.ft_freeze_modes to include freeze_cnn.")

    configured_jobs = {
        (int(getattr(job, "pattern")), int(getattr(job, "blocked_beam_index")))
        for job in getattr(config, "batch_jobs", ()) or ()
    }
    missing_jobs = sorted(_selected_plot_required_jobs(config) - configured_jobs)
    if missing_jobs:
        job_text = ", ".join(f"P{pattern} B{beam}" for pattern, beam in missing_jobs)
        issues.append(f"selected paper plots need these batch jobs: {job_text}.")
    return issues


def validate_config_for_release(config: object) -> None:
    """Validate static config selections before any runtime work starts."""
    issues: list[str] = []

    stages = tuple(str(value) for value in getattr(config, "stages", ()))
    experiments = tuple(str(value) for value in getattr(config, "experiments", ()))
    experiment_set = set(experiments)
    training = getattr(config, "training")
    kpi = getattr(config, "kpi")
    split = getattr(config, "split")

    if not stages:
        issues.append("stages is empty; choose at least one of prepare, baseline, finetune, aggregate.")
    unknown_stages = sorted(set(stages) - ALLOWED_STAGES)
    if unknown_stages:
        issues.append(f"unknown stages: {', '.join(unknown_stages)}.")

    if not experiments:
        issues.append("experiments is empty; choose at least one method to produce result rows.")
    unknown_experiments = sorted(experiment_set - ALLOWED_EXPERIMENTS)
    if unknown_experiments:
        issues.append(f"unknown experiments: {', '.join(unknown_experiments)}.")

    if "finetune" in stages and "FT" not in experiments:
        issues.append("stage 'finetune' is selected, but experiment 'FT' is not selected; add FT or remove finetune.")

    blockages = tuple(int(value) for value in getattr(config, "blockage_levels", ()))
    blockage_set = set(blockages)
    if not blockages:
        issues.append("blockage_levels is empty.")
    invalid_blockages = [value for value in blockages if value < 0 or value > 100]
    if invalid_blockages:
        issues.append(f"blockage_levels must stay between 0 and 100; invalid values: {invalid_blockages}.")

    split_values = (float(split.train_frac), float(split.val_frac), float(split.test_frac))
    if any(value <= 0.0 for value in split_values):
        issues.append("split fractions must all be positive.")
    if abs(sum(split_values) - 1.0) > 1e-6:
        issues.append(f"split fractions must sum to 1.0; got {sum(split_values):.6f}.")

    if int(training.batch_size) <= 0:
        issues.append("training.batch_size must be positive.")
    if float(training.lr) <= 0.0:
        issues.append("training.lr must be positive.")
    if float(training.weight_decay) < 0.0:
        issues.append("training.weight_decay cannot be negative.")
    if float(training.kpi_loss_weight) < 0.0:
        issues.append("training.kpi_loss_weight cannot be negative.")

    if ("BL-AG" in experiments or "FT" in experiments) and int(training.pretrain_epochs) <= 0:
        issues.append("training.pretrain_epochs must be positive when BL-AG or FT is selected.")
    if "BL-AW" in experiments and int(training.aware_epochs) <= 0:
        issues.append("training.aware_epochs must be positive when BL-AW is selected.")

    ft_epochs = tuple(int(value) for value in getattr(training, "ft_epoch_sweep", ()))
    train_fracs = tuple(float(value) for value in getattr(training, "train_fracs_sweep", ()))
    freeze_modes = tuple(str(value) for value in getattr(training, "ft_freeze_modes", ()))
    if "FT" in experiments:
        if not ft_epochs:
            issues.append("training.ft_epoch_sweep is required when FT is selected.")
        if any(value <= 0 for value in ft_epochs):
            issues.append("training.ft_epoch_sweep values must be positive.")
        if not train_fracs:
            issues.append("training.train_fracs_sweep is required when FT is selected.")
        invalid_fracs = [value for value in train_fracs if value < 0.0 or value > 1.0]
        if invalid_fracs:
            issues.append(f"training.train_fracs_sweep values must stay between 0 and 1; invalid values: {invalid_fracs}.")
        if not freeze_modes:
            issues.append("training.ft_freeze_modes is required when FT is selected.")
    unknown_freeze = sorted(set(freeze_modes) - ALLOWED_FREEZE_MODES)
    if unknown_freeze:
        issues.append(f"unknown FT freeze modes: {', '.join(unknown_freeze)}.")
    if str(training.ft_sampling) not in ALLOWED_FT_SAMPLING:
        issues.append(
            f"training.ft_sampling must be one of {', '.join(sorted(ALLOWED_FT_SAMPLING))}; "
            f"got {training.ft_sampling!r}."
        )

    topks = tuple(int(value) for value in getattr(kpi, "topks", ()))
    margins = tuple(int(value) for value in getattr(kpi, "margins_db", ()))
    if not topks:
        issues.append("kpi.topks is empty.")
    if any(value < 1 or value > 32 for value in topks):
        issues.append("kpi.topks values must be between 1 and 32.")
    if not margins:
        issues.append("kpi.margins_db is empty.")
    if any(value < 0 for value in margins):
        issues.append("kpi.margins_db values cannot be negative.")
    if int(kpi.primary_topk) not in set(topks):
        issues.append("kpi.primary_topk must be included in kpi.topks.")
    if int(kpi.primary_margin_db) not in set(margins):
        issues.append("kpi.primary_margin_db must be included in kpi.margins_db.")

    jobs = tuple(getattr(config, "batch_jobs", ()) or ())
    if not jobs:
        issues.append("batch_jobs is empty; define at least one pattern/blocked-beam job.")
    for job in jobs:
        pattern = int(getattr(job, "pattern"))
        beam = int(getattr(job, "blocked_beam_index"))
        if pattern not in {1, 2}:
            issues.append(f"batch job {getattr(job, 'description', '')!r} has unsupported pattern {pattern}; use 1 or 2.")
        if beam < 0 or beam > 7:
            issues.append(f"batch job {getattr(job, 'description', '')!r} has blocked_beam_index {beam}; use 0 through 7.")

    issues.extend(validate_postprocessing_plan(config))
    issues.extend(
        _validate_selected_plot_config(
            config,
            experiments=experiment_set,
            blockages=blockage_set,
            train_fracs=train_fracs,
            ft_epochs=ft_epochs,
            freeze_modes=freeze_modes,
        )
    )

    if issues:
        bullet_list = "\n".join(f"- {issue}" for issue in issues)
        raise ConfigSelectionError(f"Invalid Blocking V5 release config:\n{bullet_list}")


def validate_run_dependencies(config: object, job: object, *, overwrite: bool = False) -> None:
    """Validate selected stage dependencies for one run directory."""
    stages = set(str(value) for value in getattr(config, "stages", ()))
    experiments = set(str(value) for value in getattr(config, "experiments", ()))
    run_dir = config.run_dir(job)
    issues: list[str] = []
    prepared_ok = _has_prepared_data(run_dir)
    prepared_required = bool({"baseline", "finetune"} & stages)

    if "prepare" in stages and (overwrite or not prepared_ok) and not Path(getattr(config, "values_cube_path")).exists():
        issues.append(
            "prepare needs the packaged values cube, but it is missing at "
            f"{getattr(config, 'values_cube_path')}. Restore VIV0R4_116_00.npy or update values_cube_path."
        )
    if prepared_required and "prepare" not in stages and not prepared_ok:
        issues.append(
            "prepared data is missing. Add stage 'prepare' for this run, or restore the existing data/ folder "
            f"under {run_dir}."
        )

    baseline_selected = any(method in experiments for method in BASELINE_METHODS)
    if baseline_selected and "baseline" not in stages and "aggregate" in stages:
        missing_baseline: list[str] = []
        if "BL-AG" in experiments and not (run_dir / "jobs" / "baseline" / "bl_ag_eval" / "rows.csv").exists():
            missing_baseline.append("BL-AG evaluation rows")
        if "BL-AW" in experiments:
            missing_aw = [
                run_dir / "jobs" / "baseline" / f"bl_aw_blockage_{int(blockage):03d}" / "rows.csv"
                for blockage in getattr(config, "blockage_levels", ()) or ()
                if not (run_dir / "jobs" / "baseline" / f"bl_aw_blockage_{int(blockage):03d}" / "rows.csv").exists()
            ]
            if missing_aw:
                missing_baseline.append(f"BL-AW rows for {len(missing_aw)} blockage levels")
        non_ml_methods = sorted(experiments & (BASELINE_METHODS - {"BL-AG", "BL-AW"}))
        if non_ml_methods and not (run_dir / "jobs" / "baseline" / "non_ml_baselines" / "rows.csv").exists():
            missing_baseline.append(f"non-ML baseline rows for {', '.join(non_ml_methods)}")
        if missing_baseline:
            issues.append(
                "aggregate is selected without baseline, but selected baseline-family results are missing: "
                f"{'; '.join(missing_baseline)}. Add stage 'baseline', restore these rows, or remove those experiments."
            )

    if "FT" in experiments:
        weights = find_bl_ag_checkpoint(run_dir)
        if "finetune" in stages and "baseline" not in stages and weights is None:
            candidates = ", ".join(str(path) for path in bl_ag_checkpoint_candidates(run_dir))
            issues.append(
                "FT-only runs need stored BL-AG weights at one of these paths: "
                f"{candidates}. Add stage 'baseline' once, or restore the saved BL-AG checkpoint."
            )
        if "finetune" not in stages and "aggregate" in stages:
            missing_ft_rows = [path for path in _ft_branch_rows(config, run_dir) if not path.exists()]
            if missing_ft_rows:
                issues.append(
                    "aggregate is selected for FT without finetune, but FT branch rows are missing "
                    f"({len(missing_ft_rows)} missing). Add stage 'finetune' or restore the finetune rows."
                )

    if "aggregate" in stages and "baseline" not in stages and "finetune" not in stages:
        existing_rows = list((run_dir / "jobs").rglob("rows.csv")) if (run_dir / "jobs").exists() else []
        if not existing_rows:
            issues.append(
                "aggregate-only runs need existing jobs/**/rows.csv files. Add baseline/finetune stages first, "
                "or restore a previous run directory."
            )

    if issues:
        job_name = getattr(config, "run_name")(pattern=getattr(job, "pattern"), blocked_beam_index=getattr(job, "blocked_beam_index"))
        bullet_list = "\n".join(f"- {issue}" for issue in issues)
        raise MissingDependencyError(f"Cannot run selected Blocking V5 stages for {job_name}:\n{bullet_list}")
