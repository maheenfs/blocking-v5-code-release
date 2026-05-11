# Experiment Package

`experiment/` contains the training and evaluation implementation. The modules
are split by responsibility so the data pipeline, model, metrics, preflight
checks, and stages can be inspected separately.

## Module Map

- `config.py`: dataclasses, release-local path resolution, seed expansion, and
  `config.json` loading.
- `checkpoints.py`: accepted BL-AG checkpoint paths for FT-only runs.
- `io.py`: atomic file writes, CSV helpers, reproducibility seeding, formatting,
  and small path utilities.
- `geometry.py`: Set-A/Set-B beam angles, Set-B to Set-A transmitter mapping,
  and provenance table rows.
- `data.py`: value-cube loading, Set-B feature building, deterministic split
  creation, blockage application, and prepared-data loading.
- `model.py`: 1D CNN architecture, freeze-mode application, parameter counts,
  and checkpoint loading.
- `metrics.py`: KPI key construction, model KPI evaluation, and non-ML ranking
  KPI evaluation.
- `names.py`: shared folder/file naming helpers used by training, preflight,
  and dashboard artifact checks.
- `non_ml.py`: deterministic non-learning ranking baselines and their stage
  runner.
- `preflight.py`: static config checks and stage-dependency checks that stop
  missing-data or missing-weight cases before training starts.
- `run_plan.py`: optional runner postprocessing settings such as selected-only
  plotting after the full batch.
- `training.py`: training loop, epoch records, budget metrics, history files,
  and train-result summaries.
- `stages.py`: prepare, baseline, fine-tune, aggregate, and single-experiment
  orchestration.
- `system.py`: local CPU/GPU/cache settings loaded from `system_settings.json`.
  These settings affect runtime placement only, not the paper protocol.
- `pipeline.py`: shared implementation used by the modules above.

## Experiment Flow

1. `prepare_stage` loads `VIV0R4_116_00.npy`, maps Set-B beams to Set-A
   transmitter columns, builds Set-B inputs, computes clean labels, and writes
   deterministic train/validation/test split arrays.
2. `run_baseline_stage` trains BL-AG on the clean prepared data, evaluates it
   under each blockage level, trains BL-AW inside each blocked world, and writes
   baseline rows. It also exports clearly named BL-AG and BL-AW checkpoints
   under `stored_weights/baseline_pretrained/`.
3. `run_finetune_stage` loads the clean BL-AG starting weight, applies the
   configured blockage level to the selected FT training subset, then sweeps
   train fractions, freeze modes, and epoch budgets. In a full local run, the
   starting weight is the baseline job checkpoint. In this release package,
   quick verification uses the stored BL-AG weight under
   `stored_weights/baseline_pretrained/`.
4. `aggregate_stage` collects stage rows into run-level `results.csv` and
   attaches artifact metadata used by plot_code.

## FT-Only Runs

FT can be checked without retraining BL-AG or BL-AW because the release includes
clean BL-AG weights under:

```text
results/<run_name>/stored_weights/baseline_pretrained/BL-AG_clean_pretrain_weights.pt
```

### Recommended Release Path

Use `make_quick_verify_config.py`. It creates a small config with these stages:

```json
["prepare", "finetune", "aggregate"]
```

This path rebuilds the prepared arrays from `VIV0R4_116_00.npy`, copies the
stored BL-AG weight, runs one FT verification job, and writes results under
`quick_verify_results/`.

### Advanced Resume Path

Use stages `["finetune", "aggregate"]` only when the run folder already has
both prepared data and a BL-AG checkpoint. The preflight check stops before
training if either requirement is missing.

The `stored_weights/baseline_pretrained/manifest.csv` file marks
`BL-AG_clean_pretrain` as `use_for_ft_start=yes`. BL-AW checkpoints are saved
as blockage-aware baseline endpoints and are marked `use_for_ft_start=no`.
