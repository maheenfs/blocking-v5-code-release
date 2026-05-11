# Blocking V5 Code Release

This folder contains the code, data tensor, selected figures, curated result
tables, split metadata, and reusable baseline weights for the Blocking V5
beam-prediction experiments.

## What Is Included

### Code And Configuration

- `run.py`: starts experiments.
- `plots.py`: creates plots.
- `config.json`: paper experiment settings.
- `system_settings.json`: machine settings such as CPU/GPU choice.
- `experiment/`, `runner/`, `dashboard/`, and `plot_code/`: source code.
- `tests/`: small checks that do not train models.

### Dataset

- `VIV0R4_116_00.npy`: the packaged beam-response tensor used by the default
  config.

### Curated Results For Inspection And Plotting

The release includes the completed paper result tree under `results/`. This is
enough to inspect the reported runs and regenerate selected plots.

- `30` completed seed/pattern/blocked-beam run folders.
- `30` run-level `results.csv` files.
- `6000` per-job `rows.csv` files.
- `3360` training-history CSV files for convergence and audit plots.
- deterministic split metadata for all 30 runs.

### Stored Weights For Quick Verification

The release includes reusable baseline weights:

- `30` clean BL-AG pretrain checkpoints for FT reuse.
- `180` BL-AW blockage-aware checkpoints for audit/reference.

BL-AG weights are stored here:

```text
results/<run_name>/stored_weights/baseline_pretrained/BL-AG_clean_pretrain_weights.pt
```

### Selected Manuscript Plots

The release includes the selected manuscript plot bundle:

- `61` selected PNG plots.
- `10` selected source-data CSV files.
- `selected_plots_manifest.csv`, which records the selected plot plan.

The plot bundle is here:

```text
plots/selected_plots/
```

## Packaged Result Summary

Primary KPI: `test_top3_m1db_%`, meaning Top-3 within 1 dB test accuracy.

| Result slice | Mean primary KPI |
| --- | ---: |
| BL-AG, 0% blockage | 98.47% |
| BL-AG, 100% blockage | 65.53% |
| FT freeze-CNN, 1% data, 10 epochs, 100% blockage | 93.14% |
| BL-AW, 100% blockage | 97.65% |

## Quick Start

### 1. Install And Check The Package

Run from inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -B -m unittest discover tests
```

The test command checks the package setup. It does not train models and does
not change scientific results.

### 2. Verify Selected Plots Without Training

This is the fastest verification path. It uses the packaged `results/` tree and
writes new plots to a separate verification folder:

```bash
python3 -B plots.py selected \
  --results-root results \
  --config config.json \
  --selected-root plots/verification_selected_plots_paper_packaged
```

Use this command when the goal is to confirm that the packaged results can
recreate the selected manuscript plots.

### 3. Run One Quick FT Training Check

This check does not run the full paper pipeline. It uses the packaged BL-AG
weight and runs one FT job with the paper-selected verification setting:

- seed `123`;
- Pattern 1;
- blocked beam `B1`;
- 100% blockage;
- 1% FT data;
- `freeze_cnn`;
- 10 FT epochs.

Run:

```bash
python3 -B make_quick_verify_config.py
python3 -B run.py \
  --config quick_verify_configs/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10.json \
  --run-name blocking_v5_seed123_P1_blockB1 \
  --no-dashboard
```

The result is written here:

```text
quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1/results.csv
```

This quick run writes results only. It does not create selected manuscript
plots, because selected manuscript plots require the complete 30-run packaged
`results/` tree.

The helper creates separate folders for each verification setting and refuses
to overwrite an existing verification output unless `--overwrite` is passed.

### 4. Choose Plots For The Quick FT Check

If no plots are needed, stop after the quick FT command above. The result CSV is
the verification output.

To create common quick-check plots after the run finishes:

```bash
python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families blockage,epochs,loss
```

Quick-check plots are written here:

```text
plots/run_plots/blocking_v5_seed123_P1_blockB1/
```

To create only some plots, change `--families`:

```bash
# Accuracy at the configured blockage setting
python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families blockage

# Accuracy for the configured epoch setting
python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families epochs

# Training-loss bars by epoch
python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families loss
```

### 5. Know When A Full Run Is Needed

A full run is not needed for basic verification or selected-plot regeneration.
Run the full pipeline only to reproduce every model from scratch:

```bash
python3 -B run.py --dashboard
```

The default full run writes to `results/`, where the packaged paper results
already live. For a new independent rerun, copy `config.json`, change `outdir`
to a new folder, and run with that copied config.

## Output Safety

- Do not use `--overwrite` unless replacing an existing result is intentional.
- Use `--selected-root` when checking selected plots so the curated
  `plots/selected_plots/` bundle is not replaced.
- Use the quick verification helper for FT checks because it creates distinct
  `quick_verify_configs/` and `quick_verify_results/` folders.

## Documentation

This is the only README in the package. Start here, then open the linked guides
only when more detail is needed.

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md): detailed project, method, result,
  and source-code overview.
- [RUN_COMMANDS.md](RUN_COMMANDS.md): CLI options and common run/plot command
  combinations.
- [model_card.md](model_card.md): model scope, metrics, baselines, and
  reproducibility notes.
- [experiment/EXPERIMENT_GUIDE.md](experiment/EXPERIMENT_GUIDE.md): experiment
  module map.
- [plot_code/PLOTTING_CODE_GUIDE.md](plot_code/PLOTTING_CODE_GUIDE.md): plotting
  backend and style config.

For hardware changes, edit `system_settings.json`. For scientific changes,
edit `config.json` and record the change because it changes the paper protocol.
