# Blocking V5 Project Overview

Paper title: 6G Spatial Beam Prediction Under Sparse Beam-Level Blockage.

## Project Goal

Blocking V5 studies how sparse beam-level blockage affects beam prediction and
how much fine-tuning is needed to recover performance. The code compares a
clean-trained baseline (`BL-AG`), blockage-aware baselines (`BL-AW`), transfer
fine-tuning (`FT`), and deterministic non-learning baselines on two Set-B beam
patterns.

The package is self-contained: the default dataset tensor, code,
configuration, selected plots, curated results, and reusable baseline weights
are included here.

Persistent release identifier:

```text
https://doi.org/10.5281/zenodo.20128179
```

## Experiment Protocol

The default protocol is defined in `config.json`.

- Seeds: `123`, `456`, and `789`.
- Patterns: Set-B pattern 1 and Set-B pattern 2.
- Blocked beams: pattern 1 uses `B1`, `B3`, `B4`, `B6`, `B7`; pattern 2 uses
  `B0`, `B2`, `B3`, `B5`, `B7`.
- Blockage levels: `0`, `20`, `40`, `60`, `80`, and `100` percent.
- Methods: `BL-AG`, `BL-AW`, `FT`, `MAX-SETB`, and `NN-ANGLE`.
- Primary KPI: `test_top3_m1db_%`, meaning Top-3 within 1 dB test accuracy.
- Default run stages: `prepare`, `baseline`, `finetune`, and `aggregate`.

For each run, the code loads the full response cube, maps the selected Set-B
beam pattern into the original Set-A beam grid, applies blockage in the
underlying full transmitter space, rebuilds labels and Set-B features from the
blocked cube, trains/evaluates the configured methods, and aggregates results.

## Dataset And Splits

The default input tensor is:

```text
VIV0R4_116_00.npy
```

The default config points to it with:

```json
"values_cube_path": "./VIV0R4_116_00.npy"
```

Each run writes deterministic split indices under:

```text
results/<run_name>/data/train_idx.npy
results/<run_name>/data/val_idx.npy
results/<run_name>/data/test_idx.npy
results/<run_name>/data/meta.json
```

The curated release results include those split indices and metadata, but omit
the regenerated prepared arrays `X_setb.npy`, `y_tx.npy`, and `rsrp_tx.npy` to
avoid shipping duplicate data. Those arrays are rebuilt by the `prepare` stage
from the packaged input tensor.

## Methods In The Code

- `BL-AG`: clean-trained baseline. It is trained at 0% blockage and evaluated
  across blockage levels.
- `BL-AW`: blockage-aware baseline. It trains separate blockage-aware endpoints
  and is useful as an upper reference for adaptation.
- `FT`: fine-tunes from the clean BL-AG checkpoint using selected fractions,
  freeze modes, blockage levels, and epoch budgets.
- `MAX-SETB`: deterministic non-learning baseline using the strongest visible
  Set-B response.
- `NN-ANGLE`: deterministic angle-nearest baseline.

Stored BL-AG clean pretrain weights are saved for every run at:

```text
results/<run_name>/stored_weights/baseline_pretrained/BL-AG_clean_pretrain_weights.pt
```

These are the intended starting weights for FT-only researcher reruns.

## Code Structure

- `run.py`: main experiment CLI.
- `plots.py`: only plotting CLI used by the release.
- `make_quick_verify_config.py`: creates a small FT-only verification config
  from the packaged BL-AG weights.
- `experiment/config.py`: config loading and validation helpers.
- `experiment/data.py`: dataset loading, Set-B feature construction, and split
  handling.
- `experiment/model.py`: CNN model definition.
- `experiment/training.py`: training loop, device selection, and metric logging.
- `experiment/stages.py`: prepare, baseline, fine-tuning, and aggregate stages.
- `experiment/preflight.py`: dependency checks before training starts.
- `runner/`: batch planning, run-name parsing, progress snapshots, dashboard
  handoff, and postprocessing.
- `dashboard/`: optional local HTML monitor.
- `plot_code/`: publication plotting backend, style config, filters, chart
  helpers, source-data loading, and plot organization.
- `tests/`: small no-training checks for package integrity, plotting behavior,
  dashboard helpers, and preflight behavior.

`core.py` re-exports the experiment pipeline for scripts that expect a flat
entrypoint. The split `experiment.*` modules are the clearest source files for
inspection and extension.

## Packaged Results

The package includes a curated `results/` tree. This tree is meant for result
inspection, selected-plot regeneration, and quick FT verification.

### Included Under `results/`

- run-level `results.csv`, `config.json`, `aggregate_done.json`, and
  `setb_mapping.csv` for all 30 runs;
- deterministic split indices and split metadata for all 30 runs;
- per-job `rows.csv` files for result audit;
- per-job `history.csv` files for convergence plots;
- stored BL-AG and BL-AW baseline weights with manifests.

Stored BL-AG weights are included specifically so a researcher can launch a
quick FT-only check without retraining the clean baseline. The default helper
uses the paper-selected FT setting used in the selected plot bundle: seed `123`,
Pattern 1, blocked beam `B1`, 100% blockage, 1% FT data, `freeze_cnn`, and 10
FT epochs.

```bash
python3 -B make_quick_verify_config.py
python3 -B run.py \
  --config quick_verify_configs/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10.json \
  --run-name blocking_v5_seed123_P1_blockB1 \
  --no-dashboard
```

The helper writes setting-specific outputs under `quick_verify_configs/` and
`quick_verify_results/`, and it refuses to overwrite an existing verification
folder unless `--overwrite` is passed.

## Plotting

Generated plots live under `plots/`, while plotting source code lives under
`plot_code/`.

Selected manuscript plots are already included in:

```text
plots/selected_plots/
```

The selected bundle contains 61 PNG files, 10 source-data CSVs, and one
selected-plot manifest CSV. Comparison summaries are generated as bar charts.
The multi-panel beam-by-blockage heatmap is included as:

```text
plots/selected_plots/fig17_heatmap/beam_blockage_heatmap.png
```

Plot style is controlled by:

```text
plot_code/plot_config.json
```

That config covers colors, figure size, DPI, axes, legends, title rendering,
and per-plot overrides.

## Result Summary

Primary KPI: `test_top3_m1db_%`.

| Result slice | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| BL-AG, 0% blockage | 98.47% | 98.30% | 98.65% |
| BL-AG, 100% blockage | 65.53% | 40.87% | 81.87% |
| FT freeze-CNN, 1% data, 10 epochs, 100% blockage | 93.14% | 89.43% | 95.72% |
| BL-AW, 100% blockage | 97.65% | 96.75% | 98.27% |

These numbers are computed from the packaged `results/*/results.csv` files.

## Publication Notes

The manuscript availability statement should cite the exact repository, release
tag, and persistent identifier used for distribution:

- DOI: `10.5281/zenodo.20128179`
- GitHub release: `https://github.com/maheenfs/blocking-v5-code-release/releases/tag/v1.0.0`
- Repository: `https://github.com/maheenfs/blocking-v5-code-release`
