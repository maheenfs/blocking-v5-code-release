# Blocking V5 Run Commands

Run all commands from inside `blocking_v5_code/`.

## Setup And Checks

Create an environment and install the pinned runtime packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the optional fast checks without starting training:

```bash
python3 -B -m unittest discover tests
```

These checks are kept in the release because they are small and give a quick
way to verify the package before starting a long experiment. They do not change
results and are not required for normal runs.

## Which Command Should Be Used?

| Goal | Use this command path |
| --- | --- |
| Check that the package imports and basic code works | `python3 -B -m unittest discover tests` |
| Recreate selected manuscript plots from packaged results | `plots.py selected --results-root results --config config.json --selected-root <new folder>` |
| Run the paper-selected FT check without plots | `make_quick_verify_config.py`, then `run.py --config <quick config>` |
| Plot the quick FT check after it finishes | `plots.py run-plots --run-dir <quick run folder> --families <families>` |
| Reproduce every model from scratch | `run.py --dashboard` with a copied config that writes to a new `outdir` |

Show CLI help:

```bash
python3 -B run.py --help
python3 -B plots.py --help
python3 -B make_quick_verify_config.py --help
python3 -B plots.py selected --help
python3 -B plots.py compare --help
python3 -B plots.py run-plots --help
```

## Main Experiment CLI

`run.py` options:

| Option | Meaning |
| --- | --- |
| `--config PATH` | Use a config file other than `config.json`. |
| `--pattern N` | Run one Set-B pattern from the config. |
| `--blocked-beam N` | Run one blocked Set-B beam index with `--pattern`. |
| `--run-name NAME` | Run or resume a packaged run such as `blocking_v5_seed123_P1_blockB1`. |
| `--overwrite` | Rebuild outputs even if files already exist. Use carefully. |
| `--dashboard` | Write the optional local HTML dashboard. |
| `--no-dashboard` | Disable dashboard output for this command. |

Run the full configured batch:

```bash
python3 -B run.py
```

This writes to the `outdir` in `config.json`. The packaged `config.json` uses
`results/`, which already contains the curated paper results.

Run the full configured batch with dashboard:

```bash
python3 -B run.py --dashboard
```

Run the full configured batch without dashboard:

```bash
python3 -B run.py --no-dashboard
```

Run one pattern/beam job:

```bash
python3 -B run.py --pattern 1 --blocked-beam 4
python3 -B run.py --pattern 2 --blocked-beam 0
```

Run or resume one packaged seed run:

```bash
python3 -B run.py --run-name blocking_v5_seed123_P1_blockB1
python3 -B run.py --run-name blocking_v5_seed456_P2_blockB7
```

Rebuild one packaged seed run:

```bash
python3 -B run.py --run-name blocking_v5_seed123_P1_blockB1 --overwrite
```

Use `--overwrite` only for an intentional rebuild that should replace existing
outputs. Without `--overwrite`, existing complete artifacts are preserved where
the runner can safely resume or skip completed work.

For a new independent rerun, copy `config.json`, change `outdir` to a new
folder such as `results_full_rerun_2026_05_11`, and run with that copied config.
Keep `--overwrite` off unless the selected output folder is meant to be
replaced.

## Stage Combinations

Stage selection is controlled in `config.json`:

```json
"stages": ["prepare", "baseline", "finetune", "aggregate"]
```

Common stage combinations:

| Stages | Use case |
| --- | --- |
| `["prepare", "baseline", "finetune", "aggregate"]` | Full paper-style run from scratch. |
| `["prepare", "finetune", "aggregate"]` | Rebuild prepared arrays, reuse stored BL-AG weights, rerun FT. |
| `["finetune", "aggregate"]` | FT-only when full prepared data and BL-AG checkpoint are already present. |
| `["aggregate"]` | Rebuild `results.csv` from existing `jobs/**/rows.csv`. |
| `["prepare"]` | Regenerate prepared arrays and split files only. |

For FT-only reruns from this release, prefer:

```json
"stages": ["prepare", "finetune", "aggregate"],
"experiments": ["FT"]
```

That command rebuilds prepared arrays from `VIV0R4_116_00.npy` and uses the
stored BL-AG clean pretrain weights. If `prepare` is removed, the run directory
must already contain `data/X_setb.npy`, `data/y_tx.npy`, and `data/rsrp_tx.npy`.
The recommended release verification path keeps `prepare` in the stage list so
those arrays are rebuilt automatically.

## Paper-Selected Quick FT Verification

The packaged results include BL-AG clean pretrain checkpoints at:

```text
results/<run_name>/stored_weights/baseline_pretrained/BL-AG_clean_pretrain_weights.pt
```

Use the helper script for the quick training check. It does three things:

1. creates a small FT-only config;
2. copies the required packaged BL-AG weight into the quick result folder;
3. keeps the output separate from the packaged `results/` tree.

```bash
python3 -B make_quick_verify_config.py
python3 -B run.py \
  --config quick_verify_configs/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10.json \
  --run-name blocking_v5_seed123_P1_blockB1 \
  --no-dashboard
```

Default quick-verification settings match the paper-selected FT setting:

- seed `123`
- Pattern 1, blocked beam `B1`
- 100% blockage
- 1% FT data
- `freeze_cnn`
- 10 FT epochs
- config folder `quick_verify_configs/`
- result folder `quick_verify_results/<setting-label>/`

This command writes result files only. It does not create selected manuscript
plots. Selected manuscript plots require the complete 30-run packaged
`results/` tree, not the one-run quick verification folder.

Useful variants:

```bash
python3 -B make_quick_verify_config.py --epochs 3 --label smoke_epochs3
python3 -B make_quick_verify_config.py --run-name blocking_v5_seed456_P2_blockB7 --epochs 10
python3 -B make_quick_verify_config.py --blockage 80 --train-frac 0.05 --freeze-mode no_freeze --label P1_B1_block80_frac5_no_freeze
```

The helper refuses to replace an existing verification config or output folder.
Pass `--overwrite` only when intentionally replacing that verification result.

After running the default command, inspect:

```text
quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1/results.csv
```

The quick verification output is a one-run FT check. It is suitable for checking
training execution and per-run plots, but it is not the input for the full
selected manuscript plot bundle.

## Plot Choices For Quick FT Verification

No plots are created unless a plotting command is run.

### No Plots

Stop after the quick FT run. The result CSV is the verification output:

```text
quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1/results.csv
```

### Common Quick-Check Plots

This command creates blockage, epoch-budget, and training-loss plots:

```bash
python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families blockage,epochs,loss
```

The PNG files are written under:

```text
plots/run_plots/blocking_v5_seed123_P1_blockB1/
```

### Only Some Quick-Check Plots

Use one family name when only one plot type is wanted:

```bash
python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families blockage

python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families epochs

python3 -B plots.py run-plots \
  --run-dir quick_verify_results/blocking_v5_seed123_P1_blockB1_blockage100_frac0p01_freeze_cnn_epochs10/blocking_v5_seed123_P1_blockB1 \
  --families loss
```

## Scientific Config Knobs

Edit `config.json` for protocol changes:

- `values_cube_path`: input response tensor.
- `outdir`: result output root.
- `batch_jobs`: pattern/blocked-beam job list.
- `blockage_levels`: blockage percentages.
- `training.seed` and `training.seeds`: seed selection.
- `training.train_fracs_sweep`: FT data fractions.
- `training.ft_freeze_modes`: FT freeze modes.
- `training.ft_epoch_sweep`: FT epoch budgets.
- `experiments`: methods to run.
- `stages`: prepare/baseline/finetune/aggregate selection.
- `plotting.runner_postprocessing`: plots generated after a run.

Edit `system_settings.json` for local machine behavior:

- `device`: `cpu`, `auto`, `cuda`, `cuda:0`, or `mps`.
- `torch_num_threads`: optional CPU thread limit.
- `matplotlib_cache_dir`: local plot/font cache.
- `dashboard.enabled`: default dashboard setting.

Hardware settings do not change the scientific protocol. Config settings do.

## Plotting CLI

With no subcommand, `plots.py` rebuilds the selected paper plot bundle.

```bash
python3 -B plots.py
```

Regenerate selected manuscript plots:

```bash
python3 -B plots.py selected --results-root results --config config.json
```

This command uses the complete packaged 30-run `results/` tree. A one-run quick
verification folder does not contain enough runs to regenerate the complete
selected manuscript bundle.

For a non-destructive plot verification pass, write selected plots to a
separate folder:

```bash
python3 -B plots.py selected \
  --results-root results \
  --config config.json \
  --selected-root plots/verification_selected_plots_paper_packaged
```

The curated selected bundle is `plots/selected_plots/`. Use that path only when
refreshing the release plots intentionally.

Regenerate selected plots with filters:

```bash
python3 -B plots.py selected --kpi primary --families blockage,epochs --seeds 123,456,789
python3 -B plots.py selected --families heatmap,convergence --patterns 1,2
python3 -B plots.py selected --blockages 20,80,100 --train-fracs 0.01
```

Generate comparison plots:

```bash
python3 -B plots.py compare --results-root results --config config.json
python3 -B plots.py compare --families method,cross-pattern,non-ml --organize
python3 -B plots.py compare --kpi primary --blockages 80,100 --seeds 123,456,789
```

Generate per-run plots:

```bash
python3 -B plots.py run-plots --run-dir results/blocking_v5_seed123_P1_blockB4
python3 -B plots.py run-plots --run-dir results/blocking_v5_seed123_P1_blockB4 --families loss
python3 -B plots.py run-plots --run-dir results/blocking_v5_seed123_P1_blockB4 --families blockage,train-fraction --blockages 20,80,100
```

Generate the full plot library:

```bash
python3 -B plots.py full --results-root results --config config.json
```

Organize already generated plots into browse folders:

```bash
python3 -B plots.py organize-only --clean-organized
```

Generate provenance plots:

```bash
python3 -B plots.py provenance --config config.json
```

Common plotting filters:

| Filter | Example | Meaning |
| --- | --- | --- |
| `--kpi` | `--kpi primary,top1` | KPI keys to plot. |
| `--families` | `--families blockage,epochs` | Plot families to generate. |
| `--seeds` | `--seeds 123,456,789` | Seeds to include. |
| `--patterns` | `--patterns 1,2` | Set-B patterns to include. |
| `--blocked-beams` | `--blocked-beams B4,B0` | Blocked beams to include. |
| `--train-fracs` | `--train-fracs 0.01,0.1,1.0` | FT data fractions. |
| `--blockages` | `--blockages 20,80,100` | Blockage levels. |
| `--epochs` | `--epochs 1,3,10` | FT epoch budgets. |
| `--organize` | `--organize` | Copy plots into `plots/organized_plots`. |
| `--copy-mode` | `--copy-mode copy` | Copy, hardlink, or symlink organized plots. |

Common family names:

- `blockage`
- `epochs`
- `train-fraction`
- `freeze`
- `multi-kpi`
- `loss`
- `method`
- `cross-pattern`
- `non-ml`
- `heatmap`
- `convergence`
- `setup`
- `procedure`

## Non-Learning Baselines

Run deterministic baselines directly:

```bash
python3 -B non_ml_benchmarks.py --config config.json
```

These baselines are also included in the configured batch when `experiments`
contains `MAX-SETB` and `NN-ANGLE`.

## Dashboard

The dashboard is optional and writes a local file:

```text
results/dashboard.html
```

It reports selected stages, methods, seeds, current job, seed progress, ETA,
recent epoch callbacks, losses, artifacts, errors, and CPU/GPU/RAM/disk resource
status when available. It is a monitor only and does not change training or
results.
