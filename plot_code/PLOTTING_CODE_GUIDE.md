# Plotting Code Organization

`plots.py` is the plotting entrypoint. The implementation is split under
`plot_code/` so chart style, source-data loading, selected figures, comparisons,
and organization are easy to inspect separately.

## Module Map

- `cli.py`: parses commands and filters, then calls the matching builder.
- `config.py`: default release paths, `PlotSelection`, CSV flag parsers, and
  aliases that map readable family names such as `blockage`, `epochs`,
  `loss`, and `non-ml` onto canonical internal names.
- `plot_config.json`: publication style settings. Edit this file for colors,
  DPI, figure size, axes, legend placement, title rendering, and per-plot
  overrides.
- `style.py`: loads `plot_config.json`, applies Matplotlib style, cleans titles
  and legend labels, and saves figures.
- `charts.py`: shared grouped-bar, categorical-bar, and heatmap builders. This
  release intentionally renders trend-style comparisons as bars, not line
  charts, so every value is shown as its own category.
- `data_loading.py`: CSV reading, run discovery, numeric conversion, row
  filtering, and mean/std helpers.
- `metrics.py`: KPI keys, display labels, and result-column names.
- `organization.py`: copies plots into navigation folders by KPI, seed, family,
  pattern, blocked beam, method, selected-paper status, and a flat index.
- `overrides.py`: per-plot override access.
- `families/run_plots.py`: reads one `results/<run_name>/` folder and writes
  its PNGs under `plots/run_plots/<run_name>/`.
- `families/compare_plots.py`: plots across runs, beams, patterns, and seeds.
- `families/selected_plots.py`: manifest-driven selected manuscript bundle.
- `families/selected_source_data.py`: rebuilds selected-plot source CSVs from
  completed `results/` runs, but only after the configured seed/job set is
  complete.
- `families/provenance_plots.py`: beam-location and provenance outputs.
- `families/full_plots.py`: calls run, compare, provenance, selected, and
  organization builders in one workflow.

## Plot Families

`run_plots.py` builds per-run plots from one run folder:

- `blockage_vs_accuracy`: grouped bars across blockage levels.
- `train_fraction_vs_accuracy`: grouped bars across fine-tuning data fractions.
- `epoch_budget_vs_accuracy`: grouped bars across FT epoch budgets.
- `multi_kpi`: grouped bars across KPI definitions for a selected blockage.
- `train_loss`: bar diagnostics from `jobs/**/history.csv`.

`compare_plots.py` builds cross-run plots:

- `method_comparison`: BL-AG, FT freeze-CNN, and BL-AW comparisons by pattern
  and blocked beam.
- `blockage_vs_accuracy`: FT freeze-CNN grouped bars across blockage levels.
- `cross_pattern_comparison`: Pattern 1 versus Pattern 2 summaries from the
  selected source-data CSVs. These CSVs are refreshed from complete result
  tables when the configured run set is present.
- `non_ml_baselines`: deterministic MAX-SETB and NN-ANGLE baseline plots from
  `plots/selected_plots/source_data/non_ml_baseline_summary.csv`.

Comparison PNGs are written under `plots/compare_plots/`.

`selected_plots.py` rebuilds the curated main-paper bundle from
`plots/selected_plots/selected_plots_manifest.csv`. The manifest covers
procedure, setup, aggregate methods, blockage trends, freeze ablations,
train-fraction sweeps, epoch-budget sweeps, cross-pattern comparisons,
multi-KPI summaries, non-ML baselines, a combined 2x2 BL-AG-versus-FT
beam-by-blockage heatmap, and convergence diagnostics. Trend-style selected
plots are rendered as bars, not line charts.

Provenance PNGs are written under `plots/provenance/`. Plotting command
progress is written to `plots/plotting_live_progress.json`.

## Commands

Verify selected manuscript plot generation without replacing the curated
`plots/selected_plots/` bundle:

```bash
python3 -B plots.py selected \
  --results-root results \
  --config config.json \
  --selected-root plots/verification_selected_plots_paper_packaged
```

Generate comparison plots only:

```bash
python3 -B plots.py compare --results-root results --config config.json --families method,cross-pattern,non-ml
```

Generate plots for one run folder:

```bash
python3 -B plots.py run-plots --run-dir results/blocking_v5_seed123_P1_blockB4 --kpi primary
```

Generate only training-loss diagnostics for one blockage and train fraction:

```bash
python3 -B plots.py run-plots --run-dir results/blocking_v5_seed123_P1_blockB4 --families loss --blockages 100 --train-fracs 0.01
```

Generate everything after the results tree is available:

```bash
python3 -B plots.py full --results-root results --config config.json
```

Organize existing PNG files without regenerating plots:

```bash
python3 -B plots.py organize-only --results-root results --clean-organized
```

## Filters

All generation commands accept these filters:

```text
--kpi primary,top1,top3_m1db
--families blockage,epochs,train-fraction,multi-kpi,loss,method,cross-pattern,non-ml
--seeds 123,456,789
--patterns 1,2
--blocked-beams B4,B0
--train-fracs 0.01,0.1,1.0
--blockages 20,80,100
--epochs 1,3,10
--organize
--clean-organized
```

Family aliases are normalized, so `epochs`, `epoch-budget`, `train-frac`,
`nonml`, and `train-loss` are accepted.

## Organization Output

`--organize` and `organize-only` build this tree:

```text
plots/organized_plots/
  by_kpi/
  by_seed/
  by_family/
  by_pattern/
  by_blocked_beam/
  by_method/
  selected_paper_plots/
  all_plots_flat/
  organized_plot_manifest.csv
```

A plot is copied into every matching folder. For example, a multi-seed Pattern
1 B4 blockage plot can appear under `by_kpi/primary`,
`by_seed/multiseed_123_456_789`, `by_family/blockage_vs_accuracy`,
`by_pattern/pattern_1`, `by_blocked_beam/P1_B4`, and the flat index.

Builders return `PlotOutput` records so organization uses metadata known at
plot creation time. Filename inference is only a fallback for `organize-only`
when existing PNGs were not just created by a builder.

## Selected Bundle Rules

`plots/selected_plots/selected_plots_manifest.csv` is the curated traceability
table for the selected manuscript bundle. The builder first creates plots in a
temporary folder. It copies them into the selected output root only after all
requested manifest rows are regenerated. If required inputs are missing,
existing selected PNGs in that output root stay in place and
`selected_plots_generation_status.csv` records what was skipped.

When `--selected-root` points to a new verification folder, the builder seeds
that folder from the curated manifest and packaged source-data tables before
regenerating plots. This keeps non-destructive verification runs complete while
leaving `plots/selected_plots/` unchanged.

Before selected generation starts, `selected_source_data.py` checks whether the
configured seeds and batch jobs all have run-level `results.csv` files. If the
full run set is present, it refreshes the selected summary, cross-pattern, and
non-ML CSVs from those results. If the run set is incomplete, packaged
source-data CSVs are left unchanged.

## Extending Plots

- Add a new plot by writing a focused builder, returning `PlotOutput`, and
  adding a family alias if it should be callable from `--families`.
- Keep colors and layout defaults in `plot_config.json`, not hardcoded in a
  plot family.
- Keep selected-paper traceability in `selected_plots_manifest.csv`.
