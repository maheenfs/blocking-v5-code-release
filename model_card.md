# Blocking V5 Model Card

## Model Summary

Blocking V5 evaluates a compact 1D CNN beam predictor under Set-B beam blockage.
The central adaptation result is `FT freeze_cnn`: the convolutional trunk
(feature extractor) is frozen and only the fully connected prediction head is
updated using a small fraction of blocked-condition data.

## Intended Use

- Research on blockage-aware mmWave beam selection.
- Comparing clean-trained, blockage-aware, and lightweight fine-tuned models.
- Reproducing the paper claim that 1% blocked-condition adaptation data can
  recover much of the blockage-induced performance loss.

## Not Intended For

- Safety-critical real-time wireless control without independent validation.
- Deployment on channels, arrays, blockage mechanisms, or beam codebooks outside
  the evaluated VIVoR/DeepMIMO-style setup.
- Claims about general wireless robustness without the multi-seed and
  cross-scenario evidence documented with the release.

## Training Data

The pipeline reads the configured value cube from `config.json`:

- `values_cube_path`: `./VIV0R4_116_00.npy`
- The release package includes this tensor so the default run is standalone.
- Set-A output space: 32 transmitter beams.
- Set-B input features: 8 measured beams per pattern.
- Split fractions: 90% train, 5% validation, 5% test.

The blockage protocol is `underlying_tx_consistent`: the configured blocked
Set-B direction is mapped to its matching Set-A transmitter column, that full
Set-A response column is attenuated, and both the Set-B input features and the
best-beam labels are rebuilt from the blocked response.

## Model Architecture

- Input: Set-B RSRP feature vector.
- Convolutional trunk / feature extractor: three 1D convolution layers.
- Head: two fully connected layers.
- Output: logits over 32 Set-A beam classes.

Parameter counts and train timing are written to `train_result.json` and
aggregated result rows.

## Evaluation Metrics

Primary metric:

- `test_top3_m1db_%`: top-3 within 1 dB accuracy.

Additional metrics:

- top-k inclusion accuracy
- top-k within 0/1/2/3 dB accuracy
- average RSRP loss
- p95 RSRP loss

Main paper figures use raw accuracy metrics unless explicitly marked as
supplementary diagnostics.

## Baselines

Learning baselines:

- `BL-AG`: clean/general baseline evaluated under blockage.
- `BL-AW`: blockage-aware full retraining reference.
- `FT no_freeze`: full fine-tuning reference.
- `FT freeze_cnn`: lightweight adaptation method of interest.
- `FT freeze_head` and `FT freeze_all`: freeze-mode controls.

Non-ML baselines:

- `MAX-SETB`: chooses the strongest measured Set-B beam.
- `NN-ANGLE`: chooses the strongest measured Set-B beam, then ranks all Set-A
  beams by angular distance from that Set-B beam.

`RANDOM-SETB` is implemented only as an optional sanity floor. It is not part of
the main-paper baseline set.

## Reproducibility

The runner records:

- seed in result rows
- run-local `config.json`
- Set-B mapping table
- hardware snapshot in `results/hardware_snapshot.json`
- per-epoch `history.csv`
- compact `train_result.json`
- aggregate `results.csv`

Multi-seed support is configured through `training.seeds` in `config.json`. The
completed paper seed set is `123`, `456`, and `789`.

During a full local run, FT branches load the clean BL-AG checkpoint from the
run's baseline job folder. In this release package, the clean BL-AG weights
needed for FT reuse are stored in a clearer release path:

```text
results/<run_name>/stored_weights/baseline_pretrained/BL-AG_clean_pretrain_weights.pt
```

The quick verification helper copies that stored weight into a separate quick
result folder and runs `["prepare", "finetune", "aggregate"]`, so BL-AG and
BL-AW do not need to be retrained for the verification check.

## Plotting Outputs

Use `plots.py` as the single plotting entrypoint.

- `python3 -B plots.py selected --results-root results --config config.json`
  regenerates the selected paper plot bundle.
- For a non-destructive verification pass, add
  `--selected-root plots/verification_selected_plots_paper_packaged`.
- Selected bundle regeneration preserves existing selected PNGs if required
  result files are missing. Regeneration status is written to
  `selected_plots_generation_status.csv` in the selected output folder.
- `python3 -B plots.py organize-only --clean-organized` builds the navigable
  `plots/organized_plots/` tree from existing PNGs.
- `python3 -B plots.py full --results-root results --config config.json`
  regenerates the full configured plot library.

The plotting code uses adaptive heatmap annotation colors and merges repeated
grouped-bar value labels, so identical values in one bar group are shown once.

## Known Limitations

- Completed paper results aggregate the seed-123/456/789 reruns. Reruns that
  are intended to match the paper use the same seed values and result
  manifests.
- The non-ML `NN-ANGLE` baseline uses beam geometry only; it is not a learned
  spatial model.
- Results are tied to the configured Set-B patterns and blockage protocol.
- Distribution-fitting experiments that sample synthetic RSRP values do not
  directly produce beam identities and are exploratory.
