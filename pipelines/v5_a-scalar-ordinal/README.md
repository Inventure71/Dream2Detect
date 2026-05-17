# V5-A - Scalar Ordinal Regression

## Question

Does predicting a scalar severity value transfer better than direct 10-band
classification?

## Change

V5-A used a from-scratch residual GroupNorm regressor with a sigmoid scalar
head. Predictions are mapped back to the official 10 score bands for
evaluation.

## Implementation

- `scripts/train_synthetic_regressor.py`
- `src/dream2detect/training/train_regressor.py`
- `src/dream2detect/training/models.py`
- `src/dream2detect/training/metrics.py`

## Main Artifacts

- `data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed`
- `data/evaluations/model_checkpoint_comparison/v5_a_scalar_ordinal_synthetic_test_predictions.csv`
- `data/evaluations/model_checkpoint_comparison/v5_a_scalar_ordinal_real_all_raw_predictions.csv`

## Result

V5-A was mechanically correct and improved some raw-real ordinal transfer
metrics, but it weakened synthetic score-band performance. It motivated V5-B.

