# V4.2 - Score-Band Soft-Label Sigma Sweep

## Purpose

Test whether changing the score-band soft-label sigma improves the V4 baseline.

## Dataset

Use the delivery synthetic dataset:

- `dataset/synthetic/manifest.csv`
- `dataset/synthetic/images/`

## Run

```bash
python3 scripts/run_classifier_experiment_grid.py \
  --manifest dataset/synthetic/manifest.csv \
  --output-root outputs/v4_2_sigma_sweep \
  --target-label-mode score_band \
  --split-strategy metadata_family_holdout \
  --score-band-soft-label-sigmas 0.8,1.0,1.2,1.4
```

## Code

- `scripts/run_classifier_experiment_grid.py`
- `src/dream2detect/training/dataset.py`
- `src/dream2detect/training/train_classifier.py`

## Result Summary

No sigma candidate was strong enough to replace the locked V4 baseline.

