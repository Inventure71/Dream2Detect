# V4.1 - Narrow Score-Band Challenger Sweep

## Purpose

Test a small controlled set of score-band training changes against the locked
V4 baseline.

## Dataset

Use the delivery synthetic dataset:

- `dataset/synthetic/manifest.csv`
- `dataset/synthetic/images/`

## Run

```bash
python3 scripts/run_classifier_experiment_grid.py \
  --manifest dataset/synthetic/manifest.csv \
  --output-root outputs/v4_1_challenger_sweep \
  --target-label-mode score_band \
  --split-strategy metadata_family_holdout
```

## Code

- `scripts/run_classifier_experiment_grid.py`
- `scripts/train_synthetic_classifier.py`
- `src/dream2detect/training/`

## Result Summary

No V4.1 challenger earned promotion over the locked V4 baseline.

