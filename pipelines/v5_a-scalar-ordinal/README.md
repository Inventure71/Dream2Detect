# V5-A - Scalar Ordinal Regression

## Purpose

Test whether predicting a scalar severity value transfers better than direct
10-band classification.

## Dataset

Use the delivery synthetic dataset:

- `dataset/synthetic/manifest.csv`
- `dataset/synthetic/images/`

## Run

```bash
python3 scripts/train_synthetic_regressor.py \
  --manifest dataset/synthetic/manifest.csv \
  --image-size 384 \
  --model-variant residual_cnn_groupnorm_regressor \
  --split-strategy metadata_family_holdout \
  --augmentation-profile damage_safe \
  --num-epochs 160 \
  --output-dir outputs/v5_a_scalar_ordinal
```

## Code

- `scripts/train_synthetic_regressor.py`
- `src/dream2detect/training/train_regressor.py`
- `src/dream2detect/training/models.py`
- `src/dream2detect/training/metrics.py`

## Result Summary

V5-A improved some raw-real ordinal behavior but weakened synthetic 10-band
performance, motivating V5-B.

