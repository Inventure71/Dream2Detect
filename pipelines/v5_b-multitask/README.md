# V5-B - Multitask Scalar, Band, And Coarse Heads

## Purpose

Test whether one shared from-scratch model can combine scalar ordinal transfer
with explicit 10-band and coarse-class supervision.

## Dataset

Use the delivery synthetic dataset:

- `dataset/synthetic/manifest.csv`
- `dataset/synthetic/images/`

## Run

```bash
python3 scripts/train_multitask_classifier.py \
  --manifest dataset/synthetic/manifest.csv \
  --image-size 384 \
  --model-variant residual_cnn_groupnorm_multitask \
  --split-strategy metadata_family_holdout \
  --augmentation-profile damage_safe \
  --num-epochs 160 \
  --output-dir outputs/v5_b_multitask
```

## Code

- `scripts/train_multitask_classifier.py`
- `src/dream2detect/training/train_multitask_classifier.py`
- `src/dream2detect/training/models.py`
- `src/dream2detect/training/dataset.py`
- `src/dream2detect/training/metrics.py`

## Result Summary

V5-B improved broad raw-real ordinal transfer compared with V5-A, but it was
not strong enough to replace the earlier score-band baseline as a final solved
model.

