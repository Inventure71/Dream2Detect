# V3.1 - Metadata-Family Holdout Split

## Purpose

Reduce synthetic over-optimism by preventing similar prompt families from
appearing across train, validation, and test splits.

## Dataset

Use the delivery synthetic dataset:

- `dataset/synthetic/manifest.csv`
- `dataset/synthetic/images/`

## Run

```bash
python3 scripts/train_synthetic_classifier.py \
  --manifest dataset/synthetic/manifest.csv \
  --image-size 384 \
  --model-variant residual_cnn \
  --target-label-mode coarse \
  --split-strategy metadata_family_holdout \
  --augmentation-profile damage_safe \
  --num-epochs 160 \
  --output-dir outputs/v3_1_hard_split
```

## Code

- `src/dream2detect/training/splits.py`
- `src/dream2detect/training/train_classifier.py`
- `scripts/train_synthetic_classifier.py`

## Result Summary

V3.1 became the fairer synthetic-only evaluation regime used by V4 and V5.

