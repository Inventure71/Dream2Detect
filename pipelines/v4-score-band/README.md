# V4 - Official 10-Band Score-Band Baseline

## Purpose

Move from four coarse classes to the official 10-band severity target.

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
  --target-label-mode score_band \
  --score-band-soft-label-sigma 1.0 \
  --score-band-class-weight-strategy effective \
  --split-strategy metadata_family_holdout \
  --augmentation-profile damage_safe \
  --num-epochs 160 \
  --output-dir outputs/v4_score_band
```

## Code

- `scripts/train_synthetic_classifier.py`
- `scripts/analyze_classifier_checkpoint.py`
- `src/dream2detect/training/`

## Result Summary

V4 became the clean 10-band reference for later ordinal-loss and V5 target-head
experiments.

