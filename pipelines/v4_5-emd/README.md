# V4.5 - Hybrid Soft-Label Cross-Entropy Plus EMD

## Purpose

Test whether a distance-aware EMD term helps the ordered 10-band score target.

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
  --score-band-emd-weight 0.5 \
  --split-strategy metadata_family_holdout \
  --augmentation-profile damage_safe \
  --num-epochs 160 \
  --output-dir outputs/v4_5_emd
```

## Code

- `scripts/train_synthetic_classifier.py`
- `src/dream2detect/training/train_classifier.py`
- `src/dream2detect/training/metrics.py`

## Result Summary

V4.5 improved some tolerance metrics but did not cleanly beat V4 on the main
synthetic selection metrics.

