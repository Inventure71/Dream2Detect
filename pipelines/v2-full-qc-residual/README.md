# V2 - Full-QC Residual Baseline

## Purpose

Test whether the cleaned synthetic dataset, 384 px inputs, and a stronger
from-scratch residual CNN improve synthetic learning and real-transfer
diagnostics.

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
  --augmentation-profile damage_safe \
  --num-epochs 160 \
  --output-dir outputs/v2_full_qc_residual
```

## Code

- `scripts/train_synthetic_classifier.py`
- `src/dream2detect/training/train_classifier.py`
- `src/dream2detect/training/models.py`
- `src/dream2detect/training/dataset.py`

## Result Summary

V2 improved the synthetic task and became the controlled baseline for later
training-stack experiments.

