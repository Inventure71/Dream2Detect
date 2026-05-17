# V3 - Training Stack Search

## Purpose

Search for training-stack improvements around the V2 residual baseline while
holding the dataset and core task stable.

## Changes Tested

- GroupNorm residual backbone
- ordinal coarse-class loss
- damage-safe augmentation profiles
- sampler and regularization variants

## Dataset

Use the delivery synthetic dataset:

- `dataset/synthetic/manifest.csv`
- `dataset/synthetic/images/`

## Run One Representative Candidate

```bash
python3 scripts/train_synthetic_classifier.py \
  --manifest dataset/synthetic/manifest.csv \
  --image-size 384 \
  --model-variant residual_cnn_groupnorm \
  --target-label-mode coarse_ordinal \
  --ordinal-loss-weight 0.2 \
  --augmentation-profile damage_safe_ra_low \
  --num-epochs 160 \
  --output-dir outputs/v3_training_stack
```

## Run A Small Grid

```bash
python3 scripts/run_classifier_experiment_grid.py \
  --manifest dataset/synthetic/manifest.csv \
  --output-root outputs/v3_grid \
  --target-label-mode coarse
```

## Code

- `scripts/train_synthetic_classifier.py`
- `scripts/run_classifier_experiment_grid.py`
- `src/dream2detect/training/`

## Result Summary

V3 improved the coarse synthetic baseline, but split discipline was tightened
in V3.1 before the project moved to the 10-band target.
