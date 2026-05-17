# V1 - First Synthetic-Only Scratch CNN

## Purpose

Test whether package/cardboard severity labels are learnable from synthetic
images by a small from-scratch CNN.

## Dataset

Use the delivery synthetic dataset:

- `dataset/synthetic/manifest.csv`
- `dataset/synthetic/images/`

## Run

```bash
python3 scripts/train_synthetic_classifier.py \
  --manifest dataset/synthetic/manifest.csv \
  --image-size 224 \
  --model-variant simple_cnn \
  --target-label-mode coarse \
  --num-epochs 120 \
  --output-dir outputs/v1_scratch_cnn
```

## Code

- `scripts/train_synthetic_classifier.py`
- `src/dream2detect/training/dataset.py`
- `src/dream2detect/training/models.py`
- `src/dream2detect/training/train_classifier.py`

## Result Summary

V1 showed that synthetic labels were learnable, but the simple model was fragile
and was superseded by the residual full-QC setup in V2.

