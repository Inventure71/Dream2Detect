# V2 - Full-QC Residual Baseline And Targeted Scale-Up

## Question

Do cleaner labels, 384 px inputs, and a stronger from-scratch residual CNN
improve synthetic learning and real transfer?

## Dataset

- `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- delivery equivalent: `dataset/synthetic/manifest.csv`

This dataset has 899 accepted base examples after full QC plus the V2
real-failure-targeted synthetic scale-up.

## Implementation

- `scripts/train_synthetic_classifier.py`
- `scripts/build_image_cache.py`
- `src/dream2detect/training/train_classifier.py`
- `src/dream2detect/training/models.py`

## Main Artifacts

- `data/evaluations/real_transfer/v2_scale_comparison_20260513.csv`
- `data/training_runs/synthetic_classifier/v2_scale_384_residual_damage_safe_ordinal02_20260513`

## Result

V2 improved the synthetic task and some real-transfer diagnostics, but it did
not close the synthetic-to-real gap. It became the controlled baseline for V3.

