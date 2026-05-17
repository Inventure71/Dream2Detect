# V4 - Official 10-Band Score-Band Baseline

## Question

What happens when the project moves from four coarse classes to the official
10-band severity target?

## Dataset

- `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- delivery equivalent: `dataset/synthetic/manifest.csv`

## Implementation

- `src/dream2detect/training/train_classifier.py`
- `src/dream2detect/training/metrics.py`
- `scripts/train_synthetic_classifier.py`
- `scripts/analyze_classifier_checkpoint.py`

## Main Artifacts

- `data/training_runs/synthetic_classifier/synthetic_full_qc_plus_v2_scale_processed_384_384_20260514_192919`
- `data/evaluations/model_checkpoint_comparison/v4_v45_v5_within_band_comparison.csv`

## Result

V4 became the locked 10-band score-band reference. It is not the final real
transfer solution, but it is the clean reference for later score-band
experiments.

