# V4.5 - Hybrid Soft-Label Cross-Entropy Plus EMD

## Question

If score bands are ordered, does a distance-aware EMD term improve the V4
score-band objective?

## Change

V4.5 kept the V4 stack fixed and changed only the objective:

- soft-label cross-entropy
- cumulative squared EMD penalty

## Implementation

- `src/dream2detect/training/train_classifier.py`
- `src/dream2detect/training/metrics.py`
- `scripts/train_synthetic_classifier.py`

## Main Artifacts

- `data/training_runs/synthetic_classifier/v4_5_score_band_emd05_seed42_20260515`
- `data/evaluations/synthetic_only/v4_5_vs_locked_v4_20260515.csv`
- `data/evaluations/model_checkpoint_comparison/v4_v45_v5_within_band_comparison.csv`

## Result

V4.5 improved some tolerance metrics but did not cleanly beat V4 on the main
synthetic selection metrics. It remains evidence, not the promoted baseline.

