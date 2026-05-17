# V4.1 - Narrow Score-Band Challenger Sweep

## Question

Can a small controlled set of score-band training changes beat the locked V4
baseline without changing the experiment too broadly?

## Scope

V4.1 kept the V4 setup fixed and tested a narrow set of challenger settings.
The leading challenger was checked against the locked baseline instead of being
promoted from one favorable run.

## Implementation

- `scripts/run_classifier_experiment_grid.py`
- `scripts/train_synthetic_classifier.py`
- `src/dream2detect/training/train_classifier.py`

## Main Artifacts

- `data/training_runs/synthetic_classifier_grid/v4_1_local_sweep_20260514`
- `data/evaluations/synthetic_only/v4_1_score_band_challenger_comparison_20260514.csv`
- `data/evaluations/synthetic_only/v4_1_two_seed_baseline_vs_sigma125_20260514.csv`

## Result

No V4.1 challenger earned promotion over the locked V4 baseline.

