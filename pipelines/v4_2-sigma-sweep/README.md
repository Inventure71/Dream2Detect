# V4.2 - Score-Band Sigma Sweep

## Question

Does changing the score-band soft-label sigma improve the locked V4 score-band
baseline?

## Scope

V4.2 was a sigma-only sweep. It intentionally kept the rest of the V4 stack
fixed so the result was interpretable.

## Implementation

- `src/dream2detect/training/dataset.py`
- `src/dream2detect/training/train_classifier.py`
- `scripts/run_classifier_experiment_grid.py`

## Main Artifacts

- `data/training_runs/synthetic_classifier_grid/v4_2_sigma_sweep_20260515`
- `data/evaluations/synthetic_only/v4_2_sigma_sweep_vs_baseline_20260515.csv`
- `data/evaluations/synthetic_only/v4_2_sigma_candidates_vs_locked_baseline_aggregate_20260515.csv`

## Result

No sigma candidate was strong enough to replace the locked V4 baseline.

