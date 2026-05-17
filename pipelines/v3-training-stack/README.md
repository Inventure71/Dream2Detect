# V3 - Training Stack Search

## Question

Can the V2 synthetic baseline be improved by changing the training stack while
holding the dataset and core task stable?

## Changes Tested

- GroupNorm residual backbone
- ordinal coarse-class training mode
- constrained damage-safe RandAugment profiles
- sampler and regularization variants

## Implementation

- `src/dream2detect/training/models.py`
- `src/dream2detect/training/dataset.py`
- `src/dream2detect/training/train_classifier.py`
- `scripts/run_classifier_experiment_grid.py`

## Main Artifacts

- `data/evaluations/synthetic_only/v3_synthetic_training_comparison_20260513.csv`
- `data/training_runs/synthetic_classifier/v3_bn_ra_low_coarse_ord02_nosampler_20260513`

## Result

The selected V3 candidate improved the synthetic coarse-class baseline, but the
team then tightened the split discipline in V3.1 before moving to the official
10-band target.

