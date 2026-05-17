# V3.1 - Harder Split Discipline

## Question

Were earlier synthetic scores too optimistic because similar prompt families
appeared across train, validation, and test splits?

## Change

V3.1 introduced `metadata_family_holdout`, grouping examples by:

- `training_coarse_class`
- `damage_profile_primary`
- `box_form_factor`
- `background_context`

## Implementation

- `src/dream2detect/training/splits.py`
- `src/dream2detect/training/train_classifier.py`
- `scripts/run_classifier_experiment_grid.py`

## Main Artifacts

- `data/evaluations/synthetic_only/v3_1_hard_split_baseline_20260513.csv`
- `data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv`

## Result

V3.1 became the fairer synthetic-only comparison regime. Later V4 and V5
experiments use this split discipline.

