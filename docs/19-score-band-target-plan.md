# Score-Band Target Plan

## Purpose

The project target is now the official **10-band severity task**.

From this point forward:

- primary training target: `score_band`
- primary model-selection metric: **10-band validation mean band error (MAE on band index)**
- four-class evaluation: **secondary diagnostic view**

The four-class system is not discarded. It is frozen as the implementation and
comparison reference so future score-band work can still be collapsed back into
`intact / minor / moderate / severe`.

## Frozen Coarse Reference

Reference artifact:

- [coarse_reference_before_score_band_target_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/coarse_reference_before_score_band_target_20260514.csv)

Supporting baseline artifacts:

- [v3_1_hard_split_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_hard_split_baseline_20260513.csv)
- [v3_1_metadata_family_holdout_comparison_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv)
- [v3_synthetic_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_synthetic_baseline_20260513.csv)

Frozen current reference configuration:

- target mode: `coarse`
- labels: `4`
- model: `residual_cnn`
- augmentation: `damage_safe`
- dropout: `0.1`
- weight decay: `0.0001`
- ordinal loss weight: `0.2`
- balanced sampler: `False`
- split strategy: `metadata_family_holdout`

## Frozen Implementation Files

The current classifier implementation is intentionally preserved as the
reference code path before score-band-target work diverges further.

Key files:

- [src/dream2detect/training/models.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/models.py)
- [src/dream2detect/training/dataset.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/dataset.py)
- [src/dream2detect/training/metrics.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/metrics.py)
- [src/dream2detect/training/splits.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/splits.py)
- [src/dream2detect/training/train_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/train_classifier.py)
- [scripts/train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)
- [scripts/run_classifier_experiment_grid.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/run_classifier_experiment_grid.py)
- [scripts/analyze_classifier_checkpoint.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/analyze_classifier_checkpoint.py)

## New Target Definition

Primary target labels:

- `0-10`
- `11-20`
- `21-30`
- `31-35`
- `36-45`
- `46-55`
- `56-65`
- `66-75`
- `76-85`
- `86-100`

Collapsed coarse mapping for secondary evaluation:

- `0-10` -> `intact`
- `11-20`, `21-30`, `31-35` -> `minor`
- `36-45`, `46-55`, `56-65` -> `moderate`
- `66-75`, `76-85`, `86-100` -> `severe`

## Evaluation Rule

### Primary score-band reporting

Every main score-band run should report:

- exact 10-band accuracy
- 10-band macro F1
- within-one-band-or-correct accuracy
- mean band-index error
- severe band-error rate

### Secondary collapsed coarse reporting

The same score-band model should also be evaluated after collapsing to the four
coarse classes.

Preferred collapse rule:

- use the model's **10-band probabilities**
- sum probabilities inside each coarse group
- choose the coarse class from those grouped probabilities

Do not rely only on mapping the single argmax band to one coarse class if the
full probability vector is available.

## Checkpoint Selection Rule

For the score-band phase, the selected checkpoint should be chosen by:

- **validation mean band error on the 10-band task**

Lower is better. Exact 10-band macro F1 remains a required reported metric, but
it no longer decides the main checkpoint by itself because it does not reflect
ordinal distance.

## Training Rule

The main score-band path should now use:

- from-scratch `residual_cnn`
- `metadata_family_holdout`
- `damage_safe` augmentation
- **soft ordinal label distributions** across the ten bands
- **class-balanced loss** based on effective-number weighting
- optional smooth ordinal expectation penalty as a secondary term

The soft-label rule is intentional: adjacent bands should receive some target
mass, while distant bands should receive very little or none.

## First Score-Band Restart

The first serious score-band restart should keep the current coarse reference as
close as possible except for the target:

- model: `residual_cnn`
- augmentation: `damage_safe`
- dropout: `0.1`
- weight decay: `0.0001`
- ordinal loss weight: `0.2`
- balanced sampler: `False`
- split strategy: `metadata_family_holdout`
- target label mode: `score_band`
- score-band soft-label sigma: `1.0`
- score-band class-weight strategy: `effective`

This keeps the comparison interpretable.

## Historical Note

The repo already contains an older ten-band run under the earlier regime:

- [score_band_384_residual_damage_safe_ordinal02_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/score_band_384_residual_damage_safe_ordinal02_20260508)

That older score-band result was informative but not decisive. The new score-band
phase should be treated as a restart under the stricter hard-split baseline and
the current training stack, not as a direct continuation of that earlier result.

## Current V4 Baseline

The first full V4 score-band run is now recorded here:

- [synthetic_full_qc_plus_v2_scale_processed_384_384_20260514_192919](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_full_qc_plus_v2_scale_processed_384_384_20260514_192919)

Locked baseline outcome:

- best validation epoch: `67`
- validation mean band error: `1.1627`
- test exact 10-band accuracy: `0.1990`
- test 10-band macro F1: `0.1791`
- test mean band error: `1.3155`
- test `±1` band accuracy: `0.7233`
- collapsed 4-class test macro F1: `0.4557`

## Immediate V4.1 Sweep

The next narrow sweep should challenge that baseline locally, not restart the
whole design. The first V4.1 challenger set is:

- sigma `0.75`
- sigma `1.25`
- dropout `0.2`
- effective beta `0.995`

All other settings should remain fixed to the current V4 baseline.

## V4.1 Sweep Result

The first V4.1 challenger sweep is complete.

Artifacts:

- ranked sweep summary:
  [v4_1_sweep_summary_ranked.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_1_local_sweep_20260514/v4_1_sweep_summary_ranked.csv)
- baseline-plus-challengers comparison:
  [v4_1_score_band_challenger_comparison_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_1_score_band_challenger_comparison_20260514.csv)

Best challenger by the primary selection metric:

- `sigma=1.25`
- best validation mean band error: `1.1325`

Current baseline for reference:

- `sigma=1.0`
- best validation mean band error: `1.1627`

Important outcome:

- `sigma=1.25` improved the validation selection metric
- but it weakened held-out test ordinal behavior relative to the baseline

That means `sigma=1.25` is a real challenger, but not a clean replacement yet.

## Next Locked Decision

Do not broaden the search yet.

The next step should be a narrow two-way confirmation:

- current V4 baseline: `sigma=1.0`, `beta=0.999`, `dropout=0.1`
- leading V4.1 challenger: `sigma=1.25`, `beta=0.999`, `dropout=0.1`

Both should be rerun with at least one additional seed.

The next baseline lock should be based on:

1. validation mean band error
2. test mean band error
3. test `+-1` band accuracy

Do not promote a challenger on one seed and one validation win alone.

## Two-Seed Confirmation Result

The baseline versus `sigma=1.25` confirmation is now complete across two seeds.

Supporting artifacts:

- [v4_1_two_seed_baseline_vs_sigma125_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_1_two_seed_baseline_vs_sigma125_20260514.csv)
- [v4_1_two_seed_baseline_vs_sigma125_aggregate_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_1_two_seed_baseline_vs_sigma125_aggregate_20260514.csv)

What is now established:

- `sigma=1.25` beats the current baseline on validation mean band error in both
  tested seeds
- but it does not beat the baseline on average held-out test band error or
  average `+-1` band accuracy

Current lock decision:

- keep the current V4 baseline locked
- treat `sigma=1.25` as a validated challenger, not the replacement baseline

Next experiment direction:

- search narrowly between `sigma=1.0` and `sigma=1.25`
- do not broaden back into unrelated hyperparameters yet
