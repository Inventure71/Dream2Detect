# Score-Band Target Plan

## Purpose

The project target is now the official **10-band severity task**.

From this point forward:

- primary training target: `score_band`
- primary model-selection metric: **10-band validation macro F1**
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

- **validation macro F1 on the 10-band task**

The collapsed four-class metrics remain diagnostic and comparative, but they no
longer decide the main checkpoint.

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

This keeps the comparison interpretable.

## Historical Note

The repo already contains an older ten-band run under the earlier regime:

- [score_band_384_residual_damage_safe_ordinal02_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/score_band_384_residual_damage_safe_ordinal02_20260508)

That older score-band result was informative but not decisive. The new score-band
phase should be treated as a restart under the stricter hard-split baseline and
the current training stack, not as a direct continuation of that earlier result.
