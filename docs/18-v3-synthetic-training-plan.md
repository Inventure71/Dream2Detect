# V3 Synthetic-Only Training Plan

## Purpose

V3 is the next synthetic-only training phase after freezing the current V2
synthetic baseline.

The immediate objective is not real-domain transfer. The objective is to make
the synthetic classification problem itself materially stronger and more stable
before returning to cross-domain generalization.

## Frozen V2 Baseline

Frozen reference artifact:

- [v2_synthetic_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v2_synthetic_baseline_20260513.csv)

Frozen reference run:

- [v2_scale_384_residual_damage_safe_ordinal02_20260513](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v2_scale_384_residual_damage_safe_ordinal02_20260513)

Frozen synthetic-only result:

- test accuracy: `0.5333`
- test macro F1: `0.5239`
- off-by-one-or-correct rate: `0.9111`
- severe ordinal error rate: `0.0889`

## V3 Goal

Improve synthetic-only learning quality by attacking three concrete training
issues:

1. unstable normalization under small batch size
2. flat coarse-class objective on an ordinal target
3. hand-tuned augmentation that is not systematically parameterized

## V3 Implementation Scope

### 1. GroupNorm residual backbone

Add a new residual classifier variant that keeps the current V2 residual layout
but replaces `BatchNorm2d` with `GroupNorm`.

Reason:

- the project trains with batch size `8`
- BatchNorm uses mini-batch statistics
- GroupNorm is more stable when batch sizes are small

### 2. True ordinal coarse-class training path

Add a true ordinal coarse-class mode using cumulative thresholds instead of flat
4-way cross-entropy alone.

Target interpretation:

- `intact < minor < moderate < severe`

Implementation target:

- predict `K-1` ordered thresholds for `K=4` coarse classes
- decode thresholds back into one coarse prediction
- report the same coarse metrics as the current baseline so results remain
  directly comparable

### 3. Constrained RandAugment-style profiles

Add systematic synthetic-safe augmentation profiles that only use operations
unlikely to erase or fabricate the actual damage signal.

Allowed operation families:

- mild affine
- mild perspective
- brightness / contrast / saturation shifts
- blur
- low noise

Default exclusions:

- random crop
- cutout / erasing in the default RandAugment path
- invert / posterize / solarize style operations

Profiles to support:

- `damage_safe_ra_low`
- `damage_safe_ra_medium`
- `damage_safe_ra_high`

## V3 Evaluation Rule

V3 is synthetic-only first.

Primary comparison is against the frozen V2 synthetic baseline on the same
synthetic manifest family and `384 x 384` processed cache.

Primary metrics:

- synthetic test accuracy
- synthetic test macro F1
- off-by-one-or-correct rate
- severe ordinal error rate

## Completion Criteria

V3 is considered implemented when all of the following exist:

1. repo docs updated with the V3 plan and frozen V2 baseline
2. code support for GroupNorm residual training
3. code support for true ordinal coarse-class training
4. code support for constrained RandAugment-style augmentation profiles
5. tests covering the new training controls
6. one completed V3 synthetic-only training run with saved artifacts and
   comparison against the frozen V2 baseline

## V3.1 Hard-Split Completion

The V3 implementation is now complete and the synthetic-only evaluation regime
has been tightened.

New hard-split rule:

- use `metadata_family_holdout` for the primary synthetic-only comparison
- derive split groups from:
  - `training_coarse_class`
  - `damage_profile_primary`
  - `box_form_factor`
  - `background_context`

New baseline artifacts:

- random-split V3 baseline:
  [v3_synthetic_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_synthetic_baseline_20260513.csv)
- hard-split V3.1 baseline:
  [v3_1_hard_split_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_hard_split_baseline_20260513.csv)
- hard-split comparison table:
  [v3_1_metadata_family_holdout_comparison_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv)

Selected hard-split baseline configuration:

- model: `residual_cnn`
- augmentation: `damage_safe`
- dropout: `0.1`
- weight decay: `0.0001`
- ordinal loss weight: `0.2`
- balanced sampler: `False`

Selection rule:

- choose by validation macro F1 under `metadata_family_holdout`
- prefer the configuration that remains strongest across a second seed

Selected result summary:

- seed 42 best validation macro F1: `0.5869`
- seed 43 best validation macro F1: `0.5964`
- mean best validation macro F1: `0.5916`

Important observation:

- `dropout=0.3` produced a stronger single seed-42 test result, but it was less
  stable on the second seed and was not kept as the new baseline
- `weight_decay=0.0003` was pruned after clear underperformance against
  `0.0001`
- `damage_safe_ra_low` was pruned after it lost to `damage_safe` on matched
  configurations
