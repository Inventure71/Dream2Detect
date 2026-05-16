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

## V4.2 Sigma Sweep Result

The V4.2 sigma-only sweep is now complete.

Supporting artifacts:

- [v4_2_sigma_sweep_aggregate_ranked.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_2_sigma_sweep_20260515/v4_2_sigma_sweep_aggregate_ranked.csv)
- [v4_2_sigma_candidates_vs_locked_baseline_aggregate_20260515.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_2_sigma_candidates_vs_locked_baseline_aggregate_20260515.csv)

What is now established:

- `sigma=1.10` gave the best V4.2 validation average
- `sigma=1.20` gave the best V4.2 held-out test band error
- no sigma value improved both at once

Current lock decision after V4.2:

- keep the current V4 baseline locked
- stop tuning sigma for now

## Next Experiment Direction

The next narrow score-band axis should no longer be sigma.

The better next step is:

- change the **soft-label target shape** or
- change the **score-band weighting rule**

Do not broaden back into unrelated hyperparameters until one of those target
formulation changes is tested.

## V4.5 Objective-Change Challenger

The next score-band challenger is now locked as **V4.5**.

V4.5 is intentionally narrow:

- keep the current V4 score-band data split, model, optimizer, augmentation,
  and target construction
- change **only** the score-band training objective
- do **not** restart a broad hyperparameter sweep

Fixed V4.5 training definition:

- model: `residual_cnn`
- split strategy: `metadata_family_holdout`
- augmentation: `damage_safe`
- target label mode: `score_band`
- score-band soft-label sigma: `1.0`
- score-band class-weight strategy: `effective`
- score-band effective beta: `0.999`
- ordinal loss weight: `0.2`
- new score-band EMD weight: `0.5`
- random seed: `42`

The V4.5 loss should be:

- soft-label cross-entropy
- plus cumulative squared EMD across the ten ordered score bands
- plus the existing smooth ordinal expectation penalty

Operational constraints for V4.5:

- no CORAL/CORN branch yet
- no scalar-regression branch
- no new sigma sweep
- no pretrained backbone work

Expected artifact naming:

- training run:
  [v4_5_score_band_emd05_seed42_20260515](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_5_score_band_emd05_seed42_20260515)
- comparison artifact:
  [v4_5_vs_locked_v4_20260515.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_5_vs_locked_v4_20260515.csv)

## V4.5 Result

The first V4.5 objective-change challenger is now complete.

Artifacts:

- training run:
  [v4_5_score_band_emd05_seed42_20260515](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_5_score_band_emd05_seed42_20260515)
- diagnostics:
  [diagnostics](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_5_score_band_emd05_seed42_20260515/diagnostics)
- baseline comparison:
  [v4_5_vs_locked_v4_20260515.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_5_vs_locked_v4_20260515.csv)

What V4.5 changed:

- kept the V4 score-band stack fixed
- added cumulative squared EMD with `score_band_emd_weight=0.5`

Outcome versus locked V4:

- validation mean band error:
  - locked V4: `1.1627`
  - V4.5: `1.2229`
- test mean band error:
  - locked V4: `1.3155`
  - V4.5: `1.3398`
- test `+-1` band accuracy:
  - locked V4: `0.7233`
  - V4.5: `0.7330`
- test 10-band macro F1:
  - locked V4: `0.1791`
  - V4.5: `0.1574`
- collapsed 4-class macro F1:
  - locked V4: `0.4557`
  - V4.5: `0.4439`

Current decision:

- keep the locked V4 baseline
- treat V4.5 as a meaningful objective-change challenger
- do not promote the hybrid EMD objective as the new default from this run alone

## V4.5 SAM3-Cropped Synthetic Ablation

The SAM3 square-crop synthetic ablation is complete.

Artifacts:

- cropped synthetic manifest:
  [synthetic_full_qc_plus_v2_scale_sam3_square_pad01_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_full_qc_plus_v2_scale_sam3_square_pad01_384.csv)
- cropped-synthetic V4.5 run:
  [v4_5_sam3_square_pad01_synthetic_only_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_5_sam3_square_pad01_synthetic_only_20260516)
- raw-real transfer comparison:
  [v4_5_sam3_square_pad01_vs_original_20260516.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_transfer/v4_5_sam3_square_pad01_vs_original_20260516.csv)

Result:

- synthetic held-out mean band error improved from `1.3398` to `1.2670`
- synthetic held-out 10-band macro F1 improved from `0.1574` to `0.2037`
- raw-real transfer mean band error worsened from `2.5351` to `2.9558`
- raw-real transfer `+-1` band accuracy worsened from `0.3532` to `0.2468`
- raw-real transfer 10-band macro F1 worsened from `0.0834` to `0.0537`

Decision:

- do not promote SAM3-cropped synthetic training as the new baseline
- treat it as an ablation showing that cleaner synthetic framing does not
  automatically improve raw-real transfer
- do not SAM-crop the real dataset further unless the plan is explicitly
  changed

## V5 Direction

V5 is now the performance-seeking phase for the 10-band `score_band` target.

Detailed V5 design:

- [docs/21-v5-design-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/21-v5-design-plan.md)

Primary objective:

- improve the 10-band ordinal severity model as much as possible while keeping
  the experiment honest and auditable

Hard constraints:

- no pretrained backbones
- no SAM-cropping of real images unless explicitly re-approved
- `score_band` remains the primary target
- collapsed 4-class evaluation remains secondary
- raw-real transfer metrics are reported, but should not be silently used for
  model selection unless a separate real validation policy is defined

Current lessons entering V5:

- V4 remains the locked score-band reference
- V4.5 EMD was useful but not enough to replace V4
- SAM3-cropped synthetic training improved synthetic holdout but hurt raw-real
  transfer
- further gains likely require a structural target/head change, not another
  small optimizer or sigma tweak

Recommended V5 candidates:

1. Scalar ordinal regression head:
   - predict one normalized severity value in `[0, 1]`
   - train from representative score or band midpoint
   - map the scalar prediction back to the 10 score bands for evaluation
2. Multitask ordinal model:
   - shared from-scratch backbone
   - 10-band score head
   - scalar severity head
   - optional collapsed coarse auxiliary head
3. Threshold-based ordinal head:
   - CORAL/CORN-style ordered thresholds
   - same from-scratch backbone and same data split

First V5 implementation should start with the scalar ordinal regression head,
because it directly matches the ordered nature of severity and tests the
specific hypothesis that the current 10-way classifier is too classification
shaped for this task.
