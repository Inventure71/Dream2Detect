# V5 Design Plan

V5 is the performance-focused phase for the 10-band package/cardboard severity
task.

The goal is not to randomly try tricks. The goal is to apply the core ML class
workflow correctly:

- define the target precisely
- control train/validation/test usage
- choose metrics that match the task
- establish baselines
- run small sanity checks before long training
- interpret learning curves and errors before changing more variables
- change one major design axis at a time

## Current Evidence

The locked score-band reference is still V4.

V4.5 added a hybrid soft-CE + cumulative squared EMD objective. It was a useful
ordinal-loss experiment, but it did not clearly beat V4.

The SAM3 square-crop synthetic ablation improved synthetic held-out metrics but
worsened raw-real transfer. It should not be promoted as a new default.

The next likely bottleneck is the target/head formulation, not a tiny
hyperparameter detail.

## V5 Constraints

V5 keeps these constraints:

- no pretrained backbone
- primary target is still `score_band`
- collapsed 4-class metrics are secondary
- raw-real evaluation is reported separately from synthetic validation
- do not SAM-crop real images unless the plan is explicitly changed
- do not use raw-real test results as hidden model-selection feedback

## ML-Class Experimental Structure

### 1. Data Split Discipline

Use the same synthetic manifest and split strategy as the locked V4 reference
unless a split-specific experiment is explicitly declared.

Default synthetic dataset:

- [synthetic_full_qc_plus_v2_scale_processed_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv)

Default split:

- `metadata_family_holdout`

Reason:

- it reduces prompt-family leakage
- it is the fairest comparison to the locked V4 score-band setup

Raw real data is not used to pick the best epoch or tune V5-A. It is used as a
transfer evaluation after the synthetic run is complete.

### 2. Baselines

Every V5 run should be compared against:

- locked V4 score-band baseline
- V4.5 EMD challenger
- V4.5 SAM3-crop ablation
- real-only model where relevant

The minimum comparison metrics are:

- validation mean band error
- test mean band error
- test `+-1` band accuracy
- test 10-band exact accuracy
- test 10-band macro F1
- collapsed 4-class macro F1
- raw-real transfer mean band error
- raw-real transfer `+-1` band accuracy
- raw-real transfer 10-band macro F1
- raw-real collapsed 4-class macro F1

### 3. Sanity Checks Before Full Training

Before a full V5 run:

- run unit tests for target conversion and metric conversion
- overfit a tiny balanced subset
- verify training loss can drop sharply on that subset
- verify predicted scalar values move across the severity range
- verify 10-band conversion boundaries match the official band definitions

If a model cannot overfit a tiny controlled subset, do not run the full
experiment.

### 4. Bias-Variance Reading

Learning curves should be interpreted before changing the next variable:

- high train error and high validation error means underfitting
- low train error and high validation error means overfitting
- noisy validation with small batch size can justify GroupNorm or more stable
  validation selection
- improving validation while raw-real transfer worsens means the synthetic task
  got easier but domain transfer did not improve

### 5. Error Analysis

Each run should produce:

- per-band precision, recall, and F1
- predicted band distribution
- true band distribution
- confusion matrix
- mean absolute band error
- severe error rate
- example-level prediction export for inspection

The example-level export should include:

- image path
- true score band
- true representative score
- predicted scalar score
- predicted score band
- absolute band error
- collapsed true coarse class
- collapsed predicted coarse class

## V5-A: Scalar Ordinal Regression

V5-A tests whether a scalar severity target works better than a 10-way
classification head.

### Target

Train the model to predict a normalized severity value:

- `normalized_score = training_representative_score / 100.0`

At evaluation time:

- clamp prediction to `[0.0, 1.0]`
- convert back to a 0-100 severity score
- map the score into the official 10 score bands
- compute the same score-band metrics used by V4

### Model

Use the current from-scratch residual CNN backbone, not the old simple regressor.

Recommended model:

- `residual_cnn_groupnorm_regressor`

Reason:

- batch size is still small
- GroupNorm is more stable than BatchNorm for small batches
- this tests the scalar head without changing to a pretrained backbone

Output head:

- one scalar logit
- apply `sigmoid` to constrain prediction to `[0, 1]`

Do not use an unconstrained 0-100 output for V5-A. It makes the optimization
scale worse and allows invalid predictions.

### Loss

Use a robust scalar loss:

- primary: Smooth L1 / Huber loss on normalized severity

Optional later additions, not first run:

- band-boundary penalty
- asymmetric severe-error penalty
- pairwise ranking loss

First V5-A should stay clean:

- one scalar head
- one robust scalar loss
- standard regularization

### Selection Metric

Use validation mean band error after mapping scalar predictions into the ten
official bands.

Reason:

- V5-A is trained as regression
- but the project target is still ten severity bands
- selection must match the evaluation target

### Initial Training Defaults

Recommended first full V5-A run:

- manifest: `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- image size: `384`
- batch size: `8`
- epochs: `160`
- optimizer: `adamw`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.1`
- scheduler: `reduce_on_plateau`
- scheduler metric: validation mean band error
- early stopping patience: `40`
- augmentation profile: `damage_safe`
- model variant: `residual_cnn_groupnorm_regressor`
- random seed: `42`

Expected run directory:

- `data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516`

## V5-B: Multitask Ordinal Model

Only run V5-B after V5-A gives a clear answer.

V5-B should use a shared from-scratch backbone with:

- scalar severity head
- 10-band score head
- optional collapsed coarse head

Purpose:

- scalar head learns order
- score-band head preserves band boundaries
- coarse head may stabilize broad severity grouping

This is more complex than V5-A and should not be the first V5 implementation.

### V5-B Round 1 Result

V5-B round 1 has been implemented and trained as a single controlled
challenger against V5-A.

Run:

- [v5_b_multitask_scalar_band_coarse_seed42_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516)

Configuration:

- manifest: `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- image size: `384`
- model: `residual_cnn_groupnorm_multitask`
- heads:
  - scalar normalized severity head
  - auxiliary 10-band score head
  - auxiliary 4-class coarse head
- split strategy: `metadata_family_holdout`
- optimizer: `adamw`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.1`
- scheduler: `reduce_on_plateau`
- augmentation profile: `damage_safe`
- loss weights:
  - scalar: `1.0`
  - coarse: `0.3`
  - auxiliary 10-band: `0.3`
  - auxiliary band ordinal penalty: `0.2`
- seed: `42`

Synthetic test result:

- best validation epoch: `160`
- early stopped: `false`
- best validation scalar mean band error: `1.9786`
- scalar exact 10-band accuracy: `0.1461`
- scalar `+-1` band accuracy: `0.4775`
- scalar `+-2` band accuracy: `0.6910`
- scalar `+-3` band accuracy: `0.8596`
- scalar `+-4` band accuracy: `0.9213`
- scalar mean band error: `1.9663`
- auxiliary 10-band exact accuracy: `0.2191`
- auxiliary 10-band mean band error: `2.1124`
- collapsed coarse macro F1 from scalar head: `0.3881`
- direct coarse macro F1: `0.4283`

Real raw evaluation:

- scalar exact 10-band accuracy: `0.1506`
- scalar `+-1` band accuracy: `0.3948`
- scalar `+-2` band accuracy: `0.6260`
- scalar `+-3` band accuracy: `0.8130`
- scalar `+-4` band accuracy: `0.9325`
- scalar mean band error: `2.1091`
- auxiliary 10-band exact accuracy: `0.1299`
- auxiliary 10-band mean band error: `2.2909`

Comparison artifact:

- [v5_b_round1_comparison.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/model_checkpoint_comparison/v5_b_round1_comparison.csv)

Interpretation:

- V5-B round 1 does not beat V5-A on synthetic test ordinal performance.
  V5-A synthetic mean band error is `1.7079`; V5-B scalar synthetic mean band
  error is `1.9663`.
- V5-B does improve real-domain ordinal transfer at wider tolerances compared
  with V5-A. Real raw mean band error improves from `2.2727` to `2.1091`;
  real raw `+-2` improves from `0.5896` to `0.6260`; real raw `+-3` improves
  from `0.7584` to `0.8130`; real raw `+-4` improves from `0.8935` to
  `0.9325`.
- Real raw `+-1` is effectively unchanged and slightly worse than V5-A:
  `0.3948` versus `0.3974`.
- The auxiliary 10-band head improves exact band classification inside V5-B,
  but the scalar head remains the better ordinal estimator because it has lower
  mean band error on synthetic and real evaluation.
- V5-B is therefore useful evidence that multitask learning may help broad
  real-domain robustness, but this first loss-weight setting is not a better
  synthetic baseline than V5-A.

### V5-B Longer-Training Check

Because V5-B round 1 selected epoch `160`, which was also the final scheduled
epoch, the model was extended from the epoch-160 checkpoint.

Extended run:

- [v5_b_multitask_scalar_band_coarse_seed42_20260516_longer_from160](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516_longer_from160)

Extended configuration:

- resumed from `checkpoints/latest.pt` at epoch `160`
- trained to `320` scheduled epochs
- early-stopping patience increased to `80`
- optimizer state was restored from the checkpoint
- the checkpoint learning rate was already at the minimum: `1e-05`

Result:

- best validation epoch moved from `160` to `310`
- best validation scalar mean band error improved from `1.9786` to about
  `1.829`
- synthetic test scalar mean band error worsened from `1.9663` to `2.0281`
- synthetic test scalar `+-1` worsened from `0.4775` to `0.4607`
- real raw scalar mean band error worsened from `2.1091` to `2.2442`
- real raw scalar `+-1` worsened from `0.3948` to `0.3532`
- real padded-384 scalar mean band error was similar/slightly better:
  `2.1429` to `2.1143`

Comparison artifact:

- [v5_b_longer_comparison.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/model_checkpoint_comparison/v5_b_longer_comparison.csv)

Interpretation:

- The first V5-B run was under-trained with respect to validation, but the
  longer validation improvement did not generalize to the held-out synthetic
  test or raw real data.
- This suggests the current V5-B validation signal is not reliable enough for
  model selection, or the loss weighting is pushing the model toward patterns
  that fit validation families without improving broader generalization.
- V5-B should not be promoted as-is. If V5-B remains interesting, the next
  version should tune loss weights and possibly use a stricter validation
  protocol before spending more long-run training time.

## V5-C: Threshold Ordinal Model

Only run V5-C after V5-A or V5-B.

V5-C should test a CORAL/CORN-style threshold head:

- predict whether severity is above each ordered threshold
- decode threshold outputs into the 10 bands

Purpose:

- explicitly model the ordered nature of score bands
- avoid treating all class mistakes as equally wrong

## Implementation Boundaries

Do not create a giant V5 script.

Recommended code boundaries:

- dataset:
  - extend target handling only if needed for normalized score targets
- models:
  - add a residual scalar regressor using the existing residual blocks
- metrics:
  - add score-to-band conversion and scalar-regression band metrics
- trainer:
  - upgrade or replace the old regressor trainer with V5-compatible controls
- CLI:
  - expose only the controls needed for a repeatable V5-A run
- evaluation:
  - reuse existing classifier comparison patterns where possible

## Acceptance Criteria For V5-A

V5-A is complete when:

- target conversion tests pass
- score-to-band mapping tests pass
- tiny-subset overfit succeeds
- full synthetic training completes
- checkpoints and epoch metrics are written
- synthetic analyzer/comparison artifacts exist
- raw-real transfer evaluation exists
- docs record whether V5-A beats or fails to beat V4/V4.5

## First Implementation Step

Before training, implement the V5-A infrastructure:

1. add residual scalar regressor model
2. add normalized target and score-to-band metric helpers
3. upgrade the regressor trainer to use V4-style controls and artifacts
4. add tests for conversion, metrics, config persistence, and overfit smoke
5. run a tiny overfit smoke
6. run the full V5-A challenger

## Implementation Status

V5-A infrastructure has started.

Implemented:

- `fine_normalized` dataset target mode
- scalar severity-to-score-band metric helpers
- `residual_cnn_groupnorm_regressor`
- V5-compatible regressor training controls
- per-epoch regression metrics CSV/JSONL
- per-epoch/latest/best checkpoints
- checkpoint resume support via `--resume-from`
- regression training curves
- CLI controls for optimizer, scheduler, split strategy, model variant,
  target mode, overfit subset, checkpoint cadence, and resume checkpoint

Verification:

- full test suite after resume support: `87 passed`
- CLI smoke run:
  [v5_a_scalar_cli_smoke_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_cli_smoke_20260516)
- 4-example overfit gate:
  [v5_a_scalar_overfit4_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_overfit4_20260516)
- fixed full-split preflight:
  [v5_a_preflight_fullsplit_fixed_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_preflight_fullsplit_fixed_20260516)
- full V5-A run:
  [v5_a_scalar_ordinal_seed42_20260516_fixed](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed)

Current status:

- the command path runs end to end on MPS
- the two-epoch smoke proves artifact wiring
- the 4-example overfit gate reached score MAE `3.31`, mean band error `0.25`,
  `+-1` band accuracy `1.0000`, and collapsed coarse macro F1 `1.0000`
- exact 10-band accuracy was `0.7500`, with the remaining miss landing in a
  neighboring band
- metadata-family holdout was corrected so saved `split_group_id` values are
  disjoint across train, validation, and test for 10-band runs
- fixed full-split preflight verified `0` group overlaps for train/val,
  train/test, and val/test
- the full V5-A run resumed from checkpoint epoch `74` and completed cleanly
- best validation epoch: `79`
- best validation mean band error: `1.4171`
- early stopped at epoch: `119`
- final synthetic test MAE: `16.297`
- final synthetic test 10-band accuracy: `0.1685`
- final synthetic test 10-band macro F1: `0.1529`
- final synthetic test mean band error: `1.7079`
- final synthetic test `+-1` band accuracy: `0.5225`
- final synthetic collapsed coarse macro F1: `0.4285`
- saved epoch metrics contain `119` unique epochs with no duplicates
- saved split artifacts still verify `0` group overlaps across train,
  validation, and test

Longer-training check:

- copied the completed V5-A run to:
  [v5_a_scalar_ordinal_seed42_20260516_fixed_longer_from119](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed_longer_from119)
- resumed from epoch `119`
- extended `num_epochs` to `240`
- raised early-stopping patience to `100`
- stopped at epoch `179`
- overall best epoch remained `79`
- best post-119 validation mean band error: `1.5294` at epoch `123`
- original best validation mean band error remained better: `1.4171`
- final selected synthetic test metrics did not change because the best
  checkpoint did not change

Real-domain evaluation from the best V5-A checkpoint:

- raw real manifest:
  [real_labeled_dataset_current.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/real_labeled_dataset_current.csv)
  - rows: `385`
  - score MAE: `20.702`
  - 10-band accuracy: `0.1351`
  - 10-band macro F1: `0.0923`
  - mean band error: `2.2727`
  - `+-1` band accuracy: `0.3974`
  - collapsed coarse macro F1: `0.2375`
- padded 384 real manifest:
  [real_labeled_dataset_current_padded_384_full.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/real_labeled_dataset_current_padded_384_full.csv)
  - rows: `385`
  - score MAE: `20.977`
  - 10-band accuracy: `0.1481`
  - 10-band macro F1: `0.1002`
  - mean band error: `2.3013`
  - `+-1` band accuracy: `0.3662`
  - collapsed coarse macro F1: `0.2549`
- evaluation artifacts:
  [real_eval_summary.json](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed/real_eval_summary.json)

Interpretation:

- V5-A scalar regression is mechanically complete and trains correctly.
- It learns the ordinal direction better than a flat random guess, but exact
  10-band performance is weak.
- The train/validation gap shows overfitting after the best epoch, so V5-A does
  not yet solve the 10-band task by itself.
- Training longer with the same objective did not improve the best checkpoint.
  It further reduced training error but did not improve validation.
- Real transfer is weaker than synthetic test performance, so V5-A does not yet
  bridge the synthetic-to-real gap.
- The next V5 step should compare this result against locked V4/V4.5 artifacts,
  then decide whether to move to V5-B multitask or V5-C threshold ordinal.
