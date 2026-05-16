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
