# Initial Classifier Decisions

This file records the rationale for the first Dream2Detect classifier setup.

The goal is not to claim final project results yet.

The goal is to define a technically coherent **first coarse-classification training protocol** that:

- uses the synthetic data already prepared,
- is appropriate for a machine-learning course,
- is reproducible,
- and can later scale to a larger synthetic dataset without changing the core logic.

## Scope

This document applies to the **first 4-class classifier** only.

The task is:

- input: one package image
- output:
  - `intact`
  - `minor`
  - `moderate`
  - `severe`

This is the coarse-label branch of the project.

## Decision 1: Start With Coarse Classification Before Regression

### Decision

The first model should be a **4-class classifier**, not the fine-score regressor.

### Rationale

This is the correct starting point for three reasons.

1. Classification is the simpler target formulation.
   - the label space is smaller
   - the loss function is simpler
   - failure analysis is easier

2. It matches the project framing well.
   - one of the central project questions is whether coarse severity transfers more robustly than fine severity
   - starting with the coarse branch lets us validate the training stack on the easier side first

3. It is the safer debugging surface.
   - if the classifier cannot train sensibly, moving to regression would only make diagnosis harder

So the first training implementation should answer:

> can we reliably train a scratch CNN to predict the 4 coarse severity classes from the current synthetic dataset manifest?

## Decision 2: Use the Exported Synthetic Manifest, Not a Folder Scan

### Decision

Use:

- [synthetic_starting_dataset_phase1_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_starting_dataset_phase1_round1.csv)

as the dataset source for the first classifier pipeline.

### Rationale

Training should use a **manifest**, not just a directory of image files.

A folder only tells us:

- a file exists

The manifest tells us:

- the intended training label
- the QC state
- whether the label is reviewed or unreviewed
- the score band
- the image path

This matters because Dream2Detect explicitly distinguishes:

- reviewed labels
- relabeled samples
- unreviewed samples
- rejected samples

That distinction would be lost in a raw folder scan.

### Current dataset status

At the time of writing, this manifest contains `45` usable rows.

Class distribution:

- `intact`: `5`
- `minor`: `14`
- `moderate`: `13`
- `severe`: `13`

This is not large enough for strong scientific conclusions.
It is large enough to validate the training pipeline and run an exploratory synthetic-only experiment.

## Decision 3: Retire `128 x 128` For Paper-Relevant Training

### Decision

Use:

- **`224 x 224`** for paper-relevant synthetic-only classifier runs

Treat:

- **`128 x 128`** as a historical smoke-test resolution only

### Why this decision needed a check

Earlier, `128 x 128` was considered as a lighter starting size.

That is acceptable for a quick smoke test, but it is not the stronger choice for the real classifier training path, especially because:

- the project will grow to a larger synthetic dataset
- the damage signal often lives in corners, edges, punctures, short tears, and subtle deformations
- those signals can be weakened too much by aggressive downsampling

### Visual evidence

To test this, one subtle low-minor example was downsampled to both sizes and then upscaled again for human inspection.

Reference images:

- full-frame comparison:
  - [softened_corner_full_resolution_comparison.png](/Users/inventure71/VSProjects/School/Dream2Detect/docs/assets/resolution_checks/softened_corner_full_resolution_comparison.png)
- defect-area crop comparison:
  - [softened_corner_crop_resolution_comparison.png](/Users/inventure71/VSProjects/School/Dream2Detect/docs/assets/resolution_checks/softened_corner_crop_resolution_comparison.png)

### Interpretation

What the comparison shows:

- `128 x 128` preserves the global box shape
- but it visibly weakens the subtle corner damage signal
- `224 x 224` preserves noticeably more edge and corner structure

For this project, that matters.

The classifier should not learn only from:

- large crushing
- large openings
- broad deformation

It also needs a chance to learn from:

- softened corners
- light creases
- short tears
- small punctures

Those are exactly the signals that degrade first when the resolution is too low.

### Final rationale

There are two different goals here:

1. get the first training pipeline running quickly
2. preserve enough signal for subtle damage

So the practical decision is:

- keep `128 x 128` only as a fast smoke-test path
- use `224 x 224` for meaningful classifier training
- do not report `128 x 128` as the current primary experiment path

This is the correct compromise because the training stack has now been
validated and the stronger result came from preserving more image detail.

### Important boundary

The comparison images show that `224` preserves more subtle defect structure than `128`.

So the current rule is:

- `128` = historical fast smoke-test resolution
- `224` = default resolution for current classifier experiments

### Post-training update

The no-new-image escalation pass confirmed that image detail is already a
practical bottleneck. The strongest run so far uses the existing generated
images cached at `224 x 224`, not the `128 x 128` cache.

Current working default for follow-up synthetic-only classifier runs:

- image size: `224 x 224`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.1`
- balanced sampler: enabled
- augmentation: mild
- early-stopping patience: `20`

The `128 x 128` attempt failed as a paper-relevant path because it consistently
underperformed the `224 x 224` run and likely removed useful subtle-damage
signal. Keep its artifacts as historical evidence, but do not use it for the
main experiment unless a future controlled run gives a strong reason to reopen
that decision.

## Decision 3B: Cache Deterministic Preprocessing

### Decision

For repeated training at a fixed resolution, create a processed-image cache and a derived manifest.

Historical first cache:

- images:
  - `/Users/inventure71/VSProjects/School/Dream2Detect/data/cache/synthetic_all_processed_128`
- manifest:
  - `/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_starting_dataset_phase1_round1_processed_128.csv`

Current synthetic-only cache:

- images:
  - `/Users/inventure71/VSProjects/School/Dream2Detect/data/cache/synthetic_combined_phase1_plus_scaleup_200_processed_224`
- manifest:
  - `/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv`

### Rationale

Some preprocessing is deterministic and should not be repeated on every epoch:

- open image
- convert to RGB
- resize to the fixed training resolution

That work can be done once and cached.

Random augmentation should still happen during training only:

- horizontal flip
- color jitter

So the rule is:

- cache deterministic preprocessing
- keep stochastic augmentation online

This reduces unnecessary repeated image work while preserving the augmentation behavior needed for training.

## Decision 4: Split Into Train / Validation / Test

### Decision

Split the dataset into:

- `60%` train
- `20%` validation
- `20%` test

### Rationale

Even for the first exploratory run, the dataset must not be treated as one undivided pool.

We need three distinct roles:

#### Train

Used to fit model parameters.

#### Validation

Used during development to answer:

- is the model improving?
- when should training stop?
- which settings are better?

#### Test

Used only after model development for a cleaner final check on that particular run.

For a machine-learning course, this separation matters because it shows that the experiment is not using the same examples for:

- fitting
- tuning
- and final reporting

## Decision 5: Stratify The Split By Coarse Class

### Decision

The split should be:

- **stratified by `training_coarse_class`**

### Rationale

The current class distribution is imbalanced.

In particular:

- `intact` has only `5` examples

If we split purely at random, one split may accidentally receive:

- too few intact examples
- too many minor examples
- unstable class proportions

That would make the training and evaluation noisier for reasons unrelated to the model itself.

Stratification tries to preserve class proportions across:

- train
- validation
- test

This is standard good practice for small supervised datasets.

## Decision 6: Use A Fixed Random Seed

### Decision

Use a fixed seed, for example:

- `42`

### Rationale

Without a fixed seed:

- the split changes every run
- the comparison becomes harder to reproduce
- debugging gets worse

For a course project, reproducibility is not optional.

The exact numeric value of the seed is not important.

What matters is:

- it is fixed
- it is recorded
- future runs can reproduce the same split

## Decision 7: Keep Rejected Samples Out, But Allow Unreviewed Samples For Exploratory Training

### Decision

For the first exploratory synthetic-only classifier run:

- exclude `rejected` samples
- allow `unreviewed` samples
- preserve `training_label_source` and `qc_status` in the manifest

### Rationale

This is the correct compromise for the current project stage.

We do **not** yet have a fully human-reviewed synthetic pool.

If we required only reviewed samples, the usable set would be too small to be a meaningful starting point.

At the same time, we should not pretend unreviewed labels are as trustworthy as reviewed ones.

So the dataset should:

- use unreviewed examples for early training
- still record which examples are unreviewed
- stay compatible with later stricter filtering

This is why the manifest keeps:

- `qc_status`
- `training_label_source`

## Decision 8: Use Mild Augmentation On Train Only

### Decision

Apply mild image augmentation to the **training split only**.

Do not augment:

- validation
- test

### Rationale

Augmentation helps reduce overfitting by making the training data slightly more variable.

For this project, appropriate mild augmentation means things like:

- small horizontal flips
- small brightness changes
- small contrast changes

This is justified because the current synthetic dataset is still small and the classifier is trained from scratch.
Without augmentation, the model is more likely to memorize:

- exact object placement
- exact cardboard tone
- exact lighting pattern
- exact synthetic scene quirks

That would make the training loss look better than the real generalization ability.

### Why these augmentations are acceptable

#### Random horizontal flip

Horizontal flip is acceptable because overall package severity does not depend on whether the visible damage is on the left or right side of the image.

If a damaged corner or dent is mirrored left-to-right:

- the defect remains visible
- the severity class does not change

So horizontal flip reduces orientation bias without corrupting the label.

#### Mild color jitter

Mild brightness, contrast, and color variation is acceptable because the model should not depend too strongly on one exact lighting setup or one exact cardboard shade.

These changes should remain small.

If they become too strong, they risk:

- washing out subtle damage
- exaggerating shadows
- making the synthetic images less realistic

So the rule is:

- allow small nuisance variation
- do not alter the actual visible damage evidence

What we should avoid early:

- aggressive random crops that remove the defect
- heavy rotations that create unrealistic scenes
- transformations that change the meaning of the label

Validation and test should remain clean because they are supposed to measure how the model behaves on the stored dataset, not on artificial perturbations.

### Why validation and test stay unaugmented

Validation and test should answer:

- is the model improving?
- how well does it perform on the stored evaluation data?

If validation or test were augmented randomly:

- the evaluation distribution would change every run
- metric stability would get worse
- comparisons between runs would become less trustworthy

So the correct policy is:

- train: augmented
- validation: clean
- test: clean

## Decision 9: Use Class Weights In The Loss

### Decision

Use class-weighted cross-entropy loss for the first classifier.

### Rationale

The class distribution is not balanced.

If we train with an unweighted loss, the model can optimize itself by paying less attention to the rarest class.

That would be especially dangerous for:

- `intact`

Using class weights tells the optimizer:

- mistakes on underrepresented classes matter more

This does not solve the imbalance entirely, but it is the correct baseline response.

## Decision 10: Report Accuracy, Macro F1, And A Confusion Matrix

### Decision

For the first classifier runs, report at least:

- accuracy
- macro F1
- confusion matrix

### Rationale

Accuracy alone is not enough when classes are not perfectly balanced.

Macro F1 matters because:

- it gives equal importance to each class
- it shows whether one class is being ignored

The confusion matrix matters because:

- it shows which classes are getting confused
- it is much easier to interpret than a single number

For Dream2Detect, this is important because:

- `minor` vs `moderate`
- `moderate` vs `severe`

are likely to be the hardest boundaries.

## Decision 11: Treat This First Run As Pipeline Validation, Not Final Evidence

### Decision

The first classifier run should be treated as:

- an exploratory synthetic-only run
- a training-stack validation step

not as a final scientific conclusion.

### Rationale

Current limitations:

- only `45` rows in the starting manifest
- many examples are still unreviewed
- no real-data branch yet
- no synthetic-to-real comparison yet

So the first run is valuable for:

- validating the dataset loader
- validating the split logic
- validating the CNN training loop
- validating metrics and logging

It is not yet strong enough to support the final project claims.

## Locked Decisions From This Document

For the first classifier pipeline:

- task: `4-class coarse classification`
- dataset source: `synthetic_starting_dataset_phase1_round1.csv`
- image size: `224 x 224`
- split: `60 / 20 / 20`
- stratification: by coarse class
- seed: fixed
- rejected images: excluded
- unreviewed images: allowed for exploratory training
- augmentation: train only, mild
- loss: class-weighted cross-entropy
- optimizer: `AdamW`
- adaptive learning rate: `ReduceLROnPlateau` on validation macro F1
- metrics: accuracy, macro F1, confusion matrix
- artifacts: per-epoch checkpoints, epoch metrics CSV/JSONL, training curve PNG,
  and class-monitoring PNG updated during training
- monitoring: per-class precision, recall, F1, support, and predicted-class counts
- imbalance handling: inverse-frequency training sampler by default, with a CLI
  flag available for comparison runs
- debug controls: configurable dropout, weight decay, optimizer, learning-rate
  scheduler, augmentation toggle, early-stopping settings, and balanced
  overfit-subset mode

These decisions are enough to begin implementing:

- the dataset loader
- the split builder
- the classifier model
- the first training script
