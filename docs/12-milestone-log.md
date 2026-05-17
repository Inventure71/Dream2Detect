# Milestone Log

This file records what has actually been completed in Dream2Detect at each milestone.

It exists to answer questions like:

- what is already locked,
- what has been implemented,
- what can now be claimed truthfully,
- what is still only planned.

## How To Read This

- `Locked` means the project decision is documented and should be treated as current truth.
- `Implemented` means code, docs, or data artifacts exist now.
- `Verified` means the implementation has been exercised and checked.

If something is not in this file, do not assume it is finished.

## Milestone 1: Project Reframing

### Outcome

The project was reframed into a comparative ML study of synthetic-to-real transfer under domain shift.

### What is now true

- The project is not a generic image-generation demo.
- The main question is:
  - how performance changes when we vary training data source and target formulation
- The core comparisons are locked:
  - `synthetic-only`
  - `real-only`
  - `synthetic + fine-tuning`
  - `fine severity`
  - `coarse severity`

### Evidence

- [01-project-understanding.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/01-project-understanding.md)
- [05-assignment-constraints.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/05-assignment-constraints.md)

## Milestone 2: Domain And Label System Locked

### Outcome

The project returned to the package/cardboard defect domain and locked the severity system.

### What is now true

- The target domain is visible package/cardboard defect severity.
- The coarse classes are locked:
  - `intact`
  - `minor`
  - `moderate`
  - `severe`
- The fine-to-coarse mapping is locked:
  - `0-10 -> intact`
  - `11-35 -> minor`
  - `36-65 -> moderate`
  - `66-100 -> severe`
- The working prompt/annotation bands are locked.

### Evidence

- [06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md)
- [06-package-severity-bands](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands)

## Milestone 3: Real-Data Policy Locked

### Outcome

The real branch was defined around a relabeled Kaggle cardboard-defect dataset.

### What is now true

- Kaggle images are merged into one unified candidate pool before relabeling.
- ChatGPT Vision is only a labeling assistant, not the final source of truth.
- Human review is required.
- The real split policy is locked:
  - `60%` held-out real test
  - `20%` small real training
  - `20%` validation / fine-tuning support

### Evidence

- [07-real-dataset-relabeling.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/07-real-dataset-relabeling.md)
- [01-project-understanding.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/01-project-understanding.md)

## Milestone 4: Execution Contracts And Repo Scaffold

### Outcome

The repo moved from planning-only docs to an execution-ready structure.

### What is now true

- The repo has explicit execution docs.
- The repo has a working folder structure for:
  - manifests
  - registries
  - synthetic outputs
  - real-data labeling
- A SQLite execution database exists for prompt/image workflow state.
- Template CSV files exist for prompt manifests and registries.

### Evidence

- [09-data-contracts.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/09-data-contracts.md)
- [10-execution-timeline.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/10-execution-timeline.md)
- [data/templates](/Users/inventure71/VSProjects/School/Dream2Detect/data/templates)

## Milestone 5: OpenAI Prompt And Image Pipeline

### Outcome

The repo now has a working prompt-drafting and image-generation pipeline.

### What is now true

- Prompt drafting uses `gpt-5.4-mini`.
- Image generation uses direct Image API with `gpt-image-2`.
- Prompt drafting uses structured outputs, not free-form JSON parsing.
- Image generation is fail-fast:
  - if an image call fails, the prompt is marked `failed`
  - the error is stored
  - the run stops
- Prompt rows and generated-image rows are linked in SQLite.

### Verified behavior

- A paid smoke test was run successfully:
  - one prompt was approved
  - one image was generated
  - the image was saved locally
  - the prompt row was updated correctly

### Important limitation

- A successful API call does **not** mean the generated image matches the intended severity band.
- Band correctness still requires review.

### Evidence

- [src/dream2detect/services/prompt_drafting.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/services/prompt_drafting.py)
- [src/dream2detect/services/image_generation.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/services/image_generation.py)
- [src/dream2detect/storage](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/storage)

## Milestone 6: Band Prompting Tightened

### Outcome

The prompt-facing band definitions were refined after the first image overshot its target severity.

### What is now true

- Lower and middle bands now explicitly forbid stronger escalation cues such as:
  - torn open
  - exposed interior
  - major collapse
  - severe crushing
- Prompt generation is more tightly constrained by the band rubric than before.

### Why this mattered

- The first generated `36-45` image followed the prompt well but landed too high in severity.
- That showed the problem was largely prompt calibration, not only image-model behavior.

### Evidence

- [06-package-severity-bands/README.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/README.md)
- [06-package-severity-bands](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands)

## Milestone 7: Structured Feature Diversity System

### Outcome

A real feature-diversity system now exists in code.

### What is now true

- Synthetic prompt planning no longer depends only on free-text prompt creativity.
- The system now defines explicit feature axes such as:
  - `damage_profile_primary`
  - `damage_profile_secondary`
  - `damage_location_primary`
  - `box_form_factor`
  - `box_pattern`
  - `label_presence`
  - `tape_profile`
  - `background_context`
  - `camera_angle`
  - `lighting_style`
- Feature values are sampled before prompt text is written.
- Prompt rows persist those structured feature assignments in the database.

### Balance claim that is now valid

We can now truthfully say:

> the repo has a system designed to keep class and feature distributions balanced during synthetic prompt planning

More precisely:

- score bands can be assigned in a controlled way,
- feature values are balanced **per axis**,
- sampling is **band-aware**,
- the system prefers the least-used allowed values,
- exact full feature combinations are not forced to be equally common.

### Important boundary

We cannot yet truthfully say:

> the final dataset is already balanced

because the full synthetic dataset has not been generated yet.

What we can say is:

> the balancing mechanism has been implemented and can be simulated offline before spending API money

### Evidence

- [11-feature-diversity-system.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/11-feature-diversity-system.md)
- [src/dream2detect/features](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/features)
- [src/dream2detect/services/prompt_drafting.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/services/prompt_drafting.py)

### Later refinement added after first prompt reviews

- the feature system now also enforces **primary-defect to location compatibility**
- this prevents awkward pairings such as:
  - corner-split defects centered on the face
  - side-compression defects assigned to a corner
- this change moved the remaining prompt-quality problem from “sampler emits bad pairings” to normal prompt-review quality control
- a later hardening pass also added a **stress-feature policy**
  - difficult nuisance conditions such as dim or harsh lighting can still appear
  - but they should not stack together inside one prompt assignment during early calibration

## Milestone 8: Offline Balance Simulator

### Outcome

An offline simulator was created to test large-scale feature balance without calling any model API.

### What is now true

- The repo can simulate a large prompt plan, such as `1000` prompts, without:
  - drafting prompts
  - generating images
  - spending money
- The simulator writes:
  - per-simulated-prompt assignments
  - per-axis count summaries
  - per-band counts
  - an HTML report with charts

### Verified behavior

- The simulator was run successfully for `1000` simulated prompts.
- The default simulation distributes prompts evenly across the `10` score bands:
  - `100` per band
- The report computes band-aware expected counts for each feature value.

## Milestone 9: Band-Control And Image QC Hardening

### Outcome

The repo now treats post-generation image QC as a first-class part of the synthetic pipeline instead of relying only on prompt quality.

### What is now true

- the `56-65` band guidance was tightened so broad side crushing and dominant geometry loss are pushed upward into severe
- the sampler now enforces a nuisance-stress budget per assignment
- generated-image rows can now record:
  - reviewed band and class
  - final accepted band and class
  - QC decision
  - QC notes

### Why this mattered

- the first 8-image calibration batch showed that one generated image was visually useful but mislabeled for its intended band
- without explicit QC states, the system had no clean way to relabel that image or reject it systematically

### Evidence

- [docs/06-package-severity-bands/56-65-moderate-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/56-65-moderate-high.md)
- [src/dream2detect/features/sampler.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/features/sampler.py)
- [src/dream2detect/storage](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/storage)

## Milestone 10: Phase 1 Calibration Run

### Outcome

The first full calibration run passed.

### What is now true

- a clean isolated calibration database was used without disturbing the main balancing history
- `50` prompts were drafted and reviewed
- weak prompts were rejected and replaced before image spend
- `12` calibration images were generated
- image QC produced:
  - `10` accepted as labeled
  - `1` accepted with relabeling
  - `1` rejected
- the dominant discovered failure mode was not band collapse but occasional over-strong printed text/label clutter
- that failure mode was fed back into the prompt-writing instructions
- the accepted reviewed examples were merged into the permanent database

### Evidence

- [13-phase1-calibration-report.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/13-phase1-calibration-report.md)
- [data/calibration/phase1_round1.sqlite3](/Users/inventure71/VSProjects/School/Dream2Detect/data/calibration/phase1_round1.sqlite3)
- [generated_images/phase1_round1](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/phase1_round1)

### Balance claim that is now valid

We can now truthfully say:

> we have an offline tool to verify whether the current feature sampler would keep synthetic prompt planning close to balanced before we spend on actual prompt drafting or image generation

### Evidence

- [simulation/offline_feature_balance](/Users/inventure71/VSProjects/School/Dream2Detect/simulation/offline_feature_balance)
- [simulation/offline_feature_balance/outputs/report.html](/Users/inventure71/VSProjects/School/Dream2Detect/simulation/offline_feature_balance/outputs/report.html)

## Current State Summary

At this point, the project has:

- a locked ML framing,
- a locked label system,
- a locked real-data policy,
- a working OpenAI prompt/image pipeline,
- a structured feature-diversity system,
- a generated 200-image synthetic scale-up batch,
- an offline simulator for feature-balance analysis.

It does **not** yet have:

- a QC-reviewed large synthetic dataset,
- a finished relabeled real dataset,
- training results,
- evaluation results.

So the correct status is:

> the project has generated the first large synthetic image batch, but it is not training-ready until image-level QC is completed

## Milestone 11: 200-Image Synthetic Scale-Up Generation

### Outcome

The first controlled 200-image synthetic scale-up batch was generated.

### What is now true

- `200` new images were generated into [generated_images/scaleup_200_round1](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/scaleup_200_round1)
- the SQLite database has `200` registered image references for this round
- the round is balanced at `20` generated images per score band:
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
- generated images remain `unreviewed` by default and still require image-level QC before being treated as clean training data
- a training-facing manifest for exactly this batch was exported to [data/datasets/synthetic_scaleup_200_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_scaleup_200_round1.csv)
- a deterministic `128 x 128` RGB cache was built at [data/cache/synthetic_scaleup_200_round1_processed_128](/Users/inventure71/VSProjects/School/Dream2Detect/data/cache/synthetic_scaleup_200_round1_processed_128)
- the processed-cache manifest was written to [data/datasets/synthetic_scaleup_200_round1_processed_128.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_scaleup_200_round1_processed_128.csv)
- two unregistered PNGs from an interrupted concurrent generation attempt were moved into [generated_images/scaleup_200_round1_unregistered_orphans](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/scaleup_200_round1_unregistered_orphans)

### Evidence

- [data/dream2detect.sqlite3](/Users/inventure71/VSProjects/School/Dream2Detect/data/dream2detect.sqlite3)
- [generated_images/scaleup_200_round1](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/scaleup_200_round1)
- [data/datasets/synthetic_scaleup_200_round1_processed_128.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_scaleup_200_round1_processed_128.csv)

### Operational note

The first concurrent generation attempt hit the OpenAI `gpt-image-2` rate limit of `5` image requests per minute. The successful completion path used sequential batches of `5` prompts to keep the database mapping stable.

## Milestone 12: Classifier Training Monitoring Upgrade

### Outcome

The synthetic classifier training loop now records enough evidence to diagnose
class collapse during training instead of only after reading the final confusion
matrix.

### What is now true

- each epoch records per-class precision, recall, F1, and support
- each epoch records predicted-class counts for train and validation splits
- each epoch still records loss, accuracy, macro F1, and confusion matrix
- `epoch_metrics.csv` now includes flat per-class monitoring columns for quick graphing
- `epoch_metrics.jsonl` preserves the richer nested metric payload
- `class_monitoring.png` graphs validation per-class F1 and predicted-class counts
- training and evaluation image tensors now use standard RGB normalization after `ToTensor()`
- the default training loader uses inverse-frequency class sampling for the training split
- `--no-balanced-sampler` can disable balanced sampling for comparison runs

### Why this mattered

The first 20-epoch synthetic-only classifier run reached only about `30%` test
accuracy and low macro F1. Its confusion matrix showed prediction collapse:
the model did not predict all four coarse classes. Per-class monitoring and
predicted-class counts make that failure mode visible while the run is still
in progress.

### Evidence

- [src/dream2detect/training/metrics.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/metrics.py)
- [src/dream2detect/training/dataset.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/dataset.py)
- [src/dream2detect/training/train_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/train_classifier.py)
- [scripts/train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)
- [tests/test_training_monitoring.py](/Users/inventure71/VSProjects/School/Dream2Detect/tests/test_training_monitoring.py)

## Milestone 13: Training-Control Experiments Without New Images

### Outcome

The classifier pipeline now supports controlled training experiments on the
existing synthetic dataset without generating additional images.

### What is now true

- classifier dropout is configurable from the CLI
- Adam weight decay is configurable from the CLI
- training augmentation can be disabled for debug runs
- early stopping can be enabled with patience and minimum-delta settings
- a debug overfit mode can train, validate, and test on the same small balanced subset
- a grid-runner script can execute small hyperparameter grids and write a summary CSV
- a deterministic `224 x 224` cache was built from existing source image paths, not from new image generation
- a label-audit candidate CSV was exported for manual review of weak/boundary examples

### Findings

The overfit debug run on `32` balanced examples reached `100%` accuracy and
`1.0000` macro F1. This is strong evidence that the CNN, optimizer, MPS runtime,
and data-loading path can fit the training signal.

The first small grid compared `128` vs `224`, learning rates `0.001` vs `0.0003`,
dropout `0.1`, weight decay `0.0001`, balanced sampling, augmentation, and early
stopping patience `6`. That patience was too aggressive for this dataset: every
grid run stopped by epoch `12`, before the longer 60-epoch run's later validation
improvements appeared.

Best result in that small grid:

- `128 x 128`
- learning rate `0.0003`
- weight decay `0.0001`
- dropout `0.1`
- balanced sampler enabled
- test accuracy `0.2653`
- test macro F1 `0.2310`

This did not beat the previous 60-epoch run (`0.3265` test accuracy, `0.2767`
macro F1), so the current evidence says short-patience early stopping is not
yet the right default. Future early-stopping runs should use a longer patience
or a warm-up period.

A targeted follow-up with longer patience improved the best result so far:

- `128 x 128`
- learning rate `0.0003`
- weight decay `0.0001`
- dropout `0.1`
- balanced sampler enabled
- augmentation enabled
- early-stopping patience `20`
- best validation epoch `39`
- stopped epoch `59`
- test accuracy `0.3469`
- test macro F1 `0.3245`

This is still not a strong classifier, but it is a real improvement over the
earlier `0.2767` macro F1 run and confirms that training controls can move the
result without adding images.

### Evidence

- [scripts/train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)
- [scripts/run_classifier_experiment_grid.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/run_classifier_experiment_grid.py)
- [scripts/build_image_cache.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/build_image_cache.py)
- [scripts/export_label_audit_candidates.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/export_label_audit_candidates.py)
- [data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv)
- [data/datasets/label_audit_candidates.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/label_audit_candidates.csv)
- [data/training_runs/synthetic_classifier_grid/grid_controlled_small/grid_summary.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_controlled_small/grid_summary.csv)
- [data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_128_128_targeted_lr0003_wd0001_do01_patience20](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_128_128_targeted_lr0003_wd0001_do01_patience20)

## Milestone 14: No-New-Image Training Escalation Pass

### Outcome

Several training improvements were tested without generating new synthetic
images. The best run so far is now the from-scratch CNN at `224 x 224` with a
slower learning rate, mild augmentation, balanced sampling, dropout, weight
decay, and longer patience.

### What is now true

- `224 x 224` cached training manifests can be built from existing source images
- classifier runs support `mild` and `strong` augmentation profiles
- classifier runs support `simple_cnn` and optional `resnet18` model variants
- optional pretrained `resnet18` is available as an upper-bound comparison, not
  as a replacement for the core from-scratch CNN experiment
- a synthetic regressor path can train on representative severity score and
  report bucketed coarse-class metrics
- a label-audit candidate CSV exists for manual review of likely weak or
  boundary examples

### Findings

The overfit debug run on `32` balanced examples reached `1.0000` macro F1. This
means the training loop, labels, model, optimizer, and MPS path are capable of
learning when the task is deliberately easy.

Best no-new-image classifier result so far:

- manifest: `synthetic_combined_phase1_plus_scaleup_200_processed_224.csv`
- image size: `224 x 224`
- model: from-scratch `simple_cnn`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.1`
- balanced sampler: enabled
- augmentation: mild
- early-stopping patience: `20`
- best validation epoch: `55`
- test accuracy: `0.5102`
- test macro F1: `0.4862`

This is a meaningful improvement over the earlier best `128 x 128` run
(`0.3265` accuracy, `0.2767` macro F1). It also beats the targeted `128 x 128`
training-control run (`0.3469` accuracy, `0.3245` macro F1).

Other comparison runs:

- `128 x 128` with strong augmentation reached `0.4490` accuracy and `0.4458`
  macro F1
- the `224 x 224` representative-score regressor reached `0.4082` bucket
  accuracy and `0.3531` bucket macro F1
- frozen pretrained `resnet18` reached `0.3878` accuracy and `0.3738` macro F1

Repeated-seed checks for the `224 x 224` configuration showed substantial
variance:

- seed `7`: `0.4490` accuracy, `0.4000` macro F1
- seed `99`: `0.3673` accuracy, `0.2900` macro F1
- two-seed mean macro F1: `0.3450`

So the current honest conclusion is not that the classifier is solved. The
conclusion is that `224 x 224` plus conservative training controls can produce
the strongest result so far, but the dataset is still small enough that split
and seed variance are large.

### Next Decision

Do not generate more images immediately. The stronger next step is to manually
audit labels and boundary cases using `label_audit_candidates.csv`, then rerun
the `224 x 224` configuration over multiple seeds or folds to separate real
improvement from split luck.

### Evidence

- [data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv)
- [data/datasets/label_audit_candidates.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/label_audit_candidates.csv)
- [data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_224_224_targeted_lr0003_wd0001_do01_patience20](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_224_224_targeted_lr0003_wd0001_do01_patience20)
- [data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_128_128_strongaug_lr0003_wd0001_do01_patience20](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_128_128_strongaug_lr0003_wd0001_do01_patience20)
- [data/training_runs/synthetic_regressor/synthetic_combined_phase1_plus_scaleup_200_processed_224_regressor_lr0003_wd0001_do01_patience20](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/synthetic_combined_phase1_plus_scaleup_200_processed_224_regressor_lr0003_wd0001_do01_patience20)
- [data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_224_resnet18_pretrained_frozen_smoke](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_224_resnet18_pretrained_frozen_smoke)
- [data/training_runs/synthetic_classifier_grid/grid_repeated_seed_224_best/grid_summary.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_repeated_seed_224_best/grid_summary.csv)

## Milestone 15: `224 x 224` Classifier Path Locked

### Outcome

The current classifier workflow has moved definitively to `224 x 224`.
The `128 x 128` path is now historical evidence and a possible smoke-test path,
not the paper-relevant training path.

### Why `128 x 128` failed

The `128 x 128` runs were useful for validating the training stack, but they did
not produce a strong classifier:

- earlier long `128 x 128` run: `0.3265` accuracy, `0.2767` macro F1
- targeted `128 x 128` run: `0.3469` accuracy, `0.3245` macro F1
- `128 x 128` strong augmentation run: `0.4490` accuracy, `0.4458` macro F1

The best `224 x 224` run reached `0.5102` accuracy and `0.4862` macro F1. That
is still not a final scientific result because repeated seeds showed high
variance, but it is enough evidence to retire `128 x 128` as the main path.

The likely reason is signal loss: package-defect severity often depends on
small edge, corner, crease, puncture, and deformation cues. Downsampling to
`128 x 128` weakens exactly those cues.

### Training workflow changes

- classifier CLI default image size is now `224`
- classifier CLI default manifest now points to the combined `224 x 224`
  processed manifest
- default optimizer is now `AdamW`
- default weight decay is now `0.0001`
- default dropout is now `0.1`
- adaptive learning rate is available through `ReduceLROnPlateau` on validation
  macro F1
- per-epoch logs now include the active learning rate
- checkpoints now persist scheduler state when a scheduler is enabled

### Current recommended long run

Use a longer run with scheduler patience shorter than early-stopping patience so
the learning rate has time to adapt before training stops.

```bash
python3 scripts/train_synthetic_classifier.py \
  --manifest data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv \
  --image-size 224 \
  --batch-size 8 \
  --num-epochs 120 \
  --learning-rate 0.0003 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --dropout 0.1 \
  --lr-scheduler reduce_on_plateau \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 10 \
  --min-learning-rate 0.00001 \
  --early-stopping-patience 30
```

### Next Experiment Options

Before changing the model architecture, test regularization deliberately:

- dropout `0.05`, `0.1`, `0.2`
- weight decay `0.00003`, `0.0001`, `0.0003`
- keep `224 x 224`, `AdamW`, balanced sampling, mild augmentation, and the same
  seed/fold policy fixed while comparing these

Do not treat a single improved seed as final. Report mean and variance across
seeds or folds.

### Evidence

- [scripts/train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)
- [src/dream2detect/training/train_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/train_classifier.py)
- [docs/15-initial-classifier-decisions.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/15-initial-classifier-decisions.md)

### Follow-up Run Result

The first long `224 x 224` run with `AdamW` and `ReduceLROnPlateau` completed.

Run directory:

- [synthetic_combined_phase1_plus_scaleup_200_processed_224_224_20260507_231209](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_scaleup_200_processed_224_224_20260507_231209)

Result:

- best validation epoch: `53`
- best validation macro F1: `0.4504`
- early stopped at epoch: `83`
- test accuracy: `0.4490`
- test macro F1: `0.4306`
- learning-rate changes: `0.0003` at epoch `1`, `0.00015` at epoch `17`,
  `0.000075` at epoch `64`, `0.0000375` at epoch `75`

Interpretation:

- this run beat the failed `128 x 128` baseline clearly
- it did not beat the earlier best `224 x 224` run (`0.4862` macro F1)
- the scheduler worked mechanically, but the first reduction happened at epoch
  `17`, which may have been too early for this noisy validation split

Next training comparison should keep `224 x 224` fixed and compare either:

- no scheduler versus scheduler with larger patience, or
- scheduler patience `20` instead of `10`

## Milestone 16: Class-Equalizing Prompt Expansion Drafted

### Outcome

A new targeted prompt expansion was drafted to fix the intact underrepresentation
before generating any new images.

### Starting Class Counts

The current `224 x 224` combined training manifest has:

- `intact`: `25`
- `minor`: `74`
- `moderate`: `73`
- `severe`: `73`

This means the old dataset is not merely band-balanced; it is class-imbalanced
because `intact` only occupies the `0-10` band while the damaged classes each
span three bands.

### Drafted Prompt Counts

The expansion drafted `555` prompt rows and did not generate images.

Drafted pending prompts by class:

- `intact`: `175`
- `minor`: `126`
- `moderate`: `127`
- `severe`: `127`

Drafted pending prompts by band:

- `0-10`: `175`
- `11-20`: `43`
- `21-30`: `42`
- `31-35`: `41`
- `36-45`: `42`
- `46-55`: `42`
- `56-65`: `43`
- `66-75`: `42`
- `76-85`: `43`
- `86-100`: `42`

If all drafted prompts are approved, generated, and accepted by QC, the combined
dataset target becomes approximately `200` examples per coarse class.

### Verification

Database prompt status after drafting:

- total prompt rows: `774`
- already generated approved prompts: `219`
- newly drafted pending prompts: `555`
- no image generation was run for the new prompt batch

### Next Step

Review the `0-10` intact prompts before image generation. Reject any prompt that
mentions dents, tears, crushed corners, punctures, openings, collapse, or other
structural deformation. The intact fix only works if these new `0-10` prompts
are visually diverse but genuinely undamaged.

## Milestone 17: Batch Image Generation Workflow Prepared

### Outcome

The OpenAI Batch API image-generation workflow was added and a send-ready batch
input file was prepared for the `555` drafted prompts. No batch was submitted
and no images were generated.

### What is now true

- batch preparation writes an OpenAI Batch API `.jsonl` input file
- each request uses endpoint `/v1/images/generations`
- each request uses model `gpt-image-2`
- each request uses image quality `medium`
- each request uses size `1024x1024`
- each request uses a stable `custom_id` in the form `prompt_<prompt_id>`
- local metadata tracks the batch directory, input file, output file target,
  image output subdirectory, model, quality, size, and future batch IDs
- status and ingest commands exist for after the batch is submitted

### Prepared Batch

Batch directory:

- [data/batches/image_generation/targeted_555_round1_medium_ready_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/batches/image_generation/targeted_555_round1_medium_ready_20260508)

Prepared files:

- `input.jsonl`: `555` Batch API requests
- `manifest.json`: prompt ID, custom ID, band, class, UID, and title mapping
- `metadata.json`: local batch state, currently `prepared`

Prepared request counts:

- `intact`: `175`
- `minor`: `126`
- `moderate`: `127`
- `severe`: `127`

### Commands

Submit later with:

```bash
python3 scripts/image_batch.py submit \
  --batch-dir data/batches/image_generation/targeted_555_round1_medium_ready_20260508
```

Check status later with:

```bash
python3 scripts/image_batch.py status \
  --batch-dir data/batches/image_generation/targeted_555_round1_medium_ready_20260508
```

Ingest completed results with:

```bash
python3 scripts/image_batch.py ingest \
  --batch-dir data/batches/image_generation/targeted_555_round1_medium_ready_20260508
```

### Verification

The prepared batch has:

- `555` JSONL lines
- first request: `prompt_241`
- last request: `prompt_795`
- metadata status: `prepared`
- batch ID: `null`

Database state after preparation:

- prompt statuses: `219` approved, `555` drafted
- image statuses: `219` generated, `555` pending

This confirms preparation did not submit a batch and did not mark any new image
as generated.

## Milestone 18: Targeted 555-Image Batch Ingested

### Outcome

The `targeted_555_round1_medium` OpenAI Batch API job completed successfully and
all generated images were ingested into the local Dream2Detect database.

### Batch Result

Batch ID:

- `batch_69fd209fbdd48190981ad5848900e89e`

OpenAI request counts:

- completed: `555`
- failed: `0`
- total: `555`

Local ingest counts:

- generated: `555`
- failed: `0`
- skipped: `0`

Generated image folder:

- [generated_images/targeted_555_round1_medium](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/targeted_555_round1_medium)

Local folder verification:

- PNG files found: `555`
- missing or empty image files: `0`
- folder size: about `904 MB`

### Class Counts

New generated rows by intended coarse class:

- `intact`: `175`
- `minor`: `126`
- `moderate`: `127`
- `severe`: `127`

New generated rows by intended score band:

- `0-10`: `175`
- `11-20`: `43`
- `21-30`: `42`
- `31-35`: `41`
- `36-45`: `42`
- `46-55`: `42`
- `56-65`: `43`
- `66-75`: `42`
- `76-85`: `43`
- `86-100`: `42`

### Exported Review Manifest

A batch-specific unreviewed manifest was exported:

- [data/datasets/synthetic_targeted_555_round1_medium_unreviewed.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_targeted_555_round1_medium_unreviewed.csv)

An intact-only QC manifest was also exported:

- [data/datasets/synthetic_targeted_555_round1_medium_intact_qc.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_targeted_555_round1_medium_intact_qc.csv)

This manifest has:

- rows: `555`
- all image paths present on disk
- `qc_status`: `unreviewed` for all rows

### Next Step

Do not treat this batch as clean training data yet. The next step is image-level
QC, starting with the `0-10` intact rows. The intact correction only works if
the `175` new intact images are truly undamaged while still visually diverse.

## Milestone 19: Combined 800-Image Synthetic Classifier Trained

### Outcome

The synthetic classifier was retrained on a combined synthetic manifest that
includes all `555` newly ingested targeted images plus the earlier processed
synthetic images.

### What is now true

- A combined source manifest exists:
  - [data/datasets/synthetic_combined_phase1_plus_targeted_555_source.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_source.csv)
- A deterministic `224x224` processed training manifest exists:
  - [data/datasets/synthetic_combined_phase1_plus_targeted_555_processed_224.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_processed_224.csv)
- The processed manifest has `800` rows:
  - `555` from `targeted_555_round1_medium_unreviewed`
  - `200` from `scaleup_200_round1_processed_128`
  - `45` from `phase1_round1_processed_128`
- The processed manifest is class-balanced:
  - `intact`: `200`
  - `minor`: `200`
  - `moderate`: `200`
  - `severe`: `200`
- No new image-generation run was performed for this milestone; existing local
  images were combined and resized into the training cache.

### Best Current Synthetic Classifier

Best checkpoint:

- [checkpoints/best.pt](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42/checkpoints/best.pt)

Best run directory:

- [img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42)

Best run settings:

- model: from-scratch `simple_cnn`
- pretrained weights: `false`
- image size: `224`
- optimizer: `AdamW`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.2`
- augmentation: `mild`
- balanced sampler: disabled
- random seed: `42`

Best run result:

- best validation epoch: `124`
- early stopped at epoch: `169`
- test accuracy: `0.5125`
- test macro F1: `0.5053`

### Evidence

- Full training log:
  - [docs/16-overnight-training-log.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/16-overnight-training-log.md)
- Grid summary:
  - [data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/grid_summary.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/grid_summary.csv)
- Focused training tests:
  - `15 passed`

### Important Limitation

This milestone produces the best current from-scratch CNN checkpoint from the
available synthetic data, but it does not prove the model is precise enough for
the final study. The `555` new examples still use intended, unreviewed labels,
so label QC remains the likely next bottleneck before making strong accuracy
claims.

## Milestone 20: Minor-Band Synthetic QC Round 1

### Outcome

A first targeted QC pass was completed for the weakest part of the new synthetic
dataset: all `126` newly generated targeted examples originally labeled as
`minor`.

### What is now true

- A minor-band QC queue exists:
  - [data/qc/targeted_555_round1_medium/minor_qc_queue.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/targeted_555_round1_medium/minor_qc_queue.csv)
- Visual review sheets exist:
  - [data/qc/targeted_555_round1_medium](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/targeted_555_round1_medium)
- QC decisions exist:
  - [data/qc/targeted_555_round1_medium/minor_qc_decisions_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/targeted_555_round1_medium/minor_qc_decisions_round1.csv)
- New round-1 QC manifests exist:
  - [data/datasets/synthetic_targeted_555_round1_medium_qc_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_targeted_555_round1_medium_qc_round1.csv)
  - [data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round1_source.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round1_source.csv)
  - [data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round1_processed_224.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round1_processed_224.csv)

### QC Result

- Reviewed targeted images: `126`
- Accepted as originally labeled: `79`
- Accepted with relabel: `47`
- Rejected: `0`

Relabel transitions:

- `11-20 -> 0-10`: `17`
- `21-30 -> 11-20`: `13`
- `21-30 -> 31-35`: `3`
- `31-35 -> 21-30`: `2`
- `31-35 -> 36-45`: `12`

### Training Check

A focused retrain on the partial-QC manifest completed, but it did not beat the
previous best checkpoint:

- run:
  [synthetic_combined_phase1_plus_targeted_555_qc_round1_processed_224_do02_balanced_mild_seed42_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_targeted_555_qc_round1_processed_224_do02_balanced_mild_seed42_20260508)
- best validation epoch: `32`
- test accuracy: `0.4062`
- test macro F1: `0.4128`

### Important Limitation

This is a partial QC pass. It corrects the weakest known `minor` band problem,
but the rest of the new targeted batch is still mostly unreviewed. Do not
replace the previous best synthetic classifier checkpoint with the partial-QC
checkpoint. The correct next step is to finish QC across the remaining intact,
moderate, and severe bands before rerunning the full classifier comparison.

## Milestone 21: Targeted 555-Image Batch Fully QC Reviewed

### Outcome

The newly generated `targeted_555_round1_medium` batch has now been fully
reviewed at image level. All `555` targeted examples have a QC decision and
training label source.

### What is now true

- Full round-2 targeted manifest exists:
  - [data/datasets/synthetic_targeted_555_round1_medium_qc_round2.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_targeted_555_round1_medium_qc_round2.csv)
- Full round-2 combined source manifest exists:
  - [data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round2_source.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round2_source.csv)
- Full round-2 combined processed training manifest exists:
  - [data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round2_processed_224.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round2_processed_224.csv)
- QC queues, review sheets, and decision files are stored under:
  - [data/qc/targeted_555_round1_medium](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/targeted_555_round1_medium)

### QC Counts

Across the `555` targeted images:

- accepted as originally labeled: `428`
- accepted with relabel: `127`
- rejected: `0`

Targeted batch class counts after QC:

- `intact`: `191`
- `minor`: `110`
- `moderate`: `122`
- `severe`: `132`

Combined processed manifest class counts after QC:

- `intact`: `216`
- `minor`: `184`
- `moderate`: `195`
- `severe`: `205`

### Full-QC Training Check

A two-run classifier grid was trained on the full-QC combined processed
manifest:

- [grid_targeted_555_full_qc_round2_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_targeted_555_full_qc_round2_20260508)

Results:

- balanced sampler:
  - best validation epoch: `102`
  - test accuracy: `0.4625`
  - test macro F1: `0.4691`
- no sampler / class-weighted loss:
  - best validation epoch: `30`
  - test accuracy: `0.4375`
  - test macro F1: `0.4349`

### Important Interpretation

The full-QC retrain did not beat the old intended-label checkpoint. That does
not mean QC was bad. It means the target labels changed from noisy intended
labels to visually reviewed labels, so the old and new headline metrics are not
measuring exactly the same target. The full-QC results are more honest evidence
for the real four-class severity task, and they show the current scratch CNN
still struggles with fine severity boundaries.

For strict model selection on the full-QC manifest, use the validation winner:

- [checkpoints/best.pt](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_targeted_555_full_qc_round2_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42/checkpoints/best.pt)

For highest observed full-QC test macro F1 in this small grid, the balanced
sampler checkpoint scored higher:

- [checkpoints/best.pt](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_targeted_555_full_qc_round2_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_balanced_mild_seed42/checkpoints/best.pt)

## Milestone 22: Full Synthetic QC And Controlled 384/Capacity/Augmentation Upgrade

### Outcome

The full `800`-image synthetic training pool is now visually QC reviewed, and a
controlled training upgrade pass was completed across resolution, model
capacity, stronger safe augmentation, and ordinal-aware loss.

### Full Synthetic QC

The remaining `234` older unreviewed images were reviewed using contact sheets:

- [data/qc/remaining_unreviewed_234_round1](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/remaining_unreviewed_234_round1)

Decision counts:

- accepted as labeled: `185`
- accepted with relabel: `49`
- rejected: `0`

The full reviewed source manifest is:

- [data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv)

Full synthetic class counts after QC:

- `intact`: `216`
- `minor`: `192`
- `moderate`: `189`
- `severe`: `203`

### New Training Inputs

Built full-QC processed manifests:

- [synthetic_combined_phase1_plus_targeted_555_full_qc_processed_224.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_processed_224.csv)
- [synthetic_combined_phase1_plus_targeted_555_full_qc_processed_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_processed_384.csv)
- [synthetic_combined_phase1_plus_targeted_555_full_qc_padded_224.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_padded_224.csv)
- [synthetic_combined_phase1_plus_targeted_555_full_qc_padded_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_padded_384.csv)

All source images in this synthetic batch are `1024 x 1024`, so direct square
resize and padded square resize are equivalent for these synthetic images. The
main tested input-signal change is therefore `224` versus `384`, not aspect
ratio correction.

### Model And Objective Changes

Added:

- `damage_safe` augmentation profile
- augmentation preview artifact:
  - [damage_safe_384_preview.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/augmentation_previews/damage_safe_384_preview.jpg)
- `residual_cnn` from-scratch model variant
  - trainable parameters: `2,779,044`
  - no pretrained weights
- optional ordinal-aware loss:
  - `--ordinal-loss-weight`

### Controlled Results

Controlled run root:

- [grid_full_qc_resolution_capacity_aug_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_full_qc_resolution_capacity_aug_20260508)

Best controlled run:

- [E_384_residual_damage_safe_ordinal02](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_full_qc_resolution_capacity_aug_20260508/E_384_residual_damage_safe_ordinal02)

Best checkpoint:

- [checkpoints/best.pt](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_full_qc_resolution_capacity_aug_20260508/E_384_residual_damage_safe_ordinal02/checkpoints/best.pt)

Result:

- test accuracy: `0.5313`
- test macro F1: `0.5153`
- mean ordinal error: `0.6000`
- off-by-one-or-correct rate: `0.8875`
- severe ordinal-error rate: `0.1125`

Controlled comparison:

- `224` simple CNN: `0.4020` test macro F1
- `384` simple CNN: `0.4800` test macro F1
- `384` residual CNN with mild augmentation: `0.4640` test macro F1
- `384` residual CNN with `damage_safe`: `0.5096` test macro F1
- `384` residual CNN with `damage_safe` plus ordinal loss: `0.5153` test macro F1

### Interpretation

The biggest verified gains came from preserving more image detail and then
regularizing the larger model. Capacity alone overfit. The best current path is
not simply "make the model bigger"; it is:

1. full QC labels,
2. `384 x 384` inputs,
3. stronger from-scratch residual model,
4. damage-safe augmentation,
5. ordinal-aware loss.

### Next Experiment Consideration

Training directly on the official ten `score_band` labels is a valid next
experiment, but it should not replace the four-class result yet.

Rationale:

- the band labels preserve more supervision than the four coarse classes;
- collapsed band predictions can still be evaluated as `intact`, `minor`,
  `moderate`, and `severe`;
- the target is ordinal, so adjacent-band mistakes should be treated as less
  serious than distant mistakes;
- a plain ten-way classifier may perform worse if it spreads the same `800`
  examples across too many labels or overfits boundary noise.

Recommended comparison:

- current best four-class run:
  `384`, `residual_cnn`, `damage_safe`, ordinal loss `0.2`;
- ten-band classifier with ordinal-aware loss, evaluated both as ten bands and
  after collapsing predictions to the official four coarse classes;
- optional multitask model with one coarse-class head and one score-band or
  representative-score auxiliary head.

### Ten-Band Follow-Up Result

The direct ten-band comparison was run with the same setup as the best
four-class controlled result:

- run:
  [score_band_384_residual_damage_safe_ordinal02_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/score_band_384_residual_damage_safe_ordinal02_20260508)
- target label mode: `score_band`
- model: `residual_cnn`
- image size: `384`
- augmentation: `damage_safe`
- ordinal loss weight: `0.2`
- best validation epoch: `51`

Test result:

- exact ten-band accuracy: `0.2875`
- ten-band macro F1: `0.2391`
- within-one-band-or-correct accuracy: `0.5688`
- mean band-index error: `1.7125`
- severe band-error rate: `0.4313`
- collapsed coarse accuracy: `0.5000`
- collapsed coarse macro F1: `0.4936`

Conclusion:

The ten-band target is informative, but it did not beat the direct four-class
best after collapsing back to coarse classes. The current best headline
checkpoint remains the four-class `384` residual CNN with `damage_safe`
augmentation and ordinal loss. The ten-band path is better treated as an
auxiliary/ordinal diagnostic or future multitask objective, not as the new main
objective yet.

### Multitask Follow-Up Result

A multitask version was also run with a primary coarse head and an auxiliary
score-band head:

- run:
  [F_384_residual_multitask_damage_safe_auxband03_ord02_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/F_384_residual_multitask_damage_safe_auxband03_ord02_20260508)
- model: `residual_cnn_multitask`
- primary target: four coarse classes
- auxiliary target: ten official `score_band` labels
- auxiliary band loss weight: `0.3`
- auxiliary band ordinal loss weight: `0.2`
- best validation epoch: `11`
- early stopped epoch: `26`

Test result:

- primary coarse accuracy: `0.5000`
- primary coarse macro F1: `0.5042`
- auxiliary score-band exact accuracy: `0.3500`
- auxiliary score-band macro F1: `0.2909`
- auxiliary within-one-band-or-correct accuracy: `0.5625`
- auxiliary severe band-error rate: `0.4375`

Conclusion:

The multitask model improved score-band learning compared with the standalone
ten-band classifier, but it did not beat the direct four-class run E on the
headline coarse metric. Keep E as the best current checkpoint. Treat multitask
training as a useful diagnostic and possible tuning direction, not as the
current final model.

## Milestone 23: First Synthetic-To-Real Evaluation

### Outcome

The best synthetic-only checkpoint was evaluated directly on the current labeled
real dataset.

### What is now true

- The current real manifest exists and is loadable:
  - [real_labeled_dataset_current.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/real_labeled_dataset_current.csv)
- It has `164` labeled real rows.
- All referenced real image files exist.
- The checkpoint analyzer now supports:
  - `--manifest` for out-of-run evaluation
  - `--all-as-test` for evaluating every row in a manifest as one test set

### Evaluated Checkpoint

- [E_384_residual_damage_safe_ordinal02/checkpoints/best.pt](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/grid_full_qc_resolution_capacity_aug_20260508/E_384_residual_damage_safe_ordinal02/checkpoints/best.pt)

Evaluation artifacts:

- [E_384_residual_damage_safe_ordinal02_on_real_current](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_transfer/E_384_residual_damage_safe_ordinal02_on_real_current)

### Real Dataset Distribution

- `intact`: `8`
- `minor`: `45`
- `moderate`: `81`
- `severe`: `30`

### Result

- real-domain accuracy: `0.2927`
- real-domain macro F1: `0.2574`
- mean confidence: `0.6125`
- mean true-class probability: `0.2720`
- off-by-one-or-correct rate: `0.7073`
- severe ordinal-error rate: `0.2927`

Real confusion matrix:

| true \ predicted | intact | minor | moderate | severe |
| --- | ---: | ---: | ---: | ---: |
| intact | `6` | `2` | `0` | `0` |
| minor | `24` | `9` | `9` | `3` |
| moderate | `25` | `23` | `29` | `4` |
| severe | `18` | `2` | `6` | `4` |

### Interpretation

This is strong evidence of synthetic-to-real domain shift. The synthetic-only
checkpoint does not transfer cleanly to the real labeled set.

The model overpredicts `intact` on real damaged examples:

- target `intact` count: `8`
- predicted `intact` count: `73`

It also under-detects severe real damage:

- true severe examples: `30`
- predicted severe correctly: `4`

An always-`moderate` baseline would get higher accuracy on this imbalanced real
set (`0.4939`) but much lower macro F1 (`0.1653`). So the synthetic model is not
just worse in every sense; it has broader class behavior, but it is badly
miscalibrated for the real visual domain.

### Next Decision

Do not keep tuning synthetic-only models as the main path. The next controlled
experiments should use the real data:

- real-only baseline using the locked real split policy;
- synthetic checkpoint plus real fine-tuning;
- comparison against the synthetic-only real-transfer result above.

## Milestone 24: Real-Domain Debugging And Baselines

### Outcome

A real-domain debugging pass was completed after the poor synthetic-to-real
transfer result.

### What is now true

- Real images are not square like the synthetic images:
  - `163 / 164` real images have non-square dimensions
  - aspect ratios range from about `0.3866` to `2.8430`
- A padded `384 x 384` real manifest now exists:
  - [real_labeled_dataset_current_padded_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/real_labeled_dataset_current_padded_384.csv)
- The classifier training CLI now supports the documented real split policy:
  - `--train-fraction`
  - `--val-fraction`
  - `--test-fraction`
- The classifier training CLI now supports model-weight-only initialization for
  fine-tuning:
  - `--init-from-checkpoint`

### Root-Cause Checks

Re-evaluating synthetic checkpoint E on padded real images did not solve the
problem:

- raw real all-as-test: `0.2927` accuracy, `0.2574` macro F1
- padded real all-as-test: `0.3110` accuracy, `0.2445` macro F1

The overfit sanity check passed:

- run:
  [real_padded384_residual_overfit32_sanity_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/real_classifier/real_padded384_residual_overfit32_sanity_20260508)
- setup: `32` balanced real examples, same subset for train/validation/test
- result: `1.0000` accuracy, `1.0000` macro F1

This means the model and data pipeline can learn real labels. The failure is
not a basic implementation bug; it is domain shift plus limited real training
data.

### Controlled Real Split

All real training comparisons below used:

- `20%` real train
- `20%` real validation
- `60%` held-out real test
- padded `384 x 384` real inputs
- random seed `42`

Comparison summary:

| run | setup | test accuracy | test macro F1 |
| --- | --- | ---: | ---: |
| synthetic E on raw real, all-as-test | synthetic-only checkpoint | `0.2927` | `0.2574` |
| synthetic E on padded real, all-as-test | synthetic-only checkpoint | `0.3110` | `0.2445` |
| real overfit sanity | scratch residual, 32 balanced real examples | `1.0000` | `1.0000` |
| real-only scratch | padded real, balanced sampler | `0.5152` | `0.3887` |
| real-only scratch | padded real, class-weighted loss | `0.4444` | `0.2915` |
| synthetic E -> real fine-tune | padded real, balanced sampler | `0.4343` | `0.3221` |
| pretrained ResNet18 diagnostic | padded real, balanced sampler | `0.5152` | `0.4509` |

Summary artifact:

- [real_domain_experiment_summary_20260508.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_transfer/real_domain_experiment_summary_20260508.csv)

Per-class diagnostic artifacts:

- [real_only_padded384_residual_damage_safe_ord02_seed42_20260508_diagnostics](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_models/real_only_padded384_residual_damage_safe_ord02_seed42_20260508_diagnostics)
- [diagnostic_pretrained_resnet18_real_padded384_damage_safe_ord02_seed42_20260508_diagnostics](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_models/diagnostic_pretrained_resnet18_real_padded384_damage_safe_ord02_seed42_20260508_diagnostics)

### Interpretation

The current synthetic data is not helping real-domain performance under the
tested scratch-CNN setup. Real-only scratch training beats synthetic-only
transfer and synthetic-initialized real fine-tuning.

The pretrained ResNet18 diagnostic performs best on real macro F1, which points
to representation quality as a bottleneck for the tiny real train split. This
does not replace the core from-scratch experiment unless the project plan is
changed, but it is strong evidence that the current scratch model plus current
synthetic data is not enough for real transfer.

The pretrained diagnostic's macro-F1 gain comes mostly from better `minor`
recall:

- scratch real-only `minor` recall: `3 / 27`
- pretrained diagnostic `minor` recall: `17 / 27`

The scratch real-only model overpredicts `moderate` on the held-out real test
set, while the pretrained diagnostic spreads predictions more usefully across
`minor`, `moderate`, and `severe`.

### Current Best Real-Domain Models

- best core scratch model:
  [real_only_padded384_residual_damage_safe_ord02_seed42_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/real_classifier/real_only_padded384_residual_damage_safe_ord02_seed42_20260508)
- best diagnostic model:
  [diagnostic_pretrained_resnet18_real_padded384_damage_safe_ord02_seed42_20260508](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/real_classifier/diagnostic_pretrained_resnet18_real_padded384_damage_safe_ord02_seed42_20260508)

Focused verification:

- `PYTHONPATH=src python3 -m pytest -q tests/test_training_artifacts.py tests/test_training_controls.py tests/test_training_monitoring.py tests/test_experiment_grid.py`
- result: `18 passed in 11.86s`

## Milestone 25: V2 Synthetic Generation Pipeline

### Outcome

Created a versioned v2 synthetic generation preparation pipeline while
preserving the original v1 flow.

### What changed

- Added v1 compatibility namespace:
  [src/dream2detect/pipelines/v1](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/pipelines/v1)
- Added v2 pipeline package:
  [src/dream2detect/pipelines/v2](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/pipelines/v2)
- Added v2 preparation CLI:
  [scripts/v2_prepare_image_pipeline.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/v2_prepare_image_pipeline.py)
- Added v2 design documentation:
  [docs/17-v2-synthetic-generation-pipeline.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/17-v2-synthetic-generation-pipeline.md)

### Design Intent

V2 is built around the real-domain failure analysis, not generic prompt
expansion. It targets the cases where the synthetic model failed:

- damaged real packages predicted as `intact`
- flat taped mailers
- subtle minor/moderate boundary damage
- realistic tape, labels, clutter, shadows, and phone-camera compression
- hard intact negatives that look visually busy without structural damage
- severe damage that remains photographically realistic

### Preview Artifact

Prepared locally with no API submission:

- [v2_real_failure_targeted_preview_20260509_r2](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_preview_20260509_r2)

Preview settings:

- `count_per_band=3`
- total prompts: `30`
- model field: `gpt-image-2`
- quality field: `medium`
- size field: `1024x1024`

Artifacts:

- `prompt_manifest.csv`
- `input.jsonl`
- `manifest.json`
- `run_config.json`
- `summary.json`
- `qc_rubric.md`

### Verification

- `PYTHONPATH=src python3 -m pytest -q tests/test_v2_pipeline.py tests/test_image_batch.py`
- result: `7 passed in 0.27s`

## Milestone 26: Experiment State Cleaned And Baselines Frozen

### Outcome

Cleaned the current experiment state after returning to the project, froze the
current baseline table, and updated the v2 pipeline to target the exact
real-domain gaps found in the diagnostics.

### Experiment State Cleanup

The training-facing full-QC manifest is now the current synthetic source of
truth:

- [synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv)
- rows: `800`
- class counts:
  - `intact`: `216`
  - `minor`: `192`
  - `moderate`: `189`
  - `severe`: `203`

The execution SQLite database had drifted behind this manifest:

- before cleanup, `762` prompt rows were still marked `unreviewed`
- all `555` targeted medium-generation rows were still marked `unreviewed`
- the full-QC manifest itself had stable unique `prompt_uid` values, but reused
  some numeric `prompt_id` values across older source pools

Added a repeatable sync utility:

- [scripts/sync_prompt_qc_from_manifest.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/sync_prompt_qc_from_manifest.py)

The database was backed up before syncing:

- [dream2detect-before-qc-sync-20260513.sqlite3](/Users/inventure71/VSProjects/School/Dream2Detect/data/backups/dream2detect-before-qc-sync-20260513.sqlite3)

Then the prompt-level SQLite QC fields were synced from the full-QC manifest by
stable `prompt_uid`:

```bash
PYTHONPATH=src python3 scripts/sync_prompt_qc_from_manifest.py \
  --manifest data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv \
  --only-accepted \
  --skip-missing
```

Sync result:

- synced rows: `766`
- skipped rows missing from SQLite: `34`
- remaining SQLite `unreviewed` rows: `7`
- targeted 555 QC status after sync:
  - `accepted_as_labeled`: `428`
  - `accepted_relabel`: `127`

The `34` skipped rows are older `phase1_round1_processed_128` rows that exist in
the full-QC training manifest but are not present in the current execution
database. The manifest remains authoritative for training. The remaining `7`
SQLite `unreviewed` rows are older generated prompts outside the current
full-QC training manifest and should not be included in paper-facing training
unless they receive explicit QC.

### Frozen Baseline Table

Created the current baseline table:

- [current_baseline_table_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_transfer/current_baseline_table_20260513.csv)

Current frozen headline results:

| run | role | accuracy | macro F1 |
| --- | --- | ---: | ---: |
| `E_384_residual_damage_safe_ordinal02` | best current synthetic-only checkpoint on synthetic test | `0.5313` | `0.5153` |
| `synthetic_E_raw_real_all` | synthetic checkpoint on all real images | `0.2927` | `0.2574` |
| `synthetic_E_padded_real_all` | synthetic checkpoint on padded real images | `0.3110` | `0.2445` |
| `real_only_balanced_sampler_scratch` | best current scratch real baseline | `0.5152` | `0.3887` |
| `synthetic_E_finetune_real_scratch` | synthetic-initialized real fine-tune | `0.4343` | `0.3221` |
| `diagnostic_pretrained_resnet18_real` | diagnostic pretrained upper bound | `0.5152` | `0.4509` |

Interpretation remains unchanged: current synthetic data does not yet improve
real-domain performance. The next synthetic generation round must target domain
gap repair, not generic dataset growth.

### Missing Real-Domain Coverage

The diagnostics show these concrete missing capabilities:

- synthetic checkpoint overpredicts `intact` on real damaged examples:
  - true real `intact`: `8`
  - predicted `intact`: `73`
  - false-positive `intact`: `67`
- synthetic checkpoint has weak real recall:
  - `minor`: `9 / 45`
  - `moderate`: `29 / 81`
  - `severe`: `4 / 30`
- highest-risk errors are high-confidence damaged examples predicted as
  `intact`, especially real severe and moderate packages
- real-only scratch baseline also struggles with `minor` recall:
  - `3 / 27`
- pretrained diagnostic improves real `minor` recall:
  - `17 / 27`
  - this suggests representation quality and real-domain visual realism are
    important bottlenecks

### V2 Pipeline Update

Updated the v2 prompt generator so each scenario has explicit eligible coarse
classes and a `real_gap_priority`. Added targeted scenario families for:

- hard intact packages with confusing tape, label, clutter, and shadow cues
- low-contrast minor damage to repair minor recall
- moderate geometry loss that should not be predicted as intact
- severe openings/collapse that should remain visible under real capture
  artifacts

Prepared a new review-only v2 preview with no API submission:

- [v2_real_failure_targeted_preview_20260513_gap_repair](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_preview_20260513_gap_repair)

Preview summary:

- prompts: `30`
- `count_per_band`: `3`
- quality field: `medium`
- size field: `1024x1024`
- hard negative prompts: `3`
- explicit real-gap priorities are included in `summary.json`

### Verification

- `PYTHONPATH=src python3 -m pytest -q tests/test_v2_pipeline.py tests/test_image_batch.py`
- result: `8 passed in 0.38s`

## Milestone 27: V2 Prompt Preview Reviewed

### Outcome

Reviewed the May 13 V2 preview prompts before API submission and tightened the
generator to avoid wasting image-generation budget on ambiguous prompt wording.

### Prompt Review Findings

The scenario balance and real-gap targeting were acceptable, but the initial
May 13 preview had avoidable prompt risks:

- intact prompts inherited shared wording that said the assigned condition or
  defect must be visible
- intact feature hints inherited `primary damage` wording from the shared
  damage-location axis
- intact prompts still allowed generic material language mentioning dents
- some scenario-level label constraints were softer than the desired global
  unreadable/no-logo/no-address rule

These were generator issues, not hand-edited CSV issues.

### Fix

Updated the v2 generator so:

- intact prompts say the package must be structurally intact and recognizably
  undamaged at `384px`
- intact prompts explicitly prohibit dents, tears, crushed corners, holes,
  split seams, and structural deformation
- intact feature hints replace `primary damage` with `main visible package area`
- damaged prompts still require the assigned defect to be visible at `384px`
- every prompt includes a hard text-safety constraint for labels, barcodes,
  addresses, logos, and large text

Regenerated the current review candidate:

- [v2_real_failure_targeted_preview_20260513_gap_repair_r3](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_preview_20260513_gap_repair_r3)

### Review Result

Automated prompt checks on `r3` found no remaining intact/damage wording issue,
no missing global text-safety constraint, and valid Batch API JSONL fields.

Preview summary:

- prompts: `30`
- score bands: `3` prompts per band
- endpoint: `/v1/images/generations`
- model field: `gpt-image-2`
- quality field: `medium`

### Verification

- `PYTHONPATH=src python3 -m pytest -q tests/test_v2_pipeline.py tests/test_image_batch.py`
- result: `10 passed in 0.25s`

## Milestone 28: V2 Pilot Submitted To Batch API

### Outcome

Submitted the reviewed `r3` V2 pilot prompts to the OpenAI Batch API for image
generation.

### Submitted Batch

- batch directory:
  [v2_real_failure_targeted_preview_20260513_gap_repair_r3](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_preview_20260513_gap_repair_r3)
- request count: `30`
- endpoint: `/v1/images/generations`
- model field: `gpt-image-2`
- quality field: `medium`
- size field: `1024x1024`
- batch id: `batch_6a044ec544b88190949293c8e5c42e77`
- input file id: `file-PnEyLzR6E8y4JahWDU3MJt`
- final status: `completed`
- request counts: `30` completed, `0` failed, `30` total
- output file id: `file-Y5iF2V3wRHXYg4GSxvqZps`

### Retrieval Readiness

The V2 batch uses `custom_id` values like `v2_prompt_000001`, not the older
SQLite prompt ids used by V1. The ingest path was updated before completion so
V2 results can be downloaded later into:

- `generated_images/v2_real_failure_targeted_preview_20260513_gap_repair_r3/`

and written back to a batch-local:

- `generated_image_manifest.csv`

That manifest will preserve the prompt metadata needed for QC: score band,
coarse class, scenario id, real-gap priority, prompt text, output path, and
Batch ids.

### Ingest Result

The completed batch was ingested locally:

- generated images: `30`
- failed: `0`
- skipped: `0`
- image folder:
  [generated_images/v2_real_failure_targeted_preview_20260513_gap_repair_r3](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/v2_real_failure_targeted_preview_20260513_gap_repair_r3)
- generated image manifest:
  [generated_image_manifest.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_preview_20260513_gap_repair_r3/generated_image_manifest.csv)

The generated manifest has `30` rows, all `image_status=generated`, and all
referenced image files exist.

### Verification

- `PYTHONPATH=src python3 -m pytest -q tests/test_v2_pipeline.py tests/test_image_batch.py`
- result: `10 passed in 0.19s`

## Milestone 29: V2 Pilot Image QC Round 1

### Outcome

Completed image-level QC for the `30` generated V2 pilot images.

### QC Artifacts

- contact sheets:
  [data/qc/v2_real_failure_targeted_preview_20260513_gap_repair_r3](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/v2_real_failure_targeted_preview_20260513_gap_repair_r3)
- QC decisions:
  [v2_qc_decisions_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/v2_real_failure_targeted_preview_20260513_gap_repair_r3/v2_qc_decisions_round1.csv)
- reviewed generated-image manifest:
  [generated_image_manifest_qc_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_preview_20260513_gap_repair_r3/generated_image_manifest_qc_round1.csv)

### QC Result

- reviewed images: `30`
- accepted as labeled: `25`
- accepted with relabel: `5`
- rejected: `0`
- missing image files: `0`

Final class distribution after QC:

- `intact`: `3`
- `minor`: `9`
- `moderate`: `9`
- `severe`: `9`

Final band distribution after QC:

- `0-10`: `3`
- `11-20`: `2`
- `21-30`: `3`
- `31-35`: `4`
- `36-45`: `3`
- `46-55`: `3`
- `56-65`: `3`
- `66-75`: `3`
- `76-85`: `6`
- `86-100`: `0`

### Relabels

- `v2_prompt_000006`: `11-20 -> 21-30`
  - visible edge/side crease is too strong for minor-low
- `v2_prompt_000009`: `21-30 -> 31-35`
  - end/corner deformation reads as high minor
- `v2_prompt_000028`: `86-100 -> 76-85`
  - severe but not among the reserved top-band examples
- `v2_prompt_000029`: `86-100 -> 76-85`
  - severe but not extreme enough for top band
- `v2_prompt_000030`: `86-100 -> 76-85`
  - severe but not extreme enough for top band

### Interpretation

The V2 pilot is usable and materially better than the earlier synthetic images:
all images are photorealistic, package-centered, and keep labels/text
non-dominant. The main weakness is that the `86-100` severe-high prompt family
under-shoots the reserved top band; it reliably generates severe images, but
not the most extreme severity level. Before scaling V2, either avoid requesting
many `86-100` images or strengthen that scenario family.

## Milestone 30: V2 Scale-Up Candidate Prepared After QC

### Outcome

Updated the V2 generator after QC and prepared a scale-up candidate without API
submission.

### Generator Changes

- Added per-band count overrides via repeated CLI flags:
  - `--band-count BAND=COUNT`
- Added dedicated `86-100` severe-high scenarios:
  - `severe_high_extreme_collapse_anchor`
  - `severe_high_open_cavity_anchor`
- Changed scenario eligibility so if a band has exact band-specific scenarios,
  those scenarios are used for that band instead of the broader coarse-class
  scenario pool.

This directly addresses the pilot QC finding that `86-100` generated realistic
severe images but undershot the reserved top severity band.

### Prepared Candidate

Prepared locally with no API submission:

- [v2_real_failure_targeted_scale_candidate_20260513_r1](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1)

Candidate settings:

- total prompts: `132`
- quality field: `medium`
- model field: `gpt-image-2`
- size field: `1024x1024`

Band counts:

- `0-10`: `8`
- `11-20`: `16`
- `21-30`: `16`
- `31-35`: `16`
- `36-45`: `16`
- `46-55`: `16`
- `56-65`: `16`
- `66-75`: `12`
- `76-85`: `12`
- `86-100`: `4`

### Validation

- prompt rows: `132`
- JSONL rows: `132`
- unique custom ids: `132`
- all model fields: `gpt-image-2`
- all quality fields: `medium`
- prompt review checks: `0` issues
- `86-100` rows all use dedicated severe-high scenarios

### Verification

- `PYTHONPATH=src python3 -m pytest -q tests/test_v2_pipeline.py tests/test_image_batch.py`
- result: `12 passed in 0.27s`

## Milestone 31: V2 Scale-Up Candidate Submitted

### Outcome

Submitted the reviewed V2 scale-up candidate to the OpenAI Batch API.

### Submitted Batch

- batch directory:
  [v2_real_failure_targeted_scale_candidate_20260513_r1](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1)
- request count: `132`
- endpoint: `/v1/images/generations`
- model field: `gpt-image-2`
- quality field: `medium`
- size field: `1024x1024`
- batch id: `batch_6a046c93c6a48190bb5d32f4a6060d50`
- input file id: `file-Xuohw3AEwz4fy9FJfSZ8ec`
- final status: `completed`
- request counts: `132` completed, `0` failed, `132` total
- output file id: `file-QNhy6XEFs7J69dguZRWbso`

### Next Step

Refresh status later and ingest only after the Batch API reports `completed`:

```bash
python3 scripts/image_batch.py status \
  --batch-dir data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1
```

```bash
python3 scripts/image_batch.py ingest \
  --batch-dir data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1
```

### Ingest Result

The completed scale-up batch was downloaded permanently to local project
storage before cloud cleanup:

- generated images: `132`
- failed: `0`
- skipped: `0`
- missing local files after verification: `0`
- image folder:
  [generated_images/v2_real_failure_targeted_scale_candidate_20260513_r1](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/v2_real_failure_targeted_scale_candidate_20260513_r1)
- generated image manifest:
  [generated_image_manifest.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1/generated_image_manifest.csv)
- saved output JSONL:
  [output.jsonl](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1/output.jsonl)

Local storage sizes after ingest:

- image folder: about `228M`
- batch artifact folder: about `305M`

## Milestone 32: V2 Scale-Up Image QC Completed

### Outcome

Completed visual QC for the downloaded V2 real-failure-targeted scale-up batch.
The batch is usable for the next training-manifest export step.

### QC Scope

- reviewed images: `132`
- contact sheets:
  [data/qc/v2_real_failure_targeted_scale_candidate_20260513_r1](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/v2_real_failure_targeted_scale_candidate_20260513_r1)
- source generated manifest:
  [generated_image_manifest.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1/generated_image_manifest.csv)
- QC decision file:
  [v2_scale_qc_decisions_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/v2_real_failure_targeted_scale_candidate_20260513_r1/v2_scale_qc_decisions_round1.csv)
- QC-applied manifest:
  [generated_image_manifest_qc_round1.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1/generated_image_manifest_qc_round1.csv)

### QC Decisions

- accepted as labeled: `125`
- accepted with neighboring-band relabel: `7`
- rejected: `0`
- missing local images: `0`

Training band counts after QC:

- `0-10`: `8`
- `11-20`: `17`
- `21-30`: `18`
- `31-35`: `13`
- `36-45`: `16`
- `46-55`: `19`
- `56-65`: `13`
- `66-75`: `12`
- `76-85`: `12`
- `86-100`: `4`

Training coarse-class counts after QC:

- `intact`: `8`
- `minor`: `48`
- `moderate`: `48`
- `severe`: `28`

### Finding

The V2 scale-up is materially better than the earlier pilot. The dedicated
`86-100` severe-high prompts produced genuinely top-band examples rather than
ordinary severe boxes. The remaining relabels are normal neighbor-band drift,
mostly subtle high-minor and high-moderate cases that should be kept but trained
with corrected labels.

## Milestone 33: V2 Scale-Up Integrated And Controlled Model Trained

### Outcome

Integrated the QC-reviewed V2 scale-up batch into the synthetic training
pipeline and trained one controlled comparison model against the frozen baseline
family.

### Registry And Dataset Integration

- registered V2 prompt rows into SQLite: `132`
- registered V2 generated-image rows into SQLite: `132`
- synced V2 QC labels from the QC-applied manifest: `132`
- total generated accepted registry rows after sync: `899`
- exported combined reviewed source manifest:
  [synthetic_full_qc_plus_v2_scale_source.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_full_qc_plus_v2_scale_source.csv)
- source manifest rows: `899`
- missing source images: `0`
- V2 rows in combined source manifest: `132`

Combined source coarse-class counts:

- `intact`: `220`
- `minor`: `228`
- `moderate`: `229`
- `severe`: `222`

### 384 Cache

Built and verified deterministic RGB resize cache:

- processed manifest:
  [synthetic_full_qc_plus_v2_scale_processed_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv)
- processed rows: `899`
- missing processed images: `0`
- bad image size/mode checks: `0`
- preprocessing profile: `rgb_resize_384`
- V2 rows: `132`
- cache directory:
  [synthetic_full_qc_plus_v2_scale_processed_384](/Users/inventure71/VSProjects/School/Dream2Detect/data/cache/synthetic_full_qc_plus_v2_scale_processed_384)

### Controlled Training Run

Run:

- [v2_scale_384_residual_damage_safe_ordinal02_20260513](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v2_scale_384_residual_damage_safe_ordinal02_20260513)

Configuration:

- image size: `384`
- model variant: `residual_cnn`
- augmentation profile: `damage_safe`
- optimizer: `adamw`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.2`
- ordinal loss weight: `0.2`
- LR scheduler: `reduce_on_plateau`
- early stopping patience: `40`
- device: `mps`

Training result:

- best validation epoch: `87`
- early stopped at epoch: `127`
- synthetic test accuracy: `0.5333`
- synthetic test macro F1: `0.5239`
- synthetic off-by-one-or-correct rate: `0.9111`
- synthetic severe ordinal error rate: `0.0889`

Synthetic test confusion matrix:

```text
[26, 11, 4, 3]
[12, 12, 20, 2]
[4, 6, 23, 13]
[1, 2, 6, 35]
```

### Frozen Baseline Comparison

Compared against the frozen `2026-05-13` baseline table:

- previous synthetic baseline macro F1: `0.5153`
- V2 scale-up synthetic macro F1: `0.5239`
- previous synthetic off-by-one-or-correct rate: `0.8875`
- V2 scale-up off-by-one-or-correct rate: `0.9111`

Direct synthetic-to-real transfer on all current padded `384` real images:

- previous padded-real transfer accuracy: `0.3110`
- V2 padded-real transfer accuracy: `0.3110`
- previous padded-real transfer macro F1: `0.2445`
- V2 padded-real transfer macro F1: `0.2689`
- previous padded-real off-by-one-or-correct rate: `0.6768`
- V2 padded-real off-by-one-or-correct rate: `0.7195`

Comparison artifact:

- [v2_scale_comparison_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_transfer/v2_scale_comparison_20260513.csv)

### Interpretation

The V2 scale-up produced a small but real synthetic-side improvement and a small
real-transfer macro-F1 improvement. It did not solve synthetic-to-real transfer:
direct real accuracy stayed flat at `0.3110`, still far below the real-only
scratch baseline. This supports keeping the V2 data, but the next improvement
should focus on real-domain adaptation or real-only/fine-tuning strategy rather
than simply adding more synthetic images of the same type.

## Milestone 34: V2 Synthetic Baseline Frozen And V3 Plan Started

### Outcome

Froze the current V2 synthetic-only result as the baseline for the next model
iteration and wrote the V3 synthetic-only training plan into the repo before
changing the training stack.

### Frozen Baseline

- baseline artifact:
  [v2_synthetic_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v2_synthetic_baseline_20260513.csv)
- reference run:
  [v2_scale_384_residual_damage_safe_ordinal02_20260513](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v2_scale_384_residual_damage_safe_ordinal02_20260513)
- frozen synthetic test accuracy: `0.5333`
- frozen synthetic test macro F1: `0.5239`
- frozen off-by-one-or-correct rate: `0.9111`
- frozen severe ordinal error rate: `0.0889`

### V3 Plan

- plan doc:
  [18-v3-synthetic-training-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/18-v3-synthetic-training-plan.md)

Locked V3 implementation priorities:

- GroupNorm residual backbone for small-batch stability
- true ordinal coarse-class training path
- constrained RandAugment-style synthetic-safe augmentation profiles

## Milestone 35: V3 Synthetic-Only Training Implemented And Measured

### Outcome

Implemented the V3 synthetic-only training changes and measured them against the
frozen V2 synthetic baseline.

### Code Changes

Implemented:

- `residual_cnn_groupnorm` model variant
- true ordinal coarse-class training mode: `coarse_ordinal`
- constrained RandAugment-style profiles:
  - `damage_safe_ra_low`
  - `damage_safe_ra_medium`
  - `damage_safe_ra_high`
- analyzer support for evaluating saved `coarse_ordinal` checkpoints

Key files:

- [models.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/models.py)
- [dataset.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/dataset.py)
- [train_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/train_classifier.py)
- [train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)
- [analyze_classifier_checkpoint.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/analyze_classifier_checkpoint.py)

### Verification

- `python3 -m py_compile` on the changed training modules and scripts
- `PYTHONPATH=src python3 -m pytest -q tests/test_training_controls.py tests/test_training_artifacts.py tests/test_experiment_grid.py`
- result: `25 passed`

### Measured V3 Runs

Comparison artifact:

- [v3_synthetic_training_comparison_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_synthetic_training_comparison_20260513.csv)

Selected V3 synthetic-only candidate:

- run:
  [v3_bn_ra_low_coarse_ord02_nosampler_20260513](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v3_bn_ra_low_coarse_ord02_nosampler_20260513)
- model: `residual_cnn`
- augmentation: `damage_safe_ra_low`
- target: `coarse`
- ordinal loss weight: `0.2`
- balanced sampler: `False`
- evaluation method: diagnostics run on saved best checkpoint
- synthetic test accuracy: `0.5500`
- synthetic test macro F1: `0.5496`
- off-by-one-or-correct rate: `0.9167`
- severe ordinal error rate: `0.0833`

GroupNorm branch:

- run:
  [v3_gn_ra_low_coarse_ord02_nosampler_20260513](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v3_gn_ra_low_coarse_ord02_nosampler_20260513)
- synthetic test accuracy: `0.3000`
- synthetic test macro F1: `0.2457`

True ordinal coarse branch:

- run:
  [v3_gn_coarseordinal_ra_low_nosampler_20260513](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v3_gn_coarseordinal_ra_low_nosampler_20260513)
- synthetic test accuracy: `0.3056`
- synthetic test macro F1: `0.3062`

### Comparison Against Frozen V2 Baseline

Frozen V2 baseline:

- synthetic test accuracy: `0.5333`
- synthetic test macro F1: `0.5239`
- off-by-one-or-correct rate: `0.9111`
- severe ordinal error rate: `0.0889`

Selected V3 candidate vs V2:

- accuracy: `0.5500` vs `0.5333`
- macro F1: `0.5496` vs `0.5239`
- off-by-one-or-correct: `0.9167` vs `0.9111`
- severe ordinal error rate: `0.0833` vs `0.0889`

### Interpretation

The constrained low-strength RandAugment path improved synthetic-only
performance when applied to the proven BatchNorm residual baseline. The
GroupNorm hypothesis was not supported by the current synthetic dataset: both
GroupNorm runs underperformed sharply. The true ordinal coarse-class path is now
implemented and analyzable, but in its current form it is not the new baseline.

## Milestone 36: Harder Synthetic Split Added And V3.1 Baseline Frozen

### Outcome

Added a reproducible harder synthetic-only split policy, ran a compact V3.1
search around the current residual baseline, and froze the new hard-split
baseline.

### Split System

Implemented a new split strategy:

- `metadata_family_holdout`

It keeps derived prompt families disjoint across train, validation, and test
instead of relying only on random stratification.

Derived family key:

- `training_coarse_class`
- `damage_profile_primary`
- `box_form_factor`
- `background_context`

Updated training/evaluation entry points:

- [splits.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/splits.py)
- [train_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/train_classifier.py)
- [train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)
- [run_classifier_experiment_grid.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/run_classifier_experiment_grid.py)
- [analyze_classifier_checkpoint.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/analyze_classifier_checkpoint.py)

### Verification

- `python3 -m py_compile` on the changed split/training scripts
- `PYTHONPATH=src python3 -m pytest -q tests/test_training_controls.py tests/test_experiment_grid.py tests/test_training_artifacts.py tests/test_analyze_classifier_checkpoint.py`
- result: `29 passed`

### Hard-Split Search Artifacts

- hard-split comparison table:
  [v3_1_metadata_family_holdout_comparison_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv)
- hard-split baseline:
  [v3_1_hard_split_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_hard_split_baseline_20260513.csv)
- frozen random-split V3 baseline:
  [v3_synthetic_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_synthetic_baseline_20260513.csv)

### Key Results

Seed 42 completed runs on the hard split:

- `dropout=0.1`, `damage_safe`, `wd=0.0001`
  - best val macro F1: `0.5869`
  - test macro F1: `0.5607`
- `dropout=0.2`, `damage_safe`, `wd=0.0001`
  - best val macro F1: `0.5473`
  - test macro F1: `0.5983`
- `dropout=0.3`, `damage_safe`, `wd=0.0001`
  - best val macro F1: `0.5937`
  - test macro F1: `0.5953`

Seed 43 confirmation:

- `dropout=0.1`, `damage_safe`, `wd=0.0001`
  - best val macro F1: `0.5964`
  - test macro F1: `0.5309`
- `dropout=0.3`, `damage_safe`, `wd=0.0001`
  - pruned early after peaking at val macro F1 `0.5614`
  - it was already clearly below the `dropout=0.1` seed-43 validation curve

Pruned branches:

- `damage_safe_ra_low`
  - lost to `damage_safe` on matched seed-42 runs for both `dropout=0.1` and
    `dropout=0.2`
- `weight_decay=0.0003`
  - lost clearly to `weight_decay=0.0001` before the branch was completed

### Frozen V3.1 Baseline

Selected hard-split baseline configuration:

- model: `residual_cnn`
- augmentation: `damage_safe`
- dropout: `0.1`
- weight decay: `0.0001`
- ordinal loss weight: `0.2`
- balanced sampler: `False`
- split strategy: `metadata_family_holdout`

Selection reason:

- highest and most stable validation macro F1 across two seeds
- `dropout=0.3` was stronger on one seed-42 test split, but weaker on the
  follow-up validation run and therefore not the more defensible baseline

### Interpretation

The harder split matters. It lowers the easy synthetic-side optimism from the
older random split and gives a more realistic view of whether the classifier is
learning reusable damage families or only memorizing prompt-style overlap.

Under that harder regime, the BatchNorm residual model still works, but the best
configuration changed:

- the random-split V3 winner used `damage_safe_ra_low`
- the harder-split V3.1 baseline prefers plain `damage_safe`

This is the correct direction for the project. The synthetic-only setup is now
both stronger and harder to game.

### Short Paper-Ready Summary

We first improved the synthetic-only training pipeline and then tightened the
evaluation protocol so that near-duplicate synthetic prompt families could not
leak across train, validation, and test splits. Under the earlier random split,
the best synthetic-only configuration reached `0.5496` macro F1. After moving
to the harder `metadata_family_holdout` split, the strongest and most stable
configuration was a from-scratch `residual_cnn` trained with `damage_safe`
augmentation, `dropout=0.1`, `weight_decay=0.0001`, and ordinal loss weight
`0.2`.

The harder split reduced optimistic synthetic-only estimates, but the model
still remained competitive, reaching validation macro F1 around `0.59` across
two seeds (`0.5869` and `0.5964`). This suggests that the model is learning
some reusable synthetic damage structure rather than only memorizing prompt
style overlap. It also changed the preferred augmentation policy: the earlier
random-split winner used `damage_safe_ra_low`, whereas the harder-split
baseline preferred plain `damage_safe`.

In short, the main outcome of this phase is not just a better score. It is a
more credible synthetic-only baseline and a more defensible evaluation setup
for the next stage of the project.

## Milestone 37: Score-Band Target Adopted And Coarse Reference Frozen

### Outcome

The project target was explicitly changed from the four-class coarse task to
the official ten-band `score_band` task. Before starting that shift, the
current coarse implementation and baseline were frozen as the reference system.

### Frozen Reference

Reference artifact:

- [coarse_reference_before_score_band_target_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/coarse_reference_before_score_band_target_20260514.csv)

This frozen reference preserves:

- target mode: `coarse`
- labels: `4`
- model: `residual_cnn`
- augmentation: `damage_safe`
- dropout: `0.1`
- weight decay: `0.0001`
- ordinal loss weight: `0.2`
- balanced sampler: `False`
- split strategy: `metadata_family_holdout`

Supporting baseline artifacts retained for comparison:

- [v3_1_hard_split_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_hard_split_baseline_20260513.csv)
- [v3_1_metadata_family_holdout_comparison_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv)
- [v3_synthetic_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_synthetic_baseline_20260513.csv)

### Frozen Code Paths

The current implementation was intentionally preserved as the future reference
code path before score-band-target work continues:

- [src/dream2detect/training/models.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/models.py)
- [src/dream2detect/training/dataset.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/dataset.py)
- [src/dream2detect/training/metrics.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/metrics.py)
- [src/dream2detect/training/splits.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/splits.py)
- [src/dream2detect/training/train_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/train_classifier.py)
- [scripts/train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)
- [scripts/run_classifier_experiment_grid.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/run_classifier_experiment_grid.py)
- [scripts/analyze_classifier_checkpoint.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/analyze_classifier_checkpoint.py)

### New Target Rule

From this point forward:

- primary training target: `score_band`
- primary checkpoint-selection metric: 10-band validation mean band error
- four-class evaluation: secondary diagnostic view only

Collapsed four-class evaluation will still be reported, but the main model is
now judged first on the ten-band task.

### Plan Artifact

The target-shift plan is recorded in:

- [19-score-band-target-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/19-score-band-target-plan.md)

### Interpretation

This does not erase the coarse work. It makes the coarse system the stable
reference implementation while the next phase tests whether finer supervision at
the official ten-band level yields a better severity model.

## Milestone 38: Score-Band V4 Ordinal Soft-Label Training Implemented

### Outcome

The score-band training path now matches the intended ordinal problem more
closely. The new V4 implementation keeps the model from scratch, but changes
the target handling, class balancing, and checkpoint selection logic.

### What Changed

Implemented in:

- [src/dream2detect/training/train_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/train_classifier.py)
- [src/dream2detect/training/metrics.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/metrics.py)
- [src/dream2detect/training/dataset.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/training/dataset.py)
- [scripts/train_synthetic_classifier.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/train_synthetic_classifier.py)

New score-band V4 rules:

- soft ordinal targets across the 10 score bands
- effective-number class-balanced weighting by default for score-band runs
- automatic sampler policy:
  - coarse defaults stay unchanged
  - score-band defaults to no balanced sampler unless explicitly forced
- primary validation selector:
  - `val_mean_band_error`
- secondary diagnostics still reported:
  - exact 10-band accuracy
  - 10-band macro F1
  - `±1` band accuracy
  - severe band-error rate
  - collapsed 4-class accuracy and macro F1

Collapsed coarse evaluation now uses the probability-preserving rule:

- sum 10-band probabilities inside each coarse group
- choose the coarse class from the grouped probabilities

### Verification

Targeted verification passed:

- `python3 -m py_compile scripts/train_synthetic_classifier.py src/dream2detect/training/train_classifier.py src/dream2detect/training/metrics.py src/dream2detect/training/dataset.py tests/test_training_controls.py tests/test_training_artifacts.py`
- `PYTHONPATH=src python3 -m pytest -q tests/test_training_controls.py tests/test_training_artifacts.py tests/test_multitask_training.py tests/test_analyze_classifier_checkpoint.py`
- result: `36 passed`

Smoke run artifact:

- [v4_score_band_smoke_20260514](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_score_band_smoke_20260514)

Smoke run result:

- target mode: `score_band`
- device: `mps`
- selection metric: `val_mean_band_error`
- checkpoints written each epoch
- epoch metrics CSV and graphs written successfully

This was only a two-epoch overfit-subset smoke test, so its metric values are
not meaningful as model-quality evidence. Its purpose was only to prove that the
new V4 path executes end to end and emits the expected artifacts.

## Milestone 39: Score-Band Analyzer And Grid Workflow Aligned

### Outcome

The surrounding training workflow now matches the V4 score-band target. The
checkpoint analyzer and experiment-grid summary layer no longer assume a coarse
4-class primary task.

### What Changed

Updated:

- [scripts/analyze_classifier_checkpoint.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/analyze_classifier_checkpoint.py)
- [scripts/run_classifier_experiment_grid.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/run_classifier_experiment_grid.py)

Analyzer changes:

- score-band checkpoints now load with the correct 10-band output dimension
- diagnostics now report the primary task according to `target_label_mode`
- score-band runs now emit both:
  - 10-band primary diagnostics
  - collapsed coarse diagnostics

New score-band analyzer artifacts include:

- `test_collapsed_coarse_confusion_matrix.csv`
- `test_collapsed_coarse_per_class_tp_fp_tn_fn.csv`

Grid changes:

- score-band grids now preserve the V4 trainer defaults through
  `--balanced-sampler-mode auto`
- summary rows now record:
  - `selection_metric_name`
  - `best_val_metric`
  - score-band test error metrics
  - collapsed coarse test metrics
- ranked grid output is now written to:
  - `grid_summary_ranked.csv`

### Verification

Targeted verification passed:

- `python3 -m py_compile scripts/analyze_classifier_checkpoint.py scripts/run_classifier_experiment_grid.py tests/test_analyze_classifier_checkpoint.py tests/test_experiment_grid.py`
- `PYTHONPATH=src python3 -m pytest -q tests/test_analyze_classifier_checkpoint.py tests/test_experiment_grid.py tests/test_training_controls.py tests/test_training_artifacts.py tests/test_multitask_training.py`
- result: `44 passed`

Live workflow checks:

- analyzer on score-band smoke run:
  [v4_score_band_smoke_20260514/diagnostics](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_score_band_smoke_20260514/diagnostics)
- grid smoke outputs:
  [v4_score_band_grid_smoke_20260514](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_score_band_grid_smoke_20260514)

Important bug fix during verification:

- the analyzer originally rebuilt score-band splits using `training_coarse_class`
  instead of `training_score_band`
- this produced mismatched replay metrics
- the analyzer now uses the same split label column as training, and replayed
  diagnostics match the trainer's recorded test metrics

This means the codebase is now consistent across:

- training
- checkpoint diagnostics
- experiment-grid summaries

## Milestone 40: First Full V4 Score-Band Baseline Trained

### Outcome

The first full V4 score-band baseline run completed successfully on the
`384 x 384` synthetic dataset under the `metadata_family_holdout` split.

Run artifact:

- [synthetic_full_qc_plus_v2_scale_processed_384_384_20260514_192919](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_full_qc_plus_v2_scale_processed_384_384_20260514_192919)
- diagnostics:
  [diagnostics](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/synthetic_full_qc_plus_v2_scale_processed_384_384_20260514_192919/diagnostics)

### Configuration

- model: `residual_cnn`
- target: `score_band`
- image size: `384`
- augmentation: `damage_safe`
- ordinal loss weight: `0.2`
- soft-label sigma: `1.0`
- class-weight strategy: `effective`
- effective beta: `0.999`
- balanced sampler mode: `auto`
- split strategy: `metadata_family_holdout`

### Best Validation Checkpoint

- best validation epoch: `67`
- validation mean band error: `1.1627`
- validation macro F1 at best epoch: `0.2737`
- validation collapsed coarse macro F1 at best epoch: `0.4827`

### Final Test Result

Primary 10-band metrics:

- exact accuracy: `0.1990`
- macro F1: `0.1791`
- mean band error: `1.3155`
- `±1` band accuracy: `0.7233`
- severe band-error rate: `0.2767`

Collapsed 4-class diagnostic:

- accuracy: `0.4854`
- macro F1: `0.4557`

### Interpretation

This is a legitimate ordinal baseline, not a collapse or broken run. The model
is usually near the right answer, but it is still weak on exact 10-band
discrimination. The gap between:

- exact 10-band accuracy `0.1990`
- `±1` band accuracy `0.7233`

shows that the model often lands close to the correct severity band without
cleanly separating neighboring bands.

### Follow-up

The next step is not a broad redesign. It is a narrow V4.1 challenger sweep
around the current baseline:

- sigma `0.75`
- sigma `1.25`
- dropout `0.2`
- effective beta `0.995`

## Milestone 41: V4.1 Score-Band Challenger Sweep Completed

### Outcome

The narrow V4.1 sweep completed across four challengers against the full V4
score-band baseline.

Sweep artifacts:

- sweep root:
  [v4_1_local_sweep_20260514](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_1_local_sweep_20260514)
- ranked sweep summary:
  [v4_1_sweep_summary_ranked.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_1_local_sweep_20260514/v4_1_sweep_summary_ranked.csv)
- baseline-plus-challengers comparison:
  [v4_1_score_band_challenger_comparison_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_1_score_band_challenger_comparison_20260514.csv)

### Configurations Evaluated

- `sigma=0.75`
- `sigma=1.25`
- `dropout=0.2`
- `effective_beta=0.995`

All other training settings stayed fixed to the current V4 score-band baseline.

### Ranked Result

By the primary selection metric `val_mean_band_error`:

1. `sigma=1.25`: `1.1325`
2. `sigma=0.75`: `1.1747`
3. `dropout=0.2`: `1.2048`
4. `effective_beta=0.995`: `1.2530`

Reference V4 baseline:

- `sigma=1.0`, `beta=0.999`, `dropout=0.1`
- `val_mean_band_error=1.1627`

### Key Observation

`sigma=1.25` is the first challenger that beat the baseline on the primary
validation selection metric. However, it did not improve the held-out test
behavior:

- baseline test mean band error: `1.3155`
- `sigma=1.25` test mean band error: `1.3932`
- baseline test `+-1` band accuracy: `0.7233`
- `sigma=1.25` test `+-1` band accuracy: `0.6893`
- baseline test macro F1: `0.1791`
- `sigma=1.25` test macro F1: `0.1818`

So the result is mixed:

- better validation selection metric
- weaker held-out ordinal test behavior
- slightly higher exact 10-band macro F1

### Decision

Do not replace the V4 baseline yet.

The right next step is a controlled confirmation run between:

- current V4 baseline: `sigma=1.0`
- V4.1 leading challenger: `sigma=1.25`

That comparison should be repeated with at least one additional seed before
locking a new score-band baseline.

## Milestone 42: Two-Seed Confirmation for Baseline vs `sigma=1.25`

### Outcome

The baseline versus `sigma=1.25` score-band comparison has now been repeated
with a second seed.

Artifacts:

- seed-43 confirmation runs:
  [v4_1_seed43_confirmation_20260514](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_1_seed43_confirmation_20260514)
- seed-43 ranked summary:
  [seed43_confirmation_summary_ranked.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_1_seed43_confirmation_20260514/seed43_confirmation_summary_ranked.csv)
- two-seed run table:
  [v4_1_two_seed_baseline_vs_sigma125_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_1_two_seed_baseline_vs_sigma125_20260514.csv)
- two-seed aggregate:
  [v4_1_two_seed_baseline_vs_sigma125_aggregate_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_1_two_seed_baseline_vs_sigma125_aggregate_20260514.csv)

### Seed-43 Result

Baseline `sigma=1.0`:

- best validation mean band error: `1.2793`
- test mean band error: `1.4130`
- test `+-1` band accuracy: `0.6739`
- collapsed coarse macro F1: `0.4335`

Challenger `sigma=1.25`:

- best validation mean band error: `1.2514`
- test mean band error: `1.3587`
- test `+-1` band accuracy: `0.6793`
- collapsed coarse macro F1: `0.4772`

### Two-Seed Comparison

Validation selection metric:

- seed 42: `sigma=1.25` beat baseline (`1.1325` vs `1.1627`)
- seed 43: `sigma=1.25` beat baseline (`1.2514` vs `1.2793`)

So `sigma=1.25` wins the primary validation metric in both tested seeds.

However, the two-seed average held-out test behavior is still mixed:

- mean test band error:
  - baseline: `1.3643`
  - `sigma=1.25`: `1.3759`
- mean test `+-1` band accuracy:
  - baseline: `0.6986`
  - `sigma=1.25`: `0.6843`
- mean test macro F1:
  - baseline: `0.1961`
  - `sigma=1.25`: `0.1943`

### Decision

Do not lock `sigma=1.25` as the new baseline yet.

The correct conclusion is:

- `sigma=1.25` is a legitimate improvement on the validation selection metric
- but it does not yet produce a cleaner two-seed test outcome
- therefore the repo should keep the current V4 baseline locked for now

The next sweep should stay narrow and target the space between `1.0` and `1.25`
rather than promoting `1.25` outright.

## Milestone 43: V4.2 Sigma-Only Sweep Completed

### Outcome

The V4.2 sigma-only sweep is complete across:

- `sigma=1.10`
- `sigma=1.15`
- `sigma=1.20`

Each configuration was run with seeds `42` and `43`, keeping all other
training settings fixed to the locked V4 baseline.

Artifacts:

- sweep root:
  [v4_2_sigma_sweep_20260515](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_2_sigma_sweep_20260515)
- per-run table:
  [v4_2_sigma_sweep_runs.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_2_sigma_sweep_20260515/v4_2_sigma_sweep_runs.csv)
- aggregate ranking:
  [v4_2_sigma_sweep_aggregate_ranked.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier_grid/v4_2_sigma_sweep_20260515/v4_2_sigma_sweep_aggregate_ranked.csv)
- baseline comparison:
  [v4_2_sigma_sweep_vs_baseline_20260515.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_2_sigma_sweep_vs_baseline_20260515.csv)
- aggregate baseline comparison:
  [v4_2_sigma_candidates_vs_locked_baseline_aggregate_20260515.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_2_sigma_candidates_vs_locked_baseline_aggregate_20260515.csv)

### Aggregate Result

By average validation mean band error:

1. `sigma=1.10`: `1.2135`
2. `sigma=1.15`: `1.2210`
3. `sigma=1.20`: `1.3055`

Reference locked baseline:

- `sigma=1.00`: `1.2210`

### Interpretation

None of the V4.2 sigma candidates is strong enough to replace the locked
baseline.

What happened:

- `sigma=1.10` slightly improved average validation mean band error
  (`1.2135` vs `1.2210`)
  but clearly worsened held-out test behavior:
  - mean test band error: `1.4172`
  - mean test `+-1` band accuracy: `0.6529`
- `sigma=1.15` did not improve validation on average and produced the weakest
  test behavior of the sweep.
- `sigma=1.20` produced the best V4.2 held-out test band error:
  - mean test band error: `1.3490`
  - mean test `+-1` band accuracy: `0.6980`
  but its validation mean band error was materially worse than the baseline:
  - `1.3055` vs `1.2210`

So the sigma-only space between `1.0` and `1.25` is now effectively explored
enough for this phase:

- no candidate improved both validation selection and held-out ordinal test
  behavior
- therefore no V4.2 sigma value earns promotion

### Decision

Keep the current V4 baseline locked.

Stop tuning sigma.

The next score-band tuning axis should shift away from width-only changes and
move to the **soft-label target shape or weighting rule**.

## Milestone 44: V4.5 Objective-Change Challenger Locked

### Decision

The next score-band challenger is now locked as **V4.5**.

V4.5 will keep the current V4 stack fixed:

- `residual_cnn`
- `metadata_family_holdout`
- `damage_safe`
- `score_band`
- soft-label sigma `1.0`
- effective-number beta `0.999`
- ordinal loss weight `0.2`

and change **only** the training objective by adding a cumulative squared EMD
term with:

- `score_band_emd_weight=0.5`

Constraints for V4.5:

- no CORAL/CORN branch yet
- no scalar-regression branch
- no new sigma sweep
- no pretrained backbone work

The first V4.5 run is a single-seed challenger, not a new locked baseline.

## Milestone 45: V4.5 Hybrid Soft-CE + EMD Challenger Completed

### Outcome

The V4.5 score-band challenger is now fully implemented, trained, and analyzed.

Artifacts:

- training run:
  [v4_5_score_band_emd05_seed42_20260515](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_5_score_band_emd05_seed42_20260515)
- diagnostics:
  [diagnostics](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_5_score_band_emd05_seed42_20260515/diagnostics)
- comparison artifact:
  [v4_5_vs_locked_v4_20260515.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v4_5_vs_locked_v4_20260515.csv)

### Implementation

V4.5 kept the current score-band stack fixed and changed only the objective:

- soft-label cross-entropy
- cumulative squared EMD with `score_band_emd_weight=0.5`
- existing smooth ordinal expectation penalty

The score-band EMD weight is now a first-class training control in:

- trainer config
- CLI
- experiment-grid tooling

### Result

Best validation checkpoint:

- best epoch: `61`
- validation mean band error: `1.2229`

Held-out test result:

- exact accuracy: `0.1748`
- macro F1: `0.1574`
- mean band error: `1.3398`
- `+-1` band accuracy: `0.7330`
- collapsed 4-class macro F1: `0.4439`

### Comparison to Locked V4

Locked V4 remains better on the main selection and most summary metrics:

- validation mean band error:
  - locked V4: `1.1627`
  - V4.5: `1.2229`
- test mean band error:
  - locked V4: `1.3155`
  - V4.5: `1.3398`
- test 10-band macro F1:
  - locked V4: `0.1791`
  - V4.5: `0.1574`
- collapsed 4-class macro F1:
  - locked V4: `0.4557`
  - V4.5: `0.4439`

V4.5 did slightly improve one useful ordinal metric:

- test `+-1` band accuracy:
  - locked V4: `0.7233`
  - V4.5: `0.7330`

### Decision

Do not promote V4.5 to the new baseline.

This was a valid objective-change challenger, but not a clean win. The current
V4 baseline stays locked.

## Milestone 46: Object-Focused Crop Pipeline Started

### Decision

The next controlled preprocessing experiment will test whether box/object
framing explains part of the synthetic-to-real gap.

Planned pipeline:

- source manifest
- detect cardboard package region
- crop around the selected object with padding
- pad-resize to the target square size
- write a derived manifest with crop metadata
- inspect contact sheets before training

Primary backend:

- SAM 3 text-prompted segmentation with prompts such as `cardboard box`,
  `shipping box`, and `damaged cardboard package`

Required fallback:

- deterministic center-prior crop, because SAM 3 is not a normal project
  dependency and requires separate model/checkpoint setup

## Milestone 47: Object-Focused Crop Pipeline Implemented

### Outcome

The object-focused preprocessing pipeline is now implemented.

New script:

- [build_object_focused_cache.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/build_object_focused_cache.py)

Core implementation:

- [object_crop.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/preprocessing/object_crop.py)

Supported crop backends:

- `sam3`: require SAM 3 and fail if unavailable
- `center`: deterministic center-prior crop
- `hybrid`: try SAM 3 first, then fall back to center-prior crop

Correction after QC:

- strict `sam3` mode now also fails if SAM 3 returns no valid crop for an image
- the CLI default is strict `sam3`, not `hybrid`
- center-prior crops are only acceptable when the command explicitly requests
  `--crop-backend hybrid` or `--crop-backend center`

The derived manifests include crop metadata:

- crop status
- crop method
- SAM text prompt
- confidence
- mask area ratio
- crop bounding box
- crop notes

### Smoke Verification

SAM 3 is not installed in the current local environment, so the smoke runs used
the `hybrid` path and correctly fell back to center-prior crops.

Smoke artifacts:

- real smoke manifest:
  [object_focus_smoke_real_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/object_focus_smoke_real_384.csv)
- synthetic smoke manifest:
  [object_focus_smoke_synthetic_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/object_focus_smoke_synthetic_384.csv)
- real contact sheet:
  [real_object_focus_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/real_object_focus_sheet.jpg)
- synthetic contact sheet:
  [synthetic_object_focus_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/synthetic_object_focus_sheet.jpg)

Verification command:

- `PYTHONPATH=src python3 -m pytest -q tests/test_object_focused_cache.py`

Result:

- `6 passed`

## Milestone 48: SAM3 Runtime Preflight Added

### Outcome

The local Mac environment was checked against the official SAM3 runtime
requirements.

Current local result:

- Python: `3.12.2`
- platform: macOS ARM / Apple Silicon
- PyTorch: `2.6.0`
- MPS: available
- CUDA: unavailable
- `sam3`: not importable

This means strict SAM3 cropping cannot run locally in the current environment
yet. The issue is not the crop pipeline; the current environment is missing a
compatible SAM3 install and authenticated checkpoint access.

Correction:

- the official README path is CUDA-first
- an open community PR adds Apple Silicon / MPS support
- therefore MPS is a valid experimental setup target, but it must be validated
  with `scripts/check_sam3_setup.py --device mps` and a one-image smoke before
  full cache generation

New setup artifact:

- [check_sam3_setup.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/check_sam3_setup.py)
- [run_sam3_mps_python.sh](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/run_sam3_mps_python.sh)

New documentation:

- [20-sam3-setup.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/20-sam3-setup.md)

Decision:

- do not install SAM3 into the local classifier environment
- use a dedicated `sam3-mps` or CUDA environment for strict SAM3 preprocessing
- continue allowing local `center` or explicit `hybrid` fallback only for
  pipeline smoke tests, not for claiming SAM3 preprocessing

### MPS Setup Attempt

A dedicated `sam3-mps` environment was created and the Apple Silicon branch was
installed from:

- `https://github.com/provos/sam3.git@apple-silicon-support-v2`

Additional import-time dependencies installed:

- `einops`
- `decord2`
- `pycocotools`
- `opencv-python`
- `matplotlib`
- `psutil`

Because the branch wheel omits required subpackages such as `sam3.sam`, the
source checkout is kept at:

- `/Users/inventure71/.cache/dream2detect/sam3-apple-silicon-support-v2`

Validated state:

- `torch 2.12.0`
- MPS built and available
- full SAM3 source import succeeds when the source checkout is on `PYTHONPATH`
- model builder exposes a `device` parameter
- processor exposes a `device` parameter

Remaining blocker:

- Hugging Face SAM3 checkpoint access/token is not configured, so a real
  one-image SAM3 smoke cannot run yet.

## Milestone 49: SAM3 MPS Smoke Validated

### Outcome

SAM3 on Apple Silicon / MPS is now validated for this project environment.

The user authenticated Hugging Face access in the `sam3-mps` environment and
ran a one-image SAM3 smoke:

```bash
scripts/run_sam3_mps_python.sh scripts/check_sam3_setup.py \
  --device mps \
  --smoke-image data/real/source/redf0xwin_recognizing_defects_in_boxes_and_cardboard/images/0001.jpg \
  --prompt "cardboard box"
```

Result:

- Python `3.12.13`
- PyTorch `2.12.0`
- MPS built and available
- SAM3 package importable from the Apple Silicon source checkout
- Hugging Face token configured
- SAM3 returned `1` box
- first score: `0.9463`
- first box:
  `[18.3527, 16.3890, 1870.2390, 2405.3601]`

### Strict Crop Smoke

A strict SAM3/MPS crop smoke was then run on the first 12 real labeled images:

```bash
scripts/run_sam3_mps_python.sh scripts/build_object_focused_cache.py \
  --source-manifest data/datasets/real_labeled_dataset_current.csv \
  --output-dir data/cache/object_focus_smoke/real_sam3_mps_384 \
  --output-manifest data/datasets/object_focus_smoke_real_sam3_mps_384.csv \
  --image-size 384 \
  --source-image-column image_path \
  --crop-backend sam3 \
  --sam3-device mps \
  --limit 12 \
  --contact-sheet data/qc/object_focus_smoke/real_sam3_mps_sheet.jpg \
  --contact-sheet-max-rows 12
```

Result:

- rows written: `12`
- `object_crop_status=detected`: `12`
- `object_crop_method=sam3_text`: `12`
- mean confidence: `0.9024`
- min confidence: `0.7638`
- max confidence: `0.9794`

Artifacts:

- [object_focus_smoke_real_sam3_mps_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/object_focus_smoke_real_sam3_mps_384.csv)
- [real_sam3_mps_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/real_sam3_mps_sheet.jpg)

Decision:

- SAM3/MPS is now usable for object-focused preprocessing.
- The next step is visual QC of the 12-image contact sheet.
- If the crop quality is acceptable, generate full real and synthetic
  SAM3-object-focused caches before retraining.

### QC Follow-Up

Visual QC showed that SAM3 was detecting boxes, but the post-processing was not
appropriate for classifier inputs.

Problem:

- the initial crop path preserved the detected rectangle aspect ratio
- wide package detections were padded into a square canvas
- this produced thin horizontal strips with large blank bands

Correction:

- object-focused preprocessing now supports `resize_mode`
- the temporary `stretch` mode filled the square but distorted package geometry
- default is now `square_crop`, which converts the SAM box into a square crop in
  the original image before resizing
- the square side length is based on the longest SAM bbox side
- if the square is larger than a source-image dimension, the outside area is
  padded with white pixels
- default square padding is now `0.01` per side to keep the crop tight around
  the SAM box
- `pad` and `stretch` remain available for comparison/debugging

Reason:

- the object-focused experiment needs the package/damage to occupy most of the
  model input
- `square_crop` avoids both large blank bands and geometric distortion

## Milestone 50: SAM BBox to Square-Crop QC Added

### Outcome

The object-focused preprocessing path now converts the raw SAM rectangle into a
square crop before resizing.

This preserves geometry while producing square classifier inputs.

Correction:

- the square crop must not be capped to the source image's shorter side
- the shorter border is extended to the same length as the longest SAM bbox
  border
- if the source image cannot cover the full square, the missing region is
  padded white

Implementation details:

- raw SAM bbox is stored as:
  - `object_crop_raw_bbox_x0`
  - `object_crop_raw_bbox_y0`
  - `object_crop_raw_bbox_x1`
  - `object_crop_raw_bbox_y1`
- final square crop bbox is stored as:
  - `object_crop_bbox_x0`
  - `object_crop_bbox_y0`
  - `object_crop_bbox_x1`
  - `object_crop_bbox_y1`
- default `resize_mode` is now `square_crop`
- `pad` and `stretch` remain available for comparison/debugging

Requested visualization was added:

- original image
- original with raw SAM bbox overlaid
- original with square crop bbox overlaid
- resulting image

Smoke artifacts:

- [object_focus_smoke_real_sam3_mps_square_crop_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/object_focus_smoke_real_sam3_mps_square_crop_384.csv)
- [real_sam3_mps_square_crop_geometry_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/real_sam3_mps_square_crop_geometry_sheet.jpg)
- [real_sam3_mps_square_crop_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/real_sam3_mps_square_crop_sheet.jpg)

Smoke result:

- rows written: `12`
- `object_crop_status=detected`: `12`
- `object_crop_method=sam3_text`: `12`
- `object_crop_resize_mode=square_crop`: `12`

Verification:

- `38 passed`

### Synthetic Smoke

The same geometry visualization was generated on synthetic images, since the
real images are already closely cropped.

Artifacts:

- [object_focus_smoke_synthetic_sam3_mps_square_pad_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/object_focus_smoke_synthetic_sam3_mps_square_pad_384.csv)
- [synthetic_sam3_mps_square_pad_geometry_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/synthetic_sam3_mps_square_pad_geometry_sheet.jpg)
- [synthetic_sam3_mps_square_pad_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/synthetic_sam3_mps_square_pad_sheet.jpg)

Synthetic smoke result:

- rows written: `12`
- `object_crop_status=detected`: `12`
- `object_crop_method=sam3_text`: `12`
- `object_crop_resize_mode=square_crop`: `12`
- mean confidence: `0.9560`
- min confidence: `0.9143`
- max confidence: `0.9724`

### Padding Tightening

The default square-crop padding was reduced from `0.12` per side to `0.01` per
side.

Reason:

- SAM3 already returns a package-level box
- excessive padding pulls background back into the crop
- the object-focused preprocessing experiment should keep the object tightly in
  frame

Updated synthetic smoke artifacts:

- [object_focus_smoke_synthetic_sam3_mps_square_pad01_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/object_focus_smoke_synthetic_sam3_mps_square_pad01_384.csv)
- [synthetic_sam3_mps_square_pad01_geometry_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/synthetic_sam3_mps_square_pad01_geometry_sheet.jpg)
- [synthetic_sam3_mps_square_pad01_sheet.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus_smoke/synthetic_sam3_mps_square_pad01_sheet.jpg)

Updated synthetic smoke result:

- rows written: `12`
- `object_crop_status=detected`: `12`
- `object_crop_method=sam3_text`: `12`
- `object_crop_resize_mode=square_crop`: `12`
- mean confidence: `0.9560`

## Milestone 51: Full Synthetic SAM3-Cropped V4.5 Training Completed

### Goal

Test whether training V4.5 on SAM3 square-cropped synthetic images improves:

- synthetic held-out score-band performance
- transfer evaluation on the raw real labeled dataset

The real images were not SAM-cropped for the final comparison because the user
explicitly stopped that path.

### Synthetic Crop Dataset

Full synthetic SAM3/MPS preprocessing completed with strict SAM3:

- source manifest:
  [synthetic_full_qc_plus_v2_scale_processed_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv)
- cropped manifest:
  [synthetic_full_qc_plus_v2_scale_sam3_square_pad01_384.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/datasets/synthetic_full_qc_plus_v2_scale_sam3_square_pad01_384.csv)
- cache:
  [synthetic_full_qc_plus_v2_scale_sam3_square_pad01_384](/Users/inventure71/VSProjects/School/Dream2Detect/data/cache/synthetic_full_qc_plus_v2_scale_sam3_square_pad01_384)

QC result:

- rows: `899`
- image files present: `899`
- `object_crop_status=detected`: `899`
- `object_crop_method=sam3_text`: `899`
- mean SAM confidence: `0.9603`
- min SAM confidence: `0.8823`
- score-band coverage preserved

QC artifacts:

- first-40 geometry sheet:
  [synthetic_sam3_square_pad01_geometry_head40.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus/synthetic_sam3_square_pad01_geometry_head40.jpg)
- first-40 crop sheet:
  [synthetic_sam3_square_pad01_head40.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus/synthetic_sam3_square_pad01_head40.jpg)
- stratified 50-image geometry sheet:
  [synthetic_sam3_square_pad01_geometry_stratified_sample50.jpg](/Users/inventure71/VSProjects/School/Dream2Detect/data/qc/object_focus/synthetic_sam3_square_pad01_geometry_stratified_sample50.jpg)

### Training Run

The V4.5 objective was retrained on the cropped synthetic manifest using the
same controlled settings:

- `residual_cnn`
- `metadata_family_holdout`
- `damage_safe`
- `score_band`
- soft-label sigma `1.0`
- effective-number beta `0.999`
- ordinal loss weight `0.2`
- EMD weight `0.5`
- seed `42`

Run:

- [v4_5_sam3_square_pad01_synthetic_only_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_classifier/v4_5_sam3_square_pad01_synthetic_only_20260516)

Outcome:

- early stopped at epoch `134`
- best validation epoch: `94`
- best validation mean band error: `1.1687`

Synthetic held-out test:

- band accuracy: `0.2039`
- band macro F1: `0.2037`
- mean band error: `1.2670`
- `+-1` band accuracy: `0.7136`
- collapsed 4-class macro F1: `0.4619`

Compared to original V4.5 on synthetic heldout:

- band accuracy improved: `0.1748` -> `0.2039`
- band macro F1 improved: `0.1574` -> `0.2037`
- mean band error improved: `1.3398` -> `1.2670`
- collapsed coarse macro F1 improved: `0.4439` -> `0.4619`
- `+-1` band accuracy worsened slightly: `0.7330` -> `0.7136`

### Raw Real Transfer Evaluation

The cropped-synthetic model was evaluated on the existing raw real manifest:

- [v4_5_sam3_square_pad01_synthetic_on_real_all_test_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_transfer/v4_5_sam3_square_pad01_synthetic_on_real_all_test_20260516)

Comparison artifact:

- [v4_5_sam3_square_pad01_vs_original_20260516.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/real_transfer/v4_5_sam3_square_pad01_vs_original_20260516.csv)

Raw real all-test comparison against original V4.5:

- band accuracy worsened: `0.1273` -> `0.0779`
- band macro F1 worsened: `0.0834` -> `0.0537`
- mean band error worsened: `2.5351` -> `2.9558`
- `+-1` band accuracy worsened: `0.3532` -> `0.2468`
- collapsed coarse macro F1 worsened: `0.2897` -> `0.2294`

### Decision

The SAM3-cropped synthetic dataset is useful but **not a transfer improvement**
in this form.

Interpretation:

- object-focused cropping made the synthetic holdout task cleaner
- but it made raw-real transfer worse
- therefore the current crop policy should not replace the baseline training
  data for real-domain transfer

Next direction:

- use the SAM3-cropped synthetic dataset as an ablation result
- do not crop real images further unless explicitly requested
- investigate whether the crop policy removes contextual cues or changes the
  visual scale too much relative to the raw real dataset

## Milestone 52: V5 Performance Phase Opened

### Goal

Move from V4/V4.5 controlled challenger experiments into V5, where the goal is
to get the strongest possible 10-band severity model while preserving the core
experiment controls.

### Decision From The SAM3 Ablation

The SAM3 square-crop synthetic experiment is recorded as an ablation, not a new
baseline.

Reason:

- it improved synthetic held-out performance
- it worsened raw-real transfer on all main metrics
- therefore it does not solve the current synthetic-to-real gap

The real dataset should not be SAM-cropped further unless the plan is explicitly
changed.

### V5 Constraints

V5 keeps these constraints:

- no pretrained backbones
- no silent use of raw-real test results for model selection
- `score_band` remains the primary target
- collapsed 4-class metrics remain secondary diagnostics
- V4 remains the locked score-band reference until a stronger model clearly
  replaces it

### V5 Direction

The next improvement should be structural, not another small hyperparameter
tweak.

Detailed design:

- [docs/21-v5-design-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/21-v5-design-plan.md)

First target:

- V5-A scalar ordinal regression

Why:

- the task is ordered severity prediction
- the current 10-way classification framing still behaves like a class problem
- a single normalized severity output can be mapped back into the ten bands for
  direct comparison

Follow-up candidates:

- V5-B multitask ordinal model with scalar, 10-band, and optional coarse heads
- V5-C threshold-based ordinal model using a CORAL/CORN-style head

## Milestone 53: V5-A Scalar Ordinal Infrastructure Started

### Outcome

The first V5-A code path is implemented.

Implemented pieces:

- normalized fine target mode: `fine_normalized`
- scalar severity-to-score-band metrics
- from-scratch `residual_cnn_groupnorm_regressor`
- V5-compatible regressor trainer controls
- AdamW / scheduler / split / model-variant CLI controls
- per-epoch metrics CSV and JSONL
- per-epoch, latest, and best checkpoints
- training curve output

### Verification

Unit and integration tests:

- `85 passed`

CLI smoke:

```bash
python3 scripts/train_synthetic_regressor.py \
  --manifest data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv \
  --image-size 384 \
  --batch-size 2 \
  --num-epochs 2 \
  --learning-rate 0.001 \
  --lr-scheduler none \
  --no-augmentation \
  --model-variant residual_cnn_groupnorm_regressor \
  --target-mode fine_normalized \
  --overfit-subset-size 10 \
  --checkpoint-every-n-epochs 1 \
  --output-dir data/training_runs/synthetic_regressor/v5_a_scalar_cli_smoke_20260516
```

Smoke result:

- device: `mps`
- final test MAE: `23.424`
- final test 10-band accuracy: `0.1000`
- final test mean band error: `2.5000`
- artifacts:
  [v5_a_scalar_cli_smoke_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_cli_smoke_20260516)

### Interpretation

This smoke proves the V5-A command path and artifact writing work.

It does **not** prove the model can overfit yet, because the smoke only ran for
two epochs. The next gate is a proper tiny-subset overfit run before full V5-A
training.

### Overfit Gate

A stricter 4-example no-augmentation overfit gate was then run:

```bash
python3 scripts/train_synthetic_regressor.py \
  --manifest data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv \
  --image-size 384 \
  --batch-size 2 \
  --num-epochs 80 \
  --learning-rate 0.001 \
  --lr-scheduler none \
  --weight-decay 0 \
  --dropout 0 \
  --no-augmentation \
  --model-variant residual_cnn_groupnorm_regressor \
  --target-mode fine_normalized \
  --overfit-subset-size 4 \
  --checkpoint-every-n-epochs 20 \
  --output-dir data/training_runs/synthetic_regressor/v5_a_scalar_overfit4_20260516
```

Result:

- device: `mps`
- final test MAE: `3.310`
- final test 10-band accuracy: `0.7500`
- final test mean band error: `0.2500`
- final test `+-1` band accuracy: `1.0000`
- final collapsed coarse macro F1: `1.0000`
- artifacts:
  [v5_a_scalar_overfit4_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_overfit4_20260516)

Interpretation:

- the scalar model can memorize a tiny subset well enough to pass the V5-A
  ordinal overfit gate
- the only remaining exact 10-band miss landed in a neighboring band

### Preflight Bug Fix

A one-epoch full-split preflight exposed a split bug in the existing
`metadata_family_holdout` implementation:

- some `split_group_id` values appeared in multiple splits
- this happened because the old implementation split independently inside each
  label, so the same metadata family could land in different splits when it
  appeared under neighboring score bands

The split builder was corrected to choose train/validation/test groups globally
before returning indices.

Fixed preflight:

- [v5_a_preflight_fullsplit_fixed_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_preflight_fullsplit_fixed_20260516)

Verification:

- train/validation group overlap: `0`
- train/test group overlap: `0`
- validation/test group overlap: `0`
- checkpoint config, scheduler state, epoch CSV, and split artifacts were
  inspected

Decision:

- full V5-A synthetic training can now be run using the corrected split logic

## Milestone 54: V5-A Scalar Ordinal Full Run Completed

The full V5-A synthetic-only scalar ordinal regressor was trained with the
corrected metadata-family holdout split.

Run:

- [v5_a_scalar_ordinal_seed42_20260516_fixed](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed)

Configuration:

- manifest: `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- image size: `384`
- model: `residual_cnn_groupnorm_regressor`
- target: normalized representative score, evaluated as 10 score bands
- split strategy: `metadata_family_holdout`
- optimizer: `adamw`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.1`
- scheduler: `reduce_on_plateau`
- augmentation profile: `damage_safe`
- seed: `42`

During training, the local app session was interrupted after checkpoint epoch
`74`. The regressor trainer was updated with checkpoint resume support:

- CLI: `--resume-from`
- restored model, optimizer, scheduler, and early-stopping state
- rebuilt training history from existing `epoch_metrics.csv`
- added regression tests for early-stopping completion and checkpoint resume

Verification:

- full test suite: `87 passed`
- compileall: passed
- resumed from checkpoint epoch `74`
- saved epoch metrics: `119` unique epochs, no duplicates
- latest checkpoint: epoch `119`
- best checkpoint: epoch `79`
- train/validation group overlap: `0`
- train/test group overlap: `0`
- validation/test group overlap: `0`

Result:

- best validation epoch: `79`
- best validation mean band error: `1.4171`
- early stopped at epoch: `119`
- final synthetic test MAE: `16.297`
- final synthetic test 10-band accuracy: `0.1685`
- final synthetic test 10-band macro F1: `0.1529`
- final synthetic test mean band error: `1.7079`
- final synthetic test `+-1` band accuracy: `0.5225`
- final synthetic collapsed coarse macro F1: `0.4285`

Longer-training check:

- copied run:
  [v5_a_scalar_ordinal_seed42_20260516_fixed_longer_from119](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed_longer_from119)
- resumed from epoch `119`
- extended `num_epochs` to `240`
- raised early-stopping patience to `100`
- stopped at epoch `179`
- overall best epoch remained `79`
- best post-119 validation mean band error: `1.5294`
- original best validation mean band error remained `1.4171`
- selected test metrics did not change because no later checkpoint beat epoch
  `79`

Real-domain evaluation:

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
- saved artifacts:
  [real_eval_summary.json](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed/real_eval_summary.json)

Interpretation:

- V5-A is mechanically correct and completed end to end.
- The scalar model learns an ordinal signal, but exact 10-band accuracy remains
  weak.
- The model overfits after the best epoch: train band error keeps improving
  while validation band error plateaus around `1.4` to `1.6`.
- Longer training with the same objective did not improve the selected model.
- Real-domain transfer is worse than synthetic test performance. Padding real
  images to 384 does not materially close the gap.
- V5-A alone should not be treated as a solved baseline. The next comparison
  should place it against locked V4/V4.5, then decide whether V5-B multitask or
  V5-C threshold ordinal is the stronger next experiment.

## Milestone 55: V5-B Multitask Ordinal Round 1 Completed

V5-B round 1 was implemented and trained as a controlled multitask challenger
after V5-A.

Run:

- [v5_b_multitask_scalar_band_coarse_seed42_20260516](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516)

Configuration:

- manifest: `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- image size: `384`
- model: `residual_cnn_groupnorm_multitask`
- split strategy: `metadata_family_holdout`
- optimizer: `adamw`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.1`
- scheduler: `reduce_on_plateau`
- augmentation profile: `damage_safe`
- heads:
  - scalar normalized severity
  - auxiliary 10-band score band
  - auxiliary 4-class coarse class
- loss weights:
  - scalar: `1.0`
  - coarse: `0.3`
  - auxiliary 10-band: `0.3`
  - auxiliary band ordinal penalty: `0.2`
- seed: `42`

Implementation changes:

- upgraded the multitask dataset target path so each example carries
  `coarse`, `score_band`, and normalized scalar `score`
- added a GroupNorm residual multitask model variant
- made the scalar head the primary selection target using validation scalar
  mean band error
- kept the auxiliary 10-band and coarse heads as stabilizers and diagnostics
- carried metadata-family holdout through the multitask split artifacts

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
- direct coarse accuracy: `0.4382`
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

Real padded-384 evaluation:

- scalar exact 10-band accuracy: `0.1377`
- scalar `+-1` band accuracy: `0.3922`
- scalar `+-2` band accuracy: `0.6364`
- scalar `+-3` band accuracy: `0.8000`
- scalar `+-4` band accuracy: `0.9247`
- scalar mean band error: `2.1429`

Artifacts:

- evaluation summary:
  [v5_b_eval_summary.json](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516/v5_b_eval_summary.json)
- comparison table:
  [v5_b_round1_comparison.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/model_checkpoint_comparison/v5_b_round1_comparison.csv)

Interpretation:

- V5-B round 1 is not a better synthetic model than V5-A. V5-A synthetic mean
  band error is `1.7079`; V5-B scalar synthetic mean band error is `1.9663`.
- V5-B does improve real-domain broad ordinal transfer compared with V5-A.
  Real raw mean band error improves from `2.2727` to `2.1091`; real raw
  `+-2` improves from `0.5896` to `0.6260`; real raw `+-3` improves from
  `0.7584` to `0.8130`; real raw `+-4` improves from `0.8935` to `0.9325`.
- Real raw `+-1` is slightly worse than V5-A: `0.3948` versus `0.3974`.
- The auxiliary 10-band head gives higher exact band accuracy inside V5-B, but
  the scalar head is the better severity estimator because it has lower mean
  band error.
- The next useful V5 step is not to promote V5-B as-is. Either tune the
  multitask loss weights with a very small controlled grid or move to V5-C
  threshold ordinal modeling.

### Longer-Training Check

Because the first V5-B run selected epoch `160`, which was also the final
scheduled epoch, the run was extended from its latest checkpoint.

Extended run:

- [v5_b_multitask_scalar_band_coarse_seed42_20260516_longer_from160](/Users/inventure71/VSProjects/School/Dream2Detect/data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516_longer_from160)

Extended settings:

- resumed from epoch `160`
- trained to `320` scheduled epochs
- early-stopping patience: `80`
- restored model and optimizer state
- checkpoint learning rate at resume: `1e-05`

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

Artifact:

- [v5_b_longer_comparison.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/model_checkpoint_comparison/v5_b_longer_comparison.csv)

Decision:

- The user was right that the original V5-B run had not fully converged on its
  validation metric.
- However, the longer checkpoint did not improve held-out synthetic test or raw
  real evaluation.
- Treat this as validation/test mismatch evidence, not as a simple
  not-enough-epochs problem.
- Do not promote V5-B longer as the new baseline.
