# Implementation Plan

This file is the practical execution plan for the current package-based project.

## Objective

Build and evaluate a comparative ML study of synthetic-to-real transfer for package/cardboard defect severity.

## Core Deliverable

Produce evidence for how performance changes when we vary:

- training data source
- target formulation

under a held-out real evaluation set.

## Locked Design Decisions

### Targets

- fine target: overall visible package defect severity
- coarse classes:
  - `intact`
  - `minor`
  - `moderate`
  - `severe`

### Fine-to-coarse mapping

- `0-10 -> intact`
- `11-35 -> minor`
- `36-65 -> moderate`
- `66-100 -> severe`

### Prompting and annotation bands

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

### Real data policy

- relabel Kaggle `redf0xwin/recognizing-defects-in-boxes-and-cardboard` into one unified real pool first
- then split:
  - `60%` held-out real test
  - `20%` small real training
  - `20%` real validation / fine-tuning support
- own photos are optional later stress tests only

### Model stack

Baselines:

- majority class baseline
- mean predictor baseline
- HOG + logistic regression
- HOG + ridge regression

Neural models:

- custom CNN classifier from scratch
- custom CNN regressor from scratch

Training strategies for the CNNs:

- synthetic-only
- real-only
- synthetic + real fine-tuning

### OpenAI runtime choices

Chosen prompt-drafting model:

- `gpt-5.4-mini`

Chosen image-generation model:

- `gpt-image-2`

Chosen image-generation defaults for the initial pipeline:

- `quality=low`
- `size=1024x1024`

Implementation note:

- use a simple dedicated OpenAI connector/module
- use the text model for prompt drafting
- use the direct Image API for explicit image generation with `gpt-image-2`
- use structured outputs for prompt drafting, not free-form JSON parsing
- use fail-fast image generation: mark the failing prompt as `failed` and stop the run

## Workstream 1: Severity And Prompt System

1. freeze the `v1` band specs
2. define a structured diversity-axis system
3. define prompt-writing instructions from the band specs
4. sample balanced feature assignments by band
5. use an LLM to draft prompts from those assignments
6. manually review and edit the prompts
7. store prompts in a structured manifest

Expected output:

- prompt manifest by band
- prompt-review log
- feature-assignment records per prompt
- stable generation instructions

### V2 prompt update

The v1 prompt pipeline is preserved under `src/dream2detect/pipelines/v1/`.

The current improvement path is the v2 real-failure-targeted prompt pipeline in
`src/dream2detect/pipelines/v2/`. V2 is designed around the observed
synthetic-to-real failures, especially damaged real packages predicted as
`intact`, flat taped mailers, subtle boundary damage, cluttered real capture
conditions, and hard intact negatives.

V2 prepares reviewable prompt and Batch API artifacts locally. It does not
submit API jobs by itself.

The May 13 cleanup froze the current missing-coverage targets:

- reduce high-confidence real damaged examples predicted as `intact`
- increase `minor` recall without exaggerating minor damage into moderate damage
- make moderate geometry loss visible in realistic phone-photo conditions
- make severe openings/collapse realistic but unmistakable
- keep hard intact negatives structurally intact despite tape, labels, clutter,
  shadows, and scuffs

## Workstream 2: Synthetic Dataset Pipeline

1. generate a small pilot batch first
2. inspect the pilot against the band specs
3. reject prompts or images that do not fit the target band
4. revise prompts if needed
5. generate larger batches only after the pilot is acceptable
6. run image-level QC after generation:
   - `accepted_as_labeled`
   - `accepted_relabel`
   - `rejected`
7. store image metadata and quality-review state

Expected metadata per synthetic image:

- `image_id`
- `image_path`
- `score_band`
- `coarse_class`
- `band_spec_version`
- derived `representative_score`
- prompt identifier
- generation batch
- quality status

V2 synthetic images must pass a stricter domain-gap QC gate before training:

- reject render-like, stylized, or product-ad images
- reject images whose defect is hidden or semantically different from the
  manifest
- reject hard intact negatives that contain structural damage
- reject obvious score-band mismatches
- reject images dominated by readable labels, addresses, logos, or large text
- prefer ordinary smartphone-like package photos with realistic tape, labels,
  lighting, background clutter, compression, and imperfect framing

## Workstream 3: Real Dataset Relabeling

1. merge candidate Kaggle images into one pool
2. run AI-assisted, human-approved relabeling
3. mark uncertain or boundary cases
4. run second review only when needed
5. freeze final approved labels
6. rebuild the final split with balancing and leakage control

Balancing should be done at least across:

- coarse class
- score band
- structured diversity axes such as background, lighting, camera angle, and box appearance

Leakage control should keep near-duplicates and same-box near-views in the same split.

Expected output:

- relabeled real registry
- held-out real test split
- small real train split
- real validation / fine-tuning split

## Workstream 4: Modeling

### Coarse classification

Train and compare:

- majority baseline
- HOG + logistic regression
- CNN synthetic-only
- CNN real-only
- CNN synthetic + fine-tuning

### Fine regression

Train and compare:

- mean baseline
- HOG + ridge regression
- CNN synthetic-only
- CNN real-only
- CNN synthetic + fine-tuning

## Workstream 5: Evaluation

Evaluate on held-out real images.

### Classification outputs

Report at least:

- accuracy
- balanced accuracy
- macro F1
- per-class precision and recall
- confusion matrix

### Regression outputs

Report at least:

- MAE
- RMSE
- predicted vs true behavior
- error by severity range if feasible

### Comparison outputs

Show:

- synthetic-only vs real-only vs hybrid
- fine vs coarse
- where transfer fails
- where limited real data helps

## Workstream 6: Analysis And Reporting

The final report should answer:

- does synthetic-only transfer at all?
- how far is it from small real-only?
- how much does fine-tuning help?
- is coarse severity more robust than fine severity?
- what kinds of defects are hardest under shift?

Also include:

- failure examples
- labeling limitations
- dataset limitations
- likely causes of synthetic-to-real mismatch

## Immediate Next Implementation Tasks

1. define the structured diversity axes and allowed values
2. sample balanced feature assignments for pilot prompts
3. define the synthetic metadata schema
4. define the real relabeling CSV schema
5. create the pilot prompt manifest

## Completion Standard

The project is only complete when:

- the rubric is usable,
- the synthetic data pipeline is reproducible,
- the real labels are frozen under a documented protocol,
- the comparison models are trained,
- evaluation is done on held-out real data,
- the final report explains the observed differences rather than only listing scores.

## V3 Synthetic-Only Phase

Before another major real-transfer push, the current plan now includes a
synthetic-only V3 training phase:

- [18-v3-synthetic-training-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/18-v3-synthetic-training-plan.md)

This phase freezes the current V2 synthetic-only baseline and focuses on three
training changes:

- GroupNorm residual training for small-batch stability
- true ordinal coarse-class training
- constrained RandAugment-style synthetic-safe augmentation profiles

Frozen V2 comparison artifact:

- [v2_synthetic_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v2_synthetic_baseline_20260513.csv)

V3.1 completion update:

- the primary synthetic-only evaluation is now the harder
  `metadata_family_holdout` split
- the derived holdout family uses:
  `training_coarse_class | damage_profile_primary | box_form_factor | background_context`
- the current hard-split baseline artifact is:
  [v3_1_hard_split_baseline_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_hard_split_baseline_20260513.csv)
- the supporting run comparison table is:
  [v3_1_metadata_family_holdout_comparison_20260513.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv)

## Score-Band Target Phase

The next target shift is now explicit:

- primary task: `score_band` with the official 10 severity bands
- secondary view: collapse the score-band model back to the 4 coarse classes

The frozen coarse implementation/reference before this shift is:

- [coarse_reference_before_score_band_target_20260514.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/evaluations/synthetic_only/coarse_reference_before_score_band_target_20260514.csv)

The phase-specific plan is:

- [19-score-band-target-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/19-score-band-target-plan.md)

Current V4 implementation direction:

- from-scratch `residual_cnn`
- `metadata_family_holdout`
- `damage_safe` augmentation
- soft ordinal targets across neighboring score bands
- effective-number class-balanced loss
- validation mean band error as the primary checkpoint-selection metric
- collapsed coarse evaluation from summed score-band probabilities
- score-band-aware analyzer outputs and ranked grid summaries

Latest V4.1 challenger result:

- the first narrow challenger sweep is complete
- `sigma=1.25` beat the current baseline on validation mean band error
- but it did not clearly improve held-out test ordinal behavior
- therefore the baseline is not replaced yet
- the next score-band decision is a two-way seed confirmation:
  baseline `sigma=1.0` versus challenger `sigma=1.25`

Two-seed confirmation update:

- `sigma=1.25` also beat the baseline on validation mean band error in the
  second seed
- but the two-seed average test behavior remained slightly worse
- therefore the baseline still stays locked
- the next score-band tuning pass should search narrowly between `sigma=1.0`
  and `sigma=1.25`

V4.2 sigma-only sweep update:

- the `1.10 / 1.15 / 1.20` two-seed sweep is complete
- no sigma value improved both validation selection and held-out ordinal test
  behavior
- the current V4 baseline therefore remains locked
- the next score-band tuning axis should move from sigma width to target
  formulation: soft-label shape or score-band weighting

V4.5 implementation direction:

- keep the locked V4 score-band stack unchanged except for the training
  objective
- add a cumulative squared EMD distance term on top of the current soft-label
  score-band loss
- expose the new control as `score_band_emd_weight`
- train one single-seed challenger first before considering any new sweep

V4.5 outcome:

- the hybrid soft-CE + cumulative squared EMD challenger was implemented and
  trained
- it slightly improved held-out `+-1` band accuracy
- but it did not beat the locked V4 baseline on validation mean band error or
  held-out mean band error
- therefore the locked V4 baseline remains the current reference

## Object-Focused Preprocessing Phase

The next domain-shift hypothesis is that synthetic and real images differ in
object framing. Synthetic images tend to keep the package near the center and
large in frame, while real images may include more background or already-cropped
views.

The new preprocessing path should build object-focused caches before training:

- detect the cardboard box/package region with SAM 3 text prompts when SAM 3 is
  available
- use a deterministic center-prior crop fallback only in explicit `hybrid`
  mode; strict `sam3` mode must fail rather than silently producing center
  crops
- crop around the selected box region with padding, then pad-resize to the
  target square resolution
- write crop metadata into the derived manifest so failures and fallbacks are
  auditable

SAM 3 remains an optional runtime backend because the official release requires
the latest `facebookresearch/sam3` package, gated Hugging Face checkpoint access,
and a CUDA-oriented environment.

Apple Silicon / MPS support is tracked as an experimental path based on the
open upstream PR for MPS support. It is not treated as validated until
`scripts/check_sam3_setup.py --device mps` passes and a one-image smoke returns
real SAM boxes.

Validation update:

- SAM3/MPS passed the one-image smoke after Hugging Face auth was configured
- strict SAM3/MPS crop smoke on 12 real images produced 12 `sam3_text`
  detections and 0 fallbacks
- visual QC of the 12-image contact sheet is now required before full-cache
  generation

QC correction:

- the initial SAM3 smoke used aspect-preserving pad-resize after detection
- wide package detections became thin horizontal strips on a square canvas
- the temporary `stretch` mode filled the square but distorted package geometry
- object-focused preprocessing now defaults to `resize_mode=square_crop`, which
  converts the SAM box into a square crop in the original image before resizing
- square side length is based on the longest SAM bbox side and is no longer
  capped by the source image's shorter side
- when that square extends beyond the source image, the missing pixels are
  padded white
- default square padding is now `0.01` per side to keep the object crop tight
- both the raw SAM box and final square crop box are persisted in the manifest
- the geometry QC sheet shows original image, SAM bbox overlay, square bbox
  overlay, and resulting crop
- `resize_mode=pad` and `resize_mode=stretch` remain available only for
  comparison/debugging

Full synthetic crop experiment result:

- full synthetic SAM3 square-crop dataset was generated successfully
- V4.5 retraining on that dataset improved synthetic held-out band accuracy,
  band macro F1, mean band error, and collapsed 4-class macro F1 versus
  original V4.5
- raw-real transfer evaluation worsened on every main metric versus original
  V4.5
- therefore the SAM3-cropped synthetic dataset is an ablation artifact, not a
  new baseline replacement
- do not SAM-crop the real dataset further unless the plan is explicitly changed

Setup and preflight are documented in
[docs/20-sam3-setup.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/20-sam3-setup.md).

Implementation status:

- [build_object_focused_cache.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/build_object_focused_cache.py)
  now builds object-focused manifests and optional source/crop contact sheets
- `sam3`, `center`, and `hybrid` crop backends are supported
- local smoke runs passed through the `hybrid` fallback path because SAM 3 is
  not installed in the current environment
- the CLI now defaults to strict `sam3` mode and raises immediately if SAM 3 is
  not installed, so future SAM experiments cannot be mistaken for center crops
- [check_sam3_setup.py](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/check_sam3_setup.py)
  records the exact Python, PyTorch, CUDA, SAM3, and Hugging Face readiness
  checks before any strict SAM3 crop run
- [run_sam3_mps_python.sh](/Users/inventure71/VSProjects/School/Dream2Detect/scripts/run_sam3_mps_python.sh)
  runs project scripts inside the dedicated `sam3-mps` environment with the
  Apple Silicon SAM3 source checkout on `PYTHONPATH`
- `--sam3-device` is available for strict crop runs so the same pipeline can
  target `cuda`, `mps`, or `cpu` when the installed SAM3 build supports device
  selection

## V5 Performance Phase

V5 starts after the object-focused preprocessing ablation.

Detailed V5 design:

- [docs/21-v5-design-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/21-v5-design-plan.md)

The V5 goal is to improve the 10-band `score_band` task as much as possible
without abandoning the experiment controls:

- no pretrained models
- no silent real-test leakage into model selection
- no SAM-cropping of real images unless explicitly approved later
- keep the 10-band task primary and use collapsed 4-class metrics only as
  secondary diagnostics

The current evidence says the next improvement should not be another small
sigma, dropout, or learning-rate tweak. V4.5 already tested an ordinal-distance
objective, and the SAM3 crop ablation showed that cleaner synthetic framing can
improve synthetic holdout while hurting raw-real transfer.

V5 should therefore test structural target/head changes:

1. **V5-A scalar ordinal regression**
   - keep the from-scratch residual CNN backbone
   - replace the 10-way-only objective with a single normalized severity output
   - train against the representative band score normalized to `[0, 1]`
   - evaluate by mapping the scalar prediction back into the 10 score bands
2. **V5-B multitask ordinal model**
   - shared from-scratch backbone
   - scalar severity head
   - 10-band score head
   - optional collapsed coarse auxiliary head
3. **V5-C threshold ordinal model**
   - from-scratch CORAL/CORN-style ordered-threshold head
   - same data, split, and augmentation controls as V4

Initial V5 priority:

- implement V5-A first
- compare it against locked V4, V4.5, real-only, and raw-real transfer
- only move to V5-B or V5-C after the scalar head gives a clean answer
