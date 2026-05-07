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
