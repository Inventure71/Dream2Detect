# Data Contracts

This file defines the first execution-level artifacts for Dream2Detect.

The goal is simple:

- generation should write into one known structure,
- relabeling should write into one known structure,
- training should read from one known structure,
- nobody should invent ad hoc CSV columns halfway through the project.

## Folder Layout

The project now uses this starter layout:

```text
data/
  dream2detect.sqlite3
  datasets/
    synthetic_starting_dataset_phase1_round1.csv
    synthetic_reviewed_seed_set_phase1_round1.csv
  manifests/
    pilot_prompt_manifest.csv
  registries/
    synthetic_image_registry.csv
    real_relabel_registry.csv
  real/
    source/
      redf0xwin_recognizing_defects_in_boxes_and_cardboard/
        images/
        annotations/
    labeled/
  synthetic/
    pilot/
    batches/
  templates/
    prompt_writing_template.md
    synthetic_prompt_manifest_template.csv
    synthetic_image_registry_template.csv
    real_relabel_registry_template.csv
generated_images/
  synthetic_all/
```

## File Roles

### `data/manifests/pilot_prompt_manifest.csv`

The first working manifest for the pilot generation batch.

Use it to list prompt candidates before images are generated.

### `data/registries/synthetic_image_registry.csv`

The main record of generated synthetic images after review.

This should become the source of truth for:

- what was generated,
- what band it was meant to match,
- whether it passed review,
- where the image file lives,
- whether generation is `pending`, `in_progress`, `generated`, or `failed`.

### `data/registries/real_relabel_registry.csv`

The main record of relabeled real Kaggle images.

This should become the source of truth for:

- AI-assisted label suggestion,
- human-approved final band,
- final coarse class,
- final derived representative score,
- review status and disagreement.

The current imported real source is Kaggle `redf0xwin/recognizing-defects-in-boxes-and-cardboard`. Its original XML bounding-box annotations are retained as provenance, but Dream2Detect training labels must come from the relabel registry after human severity review.

### `data/dream2detect.sqlite3`

The execution database for:

- drafted prompts,
- structured feature assignments per prompt,
- generation status,
- image linkage,
- local image references.

### `data/datasets/`

Training-facing exported manifests derived from the execution database.

These exist so training code can consume a stable CSV without having to know the internal prompt/QC schema directly.

Current examples:

- `synthetic_starting_dataset_phase1_round1.csv`
  - best-available training labels
  - includes reviewed and unreviewed generated synthetic images
- `synthetic_reviewed_seed_set_phase1_round1.csv`
  - reviewed-only seed subset
  - excludes unreviewed and rejected images

### `generated_images/`

Local storage root for generated image files.

Canonical storage rule:

- generation may happen into a temporary subfolder for a run
- after the run is accepted operationally, images should be consolidated into:
  - `generated_images/synthetic_all/`

Each prompt record may have a nullable `image_ref`.

If `image_ref` is empty, that prompt still needs image generation.

## Structured Feature Assignment Rule

Prompt rows are not just free text.

Each synthetic prompt should carry explicit feature-axis fields such as:

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

Prompt text should be drafted from those fields, not used as the only place where those decisions exist.

## Core Principles

### 1. Band first

The primary label is `score_band`.

Everything else is derived from it:

- `coarse_class`
- `representative_score`

### 2. One row per image or per prompt plan

- prompt manifest: one row per intended image prompt
- synthetic registry: one row per generated synthetic image
- real relabel registry: one row per real image

### 3. Explicit review state

Nothing should silently move from idea to accepted data.

Use explicit fields like:

- `status`
- `quality_status`
- `review_status`

### 4. Stable identifiers

Every prompt and every image needs an ID.

Do not rely only on filenames.

### 5. Fail fast on image generation

If an image generation call fails:

- mark that prompt as `failed`
- store the error text
- stop the current generation run

Do not retry automatically in the same run.

## Canonical Synthetic Prompt Manifest Columns

Defined in:

- [synthetic_prompt_manifest_template.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/templates/synthetic_prompt_manifest_template.csv)

Purpose:

- plan synthetic generation by band
- persist the sampled feature assignment
- capture prompt text and prompt-review state
- keep pilot and batch generation reproducible

## Canonical Synthetic Image Registry Columns

Defined in:

- [synthetic_image_registry_template.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/templates/synthetic_image_registry_template.csv)

Purpose:

- record actual generated files
- connect each image to its prompt and target band
- preserve the structured feature assignment used to request the image
- record review acceptance, relabeling, or rejection

## Generated Image QC

Prompt approval is not enough.

After image generation, each synthetic image needs a separate QC decision because a prompt-faithful output can still land in the wrong severity band.

Each generated image should support:

- intended `score_band` and `coarse_class`
- reviewed `score_band` and `coarse_class`
- final accepted `score_band` and `coarse_class`
- QC decision:
  - `accepted_as_labeled`
  - `accepted_relabel`
  - `rejected`
- QC notes

This prevents good-but-mislabeled synthetic outputs from silently contaminating training data.

## Canonical Real Relabel Registry Columns

Defined in:

- [real_relabel_registry_template.csv](/Users/inventure71/VSProjects/School/Dream2Detect/data/templates/real_relabel_registry_template.csv)

Purpose:

- track AI suggestion
- track human review
- freeze final approved label
- preserve disagreement metadata

## Start Order

Use these artifacts in this order:

1. sample balanced feature assignments for the pilot
2. fill the pilot prompt manifest
3. generate a small pilot image batch
4. run image-level QC and register accepted, relabeled, or rejected pilot images in the synthetic image registry
5. start real Kaggle relabeling in the real relabel registry
6. only after that, scale generation and move toward training

## Export Rule For Early Training

For early synthetic-only experiments:

- use the exported starting dataset manifest
- keep `training_label_source` so reviewed and unreviewed examples remain distinguishable

For cleaner later runs:

- use the reviewed-only export
- or filter the full export by `qc_status`
