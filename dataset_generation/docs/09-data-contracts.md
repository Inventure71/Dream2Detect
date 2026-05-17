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
  cache/
    synthetic_combined_phase1_plus_scaleup_200_processed_224/
  dream2detect.sqlite3
  datasets/
    synthetic_combined_phase1_plus_scaleup_200_processed_224.csv
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
  training_runs/
    synthetic_classifier/
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

Important identity rule:

- use `prompt_uid` for cross-manifest reconciliation
- do not use numeric `prompt_id` as a global stable key across merged historical
  manifests

The current database has been synced to the full-QC synthetic manifest for all
available matching `prompt_uid` rows. The current training source of truth is
still the exported full-QC manifest under `data/datasets/`, because some older
phase-1 rows exist in the training manifest but not in the current execution
database.

### `data/datasets/`

Training-facing exported manifests derived from the execution database.

These exist so training code can consume a stable CSV without having to know the internal prompt/QC schema directly.

Current examples:

- `synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv`
  - current paper-facing synthetic source manifest
  - `800` rows
  - full image-level QC labels for the phase-1, scale-up, and targeted-555
    synthetic pools
  - should be preferred over older `unreviewed` or partial-QC manifests for
    current synthetic training
- `synthetic_combined_phase1_plus_targeted_555_full_qc_processed_384.csv`
  - current preferred processed synthetic training manifest
  - points at deterministic `384 x 384` processed image assets
- `synthetic_combined_phase1_plus_scaleup_200_processed_224.csv`
  - best-available training labels
  - includes the phase-1 seed set plus the 200-image scale-up batch
  - points `image_path` at deterministic RGB-resized `224 x 224` cached copies
  - keeps `source_image_path` as provenance for the original generated image
- `synthetic_reviewed_seed_set_phase1_round1.csv`
  - reviewed-only seed subset
  - excludes unreviewed and rejected images

### `data/cache/`

Deterministic derived image assets for training.

These are not the canonical synthetic source images.

They exist to avoid repeating the same non-random preprocessing on every epoch.

Current example:

- `synthetic_combined_phase1_plus_scaleup_200_processed_224/`
  - RGB-converted
  - resized to `224 x 224`
  - no random augmentation baked in

### `generated_images/`

Local storage root for generated image files.

Canonical storage rule:

- generation may happen into a temporary subfolder for a run
- after the run is accepted operationally, images should be consolidated into:
  - `generated_images/synthetic_all/`

Each prompt record may have a nullable `image_ref`.

If `image_ref` is empty, that prompt still needs image generation.

### `data/training_runs/`

Saved artifacts from classifier training runs.

Each run directory should contain at least:

- `run_config.json`
- `metrics.json`
- `train_split.csv`
- `val_split.csv`
- `test_split.csv`
- `best_model.pt`
- `checkpoints/epoch_XXX.pt`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `epoch_metrics.jsonl`
- `epoch_metrics.csv`
- `training_curves.png`
- `class_monitoring.png`

The per-epoch checkpoint files store model weights, optimizer state, run
configuration, and that epoch's train/validation metrics. The CSV and PNGs are
for quick inspection during training; the JSONL file preserves the richer
per-epoch metric payload.

Classifier grid runs may be stored under:

- `data/training_runs/synthetic_classifier_grid/<grid_name>/`

Each grid directory should contain:

- one subdirectory per experiment configuration
- `grid_summary.csv`

Synthetic regressor runs may be stored under:

- `data/training_runs/synthetic_regressor/<run_name>/`

Each regressor directory should contain:

- `run_config.json`
- `metrics.json`
- `train_split.csv`
- `val_split.csv`
- `test_split.csv`
- `best_model.pt`
- `epoch_metrics.jsonl`
- `epoch_metrics.csv`

The regressor predicts `representative_score` and reports both continuous
metrics such as MAE/RMSE and bucketed coarse-class metrics for comparison with
the classifier.

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
