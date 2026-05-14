# Overnight Training Log

This log records concrete actions taken for the May 8, 2026 overnight
synthetic-only classifier training pass.

Constraint: no additional synthetic images may be generated. Deterministic
preprocessing caches are allowed because they are resized copies of existing
images, not new generated training examples.

## 2026-05-08 02:45 CEST - Initial Audit

- Confirmed the new batch manifest
  `data/datasets/synthetic_targeted_555_round1_medium_unreviewed.csv` has
  `555` rows.
- Confirmed all `555` new image paths exist on disk.
- Confirmed the new batch is balanced as intended for the class-equalizing
  expansion:
  - `intact`: `175`
  - `minor`: `126`
  - `moderate`: `127`
  - `severe`: `127`
- Confirmed the existing processed combined training manifests only have `245`
  rows, so they do **not** yet include the new `555` images.
- Decision: build a new combined manifest/cache from existing images before
  launching any more classifier training. Training the old `245`-row manifest
  again would not satisfy the requirement to use the new artificial images.

## 2026-05-08 02:46 CEST - Combined 800-Image Training Manifest

- Built
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_source.csv`.
- Combined sources:
  - old phase-1 plus scale-up training set: `245` rows
  - new `targeted_555_round1_medium` batch: `555` rows
- Verified combined source manifest:
  - total rows: `800`
  - `intact`: `200`
  - `minor`: `200`
  - `moderate`: `200`
  - `severe`: `200`
  - duplicate `image_path` values: `0`
  - missing source images: `0`

## 2026-05-08 02:47 CEST - Deterministic 224 Cache

- Built a deterministic `224 x 224` RGB cache from the 800 existing source
  images:
  - cache folder:
    `data/cache/synthetic_combined_phase1_plus_targeted_555_processed_224`
  - processed manifest:
    `data/datasets/synthetic_combined_phase1_plus_targeted_555_processed_224.csv`
- Verified processed manifest:
  - total rows: `800`
  - `intact`: `200`
  - `minor`: `200`
  - `moderate`: `200`
  - `severe`: `200`
  - cache files: `800`
  - preprocessing profile: `rgb_resize_224` for all rows
  - missing processed images: `0`
  - missing source images: `0`
  - duplicate processed paths: `0`
  - corrupt processed images: `0`
  - image size check: all `800` processed images are `224 x 224`

## 2026-05-08 02:48 CEST - Focused Training Tests

- Ran:
  `PYTHONPATH=src python3 -m pytest -q tests/test_training_artifacts.py tests/test_training_controls.py tests/test_training_monitoring.py tests/test_experiment_grid.py`
- Result: `15 passed in 12.75s`.
- Coverage relevant to this overnight run:
  - per-epoch checkpoint writing creates numbered checkpoints, `latest.pt`,
    and `best.pt`
  - run config records training controls
  - epoch metrics write JSONL and CSV rows
  - training curves and class-monitoring plots write PNGs
  - dropout, augmentation toggles, overfit subset selection, sampler weights,
    and experiment-grid config generation are covered

## 2026-05-08 02:49 CEST - Exact-Manifest Smoke Training

- Ran a two-epoch smoke training run on the new 800-row manifest with
  `--overfit-subset-size 16`.
- Command target:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_processed_224.csv`
- Output directory:
  `data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_targeted_555_processed_224_224_smoke_overfit16`
- Result:
  - command completed successfully
  - device: `cpu`
  - transforms did not resize again (`used_resize_in_transforms=False`)
  - epoch metrics rows written: `2`
  - required artifacts present:
    - `run_config.json`
    - `metrics.json`
    - `train_split.csv`
    - `val_split.csv`
    - `test_split.csv`
    - `best_model.pt`
    - `epoch_metrics.jsonl`
    - `epoch_metrics.csv`
    - `training_curves.png`
    - `class_monitoring.png`
    - `checkpoints/epoch_001.pt`
    - `checkpoints/epoch_002.pt`
    - `checkpoints/latest.pt`
    - `checkpoints/best.pt`

## 2026-05-08 02:50 CEST - Long Classifier Run Started

- Started the first long training run on the new `800`-row processed manifest.
- Manifest:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_processed_224.csv`
- Output directory:
  `data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_targeted_555_processed_224_224_lr0003_wd0001_do01_no_sampler_patience60_20260508`
- Model:
  - from-scratch `simple_cnn`
  - no pretrained weights
- Training controls:
  - image size: `224`
  - batch size: `16`
  - max epochs: `240`
  - optimizer: `AdamW`
  - learning rate: `0.0003`
  - weight decay: `0.0001`
  - dropout: `0.1`
  - augmentation: `mild`
  - balanced sampler: disabled because the new manifest is already exactly
    class-balanced at `200` examples per class
  - LR scheduler: `ReduceLROnPlateau`
  - scheduler patience: `20`
  - early-stopping patience: `60`
  - random seed: `42`

## 2026-05-08 03:10 CEST - Long Classifier Run Completed

- Output directory:
  `data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_targeted_555_processed_224_224_lr0003_wd0001_do01_no_sampler_patience60_20260508`
- Completion status:
  - early stopped: `true`
  - stopped epoch: `134`
  - best validation epoch: `74`
  - best validation macro F1: `0.4973`
  - final test accuracy from best checkpoint: `0.4625`
  - final test macro F1 from best checkpoint: `0.4615`
- Test confusion matrix:
  - `[18, 14, 7, 1]`
  - `[12, 13, 11, 4]`
  - `[8, 7, 16, 9]`
  - `[1, 0, 12, 27]`
- Artifact verification:
  - `134` epoch metric rows written
  - `run_config.json`, `metrics.json`, split CSVs, plots, `best_model.pt`,
    `checkpoints/latest.pt`, `checkpoints/best.pt`, and
    `checkpoints/epoch_134.pt` all exist
  - `checkpoints/best.pt` records epoch `74`
  - `checkpoints/latest.pt` records epoch `134`
  - scheduler state is saved in the latest checkpoint
- Interpretation:
  - the model learned the new balanced manifest but overfit after the best
    validation point
  - because validation did not keep improving after scheduler reductions, run a
    compact regularization comparison before calling this the best model

## 2026-05-08 03:11 CEST - Regularization Grid Started

- Started a compact grid on the same `800`-row processed manifest.
- Output root:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508`
- Fixed settings:
  - model: from-scratch `simple_cnn`
  - no pretrained weights
  - image size: `224`
  - batch size: `16`
  - max epochs: `180`
  - optimizer: `AdamW`
  - learning rate: `0.0003`
  - weight decay: `0.0001`
  - LR scheduler: `ReduceLROnPlateau`, patience `20`
  - early-stopping patience: `45`
  - random seed: `42`
- Compared settings:
  - dropout: `0.1`, `0.2`
  - balanced sampler: enabled, disabled
  - augmentation profile: `mild`, `strong`

## 2026-05-08 06:14 CEST - Regularization Grid Completed

- Grid summary:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/grid_summary.csv`
- Completed configurations: `8`
- Best test macro F1 configuration:
  `img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42`
- Best model directory:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42`
- Best checkpoint:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42/checkpoints/best.pt`
- Exported best model:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42/best_model.pt`

### Best Run Settings

- model: from-scratch `simple_cnn`
- pretrained weights: `false`
- manifest:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_processed_224.csv`
- image size: `224`
- optimizer: `AdamW`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.2`
- augmentation profile: `mild`
- balanced sampler: disabled
- random seed: `42`
- max epochs: `180`
- early-stopping patience: `45`

### Best Run Result

- early stopped: `true`
- stopped epoch: `169`
- best validation epoch: `124`
- final test accuracy from best checkpoint: `0.5125`
- final test macro F1 from best checkpoint: `0.5053`
- Test confusion matrix:
  - `[25, 12, 2, 1]`
  - `[16, 10, 12, 2]`
  - `[10, 7, 17, 6]`
  - `[0, 2, 8, 30]`

### Grid Ranking By Test Macro F1

1. `do0p2`, unbalanced sampler, mild augmentation: `0.5053`
2. `do0p2`, balanced sampler, mild augmentation: `0.5020`
3. `do0p1`, unbalanced sampler, strong augmentation: `0.4984`
4. `do0p2`, unbalanced sampler, strong augmentation: `0.4938`
5. `do0p1`, balanced sampler, mild augmentation: `0.4853`
6. `do0p1`, balanced sampler, strong augmentation: `0.4696`
7. `do0p1`, unbalanced sampler, mild augmentation: `0.4615`
8. `do0p2`, balanced sampler, strong augmentation: `0.3444`

### Verification

- Processed manifest rows: `800`
- Training class counts:
  - `intact`: `200`
  - `minor`: `200`
  - `moderate`: `200`
  - `severe`: `200`
- Source counts:
  - `phase1_round1_processed_128`: `45`
  - `scaleup_200_round1_processed_128`: `200`
  - `targeted_555_round1_medium_unreviewed`: `555`
- Missing processed image files: `0`
- Missing source image files: `0`
- Preprocessing profile: `rgb_resize_224` for all `800` rows
- Grid artifact audit:
  - `8` run directories checked
  - each run has `run_config.json`, `metrics.json`, split CSVs,
    `epoch_metrics.jsonl`, `epoch_metrics.csv`, `training_curves.png`,
    `class_monitoring.png`, `best_model.pt`, `checkpoints/latest.pt`,
    `checkpoints/best.pt`, and numbered epoch checkpoints
- Focused training tests:
  - `PYTHONPATH=src python3 -m pytest -q tests/test_training_artifacts.py tests/test_training_controls.py tests/test_training_monitoring.py tests/test_experiment_grid.py`
  - result: `15 passed in 11.37s`

### Interpretation

The best from-scratch CNN checkpoint is now trained on the combined `800`-image
synthetic manifest that includes all `555` new targeted images. The result is
better than the first long run, but it is not strong enough to claim the
classifier is precise in an absolute sense. The main remaining bottleneck is
likely label/image quality, especially because the new `555` examples still use
`intended_unreviewed` labels.

## 2026-05-08 13:27 CEST - Best Checkpoint Diagnostic Pass

- Added repeatable checkpoint diagnostic script:
  `scripts/analyze_classifier_checkpoint.py`
- Ran diagnostics against the current best checkpoint:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42/checkpoints/best.pt`
- Diagnostic output directory:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_regularization_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42/diagnostics`

### Diagnostic Outputs

- `diagnostic_report.md`
- `split_summary.csv`
- `test_confusion_matrix.csv`
- `test_per_class_tp_fp_tn_fn.csv`
- `test_prediction_examples.csv`
- `test_high_confidence_errors.csv`
- `test_high_confidence_errors_contact_sheet.jpg`
- `test_confidence_bins.csv`
- `test_source_metrics.csv`
- `test_score_band_metrics.csv`
- `test_feature_slice_metrics.csv`

### Key Findings

- Test accuracy is `0.5125`, which is above the `0.25` random baseline for four
  balanced classes, but still weak.
- Train accuracy is `0.9313` while test accuracy is `0.5125`, so the current
  model is overfitting strongly.
- Mean test confidence is `0.7731`, while test accuracy is `0.5125`, so the
  classifier is overconfident.
- Test prediction distribution:
  - target: `40` examples per class
  - predicted: `intact=51`, `minor=31`, `moderate=39`, `severe=39`
- `minor` is the weakest class:
  - precision: `0.3226`
  - recall: `0.2500`
  - F1: `0.2817`
- `severe` is the strongest class:
  - precision: `0.7692`
  - recall: `0.7500`
  - F1: `0.7595`
- Most errors are near misses:
  - exact correct: `82 / 160`
  - off by one severity class: `61 / 160`
  - off by two or more severity classes: `17 / 160`
- Lowest score-band performance is in the minor/mild-damage range:
  - `11-20`: `0.2727`
  - `21-30`: `0.2222`
  - `31-35`: `0.2727`
- Main improvement path:
  - QC the `minor` and low-`moderate` synthetic images first
  - inspect high-confidence errors visually
  - simplify the target to `damaged` vs `not damaged` or `low/medium/high`
    only if the four-class target remains unstable after QC
  - after labels are cleaner, improve architecture or regularization

## 2026-05-08 13:58 CEST - Targeted Minor-Band QC Round 1

- Built visual QC queue for all `126` new targeted images originally labeled as
  `minor`:
  - `11-20`
  - `21-30`
  - `31-35`
- Queue and review sheets:
  - `data/qc/targeted_555_round1_medium/minor_qc_queue.csv`
  - `data/qc/targeted_555_round1_medium/minor_qc_sheet_01.jpg`
  - `data/qc/targeted_555_round1_medium/minor_qc_sheet_02.jpg`
  - `data/qc/targeted_555_round1_medium/minor_qc_sheet_03.jpg`
  - `data/qc/targeted_555_round1_medium/minor_qc_sheet_04.jpg`
  - `data/qc/targeted_555_round1_medium/minor_qc_sheet_05.jpg`
  - `data/qc/targeted_555_round1_medium/minor_qc_sheet_06.jpg`
  - `data/qc/targeted_555_round1_medium/minor_qc_sheet_07.jpg`
- QC decisions:
  - `data/qc/targeted_555_round1_medium/minor_qc_decisions_round1.csv`

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

The most important correction is that the old `minor` pool contained both
near-intact images and clearly low-moderate images. This supports the earlier
diagnostic finding that the model was struggling around the `minor` boundary
because the labels were not visually consistent.

### New QC Manifests

- Targeted batch manifest with round-1 QC applied:
  `data/datasets/synthetic_targeted_555_round1_medium_qc_round1.csv`
- Combined source manifest with round-1 QC applied:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round1_source.csv`
- Combined processed training manifest with round-1 QC applied:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round1_processed_224.csv`

Combined processed QC manifest:

- rows: `800`
- missing image files: `0`
- reviewed by this QC round: `126`
- class counts after QC:
  - `intact`: `217`
  - `minor`: `171`
  - `moderate`: `212`
  - `severe`: `200`

### QC Manifest Smoke Training

- Smoke run:
  `data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_targeted_555_qc_round1_processed_224_smoke`
- Purpose:
  - verify the QC manifest loads
  - verify labels are valid
  - verify checkpoint writing still works
- Result:
  - completed `1` epoch
  - checkpoints written

### Focused QC Retrain

- Run:
  `data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_targeted_555_qc_round1_processed_224_do02_balanced_mild_seed42_20260508`
- Settings:
  - from-scratch `simple_cnn`
  - no pretrained weights
  - dropout: `0.2`
  - augmentation: `mild`
  - balanced sampler: enabled because QC changed class counts
  - max epochs: `180`
  - early-stopping patience: `45`
- Result:
  - early stopped at epoch `77`
  - best validation epoch: `32`
  - best validation macro F1: `0.4791`
  - test accuracy: `0.4062`
  - test macro F1: `0.4128`
- Test confusion matrix:
  - `[15, 16, 10, 3]`
  - `[18, 9, 6, 1]`
  - `[11, 10, 18, 3]`
  - `[1, 5, 11, 23]`

### Interpretation

The partial QC pass improved label honesty but did not improve the classifier
metric after retraining. This is useful evidence, not a reason to discard the
QC. The old best model was partly benefiting from internally consistent but
visually wrong intended labels. After correcting only the minor bands, the
dataset became more honest but also more imbalanced and still partly noisy in
the untouched intact/moderate/severe bands.

Do not replace the previous best checkpoint with this QC-round checkpoint. The
right next step is to finish QC for the untouched bands, then retrain on a fully
QC-reviewed manifest.

## 2026-05-08 14:51 CEST - Targeted Batch Full QC Round 2

- Continued QC on the remaining new targeted labels:
  - `intact`: `175` images
  - `moderate`: `127` images
  - `severe`: `127` images
- Total newly generated targeted images now reviewed:
  - `555 / 555`
- Queue, sheet, and decision files are stored under:
  `data/qc/targeted_555_round1_medium/`

### Round-2 QC Decisions

Intact:

- reviewed: `175`
- accepted as labeled: `174`
- relabeled: `1`
- transition:
  - `0-10 -> 11-20`: `1`

Moderate:

- reviewed: `127`
- accepted as labeled: `92`
- relabeled: `35`
- important transitions:
  - `36-45 -> 21-30`: `4`
  - `36-45 -> 31-35`: `7`
  - `36-45 -> 46-55`: `2`
  - `46-55 -> 36-45`: `3`
  - `46-55 -> 56-65`: `7`
  - `46-55 -> 66-75`: `1`
  - `56-65 -> 66-75`: `7`
  - `56-65 -> 46-55`: `2`
  - `56-65 -> 31-35`: `1`
  - `56-65 -> 36-45`: `1`

Severe:

- reviewed: `127`
- accepted as labeled: `83`
- relabeled: `44`
- important transitions:
  - `66-75 -> 76-85`: `13`
  - `66-75 -> 86-100`: `2`
  - `66-75 -> 56-65`: `2`
  - `76-85 -> 66-75`: `9`
  - `76-85 -> 86-100`: `1`
  - `76-85 -> 56-65`: `1`
  - `86-100 -> 76-85`: `15`
  - `86-100 -> 66-75`: `1`

Across all `555` targeted images:

- accepted as labeled: `428`
- accepted with relabel: `127`
- rejected: `0`

### Full-QC Manifests

- Targeted batch:
  `data/datasets/synthetic_targeted_555_round1_medium_qc_round2.csv`
- Combined source:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round2_source.csv`
- Combined processed training manifest:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round2_processed_224.csv`

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

Combined processed manifest verification:

- rows: `800`
- missing image files: `0`
- newly targeted examples reviewed: `555`
- earlier non-targeted examples not part of this QC request: `245`

### Full-QC Retrain Grid

- Grid root:
  `data/training_runs/synthetic_classifier_grid/grid_targeted_555_full_qc_round2_20260508`
- Manifest:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_qc_round2_processed_224.csv`
- Fixed settings:
  - model: from-scratch `simple_cnn`
  - pretrained weights: `false`
  - image size: `224`
  - dropout: `0.2`
  - augmentation: `mild`
  - optimizer: `AdamW`
  - learning rate: `0.0003`
  - weight decay: `0.0001`
  - max epochs: `180`
  - early-stopping patience: `45`
- Compared settings:
  - balanced sampler enabled
  - balanced sampler disabled, with class-weighted loss

Results:

- Balanced sampler:
  - best validation epoch: `102`
  - early stopped at epoch: `147`
  - test accuracy: `0.4625`
  - test macro F1: `0.4691`
- No sampler / class-weighted loss:
  - best validation epoch: `30`
  - early stopped at epoch: `75`
  - test accuracy: `0.4375`
  - test macro F1: `0.4349`

Artifact verification:

- both grid runs have `run_config.json`, `metrics.json`, split CSVs,
  `epoch_metrics.jsonl`, `epoch_metrics.csv`, plots, `best_model.pt`,
  `checkpoints/latest.pt`, `checkpoints/best.pt`, and numbered epoch checkpoints
- balanced run epoch checkpoints: `147`
- no-sampler run epoch checkpoints: `75`
- diagnostics generated for both runs

Focused tests:

- `PYTHONPATH=src python3 -m pytest -q tests/test_training_artifacts.py tests/test_training_controls.py tests/test_training_monitoring.py tests/test_experiment_grid.py`
- result: `15 passed in 12.48s`

### Interpretation

The full-QC dataset is more honest than the original intended-label dataset, but
the best full-QC retrain did not beat the old unreviewed-label headline test
macro F1. This should not be interpreted as QC hurting the project. It means the
old metric was partly inflated by training and testing against noisy intended
labels. The cleaner label set exposes that the current scratch CNN still
struggles with fine four-class severity boundaries, especially around `minor`
and `moderate`.

For strict model selection on the full-QC grid, use the validation winner:

- `data/training_runs/synthetic_classifier_grid/grid_targeted_555_full_qc_round2_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_unbalanced_mild_seed42/checkpoints/best.pt`

For the highest observed full-QC test macro F1 in this small grid, the balanced
sampler run scored higher:

- `data/training_runs/synthetic_classifier_grid/grid_targeted_555_full_qc_round2_20260508/img224_lr0p0003_wd0p0001_do0p2_adamw_sched_balanced_mild_seed42/checkpoints/best.pt`

Do not silently compare the old and new metrics as if the target labels were the
same dataset. The target labels changed.

## 2026-05-08 15:48 CEST - Full Synthetic QC And Training Pipeline Upgrade

### Remaining 234-Image QC

- Built remaining-unreviewed QC queue and sheets:
  - `data/qc/remaining_unreviewed_234_round1/remaining_234_qc_queue.csv`
  - `data/qc/remaining_unreviewed_234_round1/remaining_234_qc_sheet_01.jpg`
    through `remaining_234_qc_sheet_12.jpg`
- Wrote decisions:
  - `data/qc/remaining_unreviewed_234_round1/remaining_234_qc_decisions.csv`
- Decision counts:
  - accepted as labeled: `185`
  - accepted with relabel: `49`
  - rejected: `0`

Applied these decisions to create:

- `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv`

Full synthetic source manifest verification:

- rows: `800`
- accepted as labeled: `623`
- accepted with relabel: `177`
- missing image files: `0`
- class counts:
  - `intact`: `216`
  - `minor`: `192`
  - `moderate`: `189`
  - `severe`: `203`

### 224 / 384 Cache Builds

Built full-QC deterministic caches:

- `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_processed_224.csv`
- `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_processed_384.csv`
- `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_padded_224.csv`
- `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_padded_384.csv`

Verification:

- each manifest has `800` rows
- processed image sizes match the manifest profile
- source images are all `1024 x 1024`, so padded and direct square resize are
  equivalent for this synthetic batch

### Augmentation And Model Changes

Added `damage_safe` augmentation:

- horizontal flip
- small affine rotation / translation / scale / shear
- mild perspective
- brightness / contrast / saturation / hue jitter
- slight Gaussian blur
- low-probability Gaussian noise
- cautious small random erasing
- no vertical flip

Preview sheet:

- `data/qc/augmentation_previews/damage_safe_384_preview.jpg`

Added stronger from-scratch model:

- model variant: `residual_cnn`
- trainable parameters: `2,779,044`
- no pretrained weights
- uses batch normalization and residual blocks

Added optional ordinal-aware loss:

- command argument: `--ordinal-loss-weight`
- implementation: cross-entropy plus smooth penalty on expected class index

### Controlled Experiment Matrix

Output root:

- `data/training_runs/synthetic_classifier_grid/grid_full_qc_resolution_capacity_aug_20260508`

Summary:

- `controlled_summary.csv`

| run | setup | test acc | test macro F1 | mean ordinal error | off-by-one/correct | severe ordinal errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | `224`, `simple_cnn`, `mild` | `0.4313` | `0.4020` | `0.8375` | `0.7813` | `0.2188` |
| B | `384`, `simple_cnn`, `mild` | `0.4813` | `0.4800` | `0.7688` | `0.8063` | `0.1938` |
| C | `384`, `residual_cnn`, `mild` | `0.4625` | `0.4640` | `0.6688` | `0.8813` | `0.1188` |
| D | `384`, `residual_cnn`, `damage_safe` | `0.5313` | `0.5096` | `0.6250` | `0.8688` | `0.1313` |
| E | `384`, `residual_cnn`, `damage_safe`, ordinal `0.2` | `0.5313` | `0.5153` | `0.6000` | `0.8875` | `0.1125` |

Interpretation:

- Increasing resolution from `224` to `384` helped materially for the same
  simple CNN.
- Increasing model capacity without stronger augmentation improved validation
  but overfit; test macro F1 did not improve over the simple `384` run.
- The stronger residual model plus `damage_safe` augmentation produced the best
  non-ordinal controlled result.
- Adding ordinal-aware loss gave the best test macro F1 and best ordinal
  diagnostics in this controlled pass.

### Fine-Band Target Consideration

Question considered after the controlled pass: should the next classifier train
directly on the official `score_band` target instead of only the four derived
coarse classes?

Answer: this is a reasonable next experiment, but it is not automatically
better. The score bands contain more information than the four coarse classes,
and collapsing band predictions back to coarse classes after training could
improve the final coarse result if the model learns the ordinal structure.
However, a plain ten-class cross-entropy classifier is risky because it gives
fewer examples per class and treats neighboring mistakes, such as `36-45` vs
`46-55`, as equally wrong as distant mistakes, such as `0-10` vs `86-100`.

Recommended design for this experiment:

- keep the current four-class classifier as the baseline;
- train a band-aware model that predicts the ten official `score_band` labels;
- use an ordinal-aware objective or auxiliary representative-score regression
  so near-band mistakes are penalized less than far-band mistakes;
- report both native band metrics and coarse metrics after collapsing predicted
  bands through the official mapping;
- compare against the current best controlled checkpoint, not only against the
  early `224` baseline.

This should be logged as a new experiment rather than a replacement of the core
coarse-class comparison, because the project still needs the coarse severity
result for the main synthetic-to-real study.

### Ten-Band Target Follow-Up

Ran the same setup as the best controlled four-class result, changing only the
classification target from four coarse classes to the ten official
`score_band` labels.

Run directory:

- `data/training_runs/synthetic_classifier/score_band_384_residual_damage_safe_ordinal02_20260508`

Settings:

- manifest:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_processed_384.csv`
- image size: `384`
- model: `residual_cnn`
- augmentation: `damage_safe`
- optimizer: `AdamW`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.2`
- balanced sampler: disabled
- ordinal loss weight: `0.2`
- target label mode: `score_band`
- max epochs: `60`
- best validation epoch: `51`

Test result:

- exact ten-band accuracy: `0.2875`
- ten-band macro F1: `0.2391`
- within-one-band-or-correct accuracy: `0.5688`
- mean band-index error: `1.7125`
- severe band-error rate, meaning two or more bands away: `0.4313`
- collapsed coarse accuracy after mapping predicted bands back to
  `intact/minor/moderate/severe`: `0.5000`
- collapsed coarse macro F1: `0.4936`

Per-band exact accuracy:

| band | support | correct | accuracy |
| --- | ---: | ---: | ---: |
| `0-10` | `43` | `20` | `0.4651` |
| `11-20` | `15` | `2` | `0.1333` |
| `21-30` | `11` | `0` | `0.0000` |
| `31-35` | `13` | `2` | `0.1538` |
| `36-45` | `14` | `0` | `0.0000` |
| `46-55` | `12` | `2` | `0.1667` |
| `56-65` | `12` | `5` | `0.4167` |
| `66-75` | `13` | `4` | `0.3077` |
| `76-85` | `17` | `5` | `0.2941` |
| `86-100` | `10` | `6` | `0.6000` |

Interpretation:

- The ten-band model learned useful ordinal structure: more than half of test
  predictions were exactly correct or only one neighboring band away.
- It did not beat the direct four-class model after collapsing predictions back
  to coarse classes. The direct four-class best remains `0.5313` accuracy and
  `0.5153` macro F1; the collapsed ten-band model reached `0.5000` accuracy and
  `0.4936` macro F1.
- The weakest area is still boundary/interior damaged bands, especially
  `21-30` and `36-45`, where exact accuracy was `0.0000` in this split.
- The ten-band target is useful as an auxiliary or ordinal diagnostic, but this
  single run does not justify replacing the main four-class training objective.

### Multitask Coarse + Score-Band Follow-Up

Ran a multitask version of the best controlled setup. The model keeps the
coarse four-class head as the primary prediction target and adds an auxiliary
ten-band score-band head.

Run directory:

- `data/training_runs/synthetic_classifier/F_384_residual_multitask_damage_safe_auxband03_ord02_20260508`

Settings:

- manifest:
  `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_processed_384.csv`
- image size: `384`
- shared model: from-scratch `residual_cnn_multitask`
- primary head: four coarse classes
- auxiliary head: ten official `score_band` labels
- augmentation: `damage_safe`
- optimizer: `AdamW`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.2`
- balanced sampler: disabled
- auxiliary band loss weight: `0.3`
- auxiliary band ordinal loss weight: `0.2`
- max epochs: `60`
- early stopped epoch: `26`
- best validation epoch: `11`

Test result from the best validation checkpoint:

- primary coarse accuracy: `0.5000`
- primary coarse macro F1: `0.5042`
- auxiliary score-band exact accuracy: `0.3500`
- auxiliary score-band macro F1: `0.2909`
- auxiliary within-one-band-or-correct accuracy: `0.5625`
- auxiliary mean band-index error: `1.7813`
- auxiliary severe band-error rate: `0.4375`
- auxiliary predictions collapsed back to coarse classes:
  - accuracy: `0.4875`
  - macro F1: `0.4783`

Comparison:

- best direct four-class run E:
  - coarse accuracy: `0.5313`
  - coarse macro F1: `0.5153`
- standalone ten-band run:
  - exact band accuracy: `0.2875`
  - band macro F1: `0.2391`
  - within-one-band-or-correct accuracy: `0.5688`
- multitask run F:
  - primary coarse accuracy: `0.5000`
  - primary coarse macro F1: `0.5042`
  - auxiliary exact band accuracy: `0.3500`
  - auxiliary band macro F1: `0.2909`

Interpretation:

- The multitask objective improved band learning compared with the standalone
  ten-band classifier.
- It did not improve the headline coarse result over the current best direct
  four-class model.
- The current best checkpoint therefore remains run E, not run F.
- The multitask path remains useful as a diagnostic or future tuning direction,
  but it is not yet the main model for the report.

### Synthetic-To-Real Evaluation Of Best Synthetic Checkpoint

After the real manifest became usable at
`data/datasets/real_labeled_dataset_current.csv`, evaluated the best synthetic
checkpoint directly on all labeled real rows.

Evaluation target:

- checkpoint:
  `data/training_runs/synthetic_classifier_grid/grid_full_qc_resolution_capacity_aug_20260508/E_384_residual_damage_safe_ordinal02/checkpoints/best.pt`
- real manifest:
  `data/datasets/real_labeled_dataset_current.csv`
- evaluation mode: all `164` real rows treated as a real-domain test set
- output directory:
  `data/evaluations/real_transfer/E_384_residual_damage_safe_ordinal02_on_real_current`

Real manifest distribution:

- total rows: `164`
- `intact`: `8`
- `minor`: `45`
- `moderate`: `81`
- `severe`: `30`
- missing images: `0`

Result:

- real-domain accuracy: `0.2927`
- real-domain macro F1: `0.2574`
- mean confidence: `0.6125`
- mean true-class probability: `0.2720`
- mean ordinal error: `1.1098`
- off-by-one-or-correct rate: `0.7073`
- severe ordinal-error rate: `0.2927`

Real confusion matrix:

| true \ predicted | intact | minor | moderate | severe |
| --- | ---: | ---: | ---: | ---: |
| intact | `6` | `2` | `0` | `0` |
| minor | `24` | `9` | `9` | `3` |
| moderate | `25` | `23` | `29` | `4` |
| severe | `18` | `2` | `6` | `4` |

Prediction distribution:

- predicted `intact`: `73`
- predicted `minor`: `36`
- predicted `moderate`: `44`
- predicted `severe`: `11`

Baseline context:

- always predicting the real majority class, `moderate`, would get `0.4939`
  accuracy but only `0.1653` macro F1.
- The synthetic checkpoint has worse real accuracy than this majority baseline
  but better macro F1, meaning it is not simply following the real class prior.

Interpretation:

- This is clear synthetic-to-real domain shift.
- The biggest failure is overpredicting `intact` on damaged real images.
- Severe real damage recall is poor: only `4 / 30` severe examples were
  predicted as `severe`.
- High confidence is not reliable on real data: the diagnostic report found
  `44` high-confidence wrong real predictions at confidence `>= 0.70`.
- The best current synthetic checkpoint remains useful for the synthetic-only
  result, but it is not a deployable real-domain model.

### Real-Domain Root-Cause And Baseline Pass

After the poor synthetic-to-real result, ran a real-domain debugging pass without
generating any new images.

Preprocessing finding:

- real manifest rows: `164`
- real images with non-square dimensions: `163`
- real aspect-ratio range: about `0.3866` to `2.8430`
- built padded real cache:
  `data/datasets/real_labeled_dataset_current_padded_384.csv`
- padded cache profile: `rgb_pad_resize_384`
- padded rows: `164`
- missing padded images: `0`

Re-evaluated best synthetic checkpoint E on padded real images:

- raw real all-as-test accuracy / macro F1: `0.2927` / `0.2574`
- padded real all-as-test accuracy / macro F1: `0.3110` / `0.2445`

Interpretation: aspect-ratio-safe preprocessing helps accuracy slightly but does
not solve transfer. The dominant issue is still domain shift.

Real overfit sanity check:

- run:
  `data/training_runs/real_classifier/real_padded384_residual_overfit32_sanity_20260508`
- setup: balanced 32-image real subset, `8` examples per class, same subset for
  train/validation/test
- result: `1.0000` test accuracy and `1.0000` macro F1

Interpretation: the real data path and scratch residual model can learn real
labels when the task is deliberately easy. The failure is not a broken loader or
an impossible label encoding.

Controlled real split policy:

- `20%` real train
- `20%` real validation
- `60%` held-out real test
- random seed: `42`

Real-domain comparison summary:

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

- `data/evaluations/real_transfer/real_domain_experiment_summary_20260508.csv`

Per-class diagnostic artifacts:

- scratch real-only diagnostics:
  `data/evaluations/real_models/real_only_padded384_residual_damage_safe_ord02_seed42_20260508_diagnostics`
- pretrained diagnostic:
  `data/evaluations/real_models/diagnostic_pretrained_resnet18_real_padded384_damage_safe_ord02_seed42_20260508_diagnostics`

Interpretation:

- The best core scratch result on the real split is now the real-only residual
  CNN with balanced sampling: `0.3887` macro F1.
- Fine-tuning from the synthetic E checkpoint was worse than real-only scratch
  training under this split and setting, so the current synthetic pretraining
  does not help real-domain performance.
- The pretrained ResNet18 diagnostic is the strongest real result so far:
  `0.4509` macro F1. This should be treated as an upper-bound diagnostic, not
  as a replacement for the core from-scratch experiment unless the project plan
  is explicitly changed.
- The high-confidence real error contact sheet shows many damaged real boxes
  confidently predicted as `intact`, especially flat/taped real packages whose
  defects do not look like the synthetic damage cues.
- The pretrained diagnostic improved macro F1 mainly by recovering much better
  `minor` recall (`17 / 27`) than the scratch real-only model (`3 / 27`). The
  scratch real-only model mostly collapsed real test predictions into
  `moderate`.

### Verification

- Diagnostics were generated for all five controlled runs.
- Each diagnostics folder includes:
  - `test_confusion_matrix.csv`
  - `test_per_class_tp_fp_tn_fn.csv`
  - `test_high_confidence_errors.csv`
  - `test_high_confidence_errors_contact_sheet.jpg`
  - `split_summary.csv`
- Focused tests:
  - `PYTHONPATH=src python3 -m pytest -q tests/test_training_artifacts.py tests/test_training_controls.py tests/test_training_monitoring.py tests/test_experiment_grid.py`
  - result: `18 passed in 11.86s`

### V2 Synthetic Generation Pipeline

Created a versioned v2 synthetic image-generation preparation pipeline after
the real-domain analysis showed that the current synthetic images do not
transfer well.

Versioned layout:

- v1 compatibility namespace:
  `src/dream2detect/pipelines/v1/`
- v2 implementation:
  `src/dream2detect/pipelines/v2/`
- v2 CLI:
  `scripts/v2_prepare_image_pipeline.py`
- detailed doc:
  `docs/17-v2-synthetic-generation-pipeline.md`

V2 changes the goal from bulk synthetic generation to domain-gap repair. It
targets:

- hard intact negatives with tape, labels, and clutter
- flat taped mailers with subtle damage
- minor/moderate boundary edge and seam cases
- moderate real-world crush and split cases with tape
- severe but realistic damage
- capture artifacts such as shadow, compression, imperfect framing, and mild
  blur

Prepared preview artifact without any API submission:

- `data/pipelines/v2/v2_real_failure_targeted_preview_20260509_r2/`

Preview settings:

- `count_per_band=3`
- total prompts: `30`
- image model field: `gpt-image-2`
- quality field: `medium`
- size field: `1024x1024`

Preview outputs:

- `prompt_manifest.csv`
- `input.jsonl`
- `manifest.json`
- `run_config.json`
- `summary.json`
- `qc_rubric.md`

Verification:

- `PYTHONPATH=src python3 -m pytest -q tests/test_v2_pipeline.py tests/test_image_batch.py`
- result: `7 passed in 0.27s`
