# Dream2Detect Delivery Branch

This branch combines the two project tracks into one cleaned delivery tree.

## What Is Included

- `pipelines/` contains one folder per pipeline, including the restored
  pretrained/timm pipeline from `main` under `pipelines/v0-main-pretrained/`.
- `dataset/synthetic/` contains the compact synthetic delivery dataset.
- `dataset/real/` contains the compact real delivery dataset.
- `dataset_generation/` contains the preserved prompt, image-generation, QC,
  registry, and manifest-export workflow used to create the datasets.
- `checkpoint_tools/` contains the teacher-facing script for testing trained
  checkpoints placed under a root-level `checkpoints/` folder.
- `scripts/` and `src/dream2detect/training/` contain only the shared runnable
  training/evaluation support for V1-V5B.

## Source Branches

- Base branch for this delivery branch: `exp`
- Restored code from `main` is scoped under `pipelines/v0-main-pretrained/`.

The branch was assembled in an isolated worktree so `main` and `exp` remain
unchanged.

## Dataset Contract

The delivery datasets use relative image paths so the folders can be moved as a
self-contained package.

Expected layout:

```text
dataset/
  synthetic/
    manifest.csv
    images/
    README.md
  real/
    manifest.csv
    images/
    README.md
```

## Pipeline Index

| Folder | Purpose |
|---|---|
| `pipelines/v0-main-pretrained` | Main branch pretrained/timm/SAM/ensemble pipeline |
| `pipelines/v1-scratch-cnn` | First synthetic-only scratch CNN baseline |
| `pipelines/v2-full-qc-residual` | Full-QC residual CNN and V2 targeted synthetic scale-up |
| `pipelines/v3-training-stack` | Training-stack search around the V2 baseline |
| `pipelines/v3_1-hard-split` | Metadata-family holdout split discipline |
| `pipelines/v4-score-band` | Official 10-band score-band target baseline |
| `pipelines/v4_1-challenger-sweep` | Narrow score-band challenger sweep |
| `pipelines/v4_2-sigma-sweep` | Score-band soft-label sigma sweep |
| `pipelines/v4_5-emd` | Soft-label cross-entropy plus EMD objective |
| `pipelines/v5_a-scalar-ordinal` | Scalar ordinal regression model |
| `pipelines/v5_b-multitask` | Multitask scalar, band, and coarse model |

## Basic Validation

Run:

```bash
python3 -m py_compile scripts/*.py src/dream2detect/**/*.py
python3 scripts/validate_delivery_datasets.py
```
