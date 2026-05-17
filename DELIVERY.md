# Dream2Detect Delivery Branch

This branch combines the two project tracks into one cleaned delivery tree.

## What Is Included

- `EvaluartionFiles/`, `FinalModels/`, and `TrainingFiles/` restore the
  pretrained/timm pipeline from `main`.
- `pipelines/` documents the no-pretrain experiment ladder from V1 through
  V5-B, plus the V0 main/pretrained track.
- `dataset/synthetic/` contains the compact synthetic delivery dataset.
- `dataset/real/` contains the compact real delivery dataset.
- `src/`, `scripts/`, and `tests/` preserve the newer experimental pipeline
  from `exp`.
- `PRESENTATION.md` gives the short V1-to-V5B story without requiring the long
  internal project docs.

## Source Branches

- Base branch for this delivery branch: `exp`
- Restored folders from `main`:
  - `EvaluartionFiles/`
  - `FinalModels/`
  - `TrainingFiles/`

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
PYTHONPATH=.:src pytest
python3 scripts/validate_delivery_datasets.py
```
