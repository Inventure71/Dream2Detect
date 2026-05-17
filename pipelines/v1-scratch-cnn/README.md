# V1 - First Synthetic-Only Scratch CNN

## Question

Can synthetic package/cardboard severity labels be learned at all by a small
from-scratch CNN?

## Dataset

Early runs used the first reviewed synthetic sets, then the corrected combined
synthetic manifest. The stronger V1-style run used an 800-image combined
synthetic dataset.

Relevant manifests:

- `data/datasets/synthetic_combined_phase1_plus_targeted_555_processed_224.csv`
- `dataset/synthetic/manifest.csv` for the compact delivery dataset

## Implementation

- `scripts/train_synthetic_classifier.py`
- `src/dream2detect/training/dataset.py`
- `src/dream2detect/training/models.py`
- `src/dream2detect/training/train_classifier.py`

## Result

V1 proved the labels were learnable but fragile. It was superseded by the
stronger V2 residual and full-QC setup.

