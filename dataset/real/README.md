# Real Delivery Dataset

Current labeled real-image delivery dataset, padded to 384 px.

- Source manifest: `data/datasets/real_labeled_dataset_current_padded_384_full.csv`
- Rows: `385`
- Image paths in `manifest.csv` are relative to this dataset folder.
- Use this folder as the image root for the restored main-branch scripts.

Example:

```bash
python3 scripts/validate_delivery_datasets.py --dataset real
```
