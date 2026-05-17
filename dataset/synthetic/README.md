# Synthetic Delivery Dataset

Full-QC synthetic delivery dataset with V2 scale-up examples.

- Source manifest: `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- Rows: `899`
- Image paths in `manifest.csv` are relative to this dataset folder.
- Use this folder as the image root for the restored main-branch scripts.

Example:

```bash
python3 scripts/validate_delivery_datasets.py --dataset synthetic
```
