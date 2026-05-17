# Synthetic Delivery Dataset

Full-QC synthetic delivery dataset with V2 scale-up examples.

- Rows: `899`
- Images: `images/*.png`
- Image paths in `manifest.csv` are relative to this dataset folder.
- Obsolete local source paths have been removed from the delivery manifest.
- Use this folder as the image root for the restored main-branch scripts.

Example:

```bash
python3 scripts/validate_delivery_datasets.py --dataset synthetic
```
