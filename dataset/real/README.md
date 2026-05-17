# Real Delivery Dataset

Current labeled real-image delivery dataset, padded to 384 px.

- Rows: `385`
- Images: `images/*.jpg`
- Image paths in `manifest.csv` are relative to this dataset folder.
- Obsolete local source paths have been removed from the delivery manifest.
- Use this folder as the image root for the restored main-branch scripts.

Example:

```bash
python3 scripts/validate_delivery_datasets.py --dataset real
```
