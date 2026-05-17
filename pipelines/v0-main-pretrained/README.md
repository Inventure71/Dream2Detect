# V0 - Main Branch Pretrained Pipeline

## Question

How strong is a practical pretrained/timm pipeline using EfficientNet or
ConvNeXt backbones, direct 10-band classification, EMD loss, SAM cropping, and
ensembles?

## Files

- `TrainingFiles/train_band.py`
- `TrainingFiles/train_convnext_base.py`
- `TrainingFiles/train_emd.py`
- `TrainingFiles/preprocess_sam.py`
- `EvaluartionFiles/evaluate3.py`
- `EvaluartionFiles/evaluate_ensemble.py`
- `EvaluartionFiles/evaluate_triple.py`
- `EvaluartionFiles/predict.py`
- `FinalModels/README.md`

## Dataset

Uses the final delivery datasets under `dataset/`, or the original folder names
expected by the scripts after copying/symlinking them into the command paths.

The restored scripts expect manifest/image arguments such as:

```bash
python TrainingFiles/train_emd.py \
  --manifest dataset/synthetic/manifest.csv \
  --images_dir dataset/synthetic \
  --output_dir outputs/emd_01
```

## Result Status

This path is kept as the strong pretrained comparison track. It is intentionally
separate from the V1-V5B no-pretrain class-experiment ladder.

See `FinalModels/README.md` for checkpoint download notes and reported final
model table.

