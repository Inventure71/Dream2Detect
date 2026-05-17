# V0 - Main Branch Pretrained Pipeline

## Purpose

Keep the stronger practical comparison pipeline from `main`: timm backbones,
direct 10-band classification, EMD loss, optional SAM preprocessing, and
ensembles.

## Dataset

Use the delivery datasets:

- train on `dataset/synthetic/manifest.csv`
- evaluate on `dataset/real/manifest.csv`

The V0 scripts take `--images_dir`; pass the dataset folder itself because the
manifest paths are relative, for example `images/0001.jpg`.

## Train

```bash
python3 pipelines/v0-main-pretrained/TrainingFiles/train_emd.py \
  --manifest dataset/synthetic/manifest.csv \
  --images_dir dataset/synthetic \
  --backbone convnext_small \
  --epochs 60 \
  --batch_size 16 \
  --output_dir outputs/v0_emd
```

Other V0 trainers:

- `pipelines/v0-main-pretrained/TrainingFiles/train_band.py`
- `pipelines/v0-main-pretrained/TrainingFiles/train_convnext_base.py`
- `pipelines/v0-main-pretrained/TrainingFiles/preprocess_sam.py`

## Evaluate

```bash
python3 pipelines/v0-main-pretrained/EvaluationFiles/evaluate3.py \
  --checkpoint outputs/v0_emd/best_model.pt \
  --manifest dataset/real/manifest.csv \
  --images_dir dataset/real \
  --output_dir outputs/v0_emd_real_eval \
  --model_name "V0 EMD"
```

Ensemble evaluators:

- `pipelines/v0-main-pretrained/EvaluationFiles/evaluate_ensemble.py`
- `pipelines/v0-main-pretrained/EvaluationFiles/evaluate_triple.py`
- `pipelines/v0-main-pretrained/EvaluationFiles/predict.py`

## Code

- `TrainingFiles/`
- `EvaluationFiles/`
- `FinalModels/README.md`

## Result Summary

This is the high-performing pretrained comparison track. It is separate from
the V1-V5B no-pretrain experiment ladder.
