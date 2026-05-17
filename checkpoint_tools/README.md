# Checkpoint Testing

Use this folder when the teacher wants to run the trained models instead of
retraining every pipeline.

## Expected Checkpoint Layout

Put trained checkpoint files under a root-level `checkpoints/` folder:

```text
checkpoints/
  v4_score_band/best.pt
  v5_a_scalar_ordinal/best.pt
  v5_b_multitask/best.pt
```

The script recursively finds `.pt` files, so nested folders are fine.

## Test All Checkpoints On The Real Dataset

```bash
PYTHONPATH=.:src python3 checkpoint_tools/test_checkpoints.py \
  --checkpoints-dir checkpoints \
  --manifest dataset/real/manifest.csv \
  --output-dir checkpoint_results
```

## Test One Checkpoint

```bash
PYTHONPATH=.:src python3 checkpoint_tools/test_checkpoints.py \
  --checkpoint checkpoints/v4_score_band/best.pt \
  --manifest dataset/real/manifest.csv \
  --output-dir checkpoint_results/v4_score_band
```

## Supported Checkpoints

This utility supports V1-V5B checkpoints produced by the delivery branch
training scripts. It expects either:

- a training checkpoint payload containing `model_state_dict` and `config`, or
- a raw model `state_dict` plus `--config-json path/to/run_config.json`.

V0 pretrained/timm checkpoints should still be evaluated with the V0 scripts in
`pipelines/v0-main-pretrained/EvaluationFiles/`.
