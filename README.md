# Dream2Detect

Dream2Detect studies synthetic-to-real generalization for visible package/cardboard defect severity.

Presentation: https://canva.link/4wm0l3yeq6bvidc

## Teacher Quick View

- Start here: [DELIVERY.md](DELIVERY.md)
- Pipeline summaries: [pipelines/README.md](pipelines/README.md)
- Main pretrained model notes: [pipelines/v0-main-pretrained/FinalModels/README.md](pipelines/v0-main-pretrained/FinalModels/README.md)

## Layout

```text
dataset/               compact delivery datasets, ready to run
pipelines/             one folder per pipeline, each documented
checkpoint_tools/      teacher-facing scripts for testing saved checkpoints
scripts/               shared runnable entrypoints for V1-V5B
src/dream2detect/      minimal shared training package for V1-V5B
```

## Delivery Datasets

The delivery branch includes two compact dataset folders:

- `dataset/synthetic/manifest.csv`
- `dataset/real/manifest.csv`

Each manifest uses `image_path` values relative to its own dataset folder, for
example `images/example.png`.

Validate them with:

```bash
python3 scripts/validate_delivery_datasets.py
```

## Common Commands

```bash
python3 scripts/train_synthetic_classifier.py \
  --manifest dataset/synthetic/manifest.csv \
  --image-size 384 \
  --output-dir outputs/v4_score_band

python3 scripts/train_synthetic_regressor.py \
  --manifest dataset/synthetic/manifest.csv \
  --output-dir outputs/v5_a_scalar

python3 scripts/train_multitask_classifier.py \
  --manifest dataset/synthetic/manifest.csv \
  --output-dir outputs/v5_b_multitask
```

## Test Saved Checkpoints

Place trained `.pt` files under `checkpoints/`, then run:

```bash
PYTHONPATH=.:src python3 checkpoint_tools/test_checkpoints.py \
  --checkpoints-dir checkpoints \
  --manifest dataset/real/manifest.csv \
  --output-dir checkpoint_results
```
