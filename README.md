# Dream2Detect

Dream2Detect studies synthetic-to-real generalization for visible package/cardboard defect severity.

## Teacher Quick View

- Start here: [DELIVERY.md](DELIVERY.md)
- Pipeline summaries: [pipelines/README.md](pipelines/README.md)
- Presentation narrative: [PRESENTATION.md](PRESENTATION.md)
- Main pretrained model notes: [FinalModels/README.md](FinalModels/README.md)

## Layout

```text
apps/labeling-ui/      Supabase labeling app
dataset/               compact delivery datasets, ready to run
data/                  ignored full local datasets, registries, templates, SQLite state
generated_images/      synthetic image outputs
pipelines/             documented V0-V5B pipeline ladder
scripts/               project helper scripts
src/dream2detect/      Python package code
TrainingFiles/         restored main-branch training scripts
EvaluartionFiles/      restored main-branch evaluation scripts
FinalModels/           restored main-branch model notes
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
