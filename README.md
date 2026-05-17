# Dream2Detect

Dream2Detect studies synthetic-to-real generalization for visible package/cardboard defect severity.

## Label Images

Use:

[INSTRUCTIONS_FOR_LABELLING.md](INSTRUCTIONS_FOR_LABELLING.md)

## Project Docs

- [Delivery branch guide](DELIVERY.md)
- [V0-V5B pipeline index](pipelines/README.md)
- [Project understanding](docs/01-project-understanding.md)
- [Severity scale standards](docs/06-package-severity-scale-standards.md)
- [Real dataset relabeling](docs/07-real-dataset-relabeling.md)
- [Implementation plan](docs/08-implementation-plan.md)
- [Data contracts](docs/09-data-contracts.md)

## Layout

```text
apps/labeling-ui/      Supabase labeling app
dataset/               compact delivery datasets, ready to run
data/                  ignored full local datasets, registries, templates, SQLite state
docs/                  project docs and rubric
generated_images/      synthetic image outputs
pipelines/             documented V0-V5B pipeline ladder
scripts/               project helper scripts
src/dream2detect/      Python package code
```

## Delivery Datasets

The delivery branch includes two compact dataset folders:

- `dataset/synthetic/manifest.csv`
- `dataset/real/manifest.csv`

Each manifest uses `image_path` values relative to its own dataset folder, for
example `images/example.png`.
