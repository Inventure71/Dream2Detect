# Dream2Detect

Dream2Detect studies synthetic-to-real generalization for visible package/cardboard defect severity.

## Label Images

Use:

[INSTRUCTIONS_FOR_LABELLING.md](INSTRUCTIONS_FOR_LABELLING.md)

## Project Docs

- [Project understanding](docs/01-project-understanding.md)
- [Severity scale standards](docs/06-package-severity-scale-standards.md)
- [Real dataset relabeling](docs/07-real-dataset-relabeling.md)
- [Implementation plan](docs/08-implementation-plan.md)
- [Data contracts](docs/09-data-contracts.md)

## Layout

```text
apps/labeling-ui/      Supabase labeling app
data/                  datasets, registries, templates, SQLite state
docs/                  project docs and rubric
generated_images/      synthetic image outputs
scripts/               project helper scripts
src/dream2detect/      Python package code
```
