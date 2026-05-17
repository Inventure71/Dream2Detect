# Pipeline Directory

This directory is the delivery index for the model-development ladder.

Each subfolder contains a short README with:

- the question that version tested
- the data it used
- the main implementation files
- the main artifacts and results
- whether it was promoted or kept as evidence only

The no-pretrain research path lives mainly in `src/dream2detect/training/`,
`scripts/`, and the short per-version summaries in this directory. The older
pretrained path from `main` lives in the restored top-level folders
`TrainingFiles/`, `EvaluartionFiles/`, and `FinalModels/`.
