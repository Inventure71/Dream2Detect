# Dataset Generation Workflow

This folder preserves the dataset-generation part of the project separately
from the clean training/evaluation pipeline.

It contains the scripts and supporting source copied from the `exp` branch that
were used for:

- drafting band-specific synthetic prompts,
- preparing OpenAI image-generation batches,
- ingesting generated images into the local registry,
- applying QC decisions,
- exporting synthetic and real training manifests,
- building processed image caches,
- preparing the V2 real-failure-targeted synthetic pipeline.

The ready-to-run delivery datasets are still in the root-level `dataset/`
folder. This folder is for reproducibility and inspection of how those datasets
were produced.

## Layout

```text
dataset_generation/
  scripts/              dataset-generation, QC, registry, and export commands
  src/dream2detect/     generation-specific support modules from exp
  data/templates/       CSV/manifest templates
  docs/                 band standards and generation workflow docs
  requirements.txt      extra dependencies for generation scripts
```

## Important Runtime Note

Some scripts can spend OpenAI API credits. In particular:

- `scripts/generate_images.py`
- `scripts/image_batch.py submit`
- `scripts/image_batch.py ingest` after a completed paid batch

Use the prepare/export/QC scripts freely, but do not run image generation unless
you intentionally want to spend API budget and have `OPENAI_API_KEY` configured.

## Environment

From this folder:

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Typical Synthetic Generation Flow

1. Initialize or open the local SQLite registry.

```bash
python3 scripts/draft_band_prompts.py --band 36-45 --count 10 --seed 42
```

2. Review prompts and approve/reject them.

```bash
python3 scripts/list_prompts.py
python3 scripts/show_prompt.py --id 1
python3 scripts/set_prompt_status.py --id 1 --status approved
```

3. Prepare an image-generation batch without submitting it.

```bash
python3 scripts/image_batch.py prepare \
  --prompt-status approved \
  --image-status pending \
  --limit 20 \
  --quality low \
  --output-subdir pilot
```

4. Submit only when API spend is intended.

```bash
python3 scripts/image_batch.py submit --batch-dir data/batches/image_generation/<batch_dir>
python3 scripts/image_batch.py status --batch-dir data/batches/image_generation/<batch_dir>
python3 scripts/image_batch.py ingest --batch-dir data/batches/image_generation/<batch_dir>
```

5. Review generated images and apply QC.

```bash
python3 scripts/build_qc_contact_sheets.py \
  --manifest data/datasets/synthetic_manifest.csv \
  --output-dir data/qc/synthetic_round

python3 scripts/apply_qc_decisions_to_manifest.py \
  --manifest data/datasets/synthetic_manifest.csv \
  --decisions data/qc/synthetic_round/qc_queue.csv \
  --output data/datasets/synthetic_reviewed.csv \
  --review-round round1
```

6. Export the final synthetic training manifest and build processed images.

```bash
python3 scripts/export_synthetic_training_manifest.py \
  --output data/datasets/synthetic_training_manifest.csv \
  --reviewed-only

python3 scripts/build_image_cache.py \
  --source-manifest data/datasets/synthetic_training_manifest.csv \
  --output-dir data/cache/synthetic_384 \
  --output-manifest data/datasets/synthetic_training_manifest_384.csv \
  --image-size 384
```

## V2 Targeted Pipeline Prep

The V2 script prepares prompt manifests and batch JSONL files but does not
submit an API job:

```bash
python3 scripts/v2_prepare_image_pipeline.py \
  --run-name v2_real_failure_targeted \
  --count-per-band 12 \
  --quality low
```

## Real Dataset Export

The real dataset was exported from the relabel registry into the same manifest
shape used by the training code:

```bash
python3 scripts/export_real_training_manifest.py \
  --registry data/registries/real_relabel_registry.csv \
  --output data/datasets/real_labeled_dataset_current.csv \
  --relative-image-paths
```

## What This Folder Does Not Contain

This folder does not include old generated-image outputs, SQLite databases, API
batch outputs, caches, or intermediate datasets. Those are reproducible outputs
or local artifacts and would make the delivery branch hard to read.
