# V2 Synthetic Generation Pipeline

## Purpose

The v2 pipeline is a response to the real-domain failure analysis. The earlier
synthetic pipeline produced useful synthetic accuracy, but its best checkpoint
transferred poorly to real images and often predicted real damaged packages as
`intact`.

The v2 goal is therefore not simply to create more images. It is to generate
synthetic images that specifically target the observed real-domain gap.

## Versioned Layout

- v1 compatibility package:
  `src/dream2detect/pipelines/v1/`
- v2 package:
  `src/dream2detect/pipelines/v2/`
- v2 preparation CLI:
  `scripts/v2_prepare_image_pipeline.py`

The old service modules remain available for compatibility:

- `src/dream2detect/services/prompt_drafting.py`
- `src/dream2detect/services/batch_image_generation.py`

New code should prefer the versioned namespaces.

## V2 Design

V2 prepares image-generation artifacts locally. It does not submit API jobs.

The generated artifacts are:

- `prompt_manifest.csv`
- `input.jsonl`
- `manifest.json`
- `run_config.json`
- `summary.json`
- `qc_rubric.md`

The pipeline uses deterministic scenario families:

- hard intact negatives with tape, labels, and clutter
- flat taped mailers with subtle damage
- low-contrast minor damage for minor-recall repair
- minor/moderate boundary edge and seam damage
- moderate real geometry loss that should not be predicted as intact
- moderate real-world crush and split cases with tape
- severe but photographically realistic damage
- severe openings and collapse that remain visible under phone-capture artifacts
- capture robustness cases with shadow, compression, and imperfect framing

Each prompt includes:

- score band
- coarse class
- representative score
- structured feature assignment
- real-domain scenario family
- real-gap priority
- QC checks
- label/text safety constraints
- explicit anti-render and anti-product-ad instructions

## Current Preview Artifact

Prepared without any API submission:

- `data/pipelines/v2/v2_real_failure_targeted_preview_20260509_r2/`
- `data/pipelines/v2/v2_real_failure_targeted_preview_20260513_gap_repair/`
- `data/pipelines/v2/v2_real_failure_targeted_preview_20260513_gap_repair_r3/`

The May 13 `gap_repair_r3` preview is the current candidate preview because it reflects the
frozen real-domain diagnostics and includes explicit `real_gap_priority`
metadata in the manifest and summary. It also fixes prompt-review issues from
the earlier May 13 previews:

- intact prompts no longer ask for a visible defect
- intact feature hints no longer say `primary damage`
- intact material instructions explicitly prohibit dents, tears, crushed
  corners, holes, split seams, and structural deformation
- all prompts now include a global unreadable/no-logo/no-address text-safety
  constraint

Preview settings:

- `count_per_band=3`
- total prompts: `30`
- image model field: `gpt-image-2`
- quality field: `medium`
- size field: `1024x1024`

Summary:

- `0-10`: `3`
- `11-20`: `3`
- `21-30`: `3`
- `31-35`: `3`
- `36-45`: `3`
- `46-55`: `3`
- `56-65`: `3`
- `66-75`: `3`
- `76-85`: `3`
- `86-100`: `3`

May 13 scenario coverage:

- hard intact negatives: `3`
- flat/subtle damage repair: `6`
- minor-recall repair: `3`
- minor/moderate boundary repair: `6`
- moderate-not-intact repair: `3`
- realistic moderate/low-severe damage: `3`
- severe-reality anchor: `3`
- severe-not-intact repair: `3`

After QC of the `r3` pilot, the scale-up path changed:

- `86-100` should not be generated at the same volume as other bands until the
  severe-high prompt family proves it can produce truly top-band damage
- V2 now supports repeated `--band-count BAND=COUNT` overrides
- `86-100` now uses dedicated severe-high repair scenarios instead of reusing
  ordinary severe scenarios

Current scale-up candidate:

- `data/pipelines/v2/v2_real_failure_targeted_scale_candidate_20260513_r1/`
- total prompts: `132`
- band counts:
  - `0-10`: `8`
  - `11-20`: `16`
  - `21-30`: `16`
  - `31-35`: `16`
  - `36-45`: `16`
  - `46-55`: `16`
  - `56-65`: `16`
  - `66-75`: `12`
  - `76-85`: `12`
  - `86-100`: `4`

## Usage

Prepare a reviewable v2 batch:

```bash
PYTHONPATH=src python3 scripts/v2_prepare_image_pipeline.py \
  --run-name v2_real_failure_targeted_candidate_001 \
  --count-per-band 20 \
  --seed 20260509 \
  --quality medium
```

This creates artifacts only. It does not call OpenAI and does not generate
images.

## QC Gate

Before submitting any v2 batch, inspect:

- `prompt_manifest.csv`
- `qc_rubric.md`
- at least a representative sample of the generated `input.jsonl` prompts

After images are generated, reject any image that:

- looks like a render, illustration, advertisement, or collage
- hides or changes the assigned defect
- is obviously outside the target score band
- gives an intact hard negative actual structural damage
- makes a damaged real-gap-repair scenario so subtle that it could reasonably be
  called intact
- contains dominant readable text, addresses, or logos
- has physically impossible damage geometry

## Experiment Rule

V2 should be evaluated against the current real-domain baselines:

- real-only scratch baseline
- synthetic-v1 checkpoint on real
- synthetic-v1 initialized real fine-tuning
- diagnostic pretrained upper bound

The success criterion is improved held-out real macro F1, per-class recall, and
high-confidence error behavior. Synthetic-only validation accuracy is secondary.
