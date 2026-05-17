# Package Severity Scale Standards

This document is the entry point for the package/cardboard defect severity rubric.

It defines:

- the target concept the project is trying to learn,
- the official fine-to-coarse severity mapping,
- where the detailed per-band specifications live,
- how the rubric should be used in generation and relabeling.

## Target Concept

The project does **not** try to measure literal physical breakage percentage.

The target is:

> overall visible package defect severity

This means the label should reflect the overall visible condition of the package in the image.

It may include:

- dents,
- crushing or deformation,
- tears,
- holes or openings,
- bent or softened corners,
- damaged edges or flaps,
- surface dirt or stains when they meaningfully worsen visible package condition.

It should **not** pretend to measure:

- hidden internal damage,
- shipping value loss,
- exact force damage,
- exact structural failure mechanics,
- non-visible condition.

This is a visual condition scale, not an engineering damage measurement.

## Official Coarse Classes

The official coarse classes are:

- `intact`
- `minor`
- `moderate`
- `severe`

Their high-level definitions are:

- `intact`: visually normal package, no meaningful visible defect
- `minor`: clear but light visible defect, no major structural compromise
- `moderate`: substantial visible damage, clear degradation of condition
- `severe`: major visible structural or material damage

## Official Fine Score Mapping

The official coarse mapping from the fine score is:

- `0-10 -> intact`
- `11-35 -> minor`
- `36-65 -> moderate`
- `66-100 -> severe`

The official prompting and annotation bands are:

- `0-10`
- `11-20`
- `21-30`
- `31-35`
- `36-45`
- `46-55`
- `56-65`
- `66-75`
- `76-85`
- `86-100`

## Representative Score Policy

The project labels by **band first**, not by exact score.

For any experiment, table, or model that still needs a single numeric value, use a fixed **representative score** for each band.

Default representative scores:

- `0-10 -> 5`
- `11-20 -> 15`
- `21-30 -> 25`
- `31-35 -> 33`
- `36-45 -> 40`
- `46-55 -> 50`
- `56-65 -> 60`
- `66-75 -> 70`
- `76-85 -> 80`
- `86-100 -> 93`

This keeps the regression target consistent with the band-based labeling system.

The exact band remains the primary label. The representative score is a derived numeric anchor.

## Detailed Band Files

Each band has its own detailed specification under [06-package-severity-bands](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands).

Read them in order:

- [README.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/README.md)
- [00-10-intact.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/00-10-intact.md)
- [11-20-minor-low.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/11-20-minor-low.md)
- [21-30-minor-mid.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/21-30-minor-mid.md)
- [31-35-minor-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/31-35-minor-high.md)
- [36-45-moderate-low.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/36-45-moderate-low.md)
- [46-55-moderate-mid.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/46-55-moderate-mid.md)
- [56-65-moderate-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/56-65-moderate-high.md)
- [66-75-severe-low.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/66-75-severe-low.md)
- [76-85-severe-mid.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/76-85-severe-mid.md)
- [86-100-severe-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/86-100-severe-high.md)

## How To Use This Rubric

### For synthetic generation

Do **not** prompt for exact scores like `47` or `82`.

Prompt by:

- `score_band`
- `coarse_class`
- `representative_score` if needed for metadata only
- detailed visual description from the matching band file

Workflow:

1. pick a target band,
2. generate prompt drafts from that band's specification,
3. review and edit the prompts manually,
4. run a small pilot image batch,
5. inspect the images against the same band file,
6. only then scale to larger batch generation.

### For real-image relabeling

Use the band files as the labeling rubric.

Workflow:

1. inspect the image,
2. identify the nearest band based on visible evidence,
3. assign the final `score_band`,
4. derive `coarse_class` and `representative_score` from the official mapping,
5. store any uncertainty or boundary notes.

## Band Assignment Rule

Choose the band that best matches the **overall visible condition** of the package.

Do not over-focus on one tiny defect if the rest of the package clearly belongs in a lower band.

Do not under-score an image just because the damage is localized if that localized damage is severe enough to dominate the package condition.

When an image sits between two bands:

- prefer the lower band only if the stronger defect signal is weak or uncertain,
- prefer the higher band if the defect is clearly visible and meaningfully changes the package's overall condition.

## Current Status

The `v1` scale definition is now locked enough to support:

- prompt writing,
- pilot image generation,
- human review,
- real-image relabeling,
- dataset metadata design.
