# Real Dataset Relabeling Protocol

This file defines how the real package/cardboard dataset should be relabeled into the project's severity system.

## Purpose

The Kaggle dataset is useful because it gives real package/cardboard images in the right domain.

It is not enough on its own because:

- it was not labeled with this exact severity scheme,
- it was not built for this exact comparison,
- its original splits are not the project's final splits.

So the real dataset must be rebuilt under the project's own rubric.

## Real Dataset Workflow

1. merge the candidate Kaggle images into one relabeling pool
2. run the relabeling workflow under the package severity rubric
3. freeze the final approved labels
4. rebuild train/validation/test with balancing and leakage control

## Selected Source Dataset

The current real-image source is Kaggle `redf0xwin/recognizing-defects-in-boxes-and-cardboard`.

Local imported files live under:

- `data/real/source/redf0xwin_recognizing_defects_in_boxes_and_cardboard/images`
- `data/real/source/redf0xwin_recognizing_defects_in_boxes_and_cardboard/annotations`

The source XML files contain bounding-box defect annotations. They are kept only as provenance. The project does not use those boxes as ground truth for the core experiment. Every image still needs a whole-image severity relabel under the Dream2Detect rubric.

Official split after relabeling:

- `60%` held-out real test
- `20%` small real training
- `20%` real validation / fine-tuning support

## Role Of ChatGPT Vision

ChatGPT Vision is a labeling assistant only.

Allowed role:

- suggest `score_band`
- suggest `coarse_class`
- suggest the derived representative score for that band
- explain why the image fits that suggestion

Not allowed role:

- define final ground truth on its own

## Human Review Rule

Every image gets:

- one AI suggestion
- one required human review

Some images get:

- a second human review if they are flagged

Flagged cases include:

- uncertain AI proposal
- uncertain human judgment
- boundary between two nearby bands
- multiple defect types that make the image hard to collapse into one severity band

## Labeling Source Of Truth

Use these files as the rubric:

- [06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md)
- [06-package-severity-bands](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands)

The image should be labeled by **band first**, then mapped to coarse class.

The numeric representative score should be derived from the approved band using the default mapping in [06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md).

## Recommended Labeling Steps

For each image:

1. inspect the image visually
2. compare it against the band specs
3. ask ChatGPT Vision for a proposed band and explanation
4. have a human reviewer confirm or correct the band
5. derive the coarse class and representative score from the approved band
6. store any review notes or disagreement flags

## Recommended Schema

Minimum practical schema:

```csv
image_id,image_path,llm_score_band,llm_representative_score,llm_coarse_class,llm_reasoning,human_score_band,human_representative_score,human_coarse_class,final_score_band,final_representative_score,final_coarse_class,review_status,disagreement_flag,review_notes
```

Useful `review_status` values:

- `pending_human_review`
- `approved`
- `corrected`
- `needs_second_review`

## Leakage Control Rule

When rebuilding the final split:

- near-duplicate or same-scene images should stay in the same split
- very similar views of the same box should not be spread across train and test

Otherwise the real benchmark becomes too optimistic.

## Representative Score Rule

The project labels by band, not by exact score.

If a single numeric value is needed for regression support or bookkeeping, use the fixed representative score assigned to the approved band.

The representative score is secondary to the band and should never override the band decision.
