# Phase 1 Calibration Report

This file records the verified outcome of the first full calibration run.

## Scope

Phase 1 had two goals:

1. validate that the prompt system can produce a usable 50-prompt calibration pool
2. validate that a small image run produces mostly on-band images before any large-scale generation

## Working Database Boundary

The run used an isolated calibration database:

- [phase1_round1.sqlite3](/Users/inventure71/VSProjects/School/Dream2Detect/data/calibration/phase1_round1.sqlite3)

This avoided polluting the main execution database while prompt balancing and QC policies were still being tested.

The generated images for this run were written to:

- [generated_images/phase1_round1](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/phase1_round1)

## Prompt Calibration Outcome

### Generated prompt pool

- `50` prompts were drafted:
  - `5` per band across the `10` score bands

### Prompt review outcome

Three prompts were rejected before image spend:

- `24`
  - too close to `21`
  - same primary defect family
  - same scene family
  - same stress-lighting pattern
- `27`
  - too close to `26`
  - same primary defect
  - same background family
  - same overall visual concept
- `34`
  - too likely to overshoot `56-65`
  - front-face cavity on a doorstep is a poor clean calibration example

Three replacements were drafted:

- `51` for `36-45`
- `52` for `46-55`
- `53` for `56-65`

### Prompt-system judgment

Prompt calibration passed.

The system is now good enough to support a larger prompt pool, with one remaining caution:

- prompt writing needed an extra rule to suppress dominant printed box text and oversized warning stickers

That rule has been added to:

- [prompt_writing_template.md](/Users/inventure71/VSProjects/School/Dream2Detect/data/templates/prompt_writing_template.md)

### Root-cause fixes applied after Phase 1 review

The Phase 1 failures were not left as ad hoc review notes. The following structural fixes were applied:

- near-duplicate prompt concepts are now penalized during sampling, especially repeated `primary defect + background` pairings
- `small_cavity_with_deformation` can no longer be placed on the front face center for `56-65`
- the old `76-85` collapsed-corner option was replaced with a stronger variant that explicitly includes an open seam, reducing overlap with `66-75`
- text-bearing feature values now carry a text-risk budget so combinations that tend to provoke dominant printed text are filtered out earlier
- the prompt-writing template now explicitly forbids large readable background signage and large printed package text even in work-area scenes

## Image Calibration Outcome

### Generated image subset

`12` approved prompts were used for image calibration:

- `2`
- `7`
- `13`
- `18`
- `22`
- `35`
- `37`
- `42`
- `46`
- `51`
- `52`
- `53`

### QC results

- `10` images accepted as labeled
- `1` image accepted with relabeling
- `1` image rejected

#### Accepted as labeled

- `2` -> `0-10`
- `7` -> `11-20`
- `13` -> `21-30`
- `18` -> `31-35`
- `22` -> `36-45`
- `37` -> `66-75`
- `46` -> `86-100`
- `51` -> `36-45`
- `52` -> `46-55`
- `53` -> `56-65`

#### Accepted with relabeling

- `42`
  - intended: `76-85`
  - reviewed/final: `66-75`
  - reason: the output was good, but the damage read as severe-low rather than severe-mid

#### Rejected

- `35`
  - intended: `56-65`
  - reviewed: `46-55`
  - reason: the image drifted from the intended defect mix and introduced overly dominant printed text/label cues

### Image-system judgment

Image calibration passed.

The results are strong enough to proceed, because:

- `11 / 12` generated images were usable
- only `1 / 12` needed relabeling
- only `1 / 12` was rejected

This is a strong enough pass rate for moving toward the next scale-up stage, provided that:

- large-scale generation still uses staged checkpoints
- unreviewed and reviewed pools remain separated
- post-generation QC remains available

## What Was Preserved

Reviewed good examples from Phase 1 were merged into the permanent database:

- [dream2detect.sqlite3](/Users/inventure71/VSProjects/School/Dream2Detect/data/dream2detect.sqlite3)

Merge result:

- `11` reviewed prompt rows imported
- `11` generated-image rows imported

This means the main database now contains:

- older exploratory history
- the Phase 1 reviewed examples

while the isolated calibration database still preserves the full Phase 1 working state.

## Final Phase 1 Verdict

Phase 1 is a **pass**.

More precisely:

- prompt calibration: pass
- image calibration: pass
- QC workflow: pass
- isolated calibration workflow: pass

Remaining caution before large-scale generation:

- the system should continue to treat generated images as `unreviewed` by default
- the final training/reporting dataset should still distinguish:
  - `accepted_as_labeled`
  - `accepted_relabel`
  - `rejected`
- the Phase 1 fixes should be considered part of the required preconditions for the next larger prompt pool

## Recommended Next Move

The next stage should be:

1. build the larger prompt pool
2. keep all generated images as `unreviewed` initially
3. generate in controlled batches
4. QC later without losing intended labels
