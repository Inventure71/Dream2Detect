# Feature Diversity System

This file defines the structured diversity-axis system for synthetic prompt planning.

## Goal

Do not let the LLM invent every scene detail freely.

Instead:

1. choose a target `score_band`,
2. sample a structured set of feature values,
3. draft prompt text from those values,
4. store both the feature assignment and the final prompt.

This gives better diversity control and makes balancing measurable.

## Balance Rule

Balance is done **per feature axis**, not across unrelated features as one flat pool.

Correct:

- lighting values should be balanced against other lighting values
- background values should be balanced against other background values
- camera-angle values should be balanced against other camera-angle values

Incorrect:

- comparing `bad_lighting` directly against `shipping_label_patch` as if they were the same type of thing

## Current Feature Categories

### 1. Defect content

- `damage_profile_primary`
- `damage_profile_secondary`
- `damage_location_primary`

These strongly affect what severity looks like.

They are band-aware and only some values are valid in some bands.
They are also compatibility-aware: not every primary defect is allowed at every damage location.

### 2. Package appearance

- `box_form_factor`
- `box_pattern`
- `label_presence`
- `tape_profile`

These vary the box appearance and packaging style without changing the severity target by themselves.

### 3. Scene and capture

- `background_context`
- `camera_angle`
- `lighting_style`

These are nuisance or context variables the model should not overfit to.

## Stress Policy

Some nuisance values are valid but risk making band calibration noisier if they stack together.

Current examples:

- `dim_ambient`
- `harsh_side_shadow`
- `backlit_but_readable`

These are treated as **stress features**.

Current rule:

- they still participate in axis balancing
- but the sampler should avoid stacking multiple stress-heavy nuisance values in one assignment
- the current implementation uses a one-stress-value budget per prompt assignment

This keeps the dataset realistic without letting hard lighting conditions dominate early calibration batches.

## Text-Risk Policy

Some feature combinations tend to trigger synthetic shortcuts such as:

- oversized warning stickers
- dominant barcode blocks
- readable packaging slogans
- background signage that competes with the damage signal

Current rule:

- text-bearing feature values carry a small `text_risk_weight`
- the sampler enforces a text-risk budget per assignment
- this prevents combinations such as:
  - `multiple_small_stickers` plus text-heavy scenes
  - label-heavy appearance plus additional printed-label surfaces

This policy exists because the image model can otherwise introduce strong readable text that is not part of the severity target and can become a shortcut.

## Sampler Policy

The sampler should:

1. choose only values allowed for the requested band
2. enforce compatibility rules such as primary-defect to location fit
3. avoid stacking multiple stress-heavy nuisance values in one prompt assignment
4. enforce a text-risk budget so text-bearing combinations stay bounded
5. prefer values with the lowest current count for that band and axis
6. avoid repeating the exact same full assignment when possible
7. penalize near-duplicate prompt concepts, especially repeated `primary defect + background` pairings
8. keep the balancing local to each axis

This means the dataset moves toward even marginal distributions without trying to force every full feature combination to be equally common.

It also means the sampler now actively pushes away from the two failure classes seen in Phase 1:

- near-duplicate prompts that are technically different but visually too similar
- assignments that are likely to trigger text-heavy synthetic shortcuts

## Storage Rule

Each prompt row should persist the full feature assignment.

These values should not exist only inside `prompt_text`.

At minimum, prompt rows should store:

- `damage_profile_primary`
- `damage_profile_secondary`
- `damage_location_primary`
- `box_form_factor`
- `box_pattern`
- `label_presence`
- `tape_profile`
- `background_context`
- `camera_angle`
- `lighting_style`

## Practical Workflow

For a new synthetic prompt batch:

1. choose band and count
2. sample feature assignments
3. review the assignments if needed
4. ask the LLM to write one prompt per assignment
5. review prompt text
6. approve prompts for image generation

## Current Implementation Boundary

The first version of this system is implemented in:

- [src/dream2detect/features](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/features)

The prompt execution database now stores these assignments per prompt row.

An offline simulator for checking large-scale feature balance without calling any model exists in:

- [simulation/offline_feature_balance](/Users/inventure71/VSProjects/School/Dream2Detect/simulation/offline_feature_balance)
