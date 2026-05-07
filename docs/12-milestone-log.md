# Milestone Log

This file records what has actually been completed in Dream2Detect at each milestone.

It exists to answer questions like:

- what is already locked,
- what has been implemented,
- what can now be claimed truthfully,
- what is still only planned.

## How To Read This

- `Locked` means the project decision is documented and should be treated as current truth.
- `Implemented` means code, docs, or data artifacts exist now.
- `Verified` means the implementation has been exercised and checked.

If something is not in this file, do not assume it is finished.

## Milestone 1: Project Reframing

### Outcome

The project was reframed into a comparative ML study of synthetic-to-real transfer under domain shift.

### What is now true

- The project is not a generic image-generation demo.
- The main question is:
  - how performance changes when we vary training data source and target formulation
- The core comparisons are locked:
  - `synthetic-only`
  - `real-only`
  - `synthetic + fine-tuning`
  - `fine severity`
  - `coarse severity`

### Evidence

- [01-project-understanding.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/01-project-understanding.md)
- [05-assignment-constraints.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/05-assignment-constraints.md)

## Milestone 2: Domain And Label System Locked

### Outcome

The project returned to the package/cardboard defect domain and locked the severity system.

### What is now true

- The target domain is visible package/cardboard defect severity.
- The coarse classes are locked:
  - `intact`
  - `minor`
  - `moderate`
  - `severe`
- The fine-to-coarse mapping is locked:
  - `0-10 -> intact`
  - `11-35 -> minor`
  - `36-65 -> moderate`
  - `66-100 -> severe`
- The working prompt/annotation bands are locked.

### Evidence

- [06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md)
- [06-package-severity-bands](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands)

## Milestone 3: Real-Data Policy Locked

### Outcome

The real branch was defined around a relabeled Kaggle cardboard-defect dataset.

### What is now true

- Kaggle images are merged into one unified candidate pool before relabeling.
- ChatGPT Vision is only a labeling assistant, not the final source of truth.
- Human review is required.
- The real split policy is locked:
  - `60%` held-out real test
  - `20%` small real training
  - `20%` validation / fine-tuning support

### Evidence

- [07-real-dataset-relabeling.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/07-real-dataset-relabeling.md)
- [01-project-understanding.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/01-project-understanding.md)

## Milestone 4: Execution Contracts And Repo Scaffold

### Outcome

The repo moved from planning-only docs to an execution-ready structure.

### What is now true

- The repo has explicit execution docs.
- The repo has a working folder structure for:
  - manifests
  - registries
  - synthetic outputs
  - real-data labeling
- A SQLite execution database exists for prompt/image workflow state.
- Template CSV files exist for prompt manifests and registries.

### Evidence

- [09-data-contracts.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/09-data-contracts.md)
- [10-execution-timeline.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/10-execution-timeline.md)
- [data/templates](/Users/inventure71/VSProjects/School/Dream2Detect/data/templates)

## Milestone 5: OpenAI Prompt And Image Pipeline

### Outcome

The repo now has a working prompt-drafting and image-generation pipeline.

### What is now true

- Prompt drafting uses `gpt-5.4-mini`.
- Image generation uses direct Image API with `gpt-image-2`.
- Prompt drafting uses structured outputs, not free-form JSON parsing.
- Image generation is fail-fast:
  - if an image call fails, the prompt is marked `failed`
  - the error is stored
  - the run stops
- Prompt rows and generated-image rows are linked in SQLite.

### Verified behavior

- A paid smoke test was run successfully:
  - one prompt was approved
  - one image was generated
  - the image was saved locally
  - the prompt row was updated correctly

### Important limitation

- A successful API call does **not** mean the generated image matches the intended severity band.
- Band correctness still requires review.

### Evidence

- [src/dream2detect/services/prompt_drafting.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/services/prompt_drafting.py)
- [src/dream2detect/services/image_generation.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/services/image_generation.py)
- [src/dream2detect/storage](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/storage)

## Milestone 6: Band Prompting Tightened

### Outcome

The prompt-facing band definitions were refined after the first image overshot its target severity.

### What is now true

- Lower and middle bands now explicitly forbid stronger escalation cues such as:
  - torn open
  - exposed interior
  - major collapse
  - severe crushing
- Prompt generation is more tightly constrained by the band rubric than before.

### Why this mattered

- The first generated `36-45` image followed the prompt well but landed too high in severity.
- That showed the problem was largely prompt calibration, not only image-model behavior.

### Evidence

- [06-package-severity-bands/README.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/README.md)
- [06-package-severity-bands](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands)

## Milestone 7: Structured Feature Diversity System

### Outcome

A real feature-diversity system now exists in code.

### What is now true

- Synthetic prompt planning no longer depends only on free-text prompt creativity.
- The system now defines explicit feature axes such as:
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
- Feature values are sampled before prompt text is written.
- Prompt rows persist those structured feature assignments in the database.

### Balance claim that is now valid

We can now truthfully say:

> the repo has a system designed to keep class and feature distributions balanced during synthetic prompt planning

More precisely:

- score bands can be assigned in a controlled way,
- feature values are balanced **per axis**,
- sampling is **band-aware**,
- the system prefers the least-used allowed values,
- exact full feature combinations are not forced to be equally common.

### Important boundary

We cannot yet truthfully say:

> the final dataset is already balanced

because the full synthetic dataset has not been generated yet.

What we can say is:

> the balancing mechanism has been implemented and can be simulated offline before spending API money

### Evidence

- [11-feature-diversity-system.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/11-feature-diversity-system.md)
- [src/dream2detect/features](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/features)
- [src/dream2detect/services/prompt_drafting.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/services/prompt_drafting.py)

### Later refinement added after first prompt reviews

- the feature system now also enforces **primary-defect to location compatibility**
- this prevents awkward pairings such as:
  - corner-split defects centered on the face
  - side-compression defects assigned to a corner
- this change moved the remaining prompt-quality problem from “sampler emits bad pairings” to normal prompt-review quality control
- a later hardening pass also added a **stress-feature policy**
  - difficult nuisance conditions such as dim or harsh lighting can still appear
  - but they should not stack together inside one prompt assignment during early calibration

## Milestone 8: Offline Balance Simulator

### Outcome

An offline simulator was created to test large-scale feature balance without calling any model API.

### What is now true

- The repo can simulate a large prompt plan, such as `1000` prompts, without:
  - drafting prompts
  - generating images
  - spending money
- The simulator writes:
  - per-simulated-prompt assignments
  - per-axis count summaries
  - per-band counts
  - an HTML report with charts

### Verified behavior

- The simulator was run successfully for `1000` simulated prompts.
- The default simulation distributes prompts evenly across the `10` score bands:
  - `100` per band
- The report computes band-aware expected counts for each feature value.

## Milestone 9: Band-Control And Image QC Hardening

### Outcome

The repo now treats post-generation image QC as a first-class part of the synthetic pipeline instead of relying only on prompt quality.

### What is now true

- the `56-65` band guidance was tightened so broad side crushing and dominant geometry loss are pushed upward into severe
- the sampler now enforces a nuisance-stress budget per assignment
- generated-image rows can now record:
  - reviewed band and class
  - final accepted band and class
  - QC decision
  - QC notes

### Why this mattered

- the first 8-image calibration batch showed that one generated image was visually useful but mislabeled for its intended band
- without explicit QC states, the system had no clean way to relabel that image or reject it systematically

### Evidence

- [docs/06-package-severity-bands/56-65-moderate-high.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands/56-65-moderate-high.md)
- [src/dream2detect/features/sampler.py](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/features/sampler.py)
- [src/dream2detect/storage](/Users/inventure71/VSProjects/School/Dream2Detect/src/dream2detect/storage)

## Milestone 10: Phase 1 Calibration Run

### Outcome

The first full calibration run passed.

### What is now true

- a clean isolated calibration database was used without disturbing the main balancing history
- `50` prompts were drafted and reviewed
- weak prompts were rejected and replaced before image spend
- `12` calibration images were generated
- image QC produced:
  - `10` accepted as labeled
  - `1` accepted with relabeling
  - `1` rejected
- the dominant discovered failure mode was not band collapse but occasional over-strong printed text/label clutter
- that failure mode was fed back into the prompt-writing instructions
- the accepted reviewed examples were merged into the permanent database

### Evidence

- [13-phase1-calibration-report.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/13-phase1-calibration-report.md)
- [data/calibration/phase1_round1.sqlite3](/Users/inventure71/VSProjects/School/Dream2Detect/data/calibration/phase1_round1.sqlite3)
- [generated_images/phase1_round1](/Users/inventure71/VSProjects/School/Dream2Detect/generated_images/phase1_round1)

### Balance claim that is now valid

We can now truthfully say:

> we have an offline tool to verify whether the current feature sampler would keep synthetic prompt planning close to balanced before we spend on actual prompt drafting or image generation

### Evidence

- [simulation/offline_feature_balance](/Users/inventure71/VSProjects/School/Dream2Detect/simulation/offline_feature_balance)
- [simulation/offline_feature_balance/outputs/report.html](/Users/inventure71/VSProjects/School/Dream2Detect/simulation/offline_feature_balance/outputs/report.html)

## Current State Summary

At this point, the project has:

- a locked ML framing,
- a locked label system,
- a locked real-data policy,
- a working OpenAI prompt/image pipeline,
- a structured feature-diversity system,
- an offline simulator for feature-balance analysis.

It does **not** yet have:

- a finished large synthetic dataset,
- a finished relabeled real dataset,
- training results,
- evaluation results.

So the correct status is:

> the project is operationally ready for controlled data-generation work, but not yet at the modeling-results stage
