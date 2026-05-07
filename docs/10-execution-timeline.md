# Execution Timeline

This file is the step-by-step project timeline and operational TODO.

It is meant to be specific enough that a future chat can resume work without rebuilding the plan from memory.

## Current Status

Current phase:

- Phase 1 calibration complete
- ready to begin controlled scale-up planning for the larger synthetic pool

Already completed:

- project framing locked
- severity system locked at `v1`
- detailed band specs written
- real relabeling protocol written
- data contracts and CSV templates created
- folder scaffold created
- OpenAI runtime choices selected
- structured feature-axis system implemented in code
- isolated calibration workflow verified
- Phase 1 prompt calibration completed
- Phase 1 image calibration completed
- reviewed-good-example merge path into the permanent database verified

Not started yet:

- Kaggle relabeling
- training code
- evaluation code

In progress / next:

- larger synthetic prompt-pool planning after Phase 1 pass
- reviewed vs unreviewed synthetic-pool policy during early training

## Phase 1: Prompt System Setup

Goal:

- create a reviewed pilot prompt manifest that covers the first pilot batch

Steps:

1. decide pilot coverage strategy
   - either all 10 bands lightly
   - or a smaller subset for the first debug cycle
2. define and freeze the diversity axes and allowed values
3. sample balanced feature assignments for the chosen bands
4. fill `data/manifests/pilot_prompt_manifest.csv`
5. draft prompt rows through the prompt-generation system and store them in the SQLite database
6. review all prompt rows manually
7. mark prompt row status as ready for generation

Completion condition:

- the pilot prompt set has reviewed prompt rows ready for generation

## Phase 2: Synthetic Pilot Generation

Goal:

- test whether the generation pipeline can produce images that actually match the target bands

Steps:

1. implement or use the OpenAI connector for prompt drafting and image generation
2. generate the pilot images from approved prompt rows
3. save pilot outputs into `generated_images/pilot/`
4. register each generated image in the SQLite database and, if needed, mirror accepted outputs into the CSV registries
5. review each generated image against the band specs
6. mark each image as accepted or rejected
7. record why rejected images failed

Completion condition:

- enough accepted pilot images exist to judge prompt quality and generation behavior

## Phase 3: Prompt And Rubric Correction Loop

Goal:

- correct prompt drift before scaling

Steps:

1. inspect failure patterns from rejected pilot images
2. identify whether the problem is:
   - bad prompt wording
   - bad band interpretation
   - bad scene control
   - model generation bias
3. revise prompt rows
4. if necessary, refine band guidance without changing the core label system
5. regenerate affected pilot rows

Completion condition:

- pilot prompts reliably produce visually correct band-level outputs

## Phase 4: Real Dataset Intake

Goal:

- prepare the Kaggle real dataset for relabeling

Steps:

1. place source images into `data/real/source/`
2. inventory the image set
3. assign stable `image_id` values
4. load the initial image rows into `data/registries/real_relabel_registry.csv`
5. preserve any original Kaggle labels as reference only

Completion condition:

- the real dataset exists as one unified candidate pool in the relabel registry

## Phase 5: Real Relabeling

Goal:

- freeze human-approved labels for the real dataset under the project rubric

Steps:

1. run AI-assisted band suggestion per image
2. perform the required first human review
3. flag uncertain or boundary cases
4. perform second review only on flagged cases
5. freeze:
   - `final_score_band`
   - `final_coarse_class`
   - `final_representative_score`
6. mark review status

Completion condition:

- the real relabel registry is complete enough to build the final split

## Phase 6: Real Split Construction

Goal:

- produce the official real train/validation/test structure

Steps:

1. inspect the relabeled distribution by band and coarse class
2. identify near-duplicates or same-box near-views
3. split with leakage control
4. assign:
   - `60%` held-out real test
   - `20%` small real training
   - `20%` real validation / fine-tuning support
5. verify balance is reasonable

Completion condition:

- the official real split is frozen

## Phase 7: Scaled Synthetic Generation

Goal:

- build the synthetic training corpus beyond the pilot batch

Steps:

1. expand the prompt manifest
2. generate in batches
3. register all outputs
4. review samples and spot-check quality continuously
5. remove or reject off-band outputs
6. track counts by:
   - score band
   - coarse class
   - each structured feature axis

Completion condition:

- the synthetic dataset is large enough and clean enough to support training

## Phase 8: Baseline Modeling

Goal:

- establish non-neural reference performance

Steps:

1. build majority-class baseline
2. build mean-score baseline
3. build HOG + logistic regression classifier
4. build HOG + ridge regression model
5. evaluate on held-out real data

Completion condition:

- baseline results are recorded and reproducible

## Phase 9: CNN Modeling

Goal:

- run the official from-scratch CNN comparisons

Steps:

1. implement the classifier CNN
2. implement the regressor CNN
3. run synthetic-only training
4. run real-only training
5. run synthetic + fine-tuning training
6. log results separately for:
   - coarse classification
   - fine regression

Completion condition:

- all official CNN comparison runs are complete

## Phase 10: Evaluation And Analysis

Goal:

- produce the results that answer the project question

Steps:

1. evaluate all models on held-out real data
2. compare synthetic-only vs real-only vs hybrid
3. compare fine vs coarse
4. inspect failure cases
5. identify what changed and why

Completion condition:

- the final results support a clear comparative conclusion

## Phase 11: Final Report Preparation

Goal:

- turn the technical work into a strong course deliverable

Steps:

1. summarize dataset construction
2. summarize relabeling process
3. summarize model comparisons
4. present metrics clearly
5. include failure examples
6. discuss limits and label noise honestly
7. explain whether coarse severity transferred better than fine severity

Completion condition:

- the report and presentation material can be built directly from the artifacts

## Immediate Next Step

The next action is:

> build the next controlled synthetic scale-up plan using the verified Phase 1 rules and keep generated images `unreviewed` by default until later human QC

That is the first meaningful post-Phase-1 operational task.

## Current Command Entry Points

Initialize the SQLite database:

```bash
python3 scripts/init_db.py
```

Draft prompts for one band and store them in the database:

```bash
python3 scripts/draft_band_prompts.py --band 36-45 --count 3
```

List stored prompts for review:

```bash
python3 scripts/list_prompts.py
```

Approve or reject a prompt after manual review:

```bash
python3 scripts/set_prompt_status.py --id 1 --status approved --notes "Looks on-band"
```

Show the full stored prompt before approval:

```bash
python3 scripts/show_prompt.py --id 1
```

Generate images for approved prompts with the direct Image API:

```bash
python3 scripts/generate_images.py --prompt-status approved --output-subdir pilot --limit 1
```
