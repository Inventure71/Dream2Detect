# Project Understanding

This file is the top-level description of the current Dream2Detect project.

Source artifacts behind this version:

- `ML_group_project_Guidelines.pdf`
- teacher feedback shared in chat
- [Q&A.md](/Users/inventure71/VSProjects/School/Dream2Detect/Q&A.md)

## Project In One Sentence

Dream2Detect is a comparative ML study of **synthetic-to-real generalization for package/cardboard defect severity under distribution shift**.

## Core Research Question

> What changes when we vary the training data source and the target formulation under synthetic-to-real domain shift, and why?

This is the real project question.

The project is **not** mainly about:

- generating images for its own sake,
- building the single best defect model possible,
- proving a product-ready package inspector.

It is about understanding transfer under a controlled comparison.

## Comparative Axes

The study has two main axes.

### 1. Training data source

Compare:

- `synthetic-only`
- `small real-only`
- `synthetic + small real fine-tuning`

### 2. Target formulation

Compare:

- `fine-grained severity prediction`
- `coarse severity classification`

This creates the exact comparative structure the teacher asked for.

## Domain

The domain is:

> visible package/cardboard defect severity

This includes visible conditions such as:

- dents,
- crushing,
- deformation,
- tears,
- holes,
- openings,
- edge and corner damage,
- dirt or stains when they meaningfully worsen visible package condition.

The project returned to packages after the teacher pointed out that this domain gives:

- a more natural severity spectrum,
- a harder synthetic-to-real gap,
- a stronger fine-vs-coarse comparison than the plate concept.

## Data Plan

### Synthetic branch

Main scalable training source:

- AI-generated package/cardboard images
- generated from a strict band-based severity rubric
- generated from explicit structured feature assignments before prompt text is written
- created through an LLM-assisted prompt-writing workflow followed by human review

### Real branch

Official real-data branch during development:

- Kaggle `redf0xwin/recognizing-defects-in-boxes-and-cardboard` cardboard/box defect images
- relabeled into the project rubric
- ChatGPT Vision used only as a labeling assistant
- final labels approved by humans

### Optional extra real branch

- your own package photos
- only as a later stress test
- not part of development or tuning

## Label System

### Fine target

Fine severity means:

> overall visible package defect severity

This is a visual condition scale, not a literal physical breakage percentage.

### Coarse target

The official coarse classes are:

- `intact`
- `minor`
- `moderate`
- `severe`

Official mapping:

- `0-10 -> intact`
- `11-35 -> minor`
- `36-65 -> moderate`
- `66-100 -> severe`

Official working bands:

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

The detailed rubric lives in:

- [06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md)
- [06-package-severity-bands](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-bands)

## Modeling Plan

### Baselines

- majority class baseline for coarse labels
- mean predictor for fine score
- HOG + logistic regression for coarse labels
- HOG + ridge regression for fine score

### Neural models

Two separate custom CNNs trained from scratch:

- one classifier for `intact / minor / moderate / severe`
- one regressor for fine severity

Each CNN is compared under:

- synthetic-only training
- real-only training
- synthetic + real fine-tuning

No pretrained backbone is part of the core experiment.

## Real Split Policy

After relabeling the unified Kaggle pool, split it into:

- `60%` held-out real test
- `20%` small real training
- `20%` real validation / fine-tuning support

The held-out real test set remains untouched until final evaluation.

## What A Good Result Looks Like

The project succeeds if it can show a clear, defensible pattern such as:

- synthetic-only transfers somewhat but not perfectly,
- small real-only behaves differently because real data is scarce,
- synthetic + fine-tuning improves real-world performance,
- coarse severity survives domain shift better than fine severity.

That is a valid result even if synthetic-only does not fully match real-only or hybrid performance.

## Document Map

- [02-user-perspective.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/02-user-perspective.md): why this project shape fits the user's goal
- [03-concepts-explained.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/03-concepts-explained.md): key ML concepts behind the project
- [04-thinking-model.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/04-thinking-model.md): project reasoning and guardrails
- [05-assignment-constraints.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/05-assignment-constraints.md): mapping to the course rubric
- [06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md): severity system and band index
- [07-real-dataset-relabeling.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/07-real-dataset-relabeling.md): real-data labeling workflow
- [08-implementation-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/08-implementation-plan.md): practical execution plan
- [09-data-contracts.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/09-data-contracts.md): concrete file layout and CSV contracts for starting execution
- [10-execution-timeline.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/10-execution-timeline.md): step-by-step operational timeline and next-task tracker
- [11-feature-diversity-system.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/11-feature-diversity-system.md): structured feature axes and balancing strategy for synthetic data
- [12-milestone-log.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/12-milestone-log.md): milestone-by-milestone record of what is actually completed and what can be claimed
- [13-phase1-calibration-report.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/13-phase1-calibration-report.md): verified outcome of the first full prompt-and-image calibration run
