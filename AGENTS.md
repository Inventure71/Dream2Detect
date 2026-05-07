# Dream2Detect AGENTS Instructions

This file defines repo-specific execution rules for future agent work in Dream2Detect.

## Project State

The project is a comparative ML study of synthetic-to-real generalization for package/cardboard defect severity.

Current source-of-truth docs:

- [docs/01-project-understanding.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/01-project-understanding.md)
- [docs/06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md)
- [docs/07-real-dataset-relabeling.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/07-real-dataset-relabeling.md)
- [docs/08-implementation-plan.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/08-implementation-plan.md)
- [docs/09-data-contracts.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/09-data-contracts.md)
- [docs/10-execution-timeline.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/10-execution-timeline.md)

## Repo Working Rules

### 1. Prioritize clean, organized, easy-to-read code

Default to:

- small modules,
- clear naming,
- explicit data contracts,
- code that is easy to review.

Avoid dense or overly clever implementations.

### 2. No monoliths

Do not build:

- large all-in-one scripts,
- giant notebooks that own every step,
- one file that mixes prompt generation, image generation, relabeling, training, and evaluation.

Split work by responsibility.

Minimum expected boundaries:

- prompt system
- synthetic data pipeline
- real-data relabeling
- training
- evaluation

### 3. Keep the repo organized at all times

Do not scatter outputs or temporary files around the repo root.

Use the established structure under:

- `docs/`
- `data/manifests/`
- `data/registries/`
- `data/synthetic/`
- `data/real/`
- future code folders added deliberately

If a new folder is needed, add it intentionally and document it if it becomes part of the workflow.

### 4. Do not silently diverge from the plan

The current execution plan is intentional.

If you think the plan should change:

1. stop
2. explain why the change is needed
3. ask the user for confirmation
4. after confirmation, update the plan docs before continuing

Do not keep working on a private fork of the plan in chat only.

### 5. Keep documentation synchronized with real decisions

If a design choice becomes locked and it affects workflow, update the relevant docs.

Do not let chat history become the only place where the current truth exists.

### 6. Preserve the core experiment

Do not drift away from the main comparison:

- synthetic-only
- real-only
- synthetic + real fine-tuning
- fine target
- coarse target

Do not replace the core from-scratch CNN experiment with a pretrained-backbone experiment unless the user explicitly approves that change and the plan is updated.

### 7. OpenAI runtime policy

Current chosen runtime policy:

- prompt drafting model: `gpt-5.4-mini`
- image generation model: `gpt-image-2`
- image generation settings for the first pipeline: `quality=low`, `size=1024x1024`

Implementation rule:

- use a simple dedicated OpenAI connector/module for prompt drafting
- use the direct Image API for image generation when the goal is to explicitly control the image model as `gpt-image-2`

Reason:

- the prompt model and image model should be independently configurable
- the direct Image API is the cleanest way to explicitly request `gpt-image-2`

### 8. Respect the band-first labeling system

Primary label:

- `score_band`

Derived labels:

- `coarse_class`
- `representative_score`

Do not switch to fake-precise exact-score prompting.

### 9. Execution before training

Before model training begins, the following must exist:

- usable prompt manifest
- pilot synthetic image batch
- synthetic image registry
- real relabel registry
- frozen real split policy

If those are missing, do not jump ahead into training code.
