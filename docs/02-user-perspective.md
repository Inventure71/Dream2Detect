# User Perspective

This file captures the user's intent behind the project, not just the technical setup.

## Core Goal

The user wants to prove something stronger than:

> we trained a classifier on images

The real goal is:

> test whether synthetic data can teach a model from scratch, then measure how well that learning survives when the model faces real images

## What The User Does Not Want

The user explicitly does not want the project to feel like:

- a generic computer-vision pipeline,
- an easy binary classification problem,
- a visually cute but scientifically weak dataset trick,
- a result that looks good only because the task is shallow.

## Why The Package Version Won

The project temporarily moved away from package defects toward plates and robot decisions.

That version was dropped because the teacher correctly identified the stronger ML question:

- the important part is domain shift,
- the target should have a natural severity interpretation,
- the comparison should be about what changes when we vary data source and target formulation.

Package/cardboard defects fit that much better than plate cleanliness.

## What The User Wants The Project To Demonstrate

The user wants the project to demonstrate:

- synthetic-to-real transfer is measurable,
- training source matters,
- coarse targets may survive shift better than fine targets,
- a small amount of real data may change the outcome a lot.

This makes the project a study of transfer behavior, not a demo of image generation.

## Role Of AI Assistance

The user is willing to use AI in two places:

1. **Synthetic data creation**
   - use an LLM to draft prompts from a strict rubric
   - humans review the prompts
   - images are generated in batches after a pilot run

2. **Real-data labeling assistance**
   - use ChatGPT Vision to suggest a band and coarse label
   - humans verify or correct it
   - humans freeze the final label

In both cases, the user does not want AI to replace the actual judgment standard.

## Practical Bias

The user prefers a project that is:

- methodologically strong,
- explainable in class,
- ambitious enough for a high mark,
- still feasible within the course constraints.

That is why the current design chooses:

- packages instead of plates,
- from-scratch CNNs instead of pretrained backbones,
- a comparative design instead of a single training pipeline,
- band-based labeling instead of fake-precise exact scores.
