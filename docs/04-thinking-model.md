# Thinking Model

This file captures the decision logic and guardrails behind the current project shape.

## High-Level Reasoning Pattern

The project logic is:

```text
define a meaningful severity rubric
-> generate synthetic data from that rubric
-> relabel a real dataset into the same rubric
-> train under multiple data-source setups
-> test only on held-out real images
-> compare what survives and what breaks
```

That is the backbone of the project.

## Main Design Principles

### 1. The ML question matters more than the visual idea

The project should be organized around a strong comparative question, not around whatever synthetic images are easy to generate.

### 2. The target must have a defensible interpretation

The label should mean something real enough that failure or success is interpretable.

That is why the project uses package defect severity rather than a weaker or more artificial task.

### 3. The comparison must isolate the right variable

The CNN architecture stays the same across:

- synthetic-only,
- real-only,
- synthetic + fine-tuning.

That keeps the comparison focused on the data and training strategy rather than on model differences.

### 4. The synthetic pipeline must be controlled

Synthetic images are not generated ad hoc.

They are generated from:

- a rubric,
- a band,
- reviewed prompts,
- a pilot phase,
- then larger batches.

That makes the synthetic data design auditable.

### 5. Real labels must remain human-approved

If the real benchmark is labeled only by another model, the evaluation becomes weak.

So ChatGPT can assist, but it cannot become the ground truth.

### 6. The project should stay from-scratch at its core

The user wants to test learning from scratch.

That is why:

- the main CNNs are custom and trained from scratch,
- pretrained backbones are not part of the core claim.

## What This Project Is Trying To Learn

At the project level, the real questions are:

- how much can synthetic data teach on its own?
- how far is synthetic-only from small real-only?
- how much does a small amount of real fine-tuning help?
- does coarse severity survive shift better than fine severity?

Those are the questions that should shape the experiments, tables, and discussion.

## Guardrails

The project should avoid drifting into:

- "we generated images and trained a model"
- "our synthetic validation accuracy looks good"
- "we used ChatGPT to label everything"
- "we changed multiple things at once so the comparison is unclear"

If the final write-up starts sounding like that, the project has drifted off course.
