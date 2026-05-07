# Assignment Constraints

This file maps the current project to the ML course expectations.

## Why This Project Fits The Assignment

The current version fits because it is not just a pipeline.

It is a comparative study with:

- a real ML question,
- multiple training strategies,
- multiple target formulations,
- explicit domain shift,
- documented labeling and evaluation logic.

## What The Teacher Clearly Wants

Based on the guideline and feedback, the teacher wants:

- a meaningful ML problem,
- a clear comparison,
- a justified dataset strategy,
- proper baselines,
- stronger models beyond the trivial baseline,
- rigorous evaluation,
- reflection on why results changed.

The current project matches that structure.

## Where Each Requirement Is Covered

### Problem framing

Covered by:

- [01-project-understanding.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/01-project-understanding.md)
- [03-concepts-explained.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/03-concepts-explained.md)

### Dataset design

Covered by:

- synthetic branch from controlled generation
- real branch from Kaggle relabeling
- human-in-the-loop label approval

Detailed in:

- [06-package-severity-scale-standards.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/06-package-severity-scale-standards.md)
- [07-real-dataset-relabeling.md](/Users/inventure71/VSProjects/School/Dream2Detect/docs/07-real-dataset-relabeling.md)

### Model comparison

Covered by:

- majority / mean trivial baselines
- HOG + logistic regression
- HOG + ridge regression
- scratch CNN classifier
- scratch CNN regressor
- real-only vs synthetic-only vs synthetic + fine-tuning

### Evaluation rigor

The project should report:

- classification metrics for coarse labels
- regression metrics for fine severity
- comparisons across training strategies
- real-test performance, not only synthetic validation
- failure analysis under distribution shift

### Reflection

The final report should explicitly discuss:

- why synthetic-only succeeds or fails,
- why fine vs coarse behaves differently,
- how much limited real data helps,
- where relabeling noise may affect conclusions.

## Why This Is Not Too Easy

This project remains non-trivial because:

- synthetic and real domains differ visibly,
- the real dataset is small and imperfect,
- the labels must be rebuilt carefully,
- the project compares multiple training setups,
- the fine target is genuinely harder than the coarse target.

If the project were reduced to "classify damaged vs not damaged," it would become much weaker.

## Main Risks To Manage

The biggest risks are:

- real labels become noisy or inconsistent,
- synthetic prompts create shortcut-heavy data,
- the real dataset is too small for stable conclusions,
- the final report focuses too much on generation and not enough on the comparison,
- the experiments are not isolated cleanly.

These are manageable if the implementation follows the current plan.

## What Must Be Visible In The Final Submission

To match the assignment well, the final submission should visibly include:

- the comparative research question,
- the severity rubric,
- the relabeling protocol,
- baseline and CNN comparisons,
- real held-out evaluation,
- discussion of domain shift,
- limitations and failure cases.
