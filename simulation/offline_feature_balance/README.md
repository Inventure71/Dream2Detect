# Offline Feature Balance Simulation

This folder contains an offline simulator for synthetic prompt planning diversity.

It does **not**:

- call the OpenAI API,
- draft prompts,
- generate images.

It only:

1. samples structured feature assignments with the current band-aware sampler,
2. simulates a chosen number of prompt plans,
3. writes CSV summaries and an HTML report.

## Run

From the repo root:

```bash
python3 simulation/offline_feature_balance/run.py --total 1000 --seed 7
```

Outputs are written by default to:

- [outputs](/Users/inventure71/VSProjects/School/Dream2Detect/simulation/offline_feature_balance/outputs)

Files:

- `sampled_assignments.csv`
- `axis_counts.csv`
- `band_counts.csv`
- `report.html`

## Purpose

Use this before spending API money.

It helps check whether the feature-axis system is producing balanced diversity across a large synthetic prompt plan.
