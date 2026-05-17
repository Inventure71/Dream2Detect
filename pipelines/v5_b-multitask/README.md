# V5-B - Multitask Scalar, Band, And Coarse Heads

## Question

Can a shared from-scratch model combine scalar ordinal transfer with explicit
10-band and coarse-class supervision?

## Change

V5-B uses a shared backbone with:

- scalar severity head
- 10-band classification head
- optional coarse-class head

The selected scalar checkpoint is evaluated on synthetic test data and raw real
images.

## Implementation

- `scripts/train_multitask_classifier.py`
- `src/dream2detect/training/train_multitask_classifier.py`
- `src/dream2detect/training/models.py`
- `src/dream2detect/training/dataset.py`
- `src/dream2detect/training/metrics.py`

## Main Artifacts

- `data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516`
- `data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516_longer_from160`
- `data/evaluations/model_checkpoint_comparison/v5_b_round1_comparison.csv`
- `data/evaluations/model_checkpoint_comparison/v5_b_longer_comparison.csv`

## Result

V5-B improved broad raw-real ordinal transfer compared with V5-A, but the
longer run did not justify promoting it as a final solved baseline. It is the
latest no-pretrain evidence point in the delivery ladder.

