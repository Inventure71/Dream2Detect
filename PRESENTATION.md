# V1 - First Synthetic-Only Scratch CNN

This deck explains the project model path from V1 through V5-B. It focuses on
the from-scratch, non-pretrained models. Pretrained diagnostic runs and SAM/SAM3
crop ablations are not treated as core models here.

V1 was the first real training baseline: train a small CNN from scratch on
synthetic package-defect images and see whether the generated labels contained
enough visual signal to learn severity.

The first runs used low-resolution cached images, mostly `128 x 128`, because
they were cheap to train and good for debugging the pipeline. The model was a
from-scratch `simple_cnn`, trained with Adam/AdamW-style optimization, class
balancing, RGB normalization, and basic augmentation. No pretrained backbone was
used.

The first important lesson was that the model could learn the training path, but
it was not yet a strong severity model. Early runs showed class collapse: the
model often predicted only a few classes. That is why we added per-class
precision, recall, F1, predicted-class counts, epoch CSV/JSONL logs, and class
monitoring plots.

After debugging the training loop, we moved from `128 x 128` to `224 x 224`
because damage severity depends on small visual details: crushed corners,
creases, punctures, tape deformation, and edge damage. The best no-new-image
V1-style run reached:

- `224 x 224`, from-scratch `simple_cnn`
- learning rate `0.0003`
- weight decay `0.0001`
- dropout `0.1`
- mild augmentation
- balanced sampler
- early stopping patience `20`
- synthetic test accuracy `0.5102`
- synthetic test macro F1 `0.4862`

Then the 555-image targeted synthetic batch was added to produce an 800-row
combined synthetic manifest. The best V1-style 800-image run used:

- from-scratch `simple_cnn`
- image size `224`
- AdamW
- learning rate `0.0003`
- weight decay `0.0001`
- dropout `0.2`
- mild augmentation
- no pretrained weights

Its result was:

- test accuracy `0.5125`
- test macro F1 `0.5053`

What we learned:

- The synthetic labels were learnable.
- Higher resolution mattered more than expected.
- The model was still fragile across seeds and splits.
- More images alone were not enough; label quality and visual realism became the
  next bottlenecks.

Evidence:

- `docs/12-milestone-log.md`
- `data/training_runs/synthetic_classifier/synthetic_combined_phase1_plus_targeted_555_processed_224_224_lr0003_wd0001_do01_no_sampler_patience60_20260508`

---

## V2 - Full QC, 384 Inputs, Residual CNN, And Real-Gap Synthetic Scale-Up

V2 started after we saw that V1-style synthetic-only training did not transfer
cleanly to real photos. The decision was not to blindly generate more images.
Instead, we first improved label reliability and then generated new synthetic
examples targeted at the real-domain failure modes.

Before the V2 scale-up, we fully QC-reviewed the synthetic pool. The key change
was that training labels no longer came only from intended prompt labels; they
came from visual review. That made the metric more honest, even when it lowered
some earlier optimistic results.

The model also changed:

- image size moved to `384 x 384`
- architecture moved from `simple_cnn` to from-scratch `residual_cnn`
- augmentation changed to `damage_safe`
- ordinal-aware loss was added with weight `0.2`
- no pretrained backbone was used

The best full-QC controlled run before the V2 scale-up was:

- run `E_384_residual_damage_safe_ordinal02`
- synthetic test accuracy `0.5313`
- synthetic test macro F1 `0.5153`
- mean ordinal error `0.6000`
- off-by-one-or-correct rate `0.8875`

Then we evaluated that synthetic-only checkpoint on real images. It failed to
transfer cleanly:

- raw real accuracy `0.2927`
- raw real macro F1 `0.2574`
- padded real accuracy `0.3110`
- padded real macro F1 `0.2445`

That result drove the actual V2 design decision: generate synthetic data that
targets real failures, especially damaged real packages predicted as intact,
flat mailers, subtle minor/moderate damage, realistic tape/labels/clutter, and
severe damage that still looks photographic.

After adding 132 QC-reviewed V2 scale-up examples, the V2 controlled training
run used:

- manifest `synthetic_full_qc_plus_v2_scale_processed_384.csv`
- rows `899`
- model `residual_cnn`
- image size `384`
- augmentation `damage_safe`
- AdamW, learning rate `0.0003`
- dropout `0.2`
- ordinal loss weight `0.2`
- no pretrained weights

V2 result:

- synthetic test accuracy `0.5333`
- synthetic test macro F1 `0.5239`
- synthetic off-by-one-or-correct rate `0.9111`
- synthetic severe ordinal error rate `0.0889`
- padded-real transfer accuracy stayed `0.3110`
- padded-real transfer macro F1 improved from `0.2445` to `0.2689`
- padded-real off-by-one-or-correct improved from `0.6768` to `0.7195`

What we learned:

- Full QC and 384-resolution residual training were real improvements.
- V2 targeted synthetic data improved synthetic performance and some real
  transfer diagnostics.
- But the direct real accuracy did not move. Synthetic-only training still did
  not solve the real-domain gap.
- The next step had to be better model/training structure and stricter split
  discipline, not just more generic synthetic images.

Evidence:

- `data/evaluations/real_transfer/v2_scale_comparison_20260513.csv`
- `data/training_runs/synthetic_classifier/v2_scale_384_residual_damage_safe_ordinal02_20260513`

---

## V3 - Training Stack Experiments Around The V2 Baseline

V3 asked whether the V2 model could be improved by changing the training stack
while keeping the project from-scratch constraint.

The hypotheses were:

- GroupNorm might be more stable than BatchNorm for small batches.
- A true ordinal coarse-class objective might handle severity order better than
  plain four-class cross entropy.
- Constrained RandAugment-style augmentation might improve generalization
  without destroying damage labels.

Implemented V3 options:

- `residual_cnn_groupnorm`
- `coarse_ordinal`
- `damage_safe_ra_low`
- `damage_safe_ra_medium`
- `damage_safe_ra_high`
- analyzer support for ordinal-coarse checkpoints

The selected V3 candidate was not the most exotic branch. It was still the
BatchNorm residual model, but with low-strength damage-safe RandAugment and no
balanced sampler:

- run `v3_bn_ra_low_coarse_ord02_nosampler_20260513`
- model `residual_cnn`
- augmentation `damage_safe_ra_low`
- target `coarse`
- ordinal loss weight `0.2`
- balanced sampler `False`
- no pretrained weights

V3 result:

- synthetic test accuracy `0.5500`
- synthetic test macro F1 `0.5496`
- off-by-one-or-correct rate `0.9167`
- severe ordinal error rate `0.0833`

Comparison against V2:

- V2 macro F1 `0.5239`
- V3 macro F1 `0.5496`
- V2 off-by-one-or-correct `0.9111`
- V3 off-by-one-or-correct `0.9167`

The GroupNorm and true ordinal-coarse branches did not win:

- GroupNorm branch macro F1 `0.2457`
- true ordinal coarse branch macro F1 `0.3062`

What we learned:

- Constrained augmentation helped when added to the proven residual baseline.
- GroupNorm was a reasonable hypothesis, but the data did not support it at this
  stage.
- The true ordinal coarse path was implemented and useful for learning, but not
  the new baseline.
- We should promote changes only when they survive comparison, not because they
  sound theoretically cleaner.

Evidence:

- `data/evaluations/synthetic_only/v3_synthetic_training_comparison_20260513.csv`
- `data/training_runs/synthetic_classifier/v3_bn_ra_low_coarse_ord02_nosampler_20260513`

---

## V3.1 - Harder Split Discipline

V3.1 was not mainly about a new architecture. It was about making the evaluation
harder and more credible.

The earlier random stratified split could allow visually related synthetic
prompt families into train, validation, and test. That risks optimistic
synthetic metrics because the model may learn prompt-family style rather than
general package-damage structure.

So V3.1 added:

- split strategy `metadata_family_holdout`
- family key built from:
  - `training_coarse_class`
  - `damage_profile_primary`
  - `box_form_factor`
  - `background_context`

The selected hard-split baseline was:

- from-scratch `residual_cnn`
- augmentation `damage_safe`
- dropout `0.1`
- weight decay `0.0001`
- ordinal loss weight `0.2`
- balanced sampler `False`
- split strategy `metadata_family_holdout`
- no pretrained weights

Seed-42 hard-split results:

- dropout `0.1`: test macro F1 `0.5607`
- dropout `0.2`: test macro F1 `0.5983`
- dropout `0.3`: test macro F1 `0.5953`

But the model-selection decision used validation stability, not one lucky test
score. The `dropout=0.1` configuration had the most defensible validation
behavior across seeds:

- seed 42 validation macro F1 `0.5869`
- seed 43 validation macro F1 `0.5964`
- seed 43 test macro F1 `0.5309`

What we learned:

- Split discipline matters as much as model choice.
- The harder split reduced the chance of prompt-family leakage.
- Plain `damage_safe` augmentation became preferable under the harder split,
  even though `damage_safe_ra_low` had looked better under the earlier regime.
- The next target shift should use this harder split as the default.

Evidence:

- `data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv`
- `data/evaluations/synthetic_only/v3_1_hard_split_baseline_20260513.csv`

---

## V4 - Official 10-Band Score-Band Target

V4 changed the target of the project. The four coarse classes were useful, but
the real severity rubric is the official 10-band score system:

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

The decision was to stop treating the four-class task as the primary result and
make `score_band` the main target. Four-class metrics stayed as collapsed
diagnostics.

V4 kept the from-scratch model family but changed target handling:

- model `residual_cnn`
- image size `384`
- augmentation `damage_safe`
- split `metadata_family_holdout`
- target `score_band`
- soft ordinal targets across the 10 bands
- effective-number class-balanced weighting
- ordinal loss weight `0.2`
- checkpoint selection by validation mean band error
- no pretrained weights

The V4 baseline result on synthetic test:

- exact 10-band accuracy `0.1990`
- 10-band macro F1 `0.1791`
- mean band error `1.3155`
- `+-1` band accuracy `0.7233`
- collapsed 4-class macro F1 `0.4557`

The raw-real transfer result:

- exact 10-band accuracy `0.1065`
- `+-1` band accuracy `0.3325`
- mean band error `2.6649`

What we learned:

- V4 was a legitimate score-band baseline, not a broken run.
- Exact 10-band accuracy was low, but `+-1` accuracy was high on synthetic test.
  The model often found the right neighborhood but not the exact neighboring
  band.
- Raw-real transfer was still poor, so synthetic score-band success did not
  automatically solve domain shift.
- The next changes should stay narrow and test whether ordinal loss design
  improves band distance behavior.

Evidence:

- `data/training_runs/synthetic_classifier/synthetic_full_qc_plus_v2_scale_processed_384_384_20260514_192919`
- `data/evaluations/model_checkpoint_comparison/v4_v45_v5_within_band_comparison.csv`

---

## V4.5 - Hybrid Soft-Label Cross-Entropy Plus EMD

V4.5 tested one narrow idea: if score bands are ordered, the loss should punish
far-away mistakes more than near misses.

The rule was to keep the V4 stack fixed and change only the objective:

- same from-scratch `residual_cnn`
- same `metadata_family_holdout`
- same `damage_safe` augmentation
- same `score_band` target
- same soft-label sigma `1.0`
- same effective-number beta `0.999`
- same ordinal loss weight `0.2`
- add cumulative squared EMD with `score_band_emd_weight=0.5`
- no pretrained weights

V4.5 result on synthetic test:

- exact 10-band accuracy `0.1748`
- macro F1 `0.1574`
- mean band error `1.3398`
- `+-1` band accuracy `0.7330`
- collapsed 4-class macro F1 `0.4439`

Comparison to locked V4:

- validation mean band error: V4 `1.1627`, V4.5 `1.2229`
- test mean band error: V4 `1.3155`, V4.5 `1.3398`
- test macro F1: V4 `0.1791`, V4.5 `0.1574`
- test `+-1` accuracy: V4 `0.7233`, V4.5 `0.7330`

Raw-real transfer:

- V4 real `+-1` accuracy `0.3325`, MBE `2.6649`
- V4.5 real `+-1` accuracy `0.3532`, MBE `2.5351`

What we learned:

- The EMD objective slightly improved tolerance-based real transfer and
  synthetic `+-1` accuracy.
- It did not cleanly beat V4 on validation MBE, synthetic test MBE, macro F1, or
  collapsed coarse macro F1.
- V4.5 was useful evidence, but not a new baseline.
- The bottleneck was probably not just sigma or adding one ordinal loss term.
  The next major axis should be target/head formulation.

Evidence:

- `data/training_runs/synthetic_classifier/v4_5_score_band_emd05_seed42_20260515`
- `data/evaluations/model_checkpoint_comparison/v4_v45_v5_within_band_comparison.csv`

---

## V5-A - Scalar Ordinal Regression

V5-A changed the target/head formulation. Instead of treating severity as a
10-class classification problem, the model predicts a normalized scalar
severity value.

The reason was simple: package damage is ordered. A scalar head may learn the
direction and distance of severity better than a flat 10-way head, especially
when neighboring bands are visually ambiguous.

V5-A design:

- target `training_representative_score / 100.0`
- output constrained with sigmoid to `[0, 1]`
- convert predicted scalar back to score and then to official 10-band label for
  evaluation
- model `residual_cnn_groupnorm_regressor`
- image size `384`
- augmentation `damage_safe`
- split `metadata_family_holdout`
- AdamW, learning rate `0.0003`
- weight decay `0.0001`
- dropout `0.1`
- validation selector `val_mean_band_error`
- no pretrained weights

Before the full run, V5-A had to pass sanity checks:

- target conversion tests
- score-to-band mapping tests
- tiny-subset overfit gate
- saved split group overlap checks
- checkpoint/resume checks

The 4-example overfit gate reached:

- score MAE `3.31`
- mean band error `0.25`
- `+-1` band accuracy `1.0000`
- collapsed coarse macro F1 `1.0000`

Full V5-A synthetic test result:

- exact 10-band accuracy `0.1685`
- macro F1 `0.1529`
- mean band error `1.7079`
- `+-1` band accuracy `0.5225`
- collapsed coarse macro F1 `0.4285`

Raw-real transfer result:

- exact 10-band accuracy `0.1351`
- macro F1 `0.0923`
- mean band error `2.2727`
- `+-1` band accuracy `0.3974`
- collapsed coarse macro F1 `0.2375`

What we learned:

- V5-A was mechanically correct and passed the sanity gates.
- It was worse than V4 on synthetic holdout, so scalar regression did not solve
  the synthetic 10-band task.
- It transferred better to raw real images than V4/V4.5 on mean band error and
  `+-1` accuracy.
- That tradeoff is important: optimizing synthetic exact band classification
  was not the same as improving real-domain ordinal severity.
- Longer training did not improve the best checkpoint; the original best epoch
  remained `79`.

Evidence:

- `docs/21-v5-design-plan.md`
- `data/training_runs/synthetic_regressor/v5_a_scalar_ordinal_seed42_20260516_fixed`
- `data/evaluations/model_checkpoint_comparison/v4_v45_v5_within_band_comparison.csv`

---

## V5-B - Multitask Scalar, Band, And Coarse Heads

V5-B tested whether a shared backbone could combine the strengths of scalar
ordinal learning and discrete label supervision.

The decision came from V5-A's tradeoff. V5-A improved real transfer but lost too
much synthetic score-band performance. V5-B therefore kept the scalar head but
added auxiliary heads:

- scalar normalized severity head
- auxiliary 10-band score-band head
- auxiliary 4-class coarse head

The model remained from scratch:

- `residual_cnn_groupnorm_multitask`
- image size `384`
- split `metadata_family_holdout`
- augmentation `damage_safe`
- AdamW, learning rate `0.0003`
- weight decay `0.0001`
- dropout `0.1`
- no pretrained weights

Loss weights:

- scalar loss `1.0`
- coarse loss `0.3`
- auxiliary 10-band loss `0.3`
- auxiliary band ordinal penalty `0.2`

V5-B synthetic test result from the scalar head:

- exact 10-band accuracy `0.1461`
- `+-1` band accuracy `0.4775`
- `+-2` band accuracy `0.6910`
- mean band error `1.9663`
- collapsed coarse macro F1 `0.3881`

V5-B raw-real result from the scalar head:

- exact 10-band accuracy `0.1506`
- `+-1` band accuracy `0.3948`
- `+-2` band accuracy `0.6260`
- `+-3` band accuracy `0.8130`
- `+-4` band accuracy `0.9325`
- mean band error `2.1091`

Comparison to V5-A:

- synthetic MBE worsened: V5-A `1.7079`, V5-B `1.9663`
- real MBE improved: V5-A `2.2727`, V5-B `2.1091`
- real `+-1` stayed effectively similar: V5-A `0.3974`, V5-B `0.3948`
- real wider tolerances improved:
  - `+-2`: V5-A `0.5896`, V5-B `0.6260`
  - `+-3`: V5-A `0.7584`, V5-B `0.8130`
  - `+-4`: V5-A `0.8935`, V5-B `0.9325`

Longer V5-B training did not help:

- validation improved later
- synthetic test MBE worsened from `1.9663` to `2.0281`
- raw-real MBE worsened from `2.1091` to `2.2442`
- raw-real `+-1` worsened from `0.3948` to `0.3532`

What we learned:

- Multitask learning helped broader real-domain ordinal robustness.
- It did not improve the synthetic 10-band baseline.
- The auxiliary 10-band head helped exact band classification inside V5-B, but
  the scalar head remained the better ordinal estimator.
- The validation signal was not reliable enough for longer-run model selection.
- V5-B is useful evidence, but not final. The next step should tune multitask
  loss weights or test a threshold ordinal model, not simply train longer.

Evidence:

- `docs/21-v5-design-plan.md`
- `data/training_runs/synthetic_multitask/v5_b_multitask_scalar_band_coarse_seed42_20260516`
- `data/evaluations/model_checkpoint_comparison/v5_b_round1_comparison.csv`
- `data/evaluations/model_checkpoint_comparison/v5_b_longer_comparison.csv`

V1 proved that synthetic severity labels were learnable but fragile. V2 showed
that QC, higher resolution, residual capacity, and targeted data improved the
synthetic task but did not close the real-domain gap. V3 showed that not every
"better" architectural idea actually helped, and V3.1 made the synthetic split
harder and more honest. V4 aligned the target with the official 10-band rubric.
V4.5 showed that adding EMD helped some tolerance metrics but was not a clean
baseline replacement. V5-A and V5-B showed the most important tension: the
models that looked strongest on synthetic holdout were not necessarily the ones
that transferred best to raw real photos.

The clearest project answer is:

synthetic-only training can learn package-damage severity, but synthetic
holdout performance is not enough. The real challenge is synthetic-to-real
generalization. Our best from-scratch path improved by making labels cleaner,
inputs larger, splits harder, losses more ordinal, and heads more aligned with
severity. Even then, real-domain performance remained the limiting factor.
