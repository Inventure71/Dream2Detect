# Dream2Detect Final Report Draft

Status: draft for project report writing.

This document explains the current Dream2Detect project as a final-report
source document. It is intentionally detailed. A later version can be shortened
for a slide deck, submission report, or Google Doc.

## 0. Executive Summary

Dream2Detect studies whether synthetic package/cardboard damage images can
train models that transfer to real package photos. The target is visible damage
severity, not exact physical breakage. To make the problem consistent across
synthetic and real data, the project used a shared 10-band `score_band` rubric
with 4 derived coarse classes.

The synthetic dataset was built from structured prompt metadata, not random
prompting. It was balanced across severity and visual conditions, then reviewed
image by image through QC. Real images from Kaggle were relabeled into the same
rubric with human-approved final labels.

The no-pretrain model path evolved from V1 to V5-B. V1 proved that synthetic
labels were learnable. V2 improved label quality, resolution, and architecture.
V3 and V3.1 improved the training stack and made synthetic evaluation stricter.
V4 changed the task to the official 10-band severity target. V4.5 tested a
distance-aware loss. V5-A and V5-B tested scalar and multitask ordinal heads.

The central result is that synthetic-only training worked on synthetic holdout,
but real transfer remained the bottleneck. The best no-pretrain raw-real result
in this report is V5-B at 15.06% exact 10-band accuracy, 39.48%
within-one-band accuracy, and 2.1091 mean band error. The best reported
pretrained `main` branch ensemble, `EMD60+SAM40`, reached 21.04% exact,
64.16% within-one-band, and 1.3351 mean band error on the same 385 real
images.

The final conclusion is that synthetic data was useful and learnable, but
dataset quality alone was not enough. Strong real-image performance required
better visual representations, and the pretrained solution clearly outperformed
the no-pretrain path on real photos.

## 0.1 Best Results At A Glance

### Best No-Pretrain Result

| Model | Eval set | Exact | Within-one-band | Mean band error |
|---|---|---:|---:|---:|
| V5-B multitask scalar head | real photos, all 385 | 15.06% | 39.48% | 2.1091 |

### Best Pretrained Result

| Model | Eval set | Exact | Within-one-band | Mean band error |
|---|---|---:|---:|---:|
| EMD60+SAM40 ensemble | real photos, all 385 | 21.04% | 64.16% | 1.3351 |

### Best Intermediate No-Pretrain Milestones

| Stage | Why it mattered | Best result |
|---|---|---|
| V1 | proved synthetic labels were learnable | 800-image synthetic coarse macro F1 0.5053 |
| V2 | first strong full-QC residual baseline | synthetic macro F1 0.5239; padded-real macro F1 0.2689 |
| V3 | best synthetic coarse classifier stack | synthetic macro F1 0.5496 |
| V4 | first official 10-band baseline | synthetic MBE 1.3155; real MBE 2.6649 |
| V5-A | first real scalar ordinal transfer jump | real MBE 2.2727 |
| V5-B | best no-pretrain overall raw-real score in final path | real MBE 2.1091 |

## 0.2 Metric Definitions

| Metric | Meaning | How to read it |
|---|---|---|
| Accuracy | Exact class or band match rate | Higher is better |
| Macro F1 | Mean F1 across classes, weighting classes equally | Higher is better; useful with imbalance |
| Exact 10-band accuracy | Prediction must land in the exact correct severity band | Hardest score-band metric |
| Within-one-band accuracy | Prediction is exact or one neighboring band away | Higher is better; more tolerant ordinal metric |
| Mean band error | Average distance between predicted and true band indices | Lower is better |
| Synthetic holdout | Test split drawn from synthetic data | Measures generalization within synthetic distribution |
| Raw-real transfer | Evaluation of a synthetic-trained model directly on real photos | Measures actual transfer |
| Collapsed coarse metric | Convert 10 bands back into 4 coarse classes for comparison | Easier than exact 10-band prediction |

## 0.3 How To Present This Project

If someone has never seen the project, the clearest presentation story is:

> We built a controlled synthetic-to-real experiment for package damage
> severity. We first made the labels consistent, then built and QC-reviewed a
> balanced synthetic dataset, then tested how far from-scratch CNNs could go,
> and finally compared that no-pretrain path with a stronger pretrained
> ConvNeXt-based solution.

The report should not be presented as "we generated images and trained a
classifier." That is too vague and misses the actual engineering work. The
important story is that the dataset, the labels, and the evaluation protocol
were built to answer a specific transfer question.

The one-slide thesis is:

> Synthetic data can teach visible package-damage severity, but synthetic
> holdout accuracy is not enough. The hardest problem is real-domain transfer.
> Dataset QC and balancing helped, from-scratch models improved over time, and
> pretrained visual backbones gave the strongest real-image results.

The audience needs to understand five ideas before the model versions make
sense:

1. The target is visual package condition, not exact physical damage.
2. Labels are band-first: `score_band` is primary; coarse class and numeric
   score are derived.
3. Synthetic images were generated from structured, balanced prompt metadata,
   not random free-text prompts.
4. Generated labels were reviewed and corrected before training.
5. Synthetic test results and real-photo transfer results are different
   questions.

### 0.4 Minimal Glossary

| Term | Meaning |
|---|---|
| Synthetic image | AI-generated package/cardboard image used for training. |
| Real image | Kaggle package/cardboard photo relabeled under the project rubric. |
| `score_band` | Primary 10-band severity label, for example `36-45`. |
| Coarse class | Derived 4-class label: `intact`, `minor`, `moderate`, or `severe`. |
| Representative score | Fixed numeric value assigned to each band for regression support. |
| Full QC | Image-level review where labels can be accepted, corrected, or rejected. |
| Synthetic holdout | Test split from synthetic data. Useful, but not proof of real transfer. |
| Raw-real transfer | Evaluating a synthetic-trained model directly on real photos. |
| Mean band error | Average distance between predicted and true 10-band indices. Lower is better. |
| Within-one-band accuracy | Percentage of predictions that are exact or one band away. |
| Metadata-family holdout | Split policy that keeps related prompt families out of multiple splits. |
| Pretrained backbone | Vision model initialized from external image pretraining rather than random weights. |

### 0.5 Suggested Presentation Structure

Recommended slide flow for a normal class presentation:

1. Project question:
   - Can synthetic package-damage images train a model that transfers to real
     package photos?

2. Why this is hard:
   - Synthetic-to-real transfer under domain shift.
   - Visible severity is ordered, ambiguous, and not well captured by a single
     exact score.

3. Label system:
   - Explain band-first labels, 10 fine bands, 4 coarse classes, and
     representative scores.

4. Dataset creation:
   - One slide only, or two at most.
   - Explain structured prompt balancing, image generation, and QC.
   - Show only the final counts that matter.

5. Real dataset:
   - Explain Kaggle source, human-approved relabeling, and the 385-image real
     set.

6. Evaluation metrics:
   - Explain exact accuracy, macro F1, mean band error, and within-one-band
     accuracy.

7. From-scratch model evolution:
   - Present V1 to V5-B as a sequence of questions and answers, not just a list
     of models.

8. From-scratch summary table:
   - Use the table in Section 7.

9. Pretrained solution:
   - Explain why ConvNeXt/EMD/SAM/ensembles were tested in `main`.

10. Final comparison:
    - Compare no-pretrain V5-B against pretrained EMD/SAM ensemble on real
      images.

11. Conclusion:
    - Dataset quality helped, synthetic data was learnable, but real transfer
      required stronger visual representations.

### 0.6 Suggested Importance Split

For presentation purposes, the material should not be weighted evenly. A
plausible division is:

| Topic | Suggested share of talk | Why |
|---|---:|---|
| Problem and task definition | 10% | The audience needs to understand the question first. |
| Label system and evaluation | 10% | Without band-first labels and ordinal metrics, the results are hard to interpret. |
| Dataset creation | 15% | Important, but only as setup for the model story. |
| From-scratch model evolution V1 to V5-B | 45% | This is the main experimental story of the project. |
| Pretrained `main` branch solution | 15% | This is the strongest final performance result and the natural comparison point. |
| Final comparison and conclusion | 5% | End with the actual answer to the project question. |

That means the dataset should be presented as enabling context, not as the main
story. The main story is the sequence of modeling questions and what each
version taught us.

### 0.7 Short Presentation Version

If the talk is short, compress the dataset material to this:

- We defined a shared 10-band severity rubric.
- We generated synthetic images from structured, balanced prompts.
- We QC-reviewed every generated image and corrected labels when needed.
- We relabeled 385 real Kaggle images into the same rubric.

Then move immediately into the model versions.

### 0.8 What Not To Claim

Avoid these claims:

- Do not say the synthetic-only from-scratch model solved real-image severity.
  It improved, but raw-real transfer remained weak.
- Do not say the generated prompt label was always ground truth. Labels were
  corrected through image-level QC.
- Do not say exact 10-band accuracy is the only metric. For ordered severity,
  mean band error and within-one-band accuracy are essential.
- Do not mix the no-pretrain V1 to V5-B path with the pretrained `main` branch
  solution. They answer related but different questions.
- Do not say the pretrained models were "trained from scratch." They were
  task-trained on synthetic images, but the backbones were pretrained.
- Do not use raw-real results as if they were validation-tuned model-selection
  metrics for V4/V5. They were reported as transfer evaluation.

## 1. Project Summary

Dream2Detect studies whether models trained on synthetic package/cardboard
damage images can generalize to real photos. The project target is not literal
physical breakage percentage. The target is overall visible package defect
severity: how damaged the package looks in the image.

The central project question is:

> Can synthetic data teach a model to estimate visible package/cardboard damage
> severity well enough to transfer to real images?

The project evolved through three linked tracks:

1. Dataset construction:
   - define a shared severity rubric,
   - create synthetic images with structured prompt balancing,
   - review and relabel generated images with image-level QC,
   - relabel real package photos into the same rubric.

2. From-scratch model development:
   - train custom CNN models without pretrained backbones,
   - move from simple 4-class classification to a 10-band severity target,
   - test ordinal losses, scalar severity regression, and multitask heads.

3. Pretrained model solution in the `main` branch:
   - use pretrained ConvNeXt-style backbones and ensemble methods,
   - evaluate on the 385 real photos,
   - compare against the from-scratch synthetic-only models.

The main conclusion is that synthetic data was learnable and useful, but the
hard part was synthetic-to-real generalization. The from-scratch synthetic-only
models improved as labels became cleaner, images became higher resolution,
splits became stricter, and losses became more ordinal. However, their real
performance stayed much weaker than their synthetic holdout performance. The
pretrained solution in `main` performed substantially better on real photos,
which strongly suggests that representation quality matters for this project.

## 2. Severity Rubric And Label System

The first important dataset decision was to define the label system before
training. We did not want separate label meanings for synthetic images and real
images. Both had to use the same severity scale.

The primary label is `score_band`, not an exact numeric score. The project uses
10 ordered bands:

| Band | Coarse class | Representative score |
|---|---:|---:|
| `0-10` | `intact` | 5 |
| `11-20` | `minor` | 15 |
| `21-30` | `minor` | 25 |
| `31-35` | `minor` | 33 |
| `36-45` | `moderate` | 40 |
| `46-55` | `moderate` | 50 |
| `56-65` | `moderate` | 60 |
| `66-75` | `severe` | 70 |
| `76-85` | `severe` | 80 |
| `86-100` | `severe` | 93 |

The coarse mapping is:

| Fine score range | Coarse class |
|---|---|
| `0-10` | `intact` |
| `11-35` | `minor` |
| `36-65` | `moderate` |
| `66-100` | `severe` |

The most important rule is band-first labeling. We assign a band from visible
evidence, then derive the coarse class and representative score. This avoids
fake precision. A generated or real image should not be labeled as exactly
`47`; it should be labeled as the nearest severity band, for example `46-55`.

This mattered because the project later used both classification and regression
models. A classifier could train on the band or coarse class. A regressor could
train on the representative score. But all targets still came from the same
band decision.

## 3. Synthetic Dataset Generation

Presentation note: for a report, this section can stay detailed. For a live
presentation, this should usually be compressed to one synthetic slide and one
real-data slide, with the rest moved to appendix if needed.

### 3.1 Why The Dataset Was Not Random Generation

The synthetic dataset was not created by simply asking an image model for random
damaged boxes. That would make the dataset hard to reproduce and likely create
shortcuts. For example, if every severe image were generated with dark lighting
or every intact image used a clean studio background, the model could learn the
background instead of the damage.

Instead, the dataset was generated from structured prompt rows. Each planned
image carried explicit metadata fields. The prompt text was only one part of
the record.

The structured fields included:

- `score_band`
- `coarse_class`
- `representative_score`
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

This meant that an image was not just "a damaged box." It was a specific
combination of severity, damage type, box shape, surface appearance, labels,
tape, background, camera angle, and lighting.

### 3.2 Balancing System

The balancing system worked at multiple levels.

First, we balanced severity. The project needed examples across the full
severity range, from intact to severe. The synthetic dataset therefore tracked
both the 10 `score_band` labels and the 4 derived coarse classes. This prevented
the model from mostly seeing one severity group.

Second, we balanced visual conditions inside each severity level. For each
band, the prompt planner varied damage type, damage location, box form factor,
cardboard pattern, label presence, tape style, background, camera angle, and
lighting. This was meant to reduce shortcut learning. A good severity model
should not associate "warehouse floor" with one class or "front view" with one
severity.

Third, we balanced around real failure modes after we had early real-transfer
results. When early synthetic-only models failed on real photos, we used those
failures to choose new synthetic scenarios. The V2 synthetic generation round
targeted cases the model was missing, including:

- damaged real packages predicted as `intact`,
- flat taped mailers,
- subtle minor damage,
- moderate damage visible under realistic phone-photo conditions,
- realistic tape, labels, clutter, and shadows,
- severe openings or collapse that still looked photographic,
- hard intact negatives with tape, labels, clutter, shadows, and scuffs but no
  structural damage.

Fourth, we corrected balancing after quality control. The planned label was not
the final label unless the generated image actually matched it. If a prompt
planned a moderate image but the generated image visually looked minor, the
final training label was corrected to minor. So the final balance came from
reviewed labels, not just intended prompt labels.

### 3.3 Prompt Drafting And Image Generation

Prompt drafting used `gpt-5.4-mini`. The prompt model did not invent the label
system from scratch. It drafted prompt text from the locked band definitions and
the structured feature assignments.

The image generation model was `gpt-image-2`. The initial pipeline used:

- quality: `low`
- size: `1024x1024`

The generation process was staged:

1. Define the severity band and structured feature assignment.
2. Draft the prompt from the band specification and feature assignment.
3. Review prompt candidates before generating images.
4. Generate a pilot batch first.
5. Inspect the pilot images against the severity rubric.
6. Revise prompts or rules if the images did not match.
7. Scale to larger batches only after the pilot was acceptable.

This staging was important because image generation is not guaranteed to follow
the target label. A prompt asking for `36-45` moderate damage can produce an
image that looks too minor, too severe, too stylized, or not realistic enough.

### 3.4 Image-Level QC

After image generation, every synthetic image needed an explicit quality-control
decision. The project did not blindly trust the intended prompt label.

The QC states were:

- `accepted_as_labeled`: the image matched the intended band well enough.
- `accepted_relabel`: the image was usable, but the visible severity belonged
  to a different band or class.
- `rejected`: the image was not suitable for training.

Reasons to reject or relabel included:

- the generated image was render-like, stylized, or product-ad-like,
- the visible defect was hidden or not actually present,
- the image showed a different severity than the intended band,
- an intact negative contained structural damage,
- readable labels, logos, addresses, or large text dominated the image,
- the image did not look like an ordinary package photo,
- the damage was semantically different from the manifest description.

This QC step made the dataset harder but more honest. Earlier intended-label
results could look better because they measured whether the model learned the
prompt plan, not necessarily the visible image. Full-QC training measured the
actual reviewed visual labels.

### 3.5 Synthetic Dataset Versions And Counts

The project started small. Early training used a 45-row reviewed seed set and
then a 200-image scale-up. The first larger combined training manifest had 800
rows:

- 45 from the phase-1 seed set,
- 200 from the scale-up batch,
- 555 from the targeted synthetic batch.

The initial 800-row processed manifest was intentionally class-balanced by
coarse class:

| Coarse class | Rows |
|---|---:|
| `intact` | 200 |
| `minor` | 200 |
| `moderate` | 200 |
| `severe` | 200 |

After full image-level QC, the paper-facing 800-image source manifest had:

| QC outcome | Rows |
|---|---:|
| accepted as labeled | 623 |
| accepted with relabeling | 177 |
| total accepted | 800 |

The full-QC 800-image coarse distribution became:

| Coarse class | Rows |
|---|---:|
| `intact` | 216 |
| `minor` | 192 |
| `moderate` | 189 |
| `severe` | 203 |

This was no longer perfectly balanced, but it was more honest because it used
reviewed visual labels.

Later, the V2 real-failure-targeted generation added 132 accepted synthetic
images. The expanded synthetic training manifest had 899 accepted rows:

| Coarse class | Rows |
|---|---:|
| `intact` | 220 |
| `minor` | 228 |
| `moderate` | 229 |
| `severe` | 222 |
| total | 899 |

The key point is that the final synthetic dataset was not just larger. It was
structured, reviewed, and targeted toward known real-domain failure modes.

### 3.6 Online Augmentation And Effective Dataset Size

The project does not create separate augmented image files on disk. Augmentation
is applied online during training inside the PyTorch transform pipeline. This
means the stored dataset size does not grow when augmentation is enabled. The
manifest and cached processed images remain deterministic base samples, while
each training epoch can present a new random view of those same base images.

This distinction matters because "dataset size" can mean three different
things:

- physical dataset size: how many accepted base images are stored in the
  manifest;
- train split size: how many of those base rows land in the training split for
  a given run;
- effective training draws: train rows multiplied by the number of epochs
  actually run.

In other words, V4 and V5 do not physically contain hundreds of thousands of
images. They contain 899 accepted synthetic base images, of which roughly
527-534 are in the training split depending on the split policy for that run.
What grows into the tens of thousands is the number of stochastic training
draws, not the number of stored files.

The augmentation profiles used by the training code are defined in
`src/dream2detect/training/dataset.py`. Technically, the profiles are:

| Profile | Stochastic steps | Technical definition |
|---|---:|---|
| historical `128` | 0 | No named augmentation profile was recorded for the early `128x128` baseline. In practical terms this should be treated as resize, `ToTensor()`, and RGB normalization, with no stochastic augmentation steps. |
| `mild` | 2 | `RandomHorizontalFlip(p=0.5)` plus `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)`. |
| `strong` | 6 | `RandomResizedCrop(scale=(0.82, 1.0), ratio=(0.9, 1.1))`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=8)`, `RandomPerspective(distortion_scale=0.08, p=0.25)`, `ColorJitter(brightness=0.18, contrast=0.18, saturation=0.08, hue=0.02)`, and optional `GaussianBlur(kernel_size=3, sigma=(0.1, 0.8), p=0.15)`. |
| `damage_safe` | 7 | `RandomHorizontalFlip(p=0.5)`, `RandomAffine(degrees=5, translate=(0.035, 0.035), scale=(0.94, 1.06), shear=(-2, 2), fill=245)`, `RandomPerspective(distortion_scale=0.05, p=0.2)`, `ColorJitter(brightness=0.14, contrast=0.16, saturation=0.07, hue=0.015)`, optional `GaussianBlur(kernel_size=3, sigma=(0.1, 0.55), p=0.12)`, `AddGaussianNoise(std=0.012, p=0.15)`, and `RandomErasing(p=0.08, scale=(0.004, 0.018), ratio=(0.4, 2.5))`. |

All profiles end with tensor conversion and RGB normalization. The later
versions mainly use `damage_safe`, because the project wanted augmentation that
was strong enough to increase robustness but less likely to destroy the visual
semantics of package damage.

The verified effective dataset sizes for the main training stages are:

| System | Stored rows | Train rows | Aug profile | Epochs run | Training draws |
|---|---:|---:|---|---:|---:|
| `128` historical baseline | 245 | 147 | none recorded | 60 | 8,820 |
| `224` full-QC classifier path | 800 | 480 | `mild` | 83 | 39,840 |
| V3.1 score-band ordinal classifier | 800 | 480 | `damage_safe` | 66 | 31,680 |
| V4 locked score-band model | 899 | 527 | `damage_safe` | 107 | 56,389 |
| V4.5 EMD challenger | 899 | 527 | `damage_safe` | 101 | 53,227 |
| V5-A scalar ordinal regressor | 899 | 534 | `damage_safe` | 119 | 63,546 |
| V5-B multitask scalar-band-coarse | 899 | 534 | `damage_safe` | 160 | 85,440 |
| V5-B longer resume run | 899 | 534 | `damage_safe` | 320 | 170,880 |

So the technical answer is:

- the physical synthetic dataset for the final V4/V5 experiments is 899 base
  images, not tens of thousands of saved augmented files;
- the train split for those runs is about 527-534 rows depending on the run;
- the large effective sample count comes from repeated online stochastic views
  across many epochs;
- the number of possible augmented views is extremely large in practice, but
  those views are not stored, indexed, or guaranteed to be unique.

## 4. Real Dataset Relabeling

The real dataset came from Kaggle:

`redf0xwin/recognizing-defects-in-boxes-and-cardboard`

The original XML annotations were kept as provenance, but they were not used as
the final ground truth for Dream2Detect. Those annotations did not directly
match the project's whole-image visible severity rubric.

The project therefore relabeled the real images into the same band-first system
used by the synthetic data.

The real relabeling workflow was:

1. Merge candidate Kaggle images into one real-image pool.
2. Inspect each image visually.
3. Use ChatGPT Vision as a labeling assistant.
4. Ask for a suggested `score_band`, `coarse_class`, representative score, and
   reasoning.
5. Require human approval or correction for the final label.
6. Flag uncertain or boundary cases when needed.
7. Derive the final coarse class and representative score from the approved
   band.

ChatGPT Vision was not the source of ground truth by itself. It was only an
assistant. The final label required human approval or correction.

The current real labeled dataset contains 385 labeled rows. Its class
distribution is naturally imbalanced:

| Coarse class | Rows |
|---|---:|
| `intact` | 8 |
| `minor` | 82 |
| `moderate` | 237 |
| `severe` | 58 |
| total | 385 |

This real distribution is important because it is very different from the
balanced synthetic training distribution. The synthetic dataset was constructed
to cover classes evenly. The real dataset reflects the source data, where
moderate damage dominates and intact examples are rare.

## 5. Evaluation Principles

The project used several evaluation metrics because exact accuracy alone is too
weak for an ordered severity problem.

For 4-class coarse models, the key metrics were:

- accuracy,
- macro F1,
- off-by-one-or-correct rate,
- severe ordinal error rate.

For 10-band score-band models, the key metrics were:

- exact 10-band accuracy,
- 10-band macro F1,
- mean band error,
- within-one-band accuracy,
- wider tolerance metrics such as within-two, within-three, and within-four,
- collapsed 4-class macro F1.

Mean band error was especially important because the labels are ordered. A
model that predicts `46-55` when the truth is `56-65` is much closer than a
model that predicts `0-10`.

The project also separated model selection from raw-real transfer evaluation.
For the V4/V5 synthetic-only models, raw real data was reported separately. It
was not supposed to be silently used to choose the best epoch or tune the model.
This matters because otherwise the real test set would become contaminated by
model-selection feedback.

## 5.1 Technical Protocol Definitions

Several terms in the version sections have a precise technical meaning.

- `synthetic holdout` means the synthetic test split produced from the training
  manifest. In the scratch pipeline this is normally a `60/20/20`
  train/validation/test split.
- `stratified_random` means the split was stratified by the chosen training
  label but allowed related prompt families to appear across multiple splits.
- `metadata_family_holdout` means the split was built by grouping rows with the
  derived key
  `training_coarse_class|damage_profile_primary|box_form_factor|background_context`
  and then assigning entire groups to train, validation, or test. This is the
  stricter split introduced in V3.1.
- `raw real` means evaluation on the 385 relabeled real photos in their normal
  real-image form, used as a transfer diagnostic rather than a validation set
  for the synthetic-only models.
- `padded real` means evaluation on the same real photos after square-padding
  and resizing to the synthetic training resolution, mainly to test whether a
  simple framing mismatch was hurting transfer.
- `within-one-band accuracy` means absolute band-index distance `<= 1`.
- `within-two-band`, `within-three-band`, and `within-four-band` are defined
  the same way with larger allowed ordinal distance.
- `severe ordinal error rate` means absolute band-index distance `>= 2`.
- `mean band error` means the average absolute distance between predicted and
  true 10-band indices.
- `collapsed 4-class` metrics mean 10-band predictions were mapped back into
  the four coarse severity classes before computing accuracy or macro F1.
- For scalar models such as V5-A and the scalar head of V5-B, evaluation first
  maps the normalized scalar output back to the official `0-100` severity range
  and then converts that score to the official 10-band label using the project
  band boundaries.

Two scratch-training details also need explicit definitions:

- `balanced sampler` means `WeightedRandomSampler` with replacement and
  inverse-frequency weights computed on the training split only.
- `effective-number class weighting` means score-band class weights were built
  from
  `(1 - beta) / (1 - beta^n)` with `beta=0.999`, then normalized to mean 1.0
  over the non-empty classes.

## 5.2 Data Leakage And Contamination Controls

A presenter should explicitly separate three contamination risks:

1. Label contamination:
   - Risk: training on intended prompt labels that do not match the generated
     image.
   - Control: image-level QC with `accepted_as_labeled`, `accepted_relabel`, and
     `rejected`.

2. Split leakage:
   - Risk: related prompt families appearing in train, validation, and test,
     making synthetic holdout metrics too optimistic.
   - Control: V3.1 introduced `metadata_family_holdout`, using coarse class,
     damage profile, box form factor, and background context to keep related
     families together.

3. Real-test feedback leakage:
   - Risk: tuning V4/V5 models using raw-real evaluation, which would turn the
     real test set into a hidden validation set.
   - Control: V4/V5 synthetic-only models selected checkpoints using synthetic
     validation metrics. Raw-real performance was reported separately as
     transfer evaluation.

For the main-branch pretrained solution, there is a separate distinction:

- The task-specific damage training data was synthetic.
- The image backbone was pretrained on external data.
- That means the solution was not "contaminated" with the project real images
  during task training, but it did use external visual pretraining. It should be
  described as synthetic-trained with pretrained visual representations, not as
  a pure from-scratch experiment.

## 5.3 Version Map

This is the shortest way to understand the experimental sequence.

| Version | Primary question | Main change |
|---|---|---|
| V1 | Can synthetic labels be learned at all? | First from-scratch `simple_cnn`, 128 then 224 resolution |
| V2 | Do better labels and a stronger scratch CNN help? | Full QC labels, 384 inputs, `residual_cnn`, damage-safe augmentation |
| V3 | Can training-stack changes beat the V2 baseline? | RandAugment variants, GroupNorm tests, sampler changes |
| V3.1 | Are synthetic scores too optimistic because of split leakage? | Metadata-family holdout split |
| V4 | What happens if the project moves to the official 10-band target? | Score-band model replaces coarse-class primary target |
| V4.5 | Does distance-aware ordinal loss help? | Add EMD-style loss term while keeping V4 stack fixed |
| V5-A | Is scalar severity regression better than 10-band classification? | Scalar ordinal regressor |
| V5-B | Can multitask supervision stabilize scalar severity? | Shared scalar + band + coarse heads |

For a presentation, this table is often the bridge between dataset setup and
the main model story.

## 6. Model Training: From V1 To V5-B

This section describes the from-scratch model path. These models did not use a
pretrained backbone.

### 6.1 Shared Scratch Model And Training Contracts

Before going version by version, the scratch pipeline needs a technical
baseline.

The main from-scratch model families were:

- `simple_cnn`: four plain convolution blocks with max-pooling, adaptive
  average pooling to `2x2`, then a small fully connected classifier head.
- `residual_cnn`: a deeper custom CNN with a `5x5` stride-2 stem, residual
  blocks, BatchNorm, channel growth up to 256, global average pooling, and a
  single linear prediction head.
- `residual_cnn_groupnorm_regressor`: the residual backbone with GroupNorm
  replacing BatchNorm and a single sigmoid-constrained scalar head for
  normalized severity regression.
- `residual_cnn_groupnorm_multitask`: the same residual family with GroupNorm
  and three heads: one scalar sigmoid head, one 4-class coarse head, and one
  10-band score-band head.

The scratch training protocol also standardized several mechanics:

- split fractions were normally `60% train / 20% validation / 20% test`;
- augmentation was applied only to the training split, while validation and
  test used deterministic resize, tensor conversion, and RGB normalization;
- all later scratch runs saved `run_config.json`, split artifacts, epoch
  metrics, checkpoints, and training curves;
- the scalar regressor used `SmoothL1Loss`;
- the multitask model used a weighted sum of scalar `SmoothL1Loss`, coarse
  cross-entropy, score-band cross-entropy, and an auxiliary band ordinal
  `SmoothL1` term on expected band index;
- score-band classifier runs built soft ordinal targets by spreading the hard
  band label across neighboring bands with
  `softmax(-0.5 * (distance / sigma)^2)`;
- V4 used `sigma=1.0` as the locked score-band baseline and V4.5 kept that
  stack while adding an explicit squared-EMD term.

Checkpoint selection also changed with the task:

- V1 to V3 coarse classifiers were mainly compared by validation macro F1.
- V4 and V4.5 selected checkpoints by validation mean band error.
- V5-A selected checkpoints by validation mean band error after converting
  scalar predictions back into official score bands.
- V5-B selected checkpoints by validation scalar mean band error.

This matters because later versions were not just "new models." They also
changed the target definition, the split discipline, and the selection metric.

### 6.2 V1 - First Synthetic-Only Scratch CNN

#### What V1 was

V1 was the first real training baseline. It asked:

> Can a small CNN trained from scratch learn visible package-damage severity
> from synthetic images at all?

The model was a from-scratch `simple_cnn`. The early runs used low-resolution
cached images, mainly `128x128`, because those runs were cheap and useful for
debugging the pipeline.

#### What changed during V1

The training code gained:

- configurable dropout,
- CLI controls for optimizer choice, learning rate, and weight decay,
- optional augmentation,
- early stopping,
- per-class precision/recall/F1,
- predicted-class count logging,
- CSV/JSONL epoch logs,
- training curves,
- class-monitoring plots,
- a debug overfit mode.

This was necessary because early models often collapsed to predicting only a
few classes. Accuracy alone could hide that failure.

The most important data/model change was the move from `128x128` to `224x224`.
Damage severity depends on small visual details: crushed corners, creases,
punctures, torn flaps, edge deformation, and tape damage. `128x128` was useful
for debugging but too limited for the real task.

#### V1 no-new-image result

The strongest no-new-image V1-style classifier used:

- manifest: `synthetic_combined_phase1_plus_scaleup_200_processed_224.csv`
- model: from-scratch `simple_cnn`
- image size: `224x224`
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.1`
- balanced sampler: enabled
- augmentation: mild
- early-stopping patience: `20`

Result:

| Metric | Value |
|---|---:|
| synthetic test accuracy | 0.5102 |
| synthetic test macro F1 | 0.4862 |

This was a meaningful improvement over earlier `128x128` results.

#### V1 800-image result

After adding the targeted 555-image synthetic batch, the combined manifest had
800 rows. The best V1-style run on that 800-image processed manifest used:

- model: from-scratch `simple_cnn`
- image size: `224`
- optimizer: AdamW
- learning rate: `0.0003`
- weight decay: `0.0001`
- dropout: `0.2`
- augmentation: mild
- balanced sampler: disabled
- no pretrained weights

Result:

| Metric | Value |
|---|---:|
| synthetic test accuracy | 0.5125 |
| synthetic test macro F1 | 0.5053 |

#### V1 lesson

V1 proved that synthetic severity labels were learnable. It also showed that
the model was fragile. Resolution mattered, training diagnostics mattered, and
more images alone were not enough. The next bottleneck was label quality and
realism.

### 6.3 V2 - Full QC, 384 Resolution, Residual CNN, And V2 Scale-Up

#### What V2 was

V2 was the first stronger full-QC synthetic-only model. It asked:

> If we correct the labels, increase image resolution, use a residual CNN, and
> target real failure modes, does synthetic-only training improve?

V2 moved from the V1 `simple_cnn` path to a stronger from-scratch
`residual_cnn` path.

#### What changed

V2 changed both the data and the model:

- labels came from full visual QC, not only intended prompt labels,
- image size moved to `384x384`,
- model changed from `simple_cnn` to `residual_cnn`,
- augmentation changed to `damage_safe`,
- ordinal loss was added with weight `0.2`,
- a V2 synthetic scale-up added 132 QC-reviewed examples targeted at real
  failure modes.

The `damage_safe` augmentation was chosen because some standard augmentations
can damage the label semantics. For example, overly strong rotations or visual
distortions could create unrealistic package scenes or alter visible damage in
a way that no longer matches the label.

#### Pre-scale V2 result

The best full-QC controlled run before the V2 scale-up was:

- run: `E_384_residual_damage_safe_ordinal02`
- model: from-scratch `residual_cnn`
- image size: `384`
- augmentation: `damage_safe`
- ordinal loss weight: `0.2`

Result:

| Metric | Value |
|---|---:|
| synthetic test accuracy | 0.5313 |
| synthetic test macro F1 | 0.5153 |
| mean ordinal error | 0.6000 |
| off-by-one-or-correct rate | 0.8875 |

When this synthetic checkpoint was evaluated directly on real images, transfer
was weak:

| Real input policy | Accuracy | Macro F1 |
|---|---:|---:|
| raw real | 0.2927 | 0.2574 |
| padded 384 real | 0.3110 | 0.2445 |

#### V2 scale-up result

The V2 scale-up trained on `synthetic_full_qc_plus_v2_scale_processed_384.csv`
with 899 rows, including 132 V2 real-failure-targeted images.

Result:

| Metric | Value |
|---|---:|
| synthetic test accuracy | 0.5333 |
| synthetic test macro F1 | 0.5239 |
| off-by-one-or-correct rate | 0.9111 |
| severe ordinal error rate | 0.0889 |

Real transfer on padded real images:

| Model | Accuracy | Macro F1 | Off-by-one-or-correct |
|---|---:|---:|---:|
| pre-scale V2 baseline | 0.3110 | 0.2445 | 0.6768 |
| V2 scale-up | 0.3110 | 0.2689 | 0.7195 |

#### V2 lesson

V2 improved the synthetic model and some real-transfer diagnostics, especially
macro F1 and off-by-one-or-correct on padded real images. But direct real
accuracy did not improve. Synthetic-only training still did not close the
real-domain gap.

### 6.4 V3 - Training Stack Search

#### What V3 was

V3 was a controlled search around the V2 residual baseline. It asked:

> Can we improve the from-scratch synthetic-only model by changing the training
> stack while keeping the dataset and core task stable?

#### What changed

V3 tested:

- GroupNorm instead of BatchNorm,
- true ordinal coarse-class modeling,
- constrained RandAugment-style profiles:
  - `damage_safe_ra_low`,
  - `damage_safe_ra_medium`,
  - `damage_safe_ra_high`,
- removing the balanced sampler.

The motivation was reasonable. GroupNorm can be more stable with small batches.
Ordinal coarse modeling should match the ordered class structure. RandAugment
could improve generalization if constrained enough not to break label meaning.

#### V3 result

The selected V3 candidate was:

- run: `v3_bn_ra_low_coarse_ord02_nosampler_20260513`
- model: `residual_cnn`
- normalization: BatchNorm, not GroupNorm
- augmentation: `damage_safe_ra_low`
- target: coarse
- ordinal loss weight: `0.2`
- balanced sampler: disabled
- no pretrained weights

Result:

| Metric | V2 | V3 selected |
|---|---:|---:|
| synthetic accuracy | 0.5333 | 0.5500 |
| synthetic macro F1 | 0.5239 | 0.5496 |
| off-by-one-or-correct | 0.9111 | 0.9167 |
| severe ordinal error | 0.0889 | 0.0833 |

Not every theoretically attractive branch worked:

| Branch | Macro F1 |
|---|---:|
| GroupNorm + RA low | 0.2457 |
| true ordinal coarse | 0.3062 |

#### V3 lesson

V3 showed that controlled low-strength augmentation helped the proven residual
baseline. It also showed that theory was not enough: GroupNorm and true ordinal
coarse modeling sounded reasonable but did not win in this setting.

### 6.5 V3.1 - Harder Split Discipline

#### What V3.1 was

V3.1 was mainly an evaluation correction, not an architecture change. It asked:

> Are our synthetic metrics too optimistic because related prompt families leak
> across train, validation, and test?

Earlier random stratified splits could place visually related synthetic prompt
families in multiple splits. That could make the synthetic test easier than a
true generalization test.

#### What changed

V3.1 introduced `metadata_family_holdout`.

The split group was derived from:

- `training_coarse_class`,
- `damage_profile_primary`,
- `box_form_factor`,
- `background_context`.

The selected hard-split baseline used:

- model: from-scratch `residual_cnn`,
- image size: `384`,
- augmentation: `damage_safe`,
- dropout: `0.1`,
- weight decay: `0.0001`,
- ordinal loss weight: `0.2`,
- balanced sampler: disabled,
- split strategy: `metadata_family_holdout`.

#### V3.1 result

Seed-42 hard-split candidates:

| Dropout | Best validation macro F1 | Test accuracy | Test macro F1 |
|---:|---:|---:|---:|
| 0.1 | 0.5869 | 0.5419 | 0.5607 |
| 0.2 | 0.5473 | 0.6010 | 0.5983 |
| 0.3 | 0.5937 | 0.5813 | 0.5953 |

The `dropout=0.1` configuration was selected because it was more stable across
seeds:

| Seed | Best validation macro F1 | Test accuracy | Test macro F1 |
|---:|---:|---:|---:|
| 42 | 0.5869 | 0.5419 | 0.5607 |
| 43 | 0.5964 | 0.5243 | 0.5309 |

#### V3.1 lesson

Evaluation discipline mattered as much as model choice. The harder split
reduced prompt-family leakage risk. Later versions used this hard-split policy
as the more credible synthetic evaluation setup.

### 6.6 V4 - Official 10-Band Score-Band Model

#### What V4 was

V4 changed the project target. It asked:

> What happens when we stop treating four coarse classes as the main task and
> train directly on the official 10-band severity rubric?

Before V4, the strongest models were mostly four-class coarse classifiers. V4
made `score_band` the primary target. Four-class metrics remained as collapsed
diagnostics, but they were no longer the main training target.

#### What changed

V4 used:

- model: from-scratch `residual_cnn`,
- image size: `384`,
- augmentation: `damage_safe`,
- split: `metadata_family_holdout`,
- target: `score_band`,
- soft ordinal targets across the 10 bands,
- effective-number class-balanced weighting,
- ordinal loss weight: `0.2`,
- checkpoint selection by validation mean band error,
- no pretrained weights.

#### V4 result

Synthetic test:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 0.1990 |
| 10-band macro F1 | 0.1791 |
| mean band error | 1.3155 |
| within-one-band accuracy | 0.7233 |
| collapsed 4-class accuracy | 0.4854 |
| collapsed 4-class macro F1 | 0.4557 |

Raw-real transfer:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 0.1065 |
| within-one-band accuracy | 0.3325 |
| within-two-band accuracy | 0.5195 |
| mean band error | 2.6649 |

#### V4 lesson

V4 was a legitimate score-band baseline, not a failed run. Exact 10-band
accuracy was low, but synthetic within-one-band accuracy was high. The model
often found the right severity neighborhood on synthetic images.

However, raw-real transfer was still poor. Synthetic holdout success did not
mean the model had solved real package photos.

### 6.7 V4.5 - Hybrid Soft-Label Cross-Entropy Plus EMD

#### What V4.5 was

V4.5 was a narrow objective-function challenger. It asked:

> If score bands are ordered, can an explicitly distance-aware loss reduce
> far-away severity mistakes?

#### What changed

V4.5 kept the V4 stack fixed and changed only the objective:

- same from-scratch `residual_cnn`,
- same `metadata_family_holdout` split,
- same `damage_safe` augmentation,
- same `score_band` target,
- same soft-label sigma `1.0`,
- same effective-number beta `0.999`,
- same ordinal loss weight `0.2`,
- added cumulative squared EMD with `score_band_emd_weight=0.5`.

#### V4.5 result

Synthetic test comparison:

| Metric | V4 | V4.5 |
|---|---:|---:|
| exact 10-band accuracy | 0.1990 | 0.1748 |
| macro F1 | 0.1791 | 0.1574 |
| mean band error | 1.3155 | 1.3398 |
| within-one-band accuracy | 0.7233 | 0.7330 |
| collapsed coarse macro F1 | 0.4557 | 0.4439 |

Raw-real transfer comparison:

| Metric | V4 | V4.5 |
|---|---:|---:|
| exact 10-band accuracy | 0.1065 | 0.1273 |
| within-one-band accuracy | 0.3325 | 0.3532 |
| within-two-band accuracy | 0.5195 | 0.5351 |
| mean band error | 2.6649 | 2.5351 |

#### V4.5 lesson

V4.5 improved some tolerance-based real-transfer metrics, but it did not
cleanly beat V4 on the synthetic validation/test metrics used for model
selection. It was useful evidence, but not a new locked baseline.

The lesson was that a small ordinal loss tweak was not enough. The next major
axis needed to be target/head formulation.

### 6.8 V5-A - Scalar Ordinal Regression

#### What V5-A was

V5-A changed the head and target. It asked:

> Is severity better learned as a normalized scalar value than as a flat
> 10-class classification target?

The motivation was that package damage is ordered and continuous-like.
Neighboring bands can be visually ambiguous. A scalar model might learn
severity direction and distance better than a classifier.

#### What changed

V5-A used:

- model: `residual_cnn_groupnorm_regressor`,
- target: `training_representative_score / 100.0`,
- output: sigmoid constrained to `[0, 1]`,
- evaluation: convert predicted score back to official 10-band label,
- image size: `384`,
- augmentation: `damage_safe`,
- split: `metadata_family_holdout`,
- optimizer: AdamW,
- learning rate: `0.0003`,
- weight decay: `0.0001`,
- dropout: `0.1`,
- checkpoint selection: validation mean band error,
- no pretrained weights.

V5-A also added important sanity gates before the full run:

- target conversion tests,
- score-to-band mapping tests,
- tiny-subset overfit gate,
- saved split group overlap checks,
- checkpoint/resume checks.

The 4-example overfit gate passed:

| Metric | Value |
|---|---:|
| score MAE | 3.31 |
| mean band error | 0.25 |
| within-one-band accuracy | 1.0000 |
| collapsed coarse macro F1 | 1.0000 |

#### V5-A result

Synthetic test:

| Metric | Value |
|---|---:|
| score MAE | 16.2967 |
| exact 10-band accuracy | 0.1685 |
| macro F1 | 0.1529 |
| mean band error | 1.7079 |
| within-one-band accuracy | 0.5225 |
| collapsed coarse macro F1 | 0.4285 |

Raw-real transfer:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 0.1351 |
| within-one-band accuracy | 0.3974 |
| within-two-band accuracy | 0.5896 |
| mean band error | 2.2727 |

#### V5-A lesson

V5-A was mechanically correct and passed the sanity gates. It improved raw-real
ordinal transfer compared with V4/V4.5, especially mean band error and
within-one-band accuracy.

But it was worse on synthetic holdout. It lost too much exact 10-band
classification performance. Longer training did not improve the best checkpoint;
the best epoch stayed at 79.

The important insight was that optimizing synthetic 10-band exactness and
improving raw-real ordinal transfer were not the same thing.

### 6.9 V5-B - Multitask Scalar, Band, And Coarse Heads

#### What V5-B was

V5-B was a multitask model. It asked:

> Can a shared backbone keep V5-A's scalar ordinal transfer behavior while
> using auxiliary classification heads to stabilize band and coarse supervision?

#### What changed

V5-B used:

- model: `residual_cnn_groupnorm_multitask`,
- shared from-scratch backbone,
- scalar normalized severity head,
- auxiliary 10-band score-band head,
- auxiliary 4-class coarse head,
- image size: `384`,
- split: `metadata_family_holdout`,
- augmentation: `damage_safe`,
- optimizer: AdamW,
- learning rate: `0.0003`,
- weight decay: `0.0001`,
- dropout: `0.1`,
- no pretrained weights.

Loss weights:

| Loss component | Weight |
|---|---:|
| scalar loss | 1.0 |
| coarse loss | 0.3 |
| auxiliary 10-band loss | 0.3 |
| auxiliary band ordinal penalty | 0.2 |

#### V5-B result

Synthetic test from the scalar head:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 0.1461 |
| within-one-band accuracy | 0.4775 |
| within-two-band accuracy | 0.6910 |
| within-three-band accuracy | 0.8596 |
| within-four-band accuracy | 0.9213 |
| mean band error | 1.9663 |

Raw-real transfer from the scalar head:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 0.1506 |
| within-one-band accuracy | 0.3948 |
| within-two-band accuracy | 0.6260 |
| within-three-band accuracy | 0.8130 |
| within-four-band accuracy | 0.9325 |
| mean band error | 2.1091 |

Comparison with V5-A:

| Metric | V5-A | V5-B |
|---|---:|---:|
| synthetic mean band error | 1.7079 | 1.9663 |
| real mean band error | 2.2727 | 2.1091 |
| real within-one-band | 0.3974 | 0.3948 |
| real within-two-band | 0.5896 | 0.6260 |
| real within-three-band | 0.7584 | 0.8130 |
| real within-four-band | 0.8935 | 0.9325 |

Longer V5-B training did not help:

| Metric | V5-B round 1 | V5-B longer |
|---|---:|---:|
| synthetic MBE | 1.9663 | 2.0281 |
| synthetic within-one-band | 0.4775 | 0.4607 |
| raw-real MBE | 2.1091 | 2.2442 |
| raw-real within-one-band | 0.3948 | 0.3532 |

#### V5-B lesson

V5-B improved broader real-domain ordinal robustness compared with V5-A, but it
did not improve synthetic 10-band performance. The auxiliary 10-band head was
useful inside the multitask model, but the scalar head remained the main ordinal
estimator. Simply training longer was not the answer.

The most useful next V5 direction would be loss-weight tuning or a different
ordinal-threshold formulation, not just more epochs.

## 7. Summary Of From-Scratch Model Evolution

The from-scratch path can be summarized as follows:

| Version | Main question | Main change | Synthetic result | Real result |
|---|---|---|---|---|
| V1 | Can synthetic labels be learned at all? | simple CNN, 128 to 224, logging/debugging | best 800-image macro F1 0.5053 | not the main evaluation yet |
| V2 | Do QC labels, 384 inputs, residual CNN, and targeted data help? | full QC, residual CNN, damage-safe aug, ordinal loss, V2 scale-up | macro F1 0.5239 | padded-real macro F1 0.2689 |
| V3 | Can training-stack changes improve V2? | low-strength RA, no sampler, tested GN/ordinal branches | macro F1 0.5496 | not primary |
| V3.1 | Are synthetic metrics too optimistic? | metadata-family holdout split | selected stable test macro F1 0.5607 / 0.5309 across seeds | not primary |
| V4 | What happens on official 10-band target? | score-band target, soft ordinal labels, MBE selection | exact 0.1990, MBE 1.3155, +/-1 0.7233 | exact 0.1065, MBE 2.6649, +/-1 0.3325 |
| V4.5 | Does distance-aware EMD help? | added EMD term | exact 0.1748, MBE 1.3398, +/-1 0.7330 | exact 0.1273, MBE 2.5351, +/-1 0.3532 |
| V5-A | Is scalar severity better than 10-way classification? | scalar ordinal regressor | exact 0.1685, MBE 1.7079, +/-1 0.5225 | exact 0.1351, MBE 2.2727, +/-1 0.3974 |
| V5-B | Can multitask stabilize scalar severity? | scalar + band + coarse heads | exact 0.1461, MBE 1.9663, +/-1 0.4775 | exact 0.1506, MBE 2.1091, +/-1 0.3948 |

The from-scratch path taught us that:

- synthetic labels are learnable,
- higher resolution matters,
- full QC matters,
- split discipline matters,
- ordinal metrics are more informative than exact accuracy alone,
- synthetic validation does not reliably predict raw-real performance,
- real-domain generalization remained the limiting problem.

## 8. Real-Domain Scratch And Fine-Tuning Baselines

The project also tested whether using real data directly helped. The real split
policy was:

- 60% held-out real test,
- 20% small real training,
- 20% real validation / fine-tuning support.

Relevant real-domain baseline results:

| Run | Training source | Evaluation source | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| synthetic E raw real all | full-QC synthetic | all current real images | 0.2927 | 0.2574 |
| synthetic E padded real all | full-QC synthetic | all current real images | 0.3110 | 0.2445 |
| real-only balanced sampler scratch | 20% real train + 20% real val | 60% held-out real test | 0.5152 | 0.3887 |
| real-only class-weighted scratch | 20% real train + 20% real val | 60% held-out real test | 0.4444 | 0.2915 |
| synthetic E fine-tune real scratch | synthetic initialized, then real fine-tune | 60% held-out real test | 0.4343 | 0.3221 |
| diagnostic pretrained ResNet18 real | pretrained real-domain diagnostic | 60% held-out real test | 0.5152 | 0.4509 |

The real-only scratch baseline beat the synthetic-to-real scratch transfer
baseline. The synthetic-initialized real fine-tune did not beat real-only
scratch. This was an important result: synthetic pretraining alone did not
automatically produce a better real-domain model under the tested settings.

The diagnostic pretrained ResNet18 real model improved macro F1 relative to
real-only scratch at the same accuracy. That suggested pretrained visual
representations were useful, even though the core from-scratch experiment kept
pretraining out of the main V1-V5-B comparison.

## 9. Pretrained Model Solution In The `main` Branch

### 9.1 What The Main-Branch Solution Was

The `main` branch contains a later pretrained-backbone solution under:

- `TrainingFiles/train_band.py`
- `TrainingFiles/train_emd.py`
- `TrainingFiles/train_convnext_base.py`
- `EvaluartionFiles/evaluate3.py`
- `EvaluartionFiles/evaluate_ensemble.py`
- `EvaluartionFiles/evaluate_triple.py`
- `FinalModels/README.md`

The `FinalModels/README.md` says the final models were trained exclusively on
synthetic AI-generated images and evaluated on 385 real-world photos never seen
during training.

Important interpretation: "trained exclusively on synthetic images" refers to
the task-specific damage training data. The model code still uses pretrained
image backbones, because the training scripts instantiate `BoxDamageModel` with
`pretrained=True`. So this solution is not a from-scratch model. It is a
synthetic-trained task model built on pretrained visual representations.

This difference is central:

- V1 to V5-B asked how far a no-pretrain custom CNN could go.
- The `main` solution asked how much better the project gets if we allow
  pretrained vision backbones and ensembles.

### 9.2 Why A Pretrained Backbone Was Used

The from-scratch models had a consistent problem: they could learn synthetic
holdout images but struggled on raw real photos. This means the issue was not
only the label system. It was visual representation and domain shift.

A pretrained backbone gives the model visual features learned from large image
corpora before seeing the project dataset. That can help with:

- edges,
- textures,
- material appearance,
- object shapes,
- lighting variation,
- background variation,
- camera viewpoint variation.

Those features are especially useful when the project-specific dataset is small
relative to the complexity of real-world package photos. A from-scratch CNN had
to learn both generic visual features and severity-specific features from fewer
than 1,000 synthetic images. A pretrained ConvNeXt model starts with much
stronger generic features and then adapts them to package damage.

### 9.3 Main-Branch Model Types

The main branch includes several pretrained model variants.

#### BaseModel

The README reports a `BaseModel` checkpoint. In the evaluation scripts this is
treated as one of the pretrained models evaluated on the real-world manifest.

Reported real-image result:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 15.58% |
| within-one-band accuracy | 52.47% |
| mean band error | 1.7039 |

#### EMD

The EMD model uses direct 10-band classification with Earth Mover's Distance
loss. The reason for EMD is ordinal severity: predicting a neighboring band
should be penalized less than predicting a far-away band.

The training script `train_emd.py`:

- uses `label_mode="band"`,
- uses a pretrained backbone through `pretrained=True`,
- trains direct 10-band logits,
- computes EMD by comparing cumulative distribution functions,
- reports exact band accuracy, within-one-band accuracy, and mean band error,
- freezes the backbone for early epochs and then unfreezes it.

Reported real-image result:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 20.26% |
| within-one-band accuracy | 58.44% |
| mean band error | 1.4234 |

This was much better than the from-scratch V4/V5 synthetic-only models on raw
real images.

#### Dual / ConvNeXtDual

The dual-head model predicts both:

- a continuous severity score,
- a coarse class.

The main branch `train_convnext_base.py` describes this as a ConvNeXt Base
dual-head model, with a score head and a coarse head. It uses:

- pretrained ConvNeXt Base backbone,
- image size `384`,
- batch size default `8`,
- learning rate default `1e-4`,
- dropout default `0.4`,
- freeze epochs default `5`,
- AdamW,
- a two-stage schedule: train heads first, then unfreeze backbone with lower
  backbone learning rate.

Reported real-image result for Dual:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 18.18% |
| within-one-band accuracy | 55.06% |
| mean band error | 1.6338 |

#### SAM model

The main branch README also reports a SAM model. This model expects SAM-cropped
input at evaluation time and is evaluated using `realWorld/sam_cropped/`.

Reported real-image result:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 21.04% |
| within-one-band accuracy | 59.74% |
| mean band error | 1.4182 |

For the from-scratch V1-V5-B story, SAM should be treated separately as an
input-processing/modeling variant, not as the core no-pretrain baseline. In the
main-branch pretrained solution, however, it is part of the reported final
model set.

### 9.4 Main-Branch Training Pattern

The main pretrained training scripts are more specific than the earlier summary
made them sound.

Across `train_band.py`, `train_emd.py`, and `train_convnext_base.py`, the
common pattern is:

1. Load the synthetic manifest only.
2. Build an internal random `85/15` synthetic train/validation split with
   `random_split`, using the script seed.
3. Apply resize-to-`384`, normalization, and task-specific training
   augmentation through the main-branch dataset transforms.
4. Use a sampler from `train_ds.get_sampler_weights()` to rebalance training
   examples.
5. Instantiate `BoxDamageModel(..., pretrained=True)`.
6. Freeze the pretrained backbone for the first `5` epochs by default.
7. Train only the task heads with AdamW and a OneCycle learning-rate schedule.
8. Unfreeze the backbone after the freeze stage.
9. Rebuild the optimizer with a lower backbone learning rate, typically
   `0.05 x lr` for backbone parameters and `0.5 x lr` for neck/head parameters.
10. Continue fine-tuning with a cosine-annealing scheduler.
11. Save the best checkpoint by validation loss, not by raw-real performance.
12. Evaluate the saved checkpoint on the real-world manifest.

This is important for interpretation. The main-branch models are not directly
using the later `metadata_family_holdout` protocol from the scratch V3.1-V5
pipeline. Their internal synthetic validation split is simpler, and their
selection rule is validation loss rather than validation mean band error or
raw-real transfer metrics. So the main branch should be treated as a stronger
performance-oriented pretrained benchmark, not as a perfectly apples-to-apples
replica of the later scratch protocol.

### 9.5 Main-Branch Evaluation And Ensembling

The evaluation scripts convert model outputs into 10-band predictions and report
ordered-band metrics:

- exact 10-band accuracy,
- within-one-band accuracy,
- within-two-band accuracy,
- within-three-band accuracy,
- within-four-band accuracy,
- mean band error.

They also use test-time augmentation with horizontal flips.

The ensemble mechanism is also more specific than "average the models."

- For direct band models, the evaluation code averages the softmax probability
  vectors from the original image and its horizontal flip.
- For dual score models, the code averages the predicted scalar scores from the
  original image and its horizontal flip, maps the averaged score to an
  official band index, then constructs a local Gaussian-like band distribution
  centered on that predicted band.
- The final ensemble averages those band-probability vectors with explicit
  mixture weights such as `0.60 / 0.40`, then takes `argmax` over the combined
  distribution.

The ensemble scripts average predictions from multiple checkpoints. This makes
sense because the model variants make different types of errors:

- EMD directly optimizes ordinal band distance,
- Dual combines scalar score and coarse class supervision,
- SAM changes the input framing by using cropped images,
- ConvNeXt Base adds backbone capacity.

The best reported main-branch ensemble was:

`EMD60+SAM40`

Reported real-image result:

| Metric | Value |
|---|---:|
| exact 10-band accuracy | 21.04% |
| within-one-band accuracy | 64.16% |
| mean band error | 1.3351 |

The full reported final model table from `FinalModels/README.md` is:

| Model | Exact | Within-one-band | MBE |
|---|---:|---:|---:|
| EMD60+SAM40 | 21.04% | 64.16% | 1.3351 |
| EMD65+SAM35 | 20.78% | 64.16% | 1.3299 |
| EMD+SAM 50/50 | 20.52% | 63.40% | 1.3636 |
| EMD+Dual 50/50 | 21.82% | 60.26% | 1.4338 |
| SAM alone | 21.04% | 59.74% | 1.4182 |
| EMD alone | 20.26% | 58.44% | 1.4234 |
| Dual alone | 18.18% | 55.06% | 1.6338 |
| BaseModel alone | 15.58% | 52.47% | 1.7039 |

### 9.6 Comparison To The From-Scratch V5 Models

The main pretrained solution performs much better on real images than the
from-scratch synthetic-only V4/V5 path.

| Model | Real exact | Real +/-1 | Real MBE |
|---|---:|---:|---:|
| V4 locked score-band, from scratch | 10.65% | 33.25% | 2.6649 |
| V4.5 EMD, from scratch | 12.73% | 35.32% | 2.5351 |
| V5-A scalar, from scratch | 13.51% | 39.74% | 2.2727 |
| V5-B scalar head, from scratch | 15.06% | 39.48% | 2.1091 |
| BaseModel, pretrained main branch | 15.58% | 52.47% | 1.7039 |
| EMD, pretrained main branch | 20.26% | 58.44% | 1.4234 |
| Dual, pretrained main branch | 18.18% | 55.06% | 1.6338 |
| EMD60+SAM40 ensemble, pretrained main branch | 21.04% | 64.16% | 1.3351 |

This comparison answers an important project question. The from-scratch models
showed that the synthetic dataset contains learnable signal. But the pretrained
main-branch solution shows that stronger visual representations are needed for
much better real-domain transfer.

### 9.7 Caveat About Main-Branch Evidence

The `main` branch contains final model instructions, training/evaluation
scripts, and reported results. The checkpoint folders themselves are linked via
Google Drive in `FinalModels/README.md`; they are not stored directly in the
repository branch. This report therefore treats the main-branch pretrained
results as reported project results from that README and source code, not as
freshly re-run metrics from local checkpoints.

## 10. Discussion

### 10.1 What Worked

Several decisions clearly improved the project:

- defining the severity rubric before training,
- using band-first labels instead of fake exact scores,
- creating structured prompt metadata,
- balancing severity and visual conditions,
- reviewing and relabeling generated images,
- increasing image size from 128 to 224 and then 384,
- moving from `simple_cnn` to `residual_cnn`,
- using damage-safe augmentation,
- adding ordinal-aware metrics and losses,
- using metadata-family holdout splits,
- reporting raw-real transfer separately from synthetic validation.

The project became stronger when it moved from "train a model on generated
images" to "build a controlled synthetic-to-real experiment."

### 10.2 What Did Not Work As Expected

Several plausible ideas did not become final answers:

- More synthetic images alone did not solve real transfer.
- GroupNorm did not improve the V3 coarse classifier.
- True ordinal coarse modeling did not beat the residual baseline.
- V4.5 EMD improved some real tolerance metrics but did not replace V4.
- SAM-cropped synthetic training improved synthetic holdout but worsened raw-real
  transfer in the from-scratch ablation.
- V5-A scalar regression improved real ordinal transfer but weakened synthetic
  score-band performance.
- V5-B multitask training improved wider real tolerance metrics but did not
  become a stronger synthetic model.
- Longer V5-B training worsened heldout and raw-real results.

These negative results are useful. They show that the project did not just
choose a model because it sounded better. The model path changed when evidence
supported a change.

### 10.3 Synthetic Holdout Versus Real Transfer

One of the strongest lessons is that synthetic holdout performance and real
transfer performance are related but not identical.

For example:

- V4 was stronger than V5-A on synthetic holdout MBE.
- V5-A was stronger than V4 on raw-real MBE.
- V5-B was worse than V5-A on synthetic MBE but better on wider raw-real
  tolerance metrics.
- SAM-cropped synthetic training improved synthetic heldout but worsened raw-real
  transfer.

This means the project cannot rely only on synthetic validation to claim
real-world success. Synthetic validation is still important for controlled model
selection, but final conclusions need real-image evaluation.

### 10.4 Why Pretraining Helped

The pretrained solution helped because it reduced the burden on the small
project dataset. The from-scratch models had to learn generic visual features
and package severity features from synthetic images. The pretrained ConvNeXt
models already had strong visual features and only needed to adapt them to the
damage task.

This is especially important under domain shift. Real package photos vary in
lighting, texture, background, compression, framing, and camera behavior.
Pretrained backbones are better positioned to handle that variation.

The best main-branch ensemble reached 64.16% within-one-band accuracy and
1.3351 MBE on 385 real photos. That is far ahead of the no-pretrain V5-B
scalar-head result of 39.48% within-one-band and 2.1091 MBE on raw real images.

## 11. Final Answer To The Project Question

Synthetic-only training can learn package/cardboard damage severity. The
from-scratch V1 to V5-B path proves that the synthetic dataset contains real
visual signal: performance improved with better labels, higher resolution,
stronger scratch architectures, harder splits, and ordinal training objectives.

However, synthetic-only from-scratch training did not fully solve real-image
severity estimation. The gap between synthetic holdout and raw-real performance
remained large. The limiting factor was synthetic-to-real generalization, not
just classifier capacity or the number of generated images.

The dataset process was therefore as important as the model process. The best
dataset improvements came from structured prompt balancing, image-level QC,
reviewed labels, and targeted generation based on real failure modes.

The pretrained main-branch solution gives the strongest real-image results. It
shows that when external visual representations are allowed, synthetic-trained
models transfer much better to real photos. The best reported pretrained
ensemble, EMD60+SAM40, reached 21.04% exact 10-band accuracy, 64.16%
within-one-band accuracy, and 1.3351 mean band error on 385 real images.

So the final project conclusion is:

> Synthetic data is useful and learnable, but for strong real-world package
> damage severity estimation, dataset quality and pretrained visual
> representations matter more than simply generating more synthetic images or
> training a scratch CNN longer.

## 12. Limitations And Threats To Validity

This report is strong enough to explain the project, but a careful presenter
should state the main limitations explicitly.

### 12.1 Synthetic Dataset Limitations

- The synthetic images were high-effort and reviewed, but they were still
  generated by an image model rather than captured from the real world.
- Even with balancing, synthetic textures, framing, background statistics, and
  label realism can differ from real package photos.
- Some gains on synthetic holdout may reflect becoming better at the synthetic
  distribution rather than truly solving real-domain transfer.

### 12.2 Real Dataset Limitations

- The real labeled dataset has only 385 accepted rows.
- It is heavily imbalanced toward `moderate` damage and has very few `intact`
  examples.
- Real labels were rebuilt under the Dream2Detect rubric, so they are better
  aligned to the project, but they still depend on human severity judgment for
  borderline cases.

### 12.3 Evaluation Limitations

- Synthetic holdout and raw-real transfer answer different questions.
- The V4/V5 synthetic-only models were intentionally selected on synthetic
  validation rather than tuned on raw-real performance.
- That keeps the real test cleaner, but it also means some models that transfer
  better may not look best under synthetic validation.

### 12.4 Main-Branch Pretrained Result Limitations

- The `main` branch pretrained section is based on the repo code plus the
  reported results in `FinalModels/README.md`.
- The actual final checkpoints are external Drive artifacts, not stored directly
  in the branch.
- So the pretrained result summary is evidence-backed, but it is not a fresh
  local rerun inside this report workflow.

## 13. Presenter-Ready Key Numbers

These are the numbers most useful for slides.

### Dataset Numbers

| Dataset artifact | Rows | Notes |
|---|---:|---|
| Initial combined synthetic manifest | 800 | 45 phase-1 + 200 scale-up + 555 targeted images |
| Full-QC synthetic accepted as labeled | 623 | Intended label matched visible review |
| Full-QC synthetic accepted with relabeling | 177 | Usable image, corrected training label |
| V2-expanded synthetic training manifest | 899 | 800 full-QC + 132 real-failure-targeted V2 rows minus rejected/filtered handling in final export |
| Current real labeled dataset | 385 | Kaggle real photos relabeled under project rubric |

### Best From-Scratch Model Numbers

| Version | Synthetic metric | Raw-real metric | Main interpretation |
|---|---|---|---|
| V1 800-image | macro F1 0.5053 | not primary | synthetic labels were learnable |
| V2 scale-up | macro F1 0.5239 | padded-real macro F1 0.2689 | QC and targeted data helped but did not solve transfer |
| V3 selected | macro F1 0.5496 | not primary | constrained augmentation helped |
| V3.1 hard split | seed42 macro F1 0.5607, seed43 macro F1 0.5309 | not primary | harder split made evaluation more credible |
| V4 score-band | exact 0.1990, MBE 1.3155, +/-1 0.7233 | exact 0.1065, MBE 2.6649, +/-1 0.3325 | 10-band task was hard; real transfer weak |
| V4.5 EMD | exact 0.1748, MBE 1.3398, +/-1 0.7330 | exact 0.1273, MBE 2.5351, +/-1 0.3532 | EMD helped some real tolerance metrics, not enough to replace V4 |
| V5-A scalar | exact 0.1685, MBE 1.7079, +/-1 0.5225 | exact 0.1351, MBE 2.2727, +/-1 0.3974 | scalar regression improved real ordinal transfer but hurt synthetic |
| V5-B multitask | exact 0.1461, MBE 1.9663, +/-1 0.4775 | exact 0.1506, MBE 2.1091, +/-1 0.3948 | multitask improved wider real tolerance, not synthetic score-band |

### Pretrained Main-Branch Numbers

| Model | Real exact | Real +/-1 | Real MBE |
|---|---:|---:|---:|
| BaseModel | 15.58% | 52.47% | 1.7039 |
| Dual | 18.18% | 55.06% | 1.6338 |
| EMD | 20.26% | 58.44% | 1.4234 |
| SAM | 21.04% | 59.74% | 1.4182 |
| EMD60+SAM40 ensemble | 21.04% | 64.16% | 1.3351 |

### One-Sentence Result

The best no-pretrain V5-B scalar-head model reached 39.48% within-one-band
accuracy and 2.1091 MBE on raw real images, while the best reported pretrained
main-branch ensemble reached 64.16% within-one-band accuracy and 1.3351 MBE.

## 14. Appendix: Presentation-Ready Compression

If this document needs to be turned into a presentation quickly, the most
plausible compression is:

1. Project question and why transfer is hard.
2. Shared 10-band label system.
3. One slide on synthetic dataset creation and QC.
4. One slide on real dataset relabeling.
5. One slide on metrics.
6. Two to four slides on V1 to V3.1.
7. Two to four slides on V4 to V5-B.
8. One slide on the pretrained `main` branch solution.
9. One final comparison slide.
10. One conclusion slide.

If more detail is needed, use the long tables in Sections 7, 13, and 15 as the
backup appendix rather than expanding the dataset slides.

## 15. Evidence Sources

Current branch (`exp`) sources used for the from-scratch and dataset sections:

- `docs/06-package-severity-scale-standards.md`
- `docs/07-real-dataset-relabeling.md`
- `docs/08-implementation-plan.md`
- `docs/09-data-contracts.md`
- `docs/12-milestone-log.md`
- `docs/19-score-band-target-plan.md`
- `docs/21-v5-design-plan.md`
- `data/datasets/synthetic_combined_phase1_plus_targeted_555_full_qc_source.csv`
- `data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv`
- `data/datasets/real_labeled_dataset_current.csv`
- `data/evaluations/real_transfer/v2_scale_comparison_20260513.csv`
- `data/evaluations/synthetic_only/v3_synthetic_training_comparison_20260513.csv`
- `data/evaluations/synthetic_only/v3_1_metadata_family_holdout_comparison_20260513.csv`
- `data/evaluations/model_checkpoint_comparison/v4_v45_v5_within_band_comparison.csv`
- `data/evaluations/model_checkpoint_comparison/v5_b_round1_comparison.csv`
- `data/evaluations/model_checkpoint_comparison/v5_b_longer_comparison.csv`

Main branch sources used for the pretrained solution section:

- `main:FinalModels/README.md`
- `main:TrainingFiles/train_band.py`
- `main:TrainingFiles/train_emd.py`
- `main:TrainingFiles/train_convnext_base.py`
- `main:EvaluartionFiles/evaluate3.py`
- `main:EvaluartionFiles/evaluate_ensemble.py`
- `main:EvaluartionFiles/evaluate_triple.py`
