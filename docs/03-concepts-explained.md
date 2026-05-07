# Concepts Explained

This file explains the main ML concepts that matter for the current project.

## Distribution Shift

Distribution shift means:

> the training data and the evaluation data do not come from the same distribution

In this project:

- training data is largely synthetic,
- evaluation data is real.

That gap is the main scientific difficulty.

If a model performs well on synthetic validation data but poorly on held-out real data, that is evidence of distribution shift.

## Synthetic-To-Real Generalization

Synthetic-to-real generalization asks:

> can a model learn useful visual patterns from synthetic images and still apply them to real images?

This is more meaningful than asking whether the model can fit the synthetic dataset itself.

The project uses synthetic data as the scalable source and real images as the actual test of usefulness.

## Fine-Grained Versus Coarse Targets

The project compares two ways of defining the output.

### Fine target

Predict a severity value on a `0-100` scale.

This is harder because the model must learn more detailed distinctions.

### Coarse target

Predict one of four classes:

- `intact`
- `minor`
- `moderate`
- `severe`

This is easier to learn and may transfer better under shift.

### Why This Comparison Matters

If coarse labels transfer reliably while fine scores do not, that is a real result.

It suggests that synthetic data may be useful at broad decision levels even when it is not precise enough for detailed regression.

## Training Strategy Comparison

The project compares three training setups using the same CNN architecture.

### Synthetic-only

Train on generated images only, then evaluate on real images.

This tests the main synthetic-to-real transfer claim.

### Real-only

Train only on the small real training split, then evaluate on held-out real images.

This tests what a scarce real dataset can do by itself.

### Synthetic + real fine-tuning

Train first on synthetic data, then continue training on the small real split.

This tests whether synthetic pretraining plus limited real adaptation is stronger than either source alone.

## Human-In-The-Loop Labeling

The real dataset is not originally labeled with this exact severity rubric.

So the project uses:

- ChatGPT Vision as a first-pass assistant,
- human review as the final authority.

This matters because the label process itself can become a source of noise.

If the labels are weak, the comparison becomes weak.

## Band-Based Severity

The synthetic pipeline does not ask for exact scores like `47% damaged`.

Instead, it uses bands such as:

- `36-45`
- `46-55`
- `76-85`

This is better because image generators are bad at obeying fake-precise numeric targets.

The bands give:

- more realistic prompt targets,
- more consistent human review,
- cleaner metadata.

## Why Packages Are A Stronger Domain

Packages create a harder and more meaningful transfer problem because they involve:

- geometry,
- material deformation,
- shadows,
- folds,
- tears,
- holes,
- real structural damage cues.

That makes the project scientifically stronger than a mostly surface-texture task.
