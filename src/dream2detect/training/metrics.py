from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .dataset import COARSE_CLASS_NAMES, SCORE_BAND_NAMES, SCORE_BAND_TO_COARSE_INDEX


@dataclass(frozen=True)
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]
    per_class: dict[str, PerClassMetrics]
    predicted_class_distribution: dict[str, int]
    target_class_distribution: dict[str, int]


@dataclass(frozen=True)
class ScalarScoreBandMetrics:
    normalized_mae: float
    normalized_rmse: float
    score_mae: float
    score_rmse: float
    band_metrics: ClassificationMetrics
    coarse_metrics: ClassificationMetrics
    band_ordinal_errors: dict[str, float | int]


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert model output logits into predicted class indices.

    Input shape:
    - [batch, num_classes]

    Output shape:
    - [batch]
    """
    return torch.argmax(logits, dim=1)


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute simple classification accuracy.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    correct = (predictions == targets).sum().item()
    total = len(targets)

    if total == 0:
        raise ValueError("Cannot compute accuracy on zero examples.")

    return float(correct) / float(total)


def compute_macro_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute macro F1 across all classes.
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return float(
        f1_score(
            targets_np,
            predictions_np,
            average="macro",
            zero_division=0,
        )
    )


def build_class_names(
    num_classes: int,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if class_names is None:
        return [f"class_{index}" for index in range(num_classes)]

    normalized_names = list(class_names)
    if len(normalized_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, got {len(normalized_names)}."
        )
    return normalized_names


def compute_class_distribution(
    labels: torch.Tensor,
    *,
    num_classes: int,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, int]:
    normalized_names = build_class_names(num_classes, class_names)
    labels_cpu = labels.detach().cpu().to(torch.long)
    counts = torch.bincount(labels_cpu, minlength=num_classes)

    return {
        class_name: int(counts[class_index].item())
        for class_index, class_name in enumerate(normalized_names)
    }


def compute_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, PerClassMetrics]:
    normalized_names = build_class_names(num_classes, class_names)
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np,
        predictions_np,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    return {
        class_name: PerClassMetrics(
            precision=float(precision[class_index]),
            recall=float(recall[class_index]),
            f1=float(f1[class_index]),
            support=int(support[class_index]),
        )
        for class_index, class_name in enumerate(normalized_names)
    }


def compute_confusion(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
) -> list[list[int]]:
    """
    Compute a confusion matrix as a plain nested Python list.

    Rows:
    - true class

    Columns:
    - predicted class
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    matrix = confusion_matrix(
        targets_np,
        predictions_np,
        labels=list(range(num_classes)),
    )
    return matrix.tolist()


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> ClassificationMetrics:
    """
    End-to-end helper:
    - logits -> predictions
    - predictions + targets -> accuracy, macro F1, confusion matrix
    """
    predictions = logits_to_predictions(logits)
    normalized_names = build_class_names(num_classes, class_names)

    return ClassificationMetrics(
        accuracy=compute_accuracy(predictions, targets),
        macro_f1=compute_macro_f1(predictions, targets),
        confusion=compute_confusion(predictions, targets, num_classes=num_classes),
        per_class=compute_per_class_metrics(
            predictions,
            targets,
            num_classes=num_classes,
            class_names=normalized_names,
        ),
        predicted_class_distribution=compute_class_distribution(
            predictions,
            num_classes=num_classes,
            class_names=normalized_names,
        ),
        target_class_distribution=compute_class_distribution(
            targets,
            num_classes=num_classes,
            class_names=normalized_names,
        ),
    )


def compute_classification_metrics_from_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int = 4,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> ClassificationMetrics:
    normalized_names = build_class_names(num_classes, class_names)

    return ClassificationMetrics(
        accuracy=compute_accuracy(predictions, targets),
        macro_f1=compute_macro_f1(predictions, targets),
        confusion=compute_confusion(predictions, targets, num_classes=num_classes),
        per_class=compute_per_class_metrics(
            predictions,
            targets,
            num_classes=num_classes,
            class_names=normalized_names,
        ),
        predicted_class_distribution=compute_class_distribution(
            predictions,
            num_classes=num_classes,
            class_names=normalized_names,
        ),
        target_class_distribution=compute_class_distribution(
            targets,
            num_classes=num_classes,
            class_names=normalized_names,
        ),
    )


def scores_to_band_indices(scores: torch.Tensor) -> torch.Tensor:
    """
    Map continuous 0-100 severity scores to official score-band indices.

    Boundaries follow the band upper limits:
    0-10, 11-20, 21-30, 31-35, 36-45, 46-55, 56-65, 66-75, 76-85, 86-100.
    Scores outside 0-100 are clamped before mapping.
    """
    clamped = torch.clamp(scores.to(torch.float32), 0.0, 100.0)
    boundaries = torch.tensor(
        [10.0, 20.0, 30.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0],
        dtype=torch.float32,
        device=clamped.device,
    )
    return torch.bucketize(clamped, boundaries, right=False).to(torch.long)


def compute_scalar_score_band_metrics(
    *,
    normalized_predictions: torch.Tensor,
    normalized_targets: torch.Tensor,
) -> ScalarScoreBandMetrics:
    if len(normalized_predictions) != len(normalized_targets):
        raise ValueError("Predictions and targets must have the same length.")
    if len(normalized_targets) == 0:
        raise ValueError("Cannot compute scalar metrics on zero examples.")

    predictions = torch.clamp(normalized_predictions.detach().cpu().to(torch.float32), 0.0, 1.0)
    targets = torch.clamp(normalized_targets.detach().cpu().to(torch.float32), 0.0, 1.0)
    normalized_errors = predictions - targets
    score_predictions = predictions * 100.0
    score_targets = targets * 100.0
    score_errors = score_predictions - score_targets

    predicted_bands = scores_to_band_indices(score_predictions)
    target_bands = scores_to_band_indices(score_targets)
    predicted_coarse = band_indices_to_coarse_indices(predicted_bands)
    target_coarse = band_indices_to_coarse_indices(target_bands)

    return ScalarScoreBandMetrics(
        normalized_mae=float(torch.mean(torch.abs(normalized_errors)).item()),
        normalized_rmse=float(torch.sqrt(torch.mean(normalized_errors * normalized_errors)).item()),
        score_mae=float(torch.mean(torch.abs(score_errors)).item()),
        score_rmse=float(torch.sqrt(torch.mean(score_errors * score_errors)).item()),
        band_metrics=compute_classification_metrics_from_predictions(
            predicted_bands,
            target_bands,
            num_classes=len(SCORE_BAND_NAMES),
            class_names=SCORE_BAND_NAMES,
        ),
        coarse_metrics=compute_classification_metrics_from_predictions(
            predicted_coarse,
            target_coarse,
            num_classes=len(COARSE_CLASS_NAMES),
            class_names=COARSE_CLASS_NAMES,
        ),
        band_ordinal_errors=summarize_ordinal_errors(predicted_bands, target_bands),
    )


def summarize_ordinal_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if len(targets) == 0:
        raise ValueError("Cannot summarize ordinal errors on zero examples.")

    distances = (predictions.to(torch.long) - targets.to(torch.long)).abs()
    total = int(distances.numel())

    return {
        "total": total,
        "exact_accuracy": float((distances == 0).to(torch.float32).mean().item()),
        "within_one_band_accuracy": float(
            (distances <= 1).to(torch.float32).mean().item()
        ),
        "mean_band_error": float(distances.to(torch.float32).mean().item()),
        "severe_band_error_rate": float(
            (distances >= 2).to(torch.float32).mean().item()
        ),
    }


def band_indices_to_coarse_indices(band_indices: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(
        [SCORE_BAND_TO_COARSE_INDEX[index] for index in range(len(SCORE_BAND_TO_COARSE_INDEX))],
        dtype=torch.long,
        device=band_indices.device,
    )
    return mapping[band_indices.to(torch.long)]


def collapse_score_band_probabilities_to_coarse(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected [batch, num_bands] probabilities, got shape {tuple(probabilities.shape)}."
        )
    if probabilities.size(1) != len(SCORE_BAND_TO_COARSE_INDEX):
        raise ValueError(
            "collapse_score_band_probabilities_to_coarse expects 10 score-band probabilities."
        )

    collapsed = torch.zeros(
        probabilities.size(0),
        4,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    for band_index, coarse_index in SCORE_BAND_TO_COARSE_INDEX.items():
        collapsed[:, coarse_index] += probabilities[:, band_index]
    return collapsed
