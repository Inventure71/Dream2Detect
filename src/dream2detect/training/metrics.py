from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import confusion_matrix, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]


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
    return float(f1_score(targets_np, predictions_np, average="macro"))


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
) -> ClassificationMetrics:
    """
    End-to-end helper:
    - logits -> predictions
    - predictions + targets -> accuracy, macro F1, confusion matrix
    """
    predictions = logits_to_predictions(logits)

    return ClassificationMetrics(
        accuracy=compute_accuracy(predictions, targets),
        macro_f1=compute_macro_f1(predictions, targets),
        confusion=compute_confusion(predictions, targets, num_classes=num_classes),
    )
