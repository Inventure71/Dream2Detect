from __future__ import annotations

import copy
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .dataset import (
    COARSE_CLASS_NAMES,
    SCORE_BAND_NAMES,
    SyntheticManifestDataset,
    build_eval_transform,
    build_train_transform,
)
from .metrics import compute_classification_metrics
from .metrics import compute_classification_metrics_from_predictions
from .metrics import collapse_score_band_probabilities_to_coarse
from .metrics import summarize_ordinal_errors
from .metrics import band_indices_to_coarse_indices
from .models import build_classifier_model
from .splits import (
    DEFAULT_METADATA_FAMILY_COLUMNS,
    DatasetSplit,
    build_stratified_splits,
    derive_metadata_family_groups,
    subset_frame_by_indices,
)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]
    mean_band_error: float | None = None
    within_one_band_accuracy: float | None = None
    severe_band_error_rate: float | None = None
    collapsed_coarse_accuracy: float | None = None
    collapsed_coarse_macro_f1: float | None = None
    collapsed_coarse_confusion: list[list[int]] | None = None
    per_class: dict[str, dict[str, float | int]] = field(default_factory=dict)
    predicted_class_distribution: dict[str, int] = field(default_factory=dict)
    target_class_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    train_history: list[EpochResult]
    val_history: list[EpochResult]
    test_metrics: EpochResult
    best_val_epoch: int
    device: str
    used_resize_in_transforms: bool
    output_dir: str | None
    selection_metric_name: str = "val_macro_f1"
    best_val_metric: float | None = None
    early_stopped: bool = False
    stopped_epoch: int | None = None


@dataclass(frozen=True)
class EarlyStoppingDecision:
    is_best: bool
    should_stop: bool
    best_epoch: int
    epochs_without_improvement: int


class EarlyStoppingTracker:
    def __init__(
        self,
        *,
        patience: int | None,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if patience is not None and patience <= 0:
            raise ValueError(f"patience must be positive or None, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def update(self, *, epoch_number: int, metric_value: float) -> EarlyStoppingDecision:
        if self.mode == "max":
            is_best = metric_value > self.best_metric + self.min_delta
        else:
            is_best = metric_value < self.best_metric - self.min_delta
        if is_best:
            self.best_metric = metric_value
            self.best_epoch = epoch_number
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = (
            self.patience is not None
            and self.epochs_without_improvement >= self.patience
        )

        return EarlyStoppingDecision(
            is_best=is_best,
            should_stop=should_stop,
            best_epoch=self.best_epoch,
            epochs_without_improvement=self.epochs_without_improvement,
        )


def get_target_label_spec(target_label_mode: str) -> tuple[str, tuple[str, ...], str]:
    if target_label_mode == "coarse":
        return "training_coarse_class", COARSE_CLASS_NAMES, "coarse"
    if target_label_mode == "coarse_ordinal":
        return "training_coarse_class", COARSE_CLASS_NAMES, "coarse"
    if target_label_mode == "score_band":
        return "training_score_band", SCORE_BAND_NAMES, "score_band"
    raise ValueError(f"Unsupported target_label_mode: {target_label_mode}")


def uses_ordinal_coarse_training(target_label_mode: str) -> bool:
    return target_label_mode == "coarse_ordinal"


def get_selection_metric_rule(target_label_mode: str) -> tuple[str, str]:
    if target_label_mode == "score_band":
        return "val_mean_band_error", "min"
    return "val_macro_f1", "max"


def extract_epoch_selection_metric(
    result: EpochResult,
    *,
    target_label_mode: str,
) -> float:
    metric_name, _mode = get_selection_metric_rule(target_label_mode)
    if metric_name == "val_mean_band_error":
        if result.mean_band_error is None:
            raise ValueError(
                "Score-band selection metric requires mean_band_error to be populated."
            )
        return result.mean_band_error
    return result.macro_f1


def build_cumulative_ordinal_targets(
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=targets.device)
    return (targets.unsqueeze(1) > thresholds.unsqueeze(0)).to(torch.float32)


def decode_cumulative_ordinal_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    return (probabilities > 0.5).sum(dim=1).to(torch.long)


def build_epoch_metric_field_names(class_names: tuple[str, ...]) -> list[str]:
    field_names = [
        "epoch",
        "learning_rate",
        "selection_metric_name",
        "val_selection_metric",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "train_mean_band_error",
        "train_within_one_band_accuracy",
        "train_severe_band_error_rate",
        "train_collapsed_coarse_accuracy",
        "train_collapsed_coarse_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_mean_band_error",
        "val_within_one_band_accuracy",
        "val_severe_band_error_rate",
        "val_collapsed_coarse_accuracy",
        "val_collapsed_coarse_macro_f1",
        "is_best",
    ]
    for class_name in class_names:
        field_names.extend(
            [
                f"train_pred_{class_name}",
                f"val_pred_{class_name}",
                f"train_{class_name}_precision",
                f"train_{class_name}_recall",
                f"train_{class_name}_f1",
                f"val_{class_name}_precision",
                f"val_{class_name}_recall",
                f"val_{class_name}_f1",
            ]
        )
    return field_names


EPOCH_METRIC_FIELD_NAMES = build_epoch_metric_field_names(COARSE_CLASS_NAMES)


def set_global_seed(seed: int) -> None:
    """
    Make the run more reproducible.

    This does not guarantee bit-perfect reproducibility in every environment,
    but it is the correct baseline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    """
    Pick the best available accelerator in this order:
    1. CUDA
    2. Apple Metal (MPS)
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


def infer_resize_policy(
    frame: pd.DataFrame,
    *,
    image_size: int,
) -> bool:
    """
    Decide whether transforms still need to resize images.

    Returns:
    - True  -> transforms should apply Resize
    - False -> manifest already points to images preprocessed at the target size
    """
    if "preprocessing_profile" not in frame.columns:
        return True

    profiles = {
        str(value).strip()
        for value in frame["preprocessing_profile"].dropna().unique().tolist()
        if str(value).strip()
    }
    if not profiles:
        return True

    if len(profiles) != 1:
        raise ValueError(
            "Manifest mixes multiple preprocessing_profile values. "
            "Use one consistent processed manifest per training run."
        )

    profile = next(iter(profiles))
    supported_profiles = {
        f"rgb_resize_{image_size}",
        f"rgb_pad_resize_{image_size}",
    }

    if profile in supported_profiles:
        return False

    if profile.startswith("rgb_resize_") or profile.startswith("rgb_pad_resize_"):
        raise ValueError(
            f"Manifest preprocessing profile is {profile!r}, but training requested "
            f"image_size={image_size}. Use a matching processed manifest or rebuild the cache."
        )

    return True


def validate_split_fractions(
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    fractions = {
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
    }
    for name, value in fractions.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
        if value >= 1.0:
            raise ValueError(f"{name} must be less than 1.0, got {value}")

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def build_dataloaders(
    manifest_path: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 8,
    random_seed: int = 42,
    use_balanced_sampler: bool = True,
    overfit_subset_size: int | None = None,
    use_augmentation: bool = True,
    augmentation_profile: str = "mild",
    target_label_mode: str = "coarse",
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, DatasetSplit, bool]:
    """
    Build train/validation/test dataloaders from one manifest.

    Returns:
    - train_loader
    - val_loader
    - test_loader
    - full dataframe
    - dataset split
    - whether transforms still apply Resize
    """
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    label_column, class_names, dataset_target_mode = get_target_label_spec(
        target_label_mode
    )
    validate_split_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    if overfit_subset_size is None:
        split = build_stratified_splits(
            manifest_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            random_seed=random_seed,
            label_column=label_column,
            split_strategy=split_strategy,
            split_group_column=split_group_column,
        )
    else:
        overfit_indices = build_overfit_indices(
            frame,
            subset_size=overfit_subset_size,
            random_seed=random_seed,
            label_column=label_column,
            class_names=class_names,
        )
        split = DatasetSplit(
            train_indices=overfit_indices,
            val_indices=overfit_indices,
            test_indices=overfit_indices,
        )
    include_resize = infer_resize_policy(frame, image_size=image_size)

    train_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode=dataset_target_mode,
        transform=build_train_transform(
            image_size=image_size,
            include_resize=include_resize,
            use_augmentation=use_augmentation,
            augmentation_profile="none" if not use_augmentation else augmentation_profile,
        ),
    )
    eval_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode=dataset_target_mode,
        transform=build_eval_transform(
            image_size=image_size,
            include_resize=include_resize,
        ),
    )

    train_subset = Subset(train_dataset, split.train_indices)
    val_subset = Subset(eval_dataset, split.val_indices)
    test_subset = Subset(eval_dataset, split.test_indices)

    train_sampler = None
    train_shuffle = True
    if use_balanced_sampler:
        train_sampler = WeightedRandomSampler(
            weights=build_sample_weights(
                frame,
                train_indices=split.train_indices,
                label_column=label_column,
            ),
            num_samples=len(split.train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, frame, split, include_resize


def build_class_weights(
    frame: pd.DataFrame,
    *,
    train_indices: list[int],
    device: torch.device,
    label_column: str = "training_coarse_class",
    class_names: tuple[str, ...] = COARSE_CLASS_NAMES,
) -> torch.Tensor:
    """
    Compute class weights from the training split only.
    """
    train_frame = subset_frame_by_indices(frame, train_indices)
    train_labels = train_frame[label_column].tolist()

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(class_names)),
        y=np.array(train_labels),
    )

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_effective_number_class_weights(
    frame: pd.DataFrame,
    *,
    train_indices: list[int],
    device: torch.device,
    label_column: str,
    class_names: tuple[str, ...],
    beta: float = 0.999,
) -> torch.Tensor:
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0.0, 1.0), got {beta}")

    train_frame = subset_frame_by_indices(frame, train_indices)
    counts_by_class = train_frame[label_column].value_counts().to_dict()
    counts = torch.tensor(
        [float(counts_by_class.get(class_name, 0)) for class_name in class_names],
        dtype=torch.float32,
        device=device,
    )

    if beta == 0.0:
        weights = torch.ones_like(counts)
    else:
        effective_counts = torch.zeros_like(counts)
        positive_mask = counts > 0
        effective_counts[positive_mask] = (1.0 - beta) / (
            1.0
            - torch.pow(
                torch.full_like(counts[positive_mask], beta),
                counts[positive_mask],
            )
        )
        weights = effective_counts

    positive_mask = weights > 0
    if not bool(torch.any(positive_mask)):
        raise ValueError("Could not build class weights because all class counts are zero.")

    normalized = weights.clone()
    normalized[positive_mask] = (
        normalized[positive_mask] / normalized[positive_mask].mean()
    )
    return normalized


def build_ordinal_pos_weights(
    frame: pd.DataFrame,
    *,
    train_indices: list[int],
    label_column: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    train_frame = subset_frame_by_indices(frame, train_indices)
    labels = torch.tensor(train_frame[label_column].map(COARSE_CLASS_NAMES.index).tolist())
    ordinal_targets = build_cumulative_ordinal_targets(
        labels,
        num_classes=num_classes,
    )
    positive_counts = ordinal_targets.sum(dim=0)
    negative_counts = ordinal_targets.size(0) - positive_counts
    pos_weights = negative_counts / positive_counts.clamp_min(1.0)
    return pos_weights.to(device=device, dtype=torch.float32)


def build_sample_weights(
    frame: pd.DataFrame,
    *,
    train_indices: list[int],
    label_column: str = "training_coarse_class",
) -> list[float]:
    """
    Build inverse-frequency sample weights for the training split.

    These weights are used by WeightedRandomSampler so underrepresented classes
    are sampled more often during training.
    """
    train_frame = subset_frame_by_indices(frame, train_indices)
    class_counts = train_frame[label_column].value_counts().to_dict()

    return [
        1.0 / float(class_counts[row[label_column]])
        for _, row in train_frame.iterrows()
    ]


def build_soft_ordinal_targets(
    targets: torch.Tensor,
    *,
    num_classes: int,
    sigma: float,
) -> torch.Tensor:
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    class_positions = torch.arange(
        num_classes,
        dtype=torch.float32,
        device=targets.device,
    )
    distances = class_positions.unsqueeze(0) - targets.unsqueeze(1).to(torch.float32)
    logits = -0.5 * (distances / sigma) ** 2
    return torch.softmax(logits, dim=1)


def build_cumulative_distribution(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError(
            "build_cumulative_distribution expects a 2D tensor of probabilities."
        )
    return torch.cumsum(probabilities, dim=1)


def compute_squared_emd_per_example(
    predicted_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor,
) -> torch.Tensor:
    if predicted_probabilities.shape != target_probabilities.shape:
        raise ValueError(
            "Predicted and target probability tensors must have the same shape."
        )
    predicted_cumulative = build_cumulative_distribution(predicted_probabilities)
    target_cumulative = build_cumulative_distribution(target_probabilities)
    return torch.mean((predicted_cumulative - target_cumulative) ** 2, dim=1)


def build_overfit_indices(
    frame: pd.DataFrame,
    *,
    subset_size: int,
    random_seed: int,
    label_column: str = "training_coarse_class",
    class_names: tuple[str, ...] = COARSE_CLASS_NAMES,
) -> list[int]:
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive, got {subset_size}")
    if subset_size > len(frame):
        raise ValueError(
            f"subset_size={subset_size} is larger than dataset size {len(frame)}"
        )
    if label_column not in frame.columns:
        raise ValueError(f"Frame must contain {label_column}.")

    rng = random.Random(random_seed)
    by_class: dict[str, list[int]] = {}
    for row_index, row in frame.iterrows():
        class_name = str(row[label_column])
        by_class.setdefault(class_name, []).append(int(row_index))

    for indices in by_class.values():
        rng.shuffle(indices)

    selected: list[int] = []
    available_class_names = [name for name in class_names if name in by_class]

    while len(selected) < subset_size:
        added_this_round = False
        for class_name in available_class_names:
            if len(selected) >= subset_size:
                break
            available_indices = by_class[class_name]
            if available_indices:
                selected.append(available_indices.pop())
                added_this_round = True
        if not added_this_round:
            break

    if len(selected) < subset_size:
        raise ValueError(
            f"Could only select {len(selected)} overfit examples from available classes."
        )

    return sorted(selected)


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    target_label_mode: str = "coarse",
    class_names: tuple[str, ...] = COARSE_CLASS_NAMES,
    prediction_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> EpochResult:
    """
    Run one full pass over one dataloader.

    If optimizer is provided:
    - training mode
    - gradients enabled
    - model weights updated

    If optimizer is None:
    - evaluation mode
    - no gradients
    - no weight updates
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        if is_training:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = loss_fn(logits, targets)

        current_batch_size = images.size(0)
        total_loss += float(loss.item()) * current_batch_size
        total_examples += current_batch_size

        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if total_examples == 0:
        raise ValueError("Dataloader produced zero examples.")

    epoch_loss = total_loss / total_examples
    stacked_logits = torch.cat(all_logits, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    mean_band_error: float | None = None
    within_one_band_accuracy: float | None = None
    severe_band_error_rate: float | None = None
    collapsed_coarse_accuracy: float | None = None
    collapsed_coarse_macro_f1: float | None = None
    collapsed_coarse_confusion: list[list[int]] | None = None

    if prediction_fn is None:
        metrics = compute_classification_metrics(
            stacked_logits,
            stacked_targets,
            num_classes=len(class_names),
            class_names=class_names,
        )
        if target_label_mode == "score_band":
            predictions = torch.argmax(stacked_logits, dim=1)
            ordinal_summary = summarize_ordinal_errors(predictions, stacked_targets)
            mean_band_error = float(ordinal_summary["mean_band_error"])
            within_one_band_accuracy = float(
                ordinal_summary["within_one_band_accuracy"]
            )
            severe_band_error_rate = float(
                ordinal_summary["severe_band_error_rate"]
            )

            band_probabilities = torch.softmax(stacked_logits, dim=1)
            coarse_probabilities = collapse_score_band_probabilities_to_coarse(
                band_probabilities
            )
            coarse_predictions = torch.argmax(coarse_probabilities, dim=1)
            coarse_targets = band_indices_to_coarse_indices(stacked_targets)
            coarse_metrics = compute_classification_metrics_from_predictions(
                coarse_predictions,
                coarse_targets,
                num_classes=len(COARSE_CLASS_NAMES),
                class_names=COARSE_CLASS_NAMES,
            )
            collapsed_coarse_accuracy = coarse_metrics.accuracy
            collapsed_coarse_macro_f1 = coarse_metrics.macro_f1
            collapsed_coarse_confusion = coarse_metrics.confusion
    else:
        metrics = compute_classification_metrics_from_predictions(
            prediction_fn(stacked_logits),
            stacked_targets,
            num_classes=len(class_names),
            class_names=class_names,
        )

    return EpochResult(
        loss=epoch_loss,
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        confusion=metrics.confusion,
        mean_band_error=mean_band_error,
        within_one_band_accuracy=within_one_band_accuracy,
        severe_band_error_rate=severe_band_error_rate,
        collapsed_coarse_accuracy=collapsed_coarse_accuracy,
        collapsed_coarse_macro_f1=collapsed_coarse_macro_f1,
        collapsed_coarse_confusion=collapsed_coarse_confusion,
        per_class={
            class_name: {
                "precision": class_metrics.precision,
                "recall": class_metrics.recall,
                "f1": class_metrics.f1,
                "support": class_metrics.support,
            }
            for class_name, class_metrics in metrics.per_class.items()
        },
        predicted_class_distribution=metrics.predicted_class_distribution,
        target_class_distribution=metrics.target_class_distribution,
    )


def epoch_result_to_dict(result: EpochResult) -> dict[str, object]:
    return {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "confusion": result.confusion,
        "mean_band_error": result.mean_band_error,
        "within_one_band_accuracy": result.within_one_band_accuracy,
        "severe_band_error_rate": result.severe_band_error_rate,
        "collapsed_coarse_accuracy": result.collapsed_coarse_accuracy,
        "collapsed_coarse_macro_f1": result.collapsed_coarse_macro_f1,
        "collapsed_coarse_confusion": result.collapsed_coarse_confusion,
        "per_class": result.per_class,
        "predicted_class_distribution": result.predicted_class_distribution,
        "target_class_distribution": result.target_class_distribution,
    }


def epoch_result_from_dict(payload: dict[str, object]) -> EpochResult:
    return EpochResult(
        loss=float(payload["loss"]),
        accuracy=float(payload["accuracy"]),
        macro_f1=float(payload["macro_f1"]),
        confusion=payload["confusion"],  # type: ignore[arg-type]
        mean_band_error=(
            float(payload["mean_band_error"])
            if payload.get("mean_band_error") is not None
            else None
        ),
        within_one_band_accuracy=(
            float(payload["within_one_band_accuracy"])
            if payload.get("within_one_band_accuracy") is not None
            else None
        ),
        severe_band_error_rate=(
            float(payload["severe_band_error_rate"])
            if payload.get("severe_band_error_rate") is not None
            else None
        ),
        collapsed_coarse_accuracy=(
            float(payload["collapsed_coarse_accuracy"])
            if payload.get("collapsed_coarse_accuracy") is not None
            else None
        ),
        collapsed_coarse_macro_f1=(
            float(payload["collapsed_coarse_macro_f1"])
            if payload.get("collapsed_coarse_macro_f1") is not None
            else None
        ),
        collapsed_coarse_confusion=payload.get("collapsed_coarse_confusion"),  # type: ignore[arg-type]
        per_class=payload.get("per_class", {}),  # type: ignore[arg-type]
        predicted_class_distribution=payload.get(  # type: ignore[arg-type]
            "predicted_class_distribution",
            {},
        ),
        target_class_distribution=payload.get(  # type: ignore[arg-type]
            "target_class_distribution",
            {},
        ),
    )


def training_result_to_dict(result: TrainingResult) -> dict[str, object]:
    return {
        "train_history": [epoch_result_to_dict(item) for item in result.train_history],
        "val_history": [epoch_result_to_dict(item) for item in result.val_history],
        "test_metrics": epoch_result_to_dict(result.test_metrics),
        "best_val_epoch": result.best_val_epoch,
        "device": result.device,
        "used_resize_in_transforms": result.used_resize_in_transforms,
        "output_dir": result.output_dir,
        "selection_metric_name": result.selection_metric_name,
        "best_val_metric": result.best_val_metric,
        "early_stopped": result.early_stopped,
        "stopped_epoch": result.stopped_epoch,
    }


def save_split_artifacts(
    frame: pd.DataFrame,
    split: DatasetSplit,
    *,
    output_dir: Path,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_frame = frame.copy()
    if split_strategy == "metadata_family_holdout":
        if split_group_column is not None and split_group_column in export_frame.columns:
            export_frame["split_group_id"] = (
                export_frame[split_group_column].fillna("missing").astype(str)
            )
        else:
            export_frame["split_group_id"] = derive_metadata_family_groups(export_frame)

    split_paths: dict[str, str] = {}
    for split_name, indices in (
        ("train", split.train_indices),
        ("val", split.val_indices),
        ("test", split.test_indices),
    ):
        split_frame = subset_frame_by_indices(export_frame, indices)
        split_path = output_dir / f"{split_name}_split.csv"
        split_frame.to_csv(split_path, index=False)
        split_paths[f"{split_name}_split_path"] = str(split_path)

    return split_paths


def build_run_config(
    *,
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    optimizer_name: str,
    lr_scheduler_name: str,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    min_learning_rate: float,
    random_seed: int,
    device: torch.device,
    include_resize: bool,
    use_balanced_sampler: bool,
    weight_decay: float,
    dropout_p: float,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    overfit_subset_size: int | None,
    use_augmentation: bool,
    augmentation_profile: str,
    model_variant: str,
    pretrained: bool,
    freeze_backbone: bool,
    ordinal_loss_weight: float,
    score_band_soft_label_sigma: float,
    score_band_emd_weight: float,
    score_band_class_weight_strategy: str,
    score_band_effective_beta: float,
    target_label_mode: str,
    class_names: tuple[str, ...] | list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
    init_from_checkpoint: Path | None = None,
    checkpoint_every_n_epochs: int = 1,
    plot_every_n_epochs: int = 1,
) -> dict[str, object]:
    selection_metric_name, selection_metric_mode = get_selection_metric_rule(
        target_label_mode
    )
    return {
        "manifest_path": str(manifest_path),
        "image_size": image_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer_name": optimizer_name,
        "lr_scheduler_name": lr_scheduler_name,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "min_learning_rate": min_learning_rate,
        "random_seed": random_seed,
        "device": str(device),
        "used_resize_in_transforms": include_resize,
        "use_balanced_sampler": use_balanced_sampler,
        "rgb_normalization": "imagenet_mean_std",
        "weight_decay": weight_decay,
        "dropout_p": dropout_p,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "overfit_subset_size": overfit_subset_size,
        "use_augmentation": use_augmentation,
        "augmentation_profile": augmentation_profile,
        "model_variant": model_variant,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
        "ordinal_loss_weight": ordinal_loss_weight,
        "score_band_soft_label_sigma": score_band_soft_label_sigma,
        "score_band_emd_weight": score_band_emd_weight,
        "score_band_class_weight_strategy": score_band_class_weight_strategy,
        "score_band_effective_beta": score_band_effective_beta,
        "target_label_mode": target_label_mode,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "selection_metric_name": selection_metric_name,
        "selection_metric_mode": selection_metric_mode,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_strategy": split_strategy,
        "split_group_column": split_group_column,
        "split_group_derivation": (
            list(DEFAULT_METADATA_FAMILY_COLUMNS)
            if split_strategy == "metadata_family_holdout" and split_group_column is None
            else None
        ),
        "init_from_checkpoint": (
            str(init_from_checkpoint) if init_from_checkpoint is not None else None
        ),
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
        "plot_every_n_epochs": plot_every_n_epochs,
    }


def save_run_config(
    *,
    output_dir: Path,
    config: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def flatten_epoch_metrics(
    *,
    epoch_number: int,
    train_metrics: EpochResult,
    val_metrics: EpochResult,
    is_best: bool,
    learning_rate: float | None = None,
    selection_metric_name: str = "val_macro_f1",
    val_selection_metric: float | None = None,
) -> dict[str, object]:
    return {
        "epoch": epoch_number,
        "learning_rate": learning_rate,
        "selection_metric_name": selection_metric_name,
        "val_selection_metric": val_selection_metric,
        "train_loss": train_metrics.loss,
        "train_accuracy": train_metrics.accuracy,
        "train_macro_f1": train_metrics.macro_f1,
        "train_mean_band_error": train_metrics.mean_band_error,
        "train_within_one_band_accuracy": train_metrics.within_one_band_accuracy,
        "train_severe_band_error_rate": train_metrics.severe_band_error_rate,
        "train_collapsed_coarse_accuracy": train_metrics.collapsed_coarse_accuracy,
        "train_collapsed_coarse_macro_f1": train_metrics.collapsed_coarse_macro_f1,
        "val_loss": val_metrics.loss,
        "val_accuracy": val_metrics.accuracy,
        "val_macro_f1": val_metrics.macro_f1,
        "val_mean_band_error": val_metrics.mean_band_error,
        "val_within_one_band_accuracy": val_metrics.within_one_band_accuracy,
        "val_severe_band_error_rate": val_metrics.severe_band_error_rate,
        "val_collapsed_coarse_accuracy": val_metrics.collapsed_coarse_accuracy,
        "val_collapsed_coarse_macro_f1": val_metrics.collapsed_coarse_macro_f1,
        "is_best": is_best,
    }


def append_epoch_metrics(
    *,
    output_dir: Path,
    epoch_number: int,
    train_metrics: EpochResult,
    val_metrics: EpochResult,
    is_best: bool,
    learning_rate: float | None = None,
    class_names: tuple[str, ...] = COARSE_CLASS_NAMES,
    selection_metric_name: str = "val_macro_f1",
    val_selection_metric: float | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    nested_row = {
        "epoch": epoch_number,
        "train": epoch_result_to_dict(train_metrics),
        "val": epoch_result_to_dict(val_metrics),
        "is_best": is_best,
        "learning_rate": learning_rate,
    }
    with (output_dir / "epoch_metrics.jsonl").open("a") as jsonl_file:
        jsonl_file.write(json.dumps(nested_row) + "\n")

    flat_row = flatten_epoch_metrics(
        epoch_number=epoch_number,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        is_best=is_best,
        learning_rate=learning_rate,
        selection_metric_name=selection_metric_name,
        val_selection_metric=val_selection_metric,
    )
    for class_name in class_names:
        train_class_metrics = train_metrics.per_class.get(class_name, {})
        val_class_metrics = val_metrics.per_class.get(class_name, {})
        flat_row[f"train_pred_{class_name}"] = (
            train_metrics.predicted_class_distribution.get(class_name, 0)
        )
        flat_row[f"val_pred_{class_name}"] = (
            val_metrics.predicted_class_distribution.get(class_name, 0)
        )
        flat_row[f"train_{class_name}_precision"] = train_class_metrics.get(
            "precision",
            0.0,
        )
        flat_row[f"train_{class_name}_recall"] = train_class_metrics.get(
            "recall",
            0.0,
        )
        flat_row[f"train_{class_name}_f1"] = train_class_metrics.get("f1", 0.0)
        flat_row[f"val_{class_name}_precision"] = val_class_metrics.get(
            "precision",
            0.0,
        )
        flat_row[f"val_{class_name}_recall"] = val_class_metrics.get(
            "recall",
            0.0,
        )
        flat_row[f"val_{class_name}_f1"] = val_class_metrics.get("f1", 0.0)

    csv_path = output_dir / "epoch_metrics.csv"
    should_write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=build_epoch_metric_field_names(class_names),
        )
        if should_write_header:
            writer.writeheader()
        writer.writerow(flat_row)

    return flat_row


def read_epoch_metrics(output_dir: Path) -> list[dict[str, object]]:
    jsonl_path = output_dir / "epoch_metrics.jsonl"
    if not jsonl_path.exists():
        return []

    history: list[dict[str, object]] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        history.append(
            flatten_epoch_metrics(
                epoch_number=int(row["epoch"]),
                train_metrics=epoch_result_from_dict(row["train"]),
                val_metrics=epoch_result_from_dict(row["val"]),
                is_best=bool(row["is_best"]),
                learning_rate=(
                    float(row["learning_rate"])
                    if row.get("learning_rate") is not None
                    else None
                ),
                selection_metric_name=str(
                    row.get("selection_metric_name", "val_macro_f1")
                ),
                val_selection_metric=(
                    float(row["val_selection_metric"])
                    if row.get("val_selection_metric") is not None
                    else None
                ),
            )
        )
    return history


def plot_epoch_metrics(
    *,
    output_dir: Path,
    history: list[dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "training_curves.png"

    if not history:
        raise ValueError("Cannot plot training curves with empty history.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]

    series = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_accuracy", "val_accuracy"),
        ("Macro F1", "train_macro_f1", "val_macro_f1"),
    ]
    if any(row.get("val_mean_band_error") is not None for row in history):
        series.append(("Mean Band Error", "train_mean_band_error", "val_mean_band_error"))

    figure, axes = plt.subplots(len(series), 1, figsize=(9, 3.3 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    for axis, (title, train_key, val_key) in zip(axes, series, strict=True):
        axis.plot(
            epochs,
            [float(row[train_key]) for row in history],
            marker="o",
            label="train",
        )
        axis.plot(
            epochs,
            [float(row[val_key]) for row in history],
            marker="o",
            label="val",
        )
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    return plot_path


def plot_class_monitoring(
    *,
    output_dir: Path,
    history: list[dict[str, object]],
    class_names: tuple[str, ...] = COARSE_CLASS_NAMES,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "class_monitoring.png"

    if not history:
        raise ValueError("Cannot plot class monitoring with empty history.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for class_name in class_names:
        axes[0].plot(
            epochs,
            [float(row[f"val_{class_name}_f1"]) for row in history],
            marker="o",
            label=class_name,
        )
    axes[0].set_title("Validation Per-Class F1")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    bottom = np.zeros(len(history))
    for class_name in class_names:
        values = np.array(
            [float(row[f"val_pred_{class_name}"]) for row in history]
        )
        axes[1].bar(
            epochs,
            values,
            bottom=bottom,
            label=class_name,
        )
        bottom += values
    axes[1].set_title("Validation Predicted-Class Counts")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Predictions")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    return plot_path


def save_epoch_checkpoint(
    *,
    output_dir: Path,
    epoch_number: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None = None,
    train_metrics: EpochResult,
    val_metrics: EpochResult,
    is_best: bool,
    config: dict[str, object],
) -> dict[str, str]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "train_metrics": epoch_result_to_dict(train_metrics),
        "val_metrics": epoch_result_to_dict(val_metrics),
        "is_best": is_best,
        "config": config,
    }

    epoch_path = checkpoint_dir / f"epoch_{epoch_number:03d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)

    paths = {
        "epoch_checkpoint_path": str(epoch_path),
        "latest_checkpoint_path": str(latest_path),
    }

    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(payload, best_path)
        paths["best_checkpoint_path"] = str(best_path)

    return paths


def load_training_checkpoint(checkpoint_path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_optimizer(
    *,
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def build_lr_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    lr_scheduler_name: str,
    selection_metric_mode: str,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if lr_scheduler_name == "none":
        return None
    if lr_scheduler_name != "reduce_on_plateau":
        raise ValueError(f"Unsupported lr_scheduler_name: {lr_scheduler_name}")
    if not 0.0 < lr_scheduler_factor < 1.0:
        raise ValueError(
            f"lr_scheduler_factor must be in (0.0, 1.0), got {lr_scheduler_factor}"
        )
    if lr_scheduler_patience <= 0:
        raise ValueError(
            f"lr_scheduler_patience must be positive, got {lr_scheduler_patience}"
        )
    if min_learning_rate < 0:
        raise ValueError(
            f"min_learning_rate must be non-negative, got {min_learning_rate}"
        )

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=selection_metric_mode,
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_learning_rate,
    )


def build_classification_loss(
    *,
    class_weights: torch.Tensor,
    use_balanced_sampler: bool,
    ordinal_loss_weight: float,
    device: torch.device,
    num_classes: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if ordinal_loss_weight < 0:
        raise ValueError(
            f"ordinal_loss_weight must be non-negative, got {ordinal_loss_weight}"
        )

    cross_entropy = (
        nn.CrossEntropyLoss()
        if use_balanced_sampler
        else nn.CrossEntropyLoss(weight=class_weights)
    )
    class_count = int(num_classes if num_classes is not None else len(class_weights))
    class_positions = torch.arange(class_count, dtype=torch.float32, device=device)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = cross_entropy(logits, targets)
        if ordinal_loss_weight == 0:
            return loss
        probabilities = torch.softmax(logits, dim=1)
        expected_class_index = (probabilities * class_positions).sum(dim=1)
        ordinal_loss = F.smooth_l1_loss(
            expected_class_index,
            targets.float(),
        )
        return loss + ordinal_loss_weight * ordinal_loss

    return loss_fn


def build_score_band_soft_label_loss(
    *,
    class_weights: torch.Tensor,
    use_balanced_sampler: bool,
    ordinal_loss_weight: float,
    score_band_emd_weight: float,
    device: torch.device,
    num_classes: int,
    soft_label_sigma: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if ordinal_loss_weight < 0:
        raise ValueError(
            f"ordinal_loss_weight must be non-negative, got {ordinal_loss_weight}"
        )
    if soft_label_sigma <= 0:
        raise ValueError(
            f"soft_label_sigma must be positive, got {soft_label_sigma}"
        )
    if score_band_emd_weight < 0:
        raise ValueError(
            "score_band_emd_weight must be non-negative, "
            f"got {score_band_emd_weight}"
        )

    class_positions = torch.arange(num_classes, dtype=torch.float32, device=device)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft_targets = build_soft_ordinal_targets(
            targets,
            num_classes=num_classes,
            sigma=soft_label_sigma,
        )
        if use_balanced_sampler:
            weighted_targets = soft_targets
        else:
            weighted_targets = soft_targets * class_weights.unsqueeze(0)
            weighted_targets = weighted_targets / weighted_targets.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)

        log_probabilities = F.log_softmax(logits, dim=1)
        loss = -(weighted_targets * log_probabilities).sum(dim=1).mean()
        probabilities = torch.softmax(logits, dim=1)

        if score_band_emd_weight > 0:
            emd_per_example = compute_squared_emd_per_example(
                probabilities,
                soft_targets,
            )
            if use_balanced_sampler:
                emd_loss = emd_per_example.mean()
            else:
                sample_weights = class_weights[targets]
                emd_loss = (emd_per_example * sample_weights).sum() / sample_weights.sum().clamp_min(
                    1e-8
                )
            loss = loss + score_band_emd_weight * emd_loss

        if ordinal_loss_weight == 0:
            return loss

        expected_class_index = (probabilities * class_positions).sum(dim=1)
        ordinal_loss = F.smooth_l1_loss(
            expected_class_index,
            targets.float(),
        )
        return loss + ordinal_loss_weight * ordinal_loss

    return loss_fn


def build_ordinal_classification_loss(
    *,
    pos_weights: torch.Tensor,
    num_classes: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ordinal_targets = build_cumulative_ordinal_targets(
            targets,
            num_classes=num_classes,
        )
        return bce(logits, ordinal_targets)

    return loss_fn


def model_output_dim_for_target_mode(
    target_label_mode: str,
    *,
    num_classes: int,
) -> int:
    if uses_ordinal_coarse_training(target_label_mode):
        return num_classes - 1
    return num_classes


def get_current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_training_artifacts(
    *,
    output_dir: Path,
    model: nn.Module,
    result: TrainingResult,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(training_result_to_dict(result), indent=2))

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
    }


def train_synthetic_classifier(
    manifest_path: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 8,
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    optimizer_name: str = "adamw",
    lr_scheduler_name: str = "reduce_on_plateau",
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 10,
    min_learning_rate: float = 1e-5,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    use_balanced_sampler: bool | None = None,
    weight_decay: float = 1e-4,
    dropout_p: float = 0.1,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    overfit_subset_size: int | None = None,
    use_augmentation: bool = True,
    augmentation_profile: str = "mild",
    model_variant: str = "simple_cnn",
    pretrained: bool = False,
    freeze_backbone: bool = False,
    ordinal_loss_weight: float = 0.0,
    score_band_soft_label_sigma: float = 1.0,
    score_band_emd_weight: float = 0.0,
    score_band_class_weight_strategy: str = "effective",
    score_band_effective_beta: float = 0.999,
    target_label_mode: str = "coarse",
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
    init_from_checkpoint: str | Path | None = None,
    checkpoint_every_n_epochs: int = 1,
    plot_every_n_epochs: int = 1,
) -> TrainingResult:
    """
    Train the first coarse classifier on the synthetic manifest.
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if min_learning_rate < 0:
        raise ValueError(
            f"min_learning_rate must be non-negative, got {min_learning_rate}"
        )
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")
    if early_stopping_min_delta < 0:
        raise ValueError(
            f"early_stopping_min_delta must be non-negative, got {early_stopping_min_delta}"
        )
    if ordinal_loss_weight < 0:
        raise ValueError(
            f"ordinal_loss_weight must be non-negative, got {ordinal_loss_weight}"
        )
    if score_band_soft_label_sigma <= 0:
        raise ValueError(
            "score_band_soft_label_sigma must be positive, "
            f"got {score_band_soft_label_sigma}"
        )
    if score_band_emd_weight < 0:
        raise ValueError(
            f"score_band_emd_weight must be non-negative, got {score_band_emd_weight}"
        )
    if score_band_class_weight_strategy not in {"balanced", "effective"}:
        raise ValueError(
            "score_band_class_weight_strategy must be 'balanced' or 'effective', "
            f"got {score_band_class_weight_strategy!r}"
        )
    if not 0.0 <= score_band_effective_beta < 1.0:
        raise ValueError(
            "score_band_effective_beta must be in [0.0, 1.0), "
            f"got {score_band_effective_beta}"
        )
    if checkpoint_every_n_epochs <= 0:
        raise ValueError(
            "checkpoint_every_n_epochs must be positive, "
            f"got {checkpoint_every_n_epochs}"
        )
    if plot_every_n_epochs < 0:
        raise ValueError(f"plot_every_n_epochs must be non-negative, got {plot_every_n_epochs}")
    validate_split_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    set_global_seed(random_seed)
    device = choose_device()
    manifest_path = Path(manifest_path).resolve()
    normalized_output_dir = Path(output_dir).resolve() if output_dir is not None else None
    normalized_resume_from = Path(resume_from).resolve() if resume_from is not None else None
    normalized_init_from = (
        Path(init_from_checkpoint).resolve()
        if init_from_checkpoint is not None
        else None
    )
    label_column, class_names, _dataset_target_mode = get_target_label_spec(
        target_label_mode
    )
    if use_balanced_sampler is None:
        resolved_use_balanced_sampler = target_label_mode != "score_band"
    else:
        resolved_use_balanced_sampler = use_balanced_sampler
    selection_metric_name, selection_metric_mode = get_selection_metric_rule(
        target_label_mode
    )
    output_dim = model_output_dim_for_target_mode(
        target_label_mode,
        num_classes=len(class_names),
    )

    train_loader, val_loader, test_loader, frame, split, include_resize = build_dataloaders(
        manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        random_seed=random_seed,
        use_balanced_sampler=resolved_use_balanced_sampler,
        overfit_subset_size=overfit_subset_size,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
        target_label_mode=target_label_mode,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_strategy=split_strategy,
        split_group_column=split_group_column,
    )

    run_config = build_run_config(
        manifest_path=manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        lr_scheduler_name=lr_scheduler_name,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        min_learning_rate=min_learning_rate,
        random_seed=random_seed,
        device=device,
        include_resize=include_resize,
        use_balanced_sampler=resolved_use_balanced_sampler,
        weight_decay=weight_decay,
        dropout_p=dropout_p,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        overfit_subset_size=overfit_subset_size,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
        model_variant=model_variant,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        ordinal_loss_weight=ordinal_loss_weight,
        score_band_soft_label_sigma=score_band_soft_label_sigma,
        score_band_emd_weight=score_band_emd_weight,
        score_band_class_weight_strategy=score_band_class_weight_strategy,
        score_band_effective_beta=score_band_effective_beta,
        target_label_mode=target_label_mode,
        class_names=class_names,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_strategy=split_strategy,
        split_group_column=split_group_column,
        init_from_checkpoint=normalized_init_from,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        plot_every_n_epochs=plot_every_n_epochs,
    )

    if normalized_output_dir is not None:
        save_run_config(
            output_dir=normalized_output_dir,
            config=run_config,
        )
        save_split_artifacts(
            frame,
            split,
            output_dir=normalized_output_dir,
            split_strategy=split_strategy,
            split_group_column=split_group_column,
        )

    model = build_classifier_model(
        num_classes=output_dim,
        dropout_p=dropout_p,
        model_variant=model_variant,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)
    if normalized_init_from is not None:
        checkpoint = load_training_checkpoint(normalized_init_from)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)  # type: ignore[arg-type]
    prediction_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    if uses_ordinal_coarse_training(target_label_mode):
        loss_fn = build_ordinal_classification_loss(
            pos_weights=build_ordinal_pos_weights(
                frame,
                train_indices=split.train_indices,
                label_column=label_column,
                num_classes=len(class_names),
                device=device,
            ),
            num_classes=len(class_names),
        )
        prediction_fn = decode_cumulative_ordinal_logits
    else:
        if target_label_mode == "score_band":
            if score_band_class_weight_strategy == "effective":
                class_weights = build_effective_number_class_weights(
                    frame,
                    train_indices=split.train_indices,
                    device=device,
                    label_column=label_column,
                    class_names=class_names,
                    beta=score_band_effective_beta,
                )
            else:
                class_weights = build_class_weights(
                    frame,
                    train_indices=split.train_indices,
                    device=device,
                    label_column=label_column,
                    class_names=class_names,
                )
            loss_fn = build_score_band_soft_label_loss(
                class_weights=class_weights,
                use_balanced_sampler=resolved_use_balanced_sampler,
                ordinal_loss_weight=ordinal_loss_weight,
                score_band_emd_weight=score_band_emd_weight,
                device=device,
                num_classes=len(class_names),
                soft_label_sigma=score_band_soft_label_sigma,
            )
        else:
            class_weights = build_class_weights(
                frame,
                train_indices=split.train_indices,
                device=device,
                label_column=label_column,
                class_names=class_names,
            )
            loss_fn = build_classification_loss(
                class_weights=class_weights,
                use_balanced_sampler=resolved_use_balanced_sampler,
                ordinal_loss_weight=ordinal_loss_weight,
                device=device,
                num_classes=len(class_names),
            )
    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        lr_scheduler_name=lr_scheduler_name,
        selection_metric_mode=selection_metric_mode,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        min_learning_rate=min_learning_rate,
    )

    train_history: list[EpochResult] = []
    val_history: list[EpochResult] = []

    best_val_f1 = -float("inf") if selection_metric_mode == "max" else float("inf")
    best_val_epoch = 1
    best_state_dict = copy.deepcopy(model.state_dict())
    start_epoch = 1
    epoch_metric_history: list[dict[str, object]] = []
    early_stopping = EarlyStoppingTracker(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode=selection_metric_mode,
    )
    early_stopped = False
    stopped_epoch: int | None = None

    if normalized_resume_from is not None:
        checkpoint = load_training_checkpoint(normalized_resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[arg-type]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore[arg-type]
        move_optimizer_state_to_device(optimizer, device)
        if lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict"):
            lr_scheduler.load_state_dict(  # type: ignore[arg-type]
                checkpoint["lr_scheduler_state_dict"]
            )

        loaded_epoch = int(checkpoint["epoch"])
        start_epoch = loaded_epoch + 1
        if start_epoch > num_epochs:
            raise ValueError(
                f"Checkpoint is from epoch {loaded_epoch}, but num_epochs={num_epochs}. "
                "Increase num_epochs to continue training."
            )

        if normalized_output_dir is not None:
            epoch_metric_history = [
                row
                for row in read_epoch_metrics(normalized_output_dir)
                if int(row["epoch"]) < start_epoch
            ]
            for row in epoch_metric_history:
                train_history.append(
                    epoch_result_from_dict(
                        {
                            "loss": row["train_loss"],
                            "accuracy": row["train_accuracy"],
                            "macro_f1": row["train_macro_f1"],
                            "confusion": [],
                            "mean_band_error": row.get("train_mean_band_error"),
                            "within_one_band_accuracy": row.get(
                                "train_within_one_band_accuracy"
                            ),
                            "severe_band_error_rate": row.get(
                                "train_severe_band_error_rate"
                            ),
                            "collapsed_coarse_accuracy": row.get(
                                "train_collapsed_coarse_accuracy"
                            ),
                            "collapsed_coarse_macro_f1": row.get(
                                "train_collapsed_coarse_macro_f1"
                            ),
                        }
                    )
                )
                val_history.append(
                    epoch_result_from_dict(
                        {
                            "loss": row["val_loss"],
                            "accuracy": row["val_accuracy"],
                            "macro_f1": row["val_macro_f1"],
                            "confusion": [],
                            "mean_band_error": row.get("val_mean_band_error"),
                            "within_one_band_accuracy": row.get(
                                "val_within_one_band_accuracy"
                            ),
                            "severe_band_error_rate": row.get(
                                "val_severe_band_error_rate"
                            ),
                            "collapsed_coarse_accuracy": row.get(
                                "val_collapsed_coarse_accuracy"
                            ),
                            "collapsed_coarse_macro_f1": row.get(
                                "val_collapsed_coarse_macro_f1"
                            ),
                        }
                    )
                )
                historical_metric = row.get("val_selection_metric")
                decision = early_stopping.update(
                    epoch_number=int(row["epoch"]),
                    metric_value=float(
                        historical_metric
                        if historical_metric is not None
                        else row["val_macro_f1"]
                    ),
                )
                if decision.is_best:
                    best_val_f1 = float(
                        row.get("val_selection_metric", row["val_macro_f1"])
                    )
                    best_val_epoch = int(row["epoch"])

            best_checkpoint_path = normalized_output_dir / "checkpoints/best.pt"
            if best_checkpoint_path.exists():
                best_checkpoint = load_training_checkpoint(best_checkpoint_path)
                best_state_dict = copy.deepcopy(best_checkpoint["model_state_dict"])  # type: ignore[arg-type]

    for epoch_number in range(start_epoch, num_epochs + 1):

        train_metrics = run_one_epoch(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            target_label_mode=target_label_mode,
            class_names=class_names,
            prediction_fn=prediction_fn,
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
            target_label_mode=target_label_mode,
            class_names=class_names,
            prediction_fn=prediction_fn,
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        selection_metric_value = extract_epoch_selection_metric(
            val_metrics,
            target_label_mode=target_label_mode,
        )
        decision = early_stopping.update(
            epoch_number=epoch_number,
            metric_value=selection_metric_value,
        )
        is_best = decision.is_best
        if is_best:
            best_val_f1 = selection_metric_value
            best_val_epoch = epoch_number
            best_state_dict = copy.deepcopy(model.state_dict())

        if lr_scheduler is not None:
            lr_scheduler.step(selection_metric_value)
        current_learning_rate = get_current_learning_rate(optimizer)

        if normalized_output_dir is not None:
            flat_metrics = append_epoch_metrics(
                output_dir=normalized_output_dir,
                epoch_number=epoch_number,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
                learning_rate=current_learning_rate,
                class_names=class_names,
                selection_metric_name=selection_metric_name,
                val_selection_metric=selection_metric_value,
            )
            epoch_metric_history.append(flat_metrics)
            if is_best or epoch_number % checkpoint_every_n_epochs == 0:
                save_epoch_checkpoint(
                    output_dir=normalized_output_dir,
                    epoch_number=epoch_number,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    is_best=is_best,
                    config=run_config,
                )
            if plot_every_n_epochs > 0 and epoch_number % plot_every_n_epochs == 0:
                plot_epoch_metrics(
                    output_dir=normalized_output_dir,
                    history=epoch_metric_history,
                )
                plot_class_monitoring(
                    output_dir=normalized_output_dir,
                    history=epoch_metric_history,
                    class_names=class_names,
                )

        print(
            f"Epoch {epoch_number}/{num_epochs} | "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.4f} "
            f"train_f1={train_metrics.macro_f1:.4f} | "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} "
            f"val_f1={val_metrics.macro_f1:.4f} "
            f"{selection_metric_name}={selection_metric_value:.4f} "
            f"lr={current_learning_rate:.6g}",
            flush=True,
        )

        if decision.should_stop:
            early_stopped = True
            stopped_epoch = epoch_number
            print(
                f"Early stopping at epoch {epoch_number}; "
                f"best_val_epoch={best_val_epoch} "
                f"{selection_metric_name}={best_val_f1:.4f}",
                flush=True,
            )
            break

    model.load_state_dict(best_state_dict)

    test_metrics = run_one_epoch(
        model,
        test_loader,
        device=device,
        loss_fn=loss_fn,
        optimizer=None,
        target_label_mode=target_label_mode,
        class_names=class_names,
        prediction_fn=prediction_fn,
    )

    result = TrainingResult(
        train_history=train_history,
        val_history=val_history,
        test_metrics=test_metrics,
        best_val_epoch=best_val_epoch,
        device=str(device),
        used_resize_in_transforms=include_resize,
        output_dir=str(normalized_output_dir) if normalized_output_dir is not None else None,
        selection_metric_name=selection_metric_name,
        best_val_metric=best_val_f1,
        early_stopped=early_stopped,
        stopped_epoch=stopped_epoch,
    )

    if normalized_output_dir is not None:
        if epoch_metric_history:
            plot_epoch_metrics(
                output_dir=normalized_output_dir,
                history=epoch_metric_history,
            )
            plot_class_monitoring(
                output_dir=normalized_output_dir,
                history=epoch_metric_history,
                class_names=class_names,
            )
        save_training_artifacts(
            output_dir=normalized_output_dir,
            model=model,
            result=result,
        )

    print(f"\nBest validation epoch: {best_val_epoch}")
    print(f"Selection metric ({selection_metric_name}): {best_val_f1:.4f}")
    print(
        f"Test | "
        f"loss={test_metrics.loss:.4f} "
        f"acc={test_metrics.accuracy:.4f} "
        f"macro_f1={test_metrics.macro_f1:.4f}"
    )
    if target_label_mode == "score_band":
        print(
            "Test score-band summary | "
            f"mean_band_error={test_metrics.mean_band_error:.4f} "
            f"within_one={test_metrics.within_one_band_accuracy:.4f} "
            f"severe_error_rate={test_metrics.severe_band_error_rate:.4f}"
        )
        print(
            "Collapsed coarse | "
            f"acc={test_metrics.collapsed_coarse_accuracy:.4f} "
            f"macro_f1={test_metrics.collapsed_coarse_macro_f1:.4f}"
        )
    print("Test confusion matrix:")
    for row in test_metrics.confusion:
        print(row)

    return result
