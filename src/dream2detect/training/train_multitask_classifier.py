from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
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
from .metrics import (
    ClassificationMetrics,
    ScalarScoreBandMetrics,
    compute_classification_metrics,
    compute_scalar_score_band_metrics,
)
from .models import (
    ResidualCNNMultiTaskClassifier,
    ResidualGroupNormCNNMultiTaskClassifier,
)
from .splits import DatasetSplit, build_stratified_splits
from .train_classifier import (
    EarlyStoppingTracker,
    build_class_weights,
    build_lr_scheduler,
    build_optimizer,
    build_sample_weights,
    get_current_learning_rate,
    infer_resize_policy,
    load_training_checkpoint,
    move_optimizer_state_to_device,
    save_split_artifacts,
    set_global_seed,
    validate_split_fractions,
    choose_device,
)


@dataclass(frozen=True)
class MultiTaskEpochResult:
    total_loss: float
    scalar_loss: float
    coarse_loss: float
    score_band_loss: float
    scalar_metrics: ScalarScoreBandMetrics
    coarse_metrics: ClassificationMetrics
    score_band_metrics: ClassificationMetrics


@dataclass(frozen=True)
class MultiTaskTrainingResult:
    train_history: list[MultiTaskEpochResult]
    val_history: list[MultiTaskEpochResult]
    test_metrics: MultiTaskEpochResult
    best_val_epoch: int
    device: str
    used_resize_in_transforms: bool
    output_dir: str | None
    early_stopped: bool = False
    stopped_epoch: int | None = None


def build_multitask_dataloaders(
    manifest_path: str | Path,
    *,
    image_size: int,
    batch_size: int,
    random_seed: int,
    use_balanced_sampler: bool,
    use_augmentation: bool,
    augmentation_profile: str,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_strategy: str = "metadata_family_holdout",
    split_group_column: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, DatasetSplit, bool]:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    split = build_stratified_splits(
        manifest_path,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
        label_column="training_score_band",
        split_strategy=split_strategy,
        split_group_column=split_group_column,
    )
    include_resize = infer_resize_policy(frame, image_size=image_size)

    train_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode="multitask",
        transform=build_train_transform(
            image_size=image_size,
            include_resize=include_resize,
            use_augmentation=use_augmentation,
            augmentation_profile="none" if not use_augmentation else augmentation_profile,
        ),
    )
    eval_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode="multitask",
        transform=build_eval_transform(
            image_size=image_size,
            include_resize=include_resize,
        ),
    )

    train_sampler = None
    train_shuffle = True
    if use_balanced_sampler:
        train_sampler = WeightedRandomSampler(
            weights=build_sample_weights(
                frame,
                train_indices=split.train_indices,
                label_column="training_score_band",
            ),
            num_samples=len(split.train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_shuffle = False

    train_loader = DataLoader(
        Subset(train_dataset, split.train_indices),
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, split.val_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, split.test_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, frame, split, include_resize


def build_multitask_loss(
    *,
    coarse_class_weights: torch.Tensor,
    score_band_weights: torch.Tensor,
    use_balanced_sampler: bool,
    scalar_loss_weight: float,
    coarse_loss_weight: float,
    auxiliary_band_loss_weight: float,
    band_ordinal_loss_weight: float,
    device: torch.device,
) -> Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor]:
    if scalar_loss_weight < 0:
        raise ValueError(
            f"scalar_loss_weight must be non-negative, got {scalar_loss_weight}"
        )
    if coarse_loss_weight < 0:
        raise ValueError(
            f"coarse_loss_weight must be non-negative, got {coarse_loss_weight}"
        )
    if auxiliary_band_loss_weight < 0:
        raise ValueError(
            "auxiliary_band_loss_weight must be non-negative, "
            f"got {auxiliary_band_loss_weight}"
        )
    if band_ordinal_loss_weight < 0:
        raise ValueError(
            f"band_ordinal_loss_weight must be non-negative, got {band_ordinal_loss_weight}"
        )

    coarse_loss = (
        nn.CrossEntropyLoss()
        if use_balanced_sampler
        else nn.CrossEntropyLoss(weight=coarse_class_weights)
    )
    score_band_loss = (
        nn.CrossEntropyLoss()
        if use_balanced_sampler
        else nn.CrossEntropyLoss(weight=score_band_weights)
    )
    band_positions = torch.arange(len(SCORE_BAND_NAMES), dtype=torch.float32, device=device)
    scalar_loss = nn.SmoothL1Loss()

    def loss_fn(
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        primary_loss = scalar_loss_weight * scalar_loss(
            outputs["score"],
            targets["score"],
        )
        if coarse_loss_weight > 0:
            primary_loss = primary_loss + coarse_loss_weight * coarse_loss(
                outputs["coarse_logits"],
                targets["coarse"],
            )
        if auxiliary_band_loss_weight == 0:
            return primary_loss

        auxiliary_loss = score_band_loss(
            outputs["score_band_logits"],
            targets["score_band"],
        )
        if band_ordinal_loss_weight > 0:
            probabilities = torch.softmax(outputs["score_band_logits"], dim=1)
            expected_band_index = (probabilities * band_positions).sum(dim=1)
            auxiliary_loss = auxiliary_loss + band_ordinal_loss_weight * F.smooth_l1_loss(
                expected_band_index,
                targets["score_band"].float(),
            )

        return primary_loss + auxiliary_band_loss_weight * auxiliary_loss

    return loss_fn


def _loss_parts(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    scalar_loss_fn: nn.Module,
    coarse_loss_fn: nn.Module,
    score_band_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        scalar_loss_fn(outputs["score"], targets["score"]),
        coarse_loss_fn(outputs["coarse_logits"], targets["coarse"]),
        score_band_loss_fn(outputs["score_band_logits"], targets["score_band"]),
    )


def run_multitask_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    loss_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor],
    scalar_loss_fn: nn.Module,
    coarse_loss_fn: nn.Module,
    score_band_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> MultiTaskEpochResult:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_scalar_loss = 0.0
    total_coarse_loss = 0.0
    total_score_band_loss = 0.0
    total_examples = 0

    score_predictions: list[torch.Tensor] = []
    score_targets: list[torch.Tensor] = []
    coarse_logits: list[torch.Tensor] = []
    coarse_targets: list[torch.Tensor] = []
    score_band_logits: list[torch.Tensor] = []
    score_band_targets: list[torch.Tensor] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = {
            "coarse": batch["target"]["coarse"].to(device),
            "score_band": batch["target"]["score_band"].to(device),
            "score": batch["target"]["score"].to(device),
        }

        if is_training:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, targets)

        with torch.no_grad():
            scalar_loss, coarse_loss, score_band_loss = _loss_parts(
                outputs,
                targets,
                scalar_loss_fn=scalar_loss_fn,
                coarse_loss_fn=coarse_loss_fn,
                score_band_loss_fn=score_band_loss_fn,
            )

        current_batch_size = images.size(0)
        total_loss += float(loss.item()) * current_batch_size
        total_scalar_loss += float(scalar_loss.item()) * current_batch_size
        total_coarse_loss += float(coarse_loss.item()) * current_batch_size
        total_score_band_loss += float(score_band_loss.item()) * current_batch_size
        total_examples += current_batch_size

        score_predictions.append(outputs["score"].detach().cpu())
        score_targets.append(targets["score"].detach().cpu())
        coarse_logits.append(outputs["coarse_logits"].detach().cpu())
        coarse_targets.append(targets["coarse"].detach().cpu())
        score_band_logits.append(outputs["score_band_logits"].detach().cpu())
        score_band_targets.append(targets["score_band"].detach().cpu())

    if total_examples == 0:
        raise ValueError("Dataloader produced zero examples.")

    coarse_metrics = compute_classification_metrics(
        torch.cat(coarse_logits, dim=0),
        torch.cat(coarse_targets, dim=0),
        num_classes=len(COARSE_CLASS_NAMES),
        class_names=COARSE_CLASS_NAMES,
    )
    score_band_metrics = compute_classification_metrics(
        torch.cat(score_band_logits, dim=0),
        torch.cat(score_band_targets, dim=0),
        num_classes=len(SCORE_BAND_NAMES),
        class_names=SCORE_BAND_NAMES,
    )

    return MultiTaskEpochResult(
        total_loss=total_loss / total_examples,
        scalar_loss=total_scalar_loss / total_examples,
        coarse_loss=total_coarse_loss / total_examples,
        score_band_loss=total_score_band_loss / total_examples,
        scalar_metrics=compute_scalar_score_band_metrics(
            normalized_predictions=torch.cat(score_predictions),
            normalized_targets=torch.cat(score_targets),
        ),
        coarse_metrics=coarse_metrics,
        score_band_metrics=score_band_metrics,
    )


def summarize_score_band_errors(confusion: list[list[int]]) -> dict[str, float | int]:
    total = sum(sum(row) for row in confusion)
    if total == 0:
        raise ValueError("Cannot summarize an empty confusion matrix.")

    exact = 0
    within_one = 0
    severe = 0
    ordinal_error = 0
    for true_index, row in enumerate(confusion):
        for predicted_index, count in enumerate(row):
            distance = abs(true_index - predicted_index)
            if distance == 0:
                exact += count
            if distance <= 1:
                within_one += count
            if distance >= 2:
                severe += count
            ordinal_error += distance * count

    return {
        "total": total,
        "exact_accuracy": exact / total,
        "within_one_band_accuracy": within_one / total,
        "mean_band_error": ordinal_error / total,
        "severe_band_error_rate": severe / total,
    }


def collapse_band_confusion_to_coarse(confusion: list[list[int]]) -> list[list[int]]:
    band_to_coarse = {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 3,
    }
    collapsed = [[0 for _ in COARSE_CLASS_NAMES] for _ in COARSE_CLASS_NAMES]
    for true_band, row in enumerate(confusion):
        for predicted_band, count in enumerate(row):
            collapsed[band_to_coarse[true_band]][band_to_coarse[predicted_band]] += count
    return collapsed


def macro_f1_from_confusion(confusion: list[list[int]]) -> float:
    f1_values: list[float] = []
    for class_index in range(len(confusion)):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(
            confusion[row_index][class_index]
            for row_index in range(len(confusion))
            if row_index != class_index
        )
        false_negative = sum(
            confusion[class_index][column_index]
            for column_index in range(len(confusion))
            if column_index != class_index
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(f1_values) / len(f1_values)


def within_band_accuracy_from_confusion(
    confusion: list[list[int]],
    *,
    max_distance: int,
) -> float:
    total = sum(sum(row) for row in confusion)
    if total == 0:
        raise ValueError("Cannot summarize an empty confusion matrix.")
    correct = 0
    for true_index, row in enumerate(confusion):
        for predicted_index, count in enumerate(row):
            if abs(true_index - predicted_index) <= max_distance:
                correct += count
    return correct / total


def epoch_result_to_dict(result: MultiTaskEpochResult) -> dict[str, object]:
    return {
        "total_loss": result.total_loss,
        "scalar_loss": result.scalar_loss,
        "coarse_loss": result.coarse_loss,
        "score_band_loss": result.score_band_loss,
        "scalar": {
            "score_mae": result.scalar_metrics.score_mae,
            "score_rmse": result.scalar_metrics.score_rmse,
            "band_accuracy": result.scalar_metrics.band_metrics.accuracy,
            "band_macro_f1": result.scalar_metrics.band_metrics.macro_f1,
            "band_confusion": result.scalar_metrics.band_metrics.confusion,
            "mean_band_error": result.scalar_metrics.band_ordinal_errors[
                "mean_band_error"
            ],
            "within_one_band_accuracy": result.scalar_metrics.band_ordinal_errors[
                "within_one_band_accuracy"
            ],
            "within_two_band_accuracy": within_band_accuracy_from_confusion(
                result.scalar_metrics.band_metrics.confusion,
                max_distance=2,
            ),
            "collapsed_coarse_accuracy": result.scalar_metrics.coarse_metrics.accuracy,
            "collapsed_coarse_macro_f1": result.scalar_metrics.coarse_metrics.macro_f1,
            "collapsed_coarse_confusion": result.scalar_metrics.coarse_metrics.confusion,
            "predicted_class_distribution": (
                result.scalar_metrics.band_metrics.predicted_class_distribution
            ),
            "target_class_distribution": (
                result.scalar_metrics.band_metrics.target_class_distribution
            ),
        },
        "coarse": {
            "accuracy": result.coarse_metrics.accuracy,
            "macro_f1": result.coarse_metrics.macro_f1,
            "confusion": result.coarse_metrics.confusion,
            "predicted_class_distribution": result.coarse_metrics.predicted_class_distribution,
            "target_class_distribution": result.coarse_metrics.target_class_distribution,
        },
        "score_band": {
            "accuracy": result.score_band_metrics.accuracy,
            "macro_f1": result.score_band_metrics.macro_f1,
            "confusion": result.score_band_metrics.confusion,
            "predicted_class_distribution": result.score_band_metrics.predicted_class_distribution,
            "target_class_distribution": result.score_band_metrics.target_class_distribution,
        },
    }


def training_result_to_dict(result: MultiTaskTrainingResult) -> dict[str, object]:
    return {
        "train_history": [epoch_result_to_dict(item) for item in result.train_history],
        "val_history": [epoch_result_to_dict(item) for item in result.val_history],
        "test_metrics": epoch_result_to_dict(result.test_metrics),
        "best_val_epoch": result.best_val_epoch,
        "device": result.device,
        "used_resize_in_transforms": result.used_resize_in_transforms,
        "output_dir": result.output_dir,
        "early_stopped": result.early_stopped,
        "stopped_epoch": result.stopped_epoch,
    }


def classification_metrics_from_dict(data: dict[str, object]) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(data["accuracy"]),
        macro_f1=float(data["macro_f1"]),
        confusion=data["confusion"],  # type: ignore[arg-type]
        per_class={},
        predicted_class_distribution=data.get(  # type: ignore[arg-type]
            "predicted_class_distribution",
            {},
        ),
        target_class_distribution=data.get(  # type: ignore[arg-type]
            "target_class_distribution",
            {},
        ),
    )


def scalar_metrics_from_dict(data: dict[str, object]) -> ScalarScoreBandMetrics:
    band_confusion = data["band_confusion"]  # type: ignore[assignment]
    total = sum(sum(row) for row in band_confusion)  # type: ignore[arg-type]
    band_metrics = ClassificationMetrics(
        accuracy=float(data["band_accuracy"]),
        macro_f1=float(data["band_macro_f1"]),
        confusion=band_confusion,  # type: ignore[arg-type]
        per_class={},
        predicted_class_distribution=data.get(  # type: ignore[arg-type]
            "predicted_class_distribution",
            {},
        ),
        target_class_distribution=data.get(  # type: ignore[arg-type]
            "target_class_distribution",
            {},
        ),
    )
    coarse_confusion = data["collapsed_coarse_confusion"]  # type: ignore[assignment]
    coarse_metrics = ClassificationMetrics(
        accuracy=float(data["collapsed_coarse_accuracy"]),
        macro_f1=float(data["collapsed_coarse_macro_f1"]),
        confusion=coarse_confusion,  # type: ignore[arg-type]
        per_class={},
        predicted_class_distribution={},
        target_class_distribution={},
    )
    return ScalarScoreBandMetrics(
        normalized_mae=float(data["score_mae"]) / 100.0,
        normalized_rmse=float(data["score_rmse"]) / 100.0,
        score_mae=float(data["score_mae"]),
        score_rmse=float(data["score_rmse"]),
        band_metrics=band_metrics,
        coarse_metrics=coarse_metrics,
        band_ordinal_errors={
            "total": total,
            "exact_accuracy": float(data["band_accuracy"]),
            "within_one_band_accuracy": float(data["within_one_band_accuracy"]),
            "mean_band_error": float(data["mean_band_error"]),
            "severe_band_error_rate": 1.0
            - float(data["within_two_band_accuracy"]),
        },
    )


def epoch_result_from_dict(data: dict[str, object]) -> MultiTaskEpochResult:
    return MultiTaskEpochResult(
        total_loss=float(data["total_loss"]),
        scalar_loss=float(data["scalar_loss"]),
        coarse_loss=float(data["coarse_loss"]),
        score_band_loss=float(data["score_band_loss"]),
        scalar_metrics=scalar_metrics_from_dict(data["scalar"]),  # type: ignore[arg-type]
        coarse_metrics=classification_metrics_from_dict(data["coarse"]),  # type: ignore[arg-type]
        score_band_metrics=classification_metrics_from_dict(data["score_band"]),  # type: ignore[arg-type]
    )


def read_multitask_history(
    output_dir: Path,
) -> tuple[list[MultiTaskEpochResult], list[MultiTaskEpochResult]]:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return [], []

    metrics = json.loads(metrics_path.read_text())
    return (
        [
            epoch_result_from_dict(row)
            for row in metrics.get("train_history", [])
        ],
        [
            epoch_result_from_dict(row)
            for row in metrics.get("val_history", [])
        ],
    )


def save_multitask_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    train_metrics: MultiTaskEpochResult,
    val_metrics: MultiTaskEpochResult,
    epoch_number: int,
    is_best: bool,
    config: dict[str, object],
) -> None:
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
    torch.save(payload, checkpoint_dir / "latest.pt")
    if is_best:
        torch.save(payload, checkpoint_dir / "best.pt")
    if epoch_number % int(config["checkpoint_every_n_epochs"]) == 0:
        torch.save(payload, checkpoint_dir / f"epoch_{epoch_number:03d}.pt")


def save_multitask_artifacts(
    *,
    output_dir: Path,
    model: nn.Module,
    result: MultiTaskTrainingResult,
    config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "metrics.json").write_text(
        json.dumps(training_result_to_dict(result), indent=2)
    )
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    band_summary = summarize_score_band_errors(
        result.test_metrics.score_band_metrics.confusion
    )
    collapsed = collapse_band_confusion_to_coarse(
        result.test_metrics.score_band_metrics.confusion
    )
    collapsed_total = sum(sum(row) for row in collapsed)
    collapsed_correct = sum(collapsed[index][index] for index in range(len(collapsed)))
    diagnostics = {
        "primary_scalar_score_mae": result.test_metrics.scalar_metrics.score_mae,
        "primary_scalar_band_accuracy": (
            result.test_metrics.scalar_metrics.band_metrics.accuracy
        ),
        "primary_scalar_band_macro_f1": (
            result.test_metrics.scalar_metrics.band_metrics.macro_f1
        ),
        "primary_scalar_mean_band_error": (
            result.test_metrics.scalar_metrics.band_ordinal_errors["mean_band_error"]
        ),
        "primary_scalar_within_one_band_accuracy": (
            result.test_metrics.scalar_metrics.band_ordinal_errors[
                "within_one_band_accuracy"
            ]
        ),
        "primary_scalar_within_two_band_accuracy": within_band_accuracy_from_confusion(
            result.test_metrics.scalar_metrics.band_metrics.confusion,
            max_distance=2,
        ),
        "primary_scalar_collapsed_coarse_macro_f1": (
            result.test_metrics.scalar_metrics.coarse_metrics.macro_f1
        ),
        "primary_coarse_accuracy": result.test_metrics.coarse_metrics.accuracy,
        "primary_coarse_macro_f1": result.test_metrics.coarse_metrics.macro_f1,
        "primary_coarse_confusion": result.test_metrics.coarse_metrics.confusion,
        "auxiliary_score_band_accuracy": result.test_metrics.score_band_metrics.accuracy,
        "auxiliary_score_band_macro_f1": result.test_metrics.score_band_metrics.macro_f1,
        "auxiliary_score_band_confusion": result.test_metrics.score_band_metrics.confusion,
        "auxiliary_within_one_band_accuracy": band_summary["within_one_band_accuracy"],
        "auxiliary_mean_band_error": band_summary["mean_band_error"],
        "auxiliary_severe_band_error_rate": band_summary["severe_band_error_rate"],
        "auxiliary_collapsed_coarse_accuracy": collapsed_correct / collapsed_total,
        "auxiliary_collapsed_coarse_macro_f1": macro_f1_from_confusion(collapsed),
        "auxiliary_collapsed_coarse_confusion": collapsed,
    }
    (output_dir / "multitask_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )


def train_multitask_classifier(
    manifest_path: str | Path,
    *,
    image_size: int = 384,
    batch_size: int = 16,
    num_epochs: int = 60,
    learning_rate: float = 3e-4,
    optimizer_name: str = "adamw",
    lr_scheduler_name: str = "reduce_on_plateau",
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 8,
    min_learning_rate: float = 1e-5,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    use_balanced_sampler: bool = False,
    weight_decay: float = 1e-4,
    dropout_p: float = 0.2,
    early_stopping_patience: int | None = 15,
    early_stopping_min_delta: float = 0.0,
    use_augmentation: bool = True,
    augmentation_profile: str = "damage_safe",
    model_variant: str = "residual_cnn_groupnorm_multitask",
    scalar_loss_weight: float = 1.0,
    coarse_loss_weight: float = 0.3,
    auxiliary_band_loss_weight: float = 0.3,
    band_ordinal_loss_weight: float = 0.2,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_strategy: str = "metadata_family_holdout",
    split_group_column: str | None = None,
    resume_from: str | Path | None = None,
    checkpoint_every_n_epochs: int = 10,
) -> MultiTaskTrainingResult:
    validate_split_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    if model_variant not in {
        "residual_cnn_multitask",
        "residual_cnn_groupnorm_multitask",
    }:
        raise ValueError(f"Unsupported multitask model_variant: {model_variant}")
    if scalar_loss_weight < 0:
        raise ValueError(
            f"scalar_loss_weight must be non-negative, got {scalar_loss_weight}"
        )
    if coarse_loss_weight < 0:
        raise ValueError(
            f"coarse_loss_weight must be non-negative, got {coarse_loss_weight}"
        )
    if auxiliary_band_loss_weight < 0:
        raise ValueError(
            "auxiliary_band_loss_weight must be non-negative, "
            f"got {auxiliary_band_loss_weight}"
        )
    if band_ordinal_loss_weight < 0:
        raise ValueError(
            f"band_ordinal_loss_weight must be non-negative, got {band_ordinal_loss_weight}"
        )

    set_global_seed(random_seed)
    device = choose_device()
    manifest_path = Path(manifest_path).resolve()
    normalized_output_dir = Path(output_dir).resolve() if output_dir is not None else None
    normalized_resume_from = Path(resume_from).resolve() if resume_from is not None else None

    train_loader, val_loader, test_loader, frame, split, include_resize = (
        build_multitask_dataloaders(
            manifest_path,
            image_size=image_size,
            batch_size=batch_size,
            random_seed=random_seed,
            use_balanced_sampler=use_balanced_sampler,
            use_augmentation=use_augmentation,
            augmentation_profile=augmentation_profile,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            split_strategy=split_strategy,
            split_group_column=split_group_column,
        )
    )

    config = {
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
        "weight_decay": weight_decay,
        "dropout_p": dropout_p,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "use_augmentation": use_augmentation,
        "augmentation_profile": augmentation_profile,
        "model_variant": model_variant,
        "target_label_mode": "multitask_scalar_score_band_coarse",
        "selection_metric_name": "val_scalar_mean_band_error",
        "selection_metric_mode": "min",
        "scalar_loss_weight": scalar_loss_weight,
        "coarse_loss_weight": coarse_loss_weight,
        "auxiliary_band_loss_weight": auxiliary_band_loss_weight,
        "band_ordinal_loss_weight": band_ordinal_loss_weight,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_strategy": split_strategy,
        "split_group_column": split_group_column,
        "resume_from": str(normalized_resume_from)
        if normalized_resume_from is not None
        else None,
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
    }
    if normalized_output_dir is not None:
        normalized_output_dir.mkdir(parents=True, exist_ok=True)
        (normalized_output_dir / "run_config.json").write_text(
            json.dumps(config, indent=2)
        )
        save_split_artifacts(
            frame,
            split,
            output_dir=normalized_output_dir,
            split_strategy=split_strategy,
            split_group_column=split_group_column,
        )

    if model_variant == "residual_cnn_groupnorm_multitask":
        model = ResidualGroupNormCNNMultiTaskClassifier(dropout_p=dropout_p).to(device)
    else:
        model = ResidualCNNMultiTaskClassifier(dropout_p=dropout_p).to(device)
    coarse_weights = build_class_weights(
        frame,
        train_indices=split.train_indices,
        device=device,
        label_column="training_coarse_class",
        class_names=COARSE_CLASS_NAMES,
    )
    score_band_weights = build_class_weights(
        frame,
        train_indices=split.train_indices,
        device=device,
        label_column="training_score_band",
        class_names=SCORE_BAND_NAMES,
    )
    loss_fn = build_multitask_loss(
        coarse_class_weights=coarse_weights,
        score_band_weights=score_band_weights,
        use_balanced_sampler=use_balanced_sampler,
        scalar_loss_weight=scalar_loss_weight,
        coarse_loss_weight=coarse_loss_weight,
        auxiliary_band_loss_weight=auxiliary_band_loss_weight,
        band_ordinal_loss_weight=band_ordinal_loss_weight,
        device=device,
    )
    scalar_loss_fn = nn.SmoothL1Loss()
    coarse_loss_fn = (
        nn.CrossEntropyLoss()
        if use_balanced_sampler
        else nn.CrossEntropyLoss(weight=coarse_weights)
    )
    score_band_loss_fn = (
        nn.CrossEntropyLoss()
        if use_balanced_sampler
        else nn.CrossEntropyLoss(weight=score_band_weights)
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
        selection_metric_mode="min",
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        min_learning_rate=min_learning_rate,
    )
    early_stopping = EarlyStoppingTracker(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode="min",
    )

    train_history: list[MultiTaskEpochResult] = []
    val_history: list[MultiTaskEpochResult] = []
    best_val_epoch = 1
    best_state_dict = copy.deepcopy(model.state_dict())
    start_epoch = 1
    early_stopped = False
    stopped_epoch: int | None = None

    if normalized_resume_from is not None:
        if normalized_output_dir is None:
            raise ValueError("resume_from requires output_dir so history can be restored.")
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

        loaded_train_history, loaded_val_history = read_multitask_history(
            normalized_output_dir
        )
        train_history = loaded_train_history[:loaded_epoch]
        val_history = loaded_val_history[:loaded_epoch]
        for epoch_index, val_metrics in enumerate(val_history, start=1):
            decision = early_stopping.update(
                epoch_number=epoch_index,
                metric_value=float(
                    val_metrics.scalar_metrics.band_ordinal_errors["mean_band_error"]
                ),
            )
            if decision.is_best:
                best_val_epoch = epoch_index

        best_checkpoint_path = normalized_output_dir / "checkpoints/best.pt"
        if best_checkpoint_path.exists():
            best_checkpoint = load_training_checkpoint(best_checkpoint_path)
            best_state_dict = copy.deepcopy(best_checkpoint["model_state_dict"])  # type: ignore[arg-type]
        else:
            best_state_dict = copy.deepcopy(model.state_dict())

    for epoch_number in range(start_epoch, num_epochs + 1):
        train_metrics = run_multitask_epoch(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            scalar_loss_fn=scalar_loss_fn,
            coarse_loss_fn=coarse_loss_fn,
            score_band_loss_fn=score_band_loss_fn,
            optimizer=optimizer,
        )
        val_metrics = run_multitask_epoch(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            scalar_loss_fn=scalar_loss_fn,
            coarse_loss_fn=coarse_loss_fn,
            score_band_loss_fn=score_band_loss_fn,
            optimizer=None,
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        decision = early_stopping.update(
            epoch_number=epoch_number,
            metric_value=float(
                val_metrics.scalar_metrics.band_ordinal_errors["mean_band_error"]
            ),
        )
        if decision.is_best:
            best_val_epoch = epoch_number
            best_state_dict = copy.deepcopy(model.state_dict())

        if lr_scheduler is not None:
            lr_scheduler.step(
                float(val_metrics.scalar_metrics.band_ordinal_errors["mean_band_error"])
            )
        current_learning_rate = get_current_learning_rate(optimizer)

        if normalized_output_dir is not None:
            save_multitask_checkpoint(
                output_dir=normalized_output_dir,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch_number=epoch_number,
                is_best=decision.is_best,
                config=config,
            )

        print(
            f"Epoch {epoch_number}/{num_epochs} | "
            f"train_loss={train_metrics.total_loss:.4f} "
            f"train_scalar_mbe={float(train_metrics.scalar_metrics.band_ordinal_errors['mean_band_error']):.3f} "
            f"train_coarse_f1={train_metrics.coarse_metrics.macro_f1:.4f} "
            f"train_band_acc={train_metrics.score_band_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.total_loss:.4f} "
            f"val_scalar_mbe={float(val_metrics.scalar_metrics.band_ordinal_errors['mean_band_error']):.3f} "
            f"val_coarse_f1={val_metrics.coarse_metrics.macro_f1:.4f} "
            f"val_band_acc={val_metrics.score_band_metrics.accuracy:.4f} "
            f"lr={current_learning_rate:.6g}",
            flush=True,
        )

        if decision.should_stop:
            early_stopped = True
            stopped_epoch = epoch_number
            print(
                f"Early stopping at epoch {epoch_number}; "
                f"best_val_epoch={best_val_epoch}",
                flush=True,
            )
            break

    model.load_state_dict(best_state_dict)
    test_metrics = run_multitask_epoch(
        model,
        test_loader,
        device=device,
        loss_fn=loss_fn,
        scalar_loss_fn=scalar_loss_fn,
        coarse_loss_fn=coarse_loss_fn,
        score_band_loss_fn=score_band_loss_fn,
        optimizer=None,
    )

    result = MultiTaskTrainingResult(
        train_history=train_history,
        val_history=val_history,
        test_metrics=test_metrics,
        best_val_epoch=best_val_epoch,
        device=str(device),
        used_resize_in_transforms=include_resize,
        output_dir=str(normalized_output_dir) if normalized_output_dir is not None else None,
        early_stopped=early_stopped,
        stopped_epoch=stopped_epoch,
    )
    if normalized_output_dir is not None:
        save_multitask_artifacts(
            output_dir=normalized_output_dir,
            model=model,
            result=result,
            config=config,
        )

    print(f"\nBest validation epoch: {best_val_epoch}")
    print(
        f"Test | scalar_mae={test_metrics.scalar_metrics.score_mae:.2f} "
        f"scalar_mbe={float(test_metrics.scalar_metrics.band_ordinal_errors['mean_band_error']):.4f} "
        f"scalar_within_one={float(test_metrics.scalar_metrics.band_ordinal_errors['within_one_band_accuracy']):.4f} "
        f"coarse_acc={test_metrics.coarse_metrics.accuracy:.4f} "
        f"coarse_f1={test_metrics.coarse_metrics.macro_f1:.4f} "
        f"band_acc={test_metrics.score_band_metrics.accuracy:.4f} "
        f"band_f1={test_metrics.score_band_metrics.macro_f1:.4f}"
    )

    return result
