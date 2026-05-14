from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import (
    COARSE_CLASS_NAMES,
    SyntheticManifestDataset,
    build_eval_transform,
    build_train_transform,
)
from .metrics import compute_classification_metrics
from .models import SimpleCNNRegressor
from .splits import build_stratified_splits, subset_frame_by_indices
from .train_classifier import (
    EarlyStoppingTracker,
    choose_device,
    infer_resize_policy,
    save_split_artifacts,
    set_global_seed,
)


@dataclass(frozen=True)
class RegressionEpochResult:
    loss: float
    mae: float
    rmse: float
    bucket_accuracy: float
    bucket_macro_f1: float
    bucket_confusion: list[list[int]]


@dataclass(frozen=True)
class RegressionTrainingResult:
    train_history: list[RegressionEpochResult]
    val_history: list[RegressionEpochResult]
    test_metrics: RegressionEpochResult
    best_val_epoch: int
    device: str
    output_dir: str | None
    early_stopped: bool = False
    stopped_epoch: int | None = None


def score_to_class_index(scores: torch.Tensor) -> torch.Tensor:
    clamped = torch.clamp(scores, 0.0, 100.0)
    labels = torch.zeros_like(clamped, dtype=torch.long)
    labels[(clamped >= 11.0) & (clamped <= 35.0)] = 1
    labels[(clamped >= 36.0) & (clamped <= 65.0)] = 2
    labels[clamped >= 66.0] = 3
    return labels


def build_regression_dataloaders(
    manifest_path: str | Path,
    *,
    image_size: int,
    batch_size: int,
    random_seed: int,
    use_augmentation: bool,
    augmentation_profile: str,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, bool]:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    split = build_stratified_splits(
        manifest_path,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=random_seed,
    )
    include_resize = infer_resize_policy(frame, image_size=image_size)

    train_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode="fine",
        transform=build_train_transform(
            image_size=image_size,
            include_resize=include_resize,
            use_augmentation=use_augmentation,
            augmentation_profile="none" if not use_augmentation else augmentation_profile,
        ),
    )
    eval_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode="fine",
        transform=build_eval_transform(
            image_size=image_size,
            include_resize=include_resize,
        ),
    )

    train_loader = DataLoader(
        Subset(train_dataset, split.train_indices),
        batch_size=batch_size,
        shuffle=True,
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

    return train_loader, val_loader, test_loader, frame, include_resize


def compute_regression_epoch_metrics(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss: float,
) -> RegressionEpochResult:
    errors = predictions - targets
    mae = float(torch.mean(torch.abs(errors)).item())
    rmse = float(torch.sqrt(torch.mean(errors * errors)).item())

    predicted_labels = score_to_class_index(predictions)
    target_labels = score_to_class_index(targets)
    logits = torch.zeros((len(predicted_labels), 4), dtype=torch.float32)
    logits[torch.arange(len(predicted_labels)), predicted_labels] = 1.0
    class_metrics = compute_classification_metrics(
        logits,
        target_labels,
        num_classes=4,
        class_names=COARSE_CLASS_NAMES,
    )

    return RegressionEpochResult(
        loss=loss,
        mae=mae,
        rmse=rmse,
        bucket_accuracy=class_metrics.accuracy,
        bucket_macro_f1=class_metrics.macro_f1,
        bucket_confusion=class_metrics.confusion,
    )


def run_regression_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> RegressionEpochResult:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        if is_training:
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = model(images)
                loss = loss_fn(predictions, targets)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if total_examples == 0:
        raise ValueError("Dataloader produced zero examples.")

    epoch_loss = total_loss / total_examples
    return compute_regression_epoch_metrics(
        predictions=torch.cat(all_predictions),
        targets=torch.cat(all_targets),
        loss=epoch_loss,
    )


def regression_epoch_to_dict(result: RegressionEpochResult) -> dict[str, object]:
    return {
        "loss": result.loss,
        "mae": result.mae,
        "rmse": result.rmse,
        "bucket_accuracy": result.bucket_accuracy,
        "bucket_macro_f1": result.bucket_macro_f1,
        "bucket_confusion": result.bucket_confusion,
    }


def regression_result_to_dict(result: RegressionTrainingResult) -> dict[str, object]:
    return {
        "train_history": [regression_epoch_to_dict(item) for item in result.train_history],
        "val_history": [regression_epoch_to_dict(item) for item in result.val_history],
        "test_metrics": regression_epoch_to_dict(result.test_metrics),
        "best_val_epoch": result.best_val_epoch,
        "device": result.device,
        "output_dir": result.output_dir,
        "early_stopped": result.early_stopped,
        "stopped_epoch": result.stopped_epoch,
    }


def write_regression_epoch_metrics(
    *,
    output_dir: Path,
    epoch_number: int,
    train_metrics: RegressionEpochResult,
    val_metrics: RegressionEpochResult,
    is_best: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "epoch": epoch_number,
        "train_loss": train_metrics.loss,
        "train_mae": train_metrics.mae,
        "train_rmse": train_metrics.rmse,
        "train_bucket_accuracy": train_metrics.bucket_accuracy,
        "train_bucket_macro_f1": train_metrics.bucket_macro_f1,
        "val_loss": val_metrics.loss,
        "val_mae": val_metrics.mae,
        "val_rmse": val_metrics.rmse,
        "val_bucket_accuracy": val_metrics.bucket_accuracy,
        "val_bucket_macro_f1": val_metrics.bucket_macro_f1,
        "is_best": is_best,
    }
    csv_path = output_dir / "epoch_metrics.csv"
    should_write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        import csv

        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)
    with (output_dir / "epoch_metrics.jsonl").open("a") as jsonl_file:
        jsonl_file.write(json.dumps(row) + "\n")


def train_synthetic_regressor(
    manifest_path: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 8,
    num_epochs: int = 40,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    dropout_p: float = 0.1,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    use_augmentation: bool = True,
    augmentation_profile: str = "mild",
) -> RegressionTrainingResult:
    set_global_seed(random_seed)
    device = choose_device()
    manifest_path = Path(manifest_path).resolve()
    normalized_output_dir = Path(output_dir).resolve() if output_dir is not None else None

    train_loader, val_loader, test_loader, frame, include_resize = build_regression_dataloaders(
        manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        random_seed=random_seed,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
    )

    if normalized_output_dir is not None:
        split = build_stratified_splits(
            manifest_path,
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            random_seed=random_seed,
        )
        save_split_artifacts(frame, split, output_dir=normalized_output_dir)
        (normalized_output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "manifest_path": str(manifest_path),
                    "image_size": image_size,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "dropout_p": dropout_p,
                    "random_seed": random_seed,
                    "device": str(device),
                    "used_resize_in_transforms": include_resize,
                    "target": "training_representative_score",
                    "use_augmentation": use_augmentation,
                    "augmentation_profile": augmentation_profile,
                },
                indent=2,
            )
        )

    model = SimpleCNNRegressor(dropout_p=dropout_p).to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    early_stopping = EarlyStoppingTracker(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )

    train_history: list[RegressionEpochResult] = []
    val_history: list[RegressionEpochResult] = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_epoch = 1
    early_stopped = False
    stopped_epoch: int | None = None

    for epoch_number in range(1, num_epochs + 1):
        train_metrics = run_regression_epoch(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_metrics = run_regression_epoch(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
        )
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        decision = early_stopping.update(
            epoch_number=epoch_number,
            metric_value=-val_metrics.mae,
        )
        if decision.is_best:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_epoch = epoch_number

        if normalized_output_dir is not None:
            write_regression_epoch_metrics(
                output_dir=normalized_output_dir,
                epoch_number=epoch_number,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=decision.is_best,
            )

        print(
            f"Epoch {epoch_number}/{num_epochs} | "
            f"train_mae={train_metrics.mae:.3f} "
            f"train_bucket_f1={train_metrics.bucket_macro_f1:.4f} | "
            f"val_mae={val_metrics.mae:.3f} "
            f"val_bucket_f1={val_metrics.bucket_macro_f1:.4f}",
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
    test_metrics = run_regression_epoch(
        model,
        test_loader,
        device=device,
        loss_fn=loss_fn,
        optimizer=None,
    )

    result = RegressionTrainingResult(
        train_history=train_history,
        val_history=val_history,
        test_metrics=test_metrics,
        best_val_epoch=best_val_epoch,
        device=str(device),
        output_dir=str(normalized_output_dir) if normalized_output_dir is not None else None,
        early_stopped=early_stopped,
        stopped_epoch=stopped_epoch,
    )

    if normalized_output_dir is not None:
        torch.save(model.state_dict(), normalized_output_dir / "best_model.pt")
        (normalized_output_dir / "metrics.json").write_text(
            json.dumps(regression_result_to_dict(result), indent=2)
        )

    print(f"\nBest validation epoch: {best_val_epoch}")
    print(
        f"Test | mae={test_metrics.mae:.3f} "
        f"rmse={test_metrics.rmse:.3f} "
        f"bucket_acc={test_metrics.bucket_accuracy:.4f} "
        f"bucket_macro_f1={test_metrics.bucket_macro_f1:.4f}"
    )
    print("Test bucket confusion matrix:")
    for row in test_metrics.bucket_confusion:
        print(row)

    return result
