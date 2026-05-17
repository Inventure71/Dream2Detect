from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import (
    SCORE_BAND_NAMES,
    SyntheticManifestDataset,
    build_eval_transform,
    build_train_transform,
)
from .metrics import (
    compute_scalar_score_band_metrics,
)
from .models import build_regressor_model
from .splits import DatasetSplit, build_stratified_splits
from .train_classifier import (
    DEFAULT_METADATA_FAMILY_COLUMNS,
    EarlyStoppingTracker,
    build_lr_scheduler,
    build_optimizer,
    build_overfit_indices,
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
class RegressionEpochResult:
    loss: float
    normalized_mae: float
    normalized_rmse: float
    mae: float
    rmse: float
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]
    mean_band_error: float
    within_one_band_accuracy: float
    severe_band_error_rate: float
    collapsed_coarse_accuracy: float
    collapsed_coarse_macro_f1: float
    collapsed_coarse_confusion: list[list[int]]
    per_class: dict[str, dict[str, float | int]] = field(default_factory=dict)
    predicted_class_distribution: dict[str, int] = field(default_factory=dict)
    target_class_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def bucket_accuracy(self) -> float:
        return self.collapsed_coarse_accuracy

    @property
    def bucket_macro_f1(self) -> float:
        return self.collapsed_coarse_macro_f1

    @property
    def bucket_confusion(self) -> list[list[int]]:
        return self.collapsed_coarse_confusion


@dataclass(frozen=True)
class RegressionTrainingResult:
    train_history: list[RegressionEpochResult]
    val_history: list[RegressionEpochResult]
    test_metrics: RegressionEpochResult
    best_val_epoch: int
    device: str
    used_resize_in_transforms: bool
    output_dir: str | None
    selection_metric_name: str = "val_mean_band_error"
    best_val_metric: float | None = None
    early_stopped: bool = False
    stopped_epoch: int | None = None


REGRESSION_EPOCH_FIELD_NAMES = [
    "epoch",
    "learning_rate",
    "selection_metric_name",
    "val_selection_metric",
    "train_loss",
    "train_normalized_mae",
    "train_normalized_rmse",
    "train_score_mae",
    "train_score_rmse",
    "train_band_accuracy",
    "train_band_macro_f1",
    "train_mean_band_error",
    "train_within_one_band_accuracy",
    "train_severe_band_error_rate",
    "train_collapsed_coarse_accuracy",
    "train_collapsed_coarse_macro_f1",
    "val_loss",
    "val_normalized_mae",
    "val_normalized_rmse",
    "val_score_mae",
    "val_score_rmse",
    "val_band_accuracy",
    "val_band_macro_f1",
    "val_mean_band_error",
    "val_within_one_band_accuracy",
    "val_severe_band_error_rate",
    "val_collapsed_coarse_accuracy",
    "val_collapsed_coarse_macro_f1",
    "is_best",
]


def build_regression_run_config(
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
    weight_decay: float,
    dropout_p: float,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    overfit_subset_size: int | None,
    use_augmentation: bool,
    augmentation_profile: str,
    model_variant: str,
    target_mode: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
    resume_from: Path | None = None,
    checkpoint_every_n_epochs: int = 1,
) -> dict[str, object]:
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
        "rgb_normalization": "imagenet_mean_std",
        "weight_decay": weight_decay,
        "dropout_p": dropout_p,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "overfit_subset_size": overfit_subset_size,
        "use_augmentation": use_augmentation,
        "augmentation_profile": augmentation_profile,
        "model_variant": model_variant,
        "target_mode": target_mode,
        "selection_metric_name": "val_mean_band_error",
        "selection_metric_mode": "min",
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_strategy": split_strategy,
        "split_group_column": split_group_column,
        "resume_from": str(resume_from) if resume_from is not None else None,
        "split_group_derivation": (
            list(DEFAULT_METADATA_FAMILY_COLUMNS)
            if split_strategy == "metadata_family_holdout" and split_group_column is None
            else None
        ),
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
    }


def build_regression_dataloaders(
    manifest_path: str | Path,
    *,
    image_size: int,
    batch_size: int,
    random_seed: int,
    use_augmentation: bool,
    augmentation_profile: str,
    overfit_subset_size: int | None = None,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
    target_mode: str = "fine_normalized",
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, DatasetSplit, bool]:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    label_column = "training_score_band"

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
            class_names=SCORE_BAND_NAMES,
        )
        split = DatasetSplit(
            train_indices=overfit_indices,
            val_indices=overfit_indices,
            test_indices=overfit_indices,
        )

    include_resize = infer_resize_policy(frame, image_size=image_size)

    train_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode=target_mode,  # type: ignore[arg-type]
        transform=build_train_transform(
            image_size=image_size,
            include_resize=include_resize,
            use_augmentation=use_augmentation,
            augmentation_profile="none" if not use_augmentation else augmentation_profile,
        ),
    )
    eval_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode=target_mode,  # type: ignore[arg-type]
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

    return train_loader, val_loader, test_loader, frame, split, include_resize


def compute_regression_epoch_metrics(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss: float,
) -> RegressionEpochResult:
    scalar_metrics = compute_scalar_score_band_metrics(
        normalized_predictions=predictions,
        normalized_targets=targets,
    )

    return RegressionEpochResult(
        loss=loss,
        normalized_mae=scalar_metrics.normalized_mae,
        normalized_rmse=scalar_metrics.normalized_rmse,
        mae=scalar_metrics.score_mae,
        rmse=scalar_metrics.score_rmse,
        accuracy=scalar_metrics.band_metrics.accuracy,
        macro_f1=scalar_metrics.band_metrics.macro_f1,
        confusion=scalar_metrics.band_metrics.confusion,
        mean_band_error=float(scalar_metrics.band_ordinal_errors["mean_band_error"]),
        within_one_band_accuracy=float(
            scalar_metrics.band_ordinal_errors["within_one_band_accuracy"]
        ),
        severe_band_error_rate=float(
            scalar_metrics.band_ordinal_errors["severe_band_error_rate"]
        ),
        collapsed_coarse_accuracy=scalar_metrics.coarse_metrics.accuracy,
        collapsed_coarse_macro_f1=scalar_metrics.coarse_metrics.macro_f1,
        collapsed_coarse_confusion=scalar_metrics.coarse_metrics.confusion,
        per_class={
            class_name: {
                "precision": class_metrics.precision,
                "recall": class_metrics.recall,
                "f1": class_metrics.f1,
                "support": class_metrics.support,
            }
            for class_name, class_metrics in scalar_metrics.band_metrics.per_class.items()
        },
        predicted_class_distribution=(
            scalar_metrics.band_metrics.predicted_class_distribution
        ),
        target_class_distribution=scalar_metrics.band_metrics.target_class_distribution,
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

    return compute_regression_epoch_metrics(
        predictions=torch.cat(all_predictions),
        targets=torch.cat(all_targets),
        loss=total_loss / total_examples,
    )


def regression_epoch_to_dict(result: RegressionEpochResult) -> dict[str, object]:
    return {
        "loss": result.loss,
        "normalized_mae": result.normalized_mae,
        "normalized_rmse": result.normalized_rmse,
        "mae": result.mae,
        "rmse": result.rmse,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "confusion": result.confusion,
        "mean_band_error": result.mean_band_error,
        "within_one_band_accuracy": result.within_one_band_accuracy,
        "severe_band_error_rate": result.severe_band_error_rate,
        "collapsed_coarse_accuracy": result.collapsed_coarse_accuracy,
        "collapsed_coarse_macro_f1": result.collapsed_coarse_macro_f1,
        "collapsed_coarse_confusion": result.collapsed_coarse_confusion,
        "bucket_accuracy": result.bucket_accuracy,
        "bucket_macro_f1": result.bucket_macro_f1,
        "bucket_confusion": result.bucket_confusion,
        "per_class": result.per_class,
        "predicted_class_distribution": result.predicted_class_distribution,
        "target_class_distribution": result.target_class_distribution,
    }


def regression_result_to_dict(result: RegressionTrainingResult) -> dict[str, object]:
    return {
        "train_history": [regression_epoch_to_dict(item) for item in result.train_history],
        "val_history": [regression_epoch_to_dict(item) for item in result.val_history],
        "test_metrics": regression_epoch_to_dict(result.test_metrics),
        "best_val_epoch": result.best_val_epoch,
        "device": result.device,
        "used_resize_in_transforms": result.used_resize_in_transforms,
        "output_dir": result.output_dir,
        "selection_metric_name": result.selection_metric_name,
        "best_val_metric": result.best_val_metric,
        "early_stopped": result.early_stopped,
        "stopped_epoch": result.stopped_epoch,
    }


def flatten_regression_epoch_metrics(
    *,
    epoch_number: int,
    train_metrics: RegressionEpochResult,
    val_metrics: RegressionEpochResult,
    is_best: bool,
    learning_rate: float | None = None,
    val_selection_metric: float | None = None,
) -> dict[str, object]:
    return {
        "epoch": epoch_number,
        "learning_rate": learning_rate,
        "selection_metric_name": "val_mean_band_error",
        "val_selection_metric": val_selection_metric,
        "train_loss": train_metrics.loss,
        "train_normalized_mae": train_metrics.normalized_mae,
        "train_normalized_rmse": train_metrics.normalized_rmse,
        "train_score_mae": train_metrics.mae,
        "train_score_rmse": train_metrics.rmse,
        "train_band_accuracy": train_metrics.accuracy,
        "train_band_macro_f1": train_metrics.macro_f1,
        "train_mean_band_error": train_metrics.mean_band_error,
        "train_within_one_band_accuracy": train_metrics.within_one_band_accuracy,
        "train_severe_band_error_rate": train_metrics.severe_band_error_rate,
        "train_collapsed_coarse_accuracy": train_metrics.collapsed_coarse_accuracy,
        "train_collapsed_coarse_macro_f1": train_metrics.collapsed_coarse_macro_f1,
        "val_loss": val_metrics.loss,
        "val_normalized_mae": val_metrics.normalized_mae,
        "val_normalized_rmse": val_metrics.normalized_rmse,
        "val_score_mae": val_metrics.mae,
        "val_score_rmse": val_metrics.rmse,
        "val_band_accuracy": val_metrics.accuracy,
        "val_band_macro_f1": val_metrics.macro_f1,
        "val_mean_band_error": val_metrics.mean_band_error,
        "val_within_one_band_accuracy": val_metrics.within_one_band_accuracy,
        "val_severe_band_error_rate": val_metrics.severe_band_error_rate,
        "val_collapsed_coarse_accuracy": val_metrics.collapsed_coarse_accuracy,
        "val_collapsed_coarse_macro_f1": val_metrics.collapsed_coarse_macro_f1,
        "is_best": is_best,
    }


def append_regression_epoch_metrics(
    *,
    output_dir: Path,
    epoch_number: int,
    train_metrics: RegressionEpochResult,
    val_metrics: RegressionEpochResult,
    is_best: bool,
    learning_rate: float | None = None,
    val_selection_metric: float | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    nested_row = {
        "epoch": epoch_number,
        "train": regression_epoch_to_dict(train_metrics),
        "val": regression_epoch_to_dict(val_metrics),
        "is_best": is_best,
        "learning_rate": learning_rate,
        "selection_metric_name": "val_mean_band_error",
        "val_selection_metric": val_selection_metric,
    }
    with (output_dir / "epoch_metrics.jsonl").open("a") as jsonl_file:
        jsonl_file.write(json.dumps(nested_row) + "\n")

    flat_row = flatten_regression_epoch_metrics(
        epoch_number=epoch_number,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        is_best=is_best,
        learning_rate=learning_rate,
        val_selection_metric=val_selection_metric,
    )
    csv_path = output_dir / "epoch_metrics.csv"
    should_write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REGRESSION_EPOCH_FIELD_NAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(flat_row)

    return flat_row


def read_regression_epoch_metrics(output_dir: Path) -> list[dict[str, object]]:
    csv_path = output_dir / "epoch_metrics.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def regression_epoch_from_flat_row(
    row: dict[str, object],
    *,
    prefix: str,
) -> RegressionEpochResult:
    return RegressionEpochResult(
        loss=float(row[f"{prefix}_loss"]),
        normalized_mae=float(row[f"{prefix}_normalized_mae"]),
        normalized_rmse=float(row[f"{prefix}_normalized_rmse"]),
        mae=float(row[f"{prefix}_score_mae"]),
        rmse=float(row[f"{prefix}_score_rmse"]),
        accuracy=float(row[f"{prefix}_band_accuracy"]),
        macro_f1=float(row[f"{prefix}_band_macro_f1"]),
        confusion=[],
        mean_band_error=float(row[f"{prefix}_mean_band_error"]),
        within_one_band_accuracy=float(row[f"{prefix}_within_one_band_accuracy"]),
        severe_band_error_rate=float(row[f"{prefix}_severe_band_error_rate"]),
        collapsed_coarse_accuracy=float(row[f"{prefix}_collapsed_coarse_accuracy"]),
        collapsed_coarse_macro_f1=float(row[f"{prefix}_collapsed_coarse_macro_f1"]),
        collapsed_coarse_confusion=[],
    )


def plot_regression_epoch_metrics(
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
        ("Score MAE", "train_score_mae", "val_score_mae"),
        ("Mean Band Error", "train_mean_band_error", "val_mean_band_error"),
        ("+-1 Band Accuracy", "train_within_one_band_accuracy", "val_within_one_band_accuracy"),
    ]

    figure, axes = plt.subplots(len(series), 1, figsize=(9, 3.3 * len(series)), sharex=True)
    for axis, (title, train_key, val_key) in zip(axes, series, strict=True):
        axis.plot(epochs, [float(row[train_key]) for row in history], marker="o", label="train")
        axis.plot(epochs, [float(row[val_key]) for row in history], marker="o", label="val")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def save_regression_epoch_checkpoint(
    *,
    output_dir: Path,
    epoch_number: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    train_metrics: RegressionEpochResult,
    val_metrics: RegressionEpochResult,
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
        "train_metrics": regression_epoch_to_dict(train_metrics),
        "val_metrics": regression_epoch_to_dict(val_metrics),
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


def train_synthetic_regressor(
    manifest_path: str | Path,
    *,
    image_size: int = 224,
    batch_size: int = 8,
    num_epochs: int = 40,
    learning_rate: float = 3e-4,
    optimizer_name: str = "adamw",
    lr_scheduler_name: str = "reduce_on_plateau",
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 10,
    min_learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    dropout_p: float = 0.1,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    use_augmentation: bool = True,
    augmentation_profile: str = "mild",
    model_variant: str = "simple_cnn_regressor",
    target_mode: str = "fine_normalized",
    overfit_subset_size: int | None = None,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
    checkpoint_every_n_epochs: int = 1,
) -> RegressionTrainingResult:
    if target_mode != "fine_normalized":
        raise ValueError(
            "V5 scalar regression requires target_mode='fine_normalized', "
            f"got {target_mode!r}"
        )
    if checkpoint_every_n_epochs <= 0:
        raise ValueError(
            "checkpoint_every_n_epochs must be positive, "
            f"got {checkpoint_every_n_epochs}"
        )
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

    train_loader, val_loader, test_loader, frame, split, include_resize = (
        build_regression_dataloaders(
            manifest_path,
            image_size=image_size,
            batch_size=batch_size,
            random_seed=random_seed,
            use_augmentation=use_augmentation,
            augmentation_profile=augmentation_profile,
            overfit_subset_size=overfit_subset_size,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            split_strategy=split_strategy,
            split_group_column=split_group_column,
            target_mode=target_mode,
        )
    )

    config = build_regression_run_config(
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
        weight_decay=weight_decay,
        dropout_p=dropout_p,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        overfit_subset_size=overfit_subset_size,
        use_augmentation=use_augmentation,
        augmentation_profile=augmentation_profile,
        model_variant=model_variant,
        target_mode=target_mode,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        split_strategy=split_strategy,
        split_group_column=split_group_column,
        resume_from=normalized_resume_from,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
    )

    if normalized_output_dir is not None:
        normalized_output_dir.mkdir(parents=True, exist_ok=True)
        save_split_artifacts(
            frame,
            split,
            output_dir=normalized_output_dir,
            split_strategy=split_strategy,
            split_group_column=split_group_column,
        )
        (normalized_output_dir / "run_config.json").write_text(
            json.dumps(config, indent=2)
        )

    model = build_regressor_model(
        dropout_p=dropout_p,
        model_variant=model_variant,
    ).to(device)
    loss_fn = nn.SmoothL1Loss()
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

    train_history: list[RegressionEpochResult] = []
    val_history: list[RegressionEpochResult] = []
    flat_history: list[dict[str, object]] = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_epoch = 1
    best_val_metric: float | None = None
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

        flat_history = [
            row
            for row in read_regression_epoch_metrics(normalized_output_dir)
            if int(row["epoch"]) < start_epoch
        ]
        for row in flat_history:
            train_history.append(regression_epoch_from_flat_row(row, prefix="train"))
            val_history.append(regression_epoch_from_flat_row(row, prefix="val"))
            metric_value = float(row["val_selection_metric"])
            decision = early_stopping.update(
                epoch_number=int(row["epoch"]),
                metric_value=metric_value,
            )
            if decision.is_best:
                best_val_epoch = int(row["epoch"])
                best_val_metric = metric_value

        best_checkpoint_path = normalized_output_dir / "checkpoints/best.pt"
        if best_checkpoint_path.exists():
            best_checkpoint = load_training_checkpoint(best_checkpoint_path)
            best_state_dict = copy.deepcopy(best_checkpoint["model_state_dict"])  # type: ignore[arg-type]
        else:
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Resuming from checkpoint epoch {loaded_epoch}; "
            f"next epoch={start_epoch}/{num_epochs}",
            flush=True,
        )

    for epoch_number in range(start_epoch, num_epochs + 1):
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

        val_selection_metric = val_metrics.mean_band_error
        if lr_scheduler is not None:
            lr_scheduler.step(val_selection_metric)
        decision = early_stopping.update(
            epoch_number=epoch_number,
            metric_value=val_selection_metric,
        )
        if decision.is_best:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_epoch = epoch_number
            best_val_metric = val_selection_metric

        if normalized_output_dir is not None:
            flat_row = append_regression_epoch_metrics(
                output_dir=normalized_output_dir,
                epoch_number=epoch_number,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=decision.is_best,
                learning_rate=get_current_learning_rate(optimizer),
                val_selection_metric=val_selection_metric,
            )
            flat_history.append(flat_row)
            if checkpoint_every_n_epochs > 0 and (
                epoch_number % checkpoint_every_n_epochs == 0 or decision.is_best
            ):
                save_regression_epoch_checkpoint(
                    output_dir=normalized_output_dir,
                    epoch_number=epoch_number,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    is_best=decision.is_best,
                    config=config,
                )

        print(
            f"Epoch {epoch_number}/{num_epochs} | "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_mae={train_metrics.mae:.2f} "
            f"train_band_err={train_metrics.mean_band_error:.3f} | "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_mae={val_metrics.mae:.2f} "
            f"val_band_err={val_metrics.mean_band_error:.3f} "
            f"lr={get_current_learning_rate(optimizer):g}",
            flush=True,
        )

        if decision.should_stop:
            early_stopped = True
            stopped_epoch = epoch_number
            print(
                f"Early stopping at epoch {epoch_number}; "
                f"best_val_epoch={best_val_epoch} "
                f"best_val_mean_band_error={early_stopping.best_metric:.4f}",
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
        used_resize_in_transforms=include_resize,
        output_dir=str(normalized_output_dir) if normalized_output_dir is not None else None,
        best_val_metric=best_val_metric,
        early_stopped=early_stopped,
        stopped_epoch=stopped_epoch,
    )

    if normalized_output_dir is not None:
        torch.save(model.state_dict(), normalized_output_dir / "best_model.pt")
        (normalized_output_dir / "metrics.json").write_text(
            json.dumps(regression_result_to_dict(result), indent=2)
        )
        if flat_history:
            plot_regression_epoch_metrics(
                output_dir=normalized_output_dir,
                history=flat_history,
            )

    print(f"\nBest validation epoch: {best_val_epoch}")
    print(
        f"Test | mae={test_metrics.mae:.2f} "
        f"rmse={test_metrics.rmse:.2f} "
        f"band_acc={test_metrics.accuracy:.4f} "
        f"band_f1={test_metrics.macro_f1:.4f} "
        f"mean_band_error={test_metrics.mean_band_error:.4f} "
        f"within_one={test_metrics.within_one_band_accuracy:.4f} "
        f"coarse_f1={test_metrics.collapsed_coarse_macro_f1:.4f}"
    )
    print("Test band confusion matrix:")
    for row in test_metrics.confusion:
        print(row)

    return result
