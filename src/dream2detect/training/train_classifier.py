from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import (
    SyntheticManifestDataset,
    build_eval_transform,
    build_train_transform,
)
from .metrics import compute_classification_metrics
from .models import SimpleCNNClassifier
from .splits import build_stratified_splits, subset_frame_by_indices


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]


@dataclass(frozen=True)
class TrainingResult:
    train_history: list[EpochResult]
    val_history: list[EpochResult]
    test_metrics: EpochResult
    best_val_epoch: int


def set_global_seed(seed: int) -> None:
    """
    Make the run more reproducible.

    This does not guarantee bit-perfect reproducibility in every environment,
    but it is the correct baseline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def build_dataloaders(
    manifest_path: str | Path,
    *,
    image_size: int = 128,
    batch_size: int = 8,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """
    Build train/validation/test dataloaders from one manifest.

    Returns:
    - train_loader
    - val_loader
    - test_loader
    - full dataframe
    """
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)

    split = build_stratified_splits(
        manifest_path,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=random_seed,
    )

    train_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode="coarse",
        transform=build_train_transform(image_size=image_size),
    )
    eval_dataset = SyntheticManifestDataset(
        manifest_path,
        target_mode="coarse",
        transform=build_eval_transform(image_size=image_size),
    )

    train_subset = Subset(train_dataset, split.train_indices)
    val_subset = Subset(eval_dataset, split.val_indices)
    test_subset = Subset(eval_dataset, split.test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
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

    return train_loader, val_loader, test_loader, frame


def build_class_weights(
    frame: pd.DataFrame,
    *,
    train_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute class weights from the training split only.
    """
    train_frame = subset_frame_by_indices(frame, train_indices)
    train_labels = train_frame["training_coarse_class"].tolist()

    class_names = ["intact", "minor", "moderate", "severe"]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_names),
        y=np.array(train_labels),
    )

    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
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
    metrics = compute_classification_metrics(
        stacked_logits,
        stacked_targets,
        num_classes=4,
    )

    return EpochResult(
        loss=epoch_loss,
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        confusion=metrics.confusion,
    )


def train_synthetic_classifier(
    manifest_path: str | Path,
    *,
    image_size: int = 128,
    batch_size: int = 8,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> TrainingResult:
    """
    Train the first coarse classifier on the synthetic manifest.
    """
    set_global_seed(random_seed)
    device = choose_device()

    manifest_path = Path(manifest_path)
    split = build_stratified_splits(
        manifest_path,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        random_seed=random_seed,
    )

    train_loader, val_loader, test_loader, frame = build_dataloaders(
        manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        random_seed=random_seed,
    )

    model = SimpleCNNClassifier(num_classes=4).to(device)
    class_weights = build_class_weights(
        frame,
        train_indices=split.train_indices,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_history: list[EpochResult] = []
    val_history: list[EpochResult] = []

    best_val_f1 = -1.0
    best_val_epoch = 1
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch_index in range(num_epochs):
        epoch_number = epoch_index + 1

        train_metrics = run_one_epoch(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=None,
        )

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if val_metrics.macro_f1 > best_val_f1:
            best_val_f1 = val_metrics.macro_f1
            best_val_epoch = epoch_number
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch_number}/{num_epochs} | "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.4f} "
            f"train_f1={train_metrics.macro_f1:.4f} | "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} "
            f"val_f1={val_metrics.macro_f1:.4f}"
        )

    model.load_state_dict(best_state_dict)

    test_metrics = run_one_epoch(
        model,
        test_loader,
        device=device,
        loss_fn=loss_fn,
        optimizer=None,
    )

    print(f"\nBest validation epoch: {best_val_epoch}")
    print(
        f"Test | "
        f"loss={test_metrics.loss:.4f} "
        f"acc={test_metrics.accuracy:.4f} "
        f"macro_f1={test_metrics.macro_f1:.4f}"
    )
    print("Test confusion matrix:")
    for row in test_metrics.confusion:
        print(row)

    return TrainingResult(
        train_history=train_history,
        val_history=val_history,
        test_metrics=test_metrics,
        best_val_epoch=best_val_epoch,
    )
