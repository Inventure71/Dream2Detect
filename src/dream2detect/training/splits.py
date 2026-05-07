from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


def build_stratified_splits(
    manifest_path: str | Path,
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> DatasetSplit:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)

    if len(frame) == 0:
        raise ValueError("Manifest is empty. Cannot build splits.")

    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total_fraction}"
        )

    labels = frame["training_coarse_class"]

    all_indices = list(range(len(frame)))

    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(1.0 - train_fraction),
        random_state=random_seed,
        stratify=labels,
    )

    temp_frame = frame.iloc[temp_indices].reset_index(drop=True)
    temp_labels = temp_frame["training_coarse_class"]

    val_relative_indices, test_relative_indices = train_test_split(
        list(range(len(temp_indices))),
        test_size=(test_fraction / (val_fraction + test_fraction)),
        random_state=random_seed,
        stratify=temp_labels,
    )

    val_indices = [temp_indices[i] for i in val_relative_indices]
    test_indices = [temp_indices[i] for i in test_relative_indices]

    return DatasetSplit(
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
    )


def subset_frame_by_indices(frame: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """
    Return a dataframe subset while keeping the original column structure.
    """
    return frame.iloc[indices].reset_index(drop=True)


def describe_split_distribution(
    frame: pd.DataFrame,
    indices: list[int],
) -> pd.Series:
    """
    Count how many examples of each coarse class appear in one split.
    """
    subset = subset_frame_by_indices(frame, indices)
    return subset["training_coarse_class"].value_counts().sort_index()
