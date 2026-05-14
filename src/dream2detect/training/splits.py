from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


DEFAULT_METADATA_FAMILY_COLUMNS = (
    "training_coarse_class",
    "damage_profile_primary",
    "box_form_factor",
    "background_context",
)


def derive_metadata_family_groups(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = DEFAULT_METADATA_FAMILY_COLUMNS,
) -> pd.Series:
    missing_columns = [column for column in group_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Manifest is missing metadata columns required for metadata family holdout: "
            + ", ".join(missing_columns)
        )

    normalized = frame.loc[:, list(group_columns)].fillna("missing").astype(str)
    return normalized.agg("|".join, axis=1)


def _build_group_holdout_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
    label_column: str,
    group_ids: pd.Series,
) -> DatasetSplit:
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for _label, label_frame in frame.groupby(label_column, sort=True):
        label_indices = label_frame.index.to_list()
        label_groups = group_ids.loc[label_indices]
        unique_group_count = label_groups.nunique()
        if unique_group_count < 3:
            raise ValueError(
                "metadata_family_holdout requires at least 3 distinct groups per label, "
                f"but label {_label!r} only has {unique_group_count}"
            )

        label_train_fraction = train_fraction
        label_val_share = val_fraction / (val_fraction + test_fraction)

        label_index_positions = list(range(len(label_indices)))
        first_split = GroupShuffleSplit(
            n_splits=1,
            train_size=label_train_fraction,
            random_state=random_seed,
        )
        label_train_positions, label_temp_positions = next(
            first_split.split(
                X=label_index_positions,
                y=None,
                groups=label_groups.to_numpy(),
            )
        )

        temp_groups = label_groups.iloc[label_temp_positions]
        temp_group_count = temp_groups.nunique()
        if temp_group_count < 2:
            raise ValueError(
                "metadata_family_holdout requires at least 2 held-out groups per label "
                f"for val/test, but label {_label!r} only has {temp_group_count}"
            )

        second_split = GroupShuffleSplit(
            n_splits=1,
            train_size=label_val_share,
            random_state=random_seed + 1,
        )
        label_val_positions, label_test_positions = next(
            second_split.split(
                X=list(range(len(label_temp_positions))),
                y=None,
                groups=temp_groups.to_numpy(),
            )
        )

        train_indices.extend(label_indices[position] for position in label_train_positions)
        temp_indices = [label_indices[position] for position in label_temp_positions]
        val_indices.extend(temp_indices[position] for position in label_val_positions)
        test_indices.extend(temp_indices[position] for position in label_test_positions)

    return DatasetSplit(
        train_indices=sorted(train_indices),
        val_indices=sorted(val_indices),
        test_indices=sorted(test_indices),
    )


def build_stratified_splits(
    manifest_path: str | Path,
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_seed: int = 42,
    label_column: str = "training_coarse_class",
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
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

    if label_column not in frame.columns:
        raise ValueError(f"Manifest is missing split label column: {label_column}")

    if split_strategy == "metadata_family_holdout":
        if split_group_column is not None:
            if split_group_column not in frame.columns:
                raise ValueError(
                    f"Manifest is missing requested split group column: {split_group_column}"
                )
            group_ids = frame[split_group_column].fillna("missing").astype(str)
        else:
            group_ids = derive_metadata_family_groups(frame)
        return _build_group_holdout_splits(
            frame,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            random_seed=random_seed,
            label_column=label_column,
            group_ids=group_ids,
        )

    if split_strategy != "stratified_random":
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    labels = frame[label_column]

    all_indices = list(range(len(frame)))

    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(1.0 - train_fraction),
        random_state=random_seed,
        stratify=labels,
    )

    temp_frame = frame.iloc[temp_indices].reset_index(drop=True)
    temp_labels = temp_frame[label_column]

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
    *,
    label_column: str = "training_coarse_class",
) -> pd.Series:
    """
    Count how many examples of each target label appear in one split.
    """
    if label_column not in frame.columns:
        raise ValueError(f"Frame is missing split label column: {label_column}")
    subset = subset_frame_by_indices(frame, indices)
    return subset[label_column].value_counts().sort_index()
