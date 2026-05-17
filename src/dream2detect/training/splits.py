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
    unique_group_count = group_ids.nunique()
    if unique_group_count < 3:
        raise ValueError(
            "metadata_family_holdout requires at least 3 distinct metadata groups, "
            f"but only found {unique_group_count}"
        )

    val_share_of_heldout = val_fraction / (val_fraction + test_fraction)
    all_indices = list(range(len(frame)))
    labels = frame[label_column].astype(str)
    all_label_names = sorted(labels.unique().tolist())
    global_distribution = labels.value_counts(normalize=True).to_dict()

    def score_split(
        train_indices: list[int],
        val_indices: list[int],
        test_indices: list[int],
    ) -> float:
        score = 0.0
        split_specs = (
            (train_indices, train_fraction),
            (val_indices, val_fraction),
            (test_indices, test_fraction),
        )
        for indices, expected_fraction in split_specs:
            actual_fraction = len(indices) / len(frame)
            score += abs(actual_fraction - expected_fraction) * len(all_label_names)
            split_labels = labels.iloc[indices]
            split_distribution = split_labels.value_counts(normalize=True).to_dict()
            for label_name in all_label_names:
                observed = split_distribution.get(label_name, 0.0)
                expected = global_distribution.get(label_name, 0.0)
                score += abs(observed - expected)
                if expected > 0 and observed == 0:
                    score += 10.0
        return score

    best_split: DatasetSplit | None = None
    best_score = float("inf")

    for attempt in range(128):
        first_split = GroupShuffleSplit(
            n_splits=1,
            train_size=train_fraction,
            random_state=random_seed + attempt,
        )
        train_positions, temp_positions = next(
            first_split.split(
                X=all_indices,
                y=labels.to_numpy(),
                groups=group_ids.to_numpy(),
            )
        )
        temp_positions_list = [int(position) for position in temp_positions]
        if len(set(group_ids.iloc[temp_positions_list].tolist())) < 2:
            continue

        temp_groups = group_ids.iloc[temp_positions_list].reset_index(drop=True)
        temp_labels = labels.iloc[temp_positions_list].reset_index(drop=True)
        second_split = GroupShuffleSplit(
            n_splits=1,
            train_size=val_share_of_heldout,
            random_state=random_seed + 10_000 + attempt,
        )
        val_relative_positions, test_relative_positions = next(
            second_split.split(
                X=list(range(len(temp_positions_list))),
                y=temp_labels.to_numpy(),
                groups=temp_groups.to_numpy(),
            )
        )

        train_indices = sorted(int(position) for position in train_positions)
        val_indices = sorted(
            temp_positions_list[int(position)] for position in val_relative_positions
        )
        test_indices = sorted(
            temp_positions_list[int(position)] for position in test_relative_positions
        )

        split = DatasetSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
        current_score = score_split(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )
        if current_score < best_score:
            best_score = current_score
            best_split = split

    if best_split is None:
        raise ValueError("Could not build metadata-family holdout split.")

    return best_split


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
