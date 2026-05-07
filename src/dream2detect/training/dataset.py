from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CoarseLabel = Literal["intact", "minor", "moderate", "severe"]
TargetMode = Literal["coarse", "fine"]


COARSE_CLASS_TO_INDEX: dict[CoarseLabel, int] = {
    "intact": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}

INDEX_TO_COARSE_CLASS: dict[int, CoarseLabel] = {
    0: "intact",
    1: "minor",
    2: "moderate",
    3: "severe",
}


def build_train_transform(
    image_size: int = 128,
    *,
    include_resize: bool = True,
) -> transforms.Compose:
    transform_steps: list[transforms.Transform] = []

    if include_resize:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
        ]
    )

    return transforms.Compose(transform_steps)


def build_eval_transform(
    image_size: int = 128,
    *,
    include_resize: bool = True,
) -> transforms.Compose:
    transform_steps: list[transforms.Transform] = []

    if include_resize:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    transform_steps.append(transforms.ToTensor())

    return transforms.Compose(transform_steps)


class SyntheticManifestDataset(Dataset):
    """
    PyTorch dataset for Dream2Detect synthetic training manifests.

    Each item returns:
    - image: torch tensor
    - target: class index or regression score
    - metadata: helpful debug fields
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        target_mode: TargetMode,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.target_mode = target_mode
        self.transform = transform or build_eval_transform()

        self.frame = pd.read_csv(self.manifest_path)

        required_columns = {
            "prompt_id",
            "image_path",
            "training_score_band",
            "training_coarse_class",
            "training_representative_score",
            "training_label_source",
            "qc_status",
        }
        missing_columns = required_columns - set(self.frame.columns)
        if missing_columns:
            missing_str = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Manifest is missing required columns: {missing_str}"
            )

        if len(self.frame) == 0:
            raise ValueError("Manifest is empty. Cannot train with zero rows.")

        self._validate_rows()

    def _validate_rows(self) -> None:
        """
        Fail early if the manifest points to missing files or invalid labels.
        """
        for row_index, row in self.frame.iterrows():
            image_path = Path(row["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Row {row_index} points to a missing image: {image_path}"
                )

            coarse_class = row["training_coarse_class"]
            if coarse_class not in COARSE_CLASS_TO_INDEX:
                raise ValueError(
                    f"Row {row_index} has invalid training_coarse_class={coarse_class!r}"
                )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]

        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        if self.target_mode == "coarse":
            target = torch.tensor(
                COARSE_CLASS_TO_INDEX[row["training_coarse_class"]],
                dtype=torch.long,
            )
        elif self.target_mode == "fine":
            target = torch.tensor(
                float(row["training_representative_score"]),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")

        metadata = {
            "prompt_id": int(row["prompt_id"]),
            "image_path": str(image_path),
            "training_score_band": row["training_score_band"],
            "training_coarse_class": row["training_coarse_class"],
            "training_label_source": row["training_label_source"],
            "qc_status": row["qc_status"],
        }

        return {
            "image": image_tensor,
            "target": target,
            "metadata": metadata,
        }
