from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CoarseLabel = Literal["intact", "minor", "moderate", "severe"]
ScoreBandLabel = Literal[
    "0-10",
    "11-20",
    "21-30",
    "31-35",
    "36-45",
    "46-55",
    "56-65",
    "66-75",
    "76-85",
    "86-100",
]
TargetMode = Literal["coarse", "coarse_ordinal", "score_band", "multitask", "fine"]
AugmentationProfile = Literal[
    "none",
    "mild",
    "strong",
    "damage_safe",
    "damage_safe_ra_low",
    "damage_safe_ra_medium",
    "damage_safe_ra_high",
]


COARSE_CLASS_TO_INDEX: dict[CoarseLabel, int] = {
    "intact": 0,
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}

COARSE_CLASS_NAMES = ("intact", "minor", "moderate", "severe")
SCORE_BAND_NAMES = (
    "0-10",
    "11-20",
    "21-30",
    "31-35",
    "36-45",
    "46-55",
    "56-65",
    "66-75",
    "76-85",
    "86-100",
)
SCORE_BAND_TO_INDEX: dict[ScoreBandLabel, int] = {
    band_name: index for index, band_name in enumerate(SCORE_BAND_NAMES)
}
RGB_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
RGB_NORMALIZATION_STD = (0.229, 0.224, 0.225)

INDEX_TO_COARSE_CLASS: dict[int, CoarseLabel] = {
    0: "intact",
    1: "minor",
    2: "moderate",
    3: "severe",
}


class AddGaussianNoise:
    def __init__(self, *, std: float = 0.015, p: float = 0.15) -> None:
        if std < 0:
            raise ValueError(f"std must be non-negative, got {std}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0.0, 1.0], got {p}")
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std == 0 or torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class DamageSafeRandAugment:
    """
    RandAugment-style augmentation restricted to operations that preserve
    package-defect semantics more reliably than the default torchvision pool.
    """

    def __init__(
        self,
        *,
        num_ops: int,
        magnitude: float,
        fill: int = 245,
    ) -> None:
        if num_ops <= 0:
            raise ValueError(f"num_ops must be positive, got {num_ops}")
        if not 0.0 <= magnitude <= 1.0:
            raise ValueError(f"magnitude must be in [0.0, 1.0], got {magnitude}")
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.fill = fill

    def _signed(self, scale: float) -> float:
        sign = -1.0 if torch.rand(1).item() < 0.5 else 1.0
        return sign * scale * self.magnitude

    def _factor_around_one(self, scale: float) -> float:
        return 1.0 + self._signed(scale)

    def _op_affine(self, image: Image.Image) -> Image.Image:
        angle = self._signed(6.0)
        translate_x = int(round(image.width * self._signed(0.04)))
        translate_y = int(round(image.height * self._signed(0.04)))
        scale = 1.0 + self._signed(0.08)
        shear = [self._signed(3.0), self._signed(3.0)]
        return TF.affine(
            image,
            angle=angle,
            translate=[translate_x, translate_y],
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=self.fill,
        )

    def _op_perspective(self, image: Image.Image) -> Image.Image:
        distortion = 0.02 + 0.07 * self.magnitude
        startpoints, endpoints = transforms.RandomPerspective.get_params(
            image.width,
            image.height,
            distortion_scale=distortion,
        )
        return TF.perspective(
            image,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=InterpolationMode.BILINEAR,
            fill=self.fill,
        )

    def _op_brightness(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(self._factor_around_one(0.18))

    def _op_contrast(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self._factor_around_one(0.2))

    def _op_saturation(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Color(image).enhance(self._factor_around_one(0.12))

    def _op_blur(self, image: Image.Image) -> Image.Image:
        sigma = 0.08 + 0.55 * self.magnitude * torch.rand(1).item()
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def __call__(self, image: Image.Image) -> Image.Image:
        operations = [
            self._op_affine,
            self._op_perspective,
            self._op_brightness,
            self._op_contrast,
            self._op_saturation,
            self._op_blur,
        ]
        selected = torch.randperm(len(operations))[: self.num_ops].tolist()
        for op_index in selected:
            image = operations[op_index](image)
        return image


def build_damage_safe_randaugment_profile(
    profile: AugmentationProfile,
) -> tuple[DamageSafeRandAugment, float]:
    if profile == "damage_safe_ra_low":
        return DamageSafeRandAugment(num_ops=2, magnitude=0.35), 0.010
    if profile == "damage_safe_ra_medium":
        return DamageSafeRandAugment(num_ops=2, magnitude=0.55), 0.014
    if profile == "damage_safe_ra_high":
        return DamageSafeRandAugment(num_ops=3, magnitude=0.75), 0.018
    raise ValueError(f"Unsupported RandAugment profile: {profile}")


def build_train_transform(
    image_size: int = 224,
    *,
    include_resize: bool = True,
    use_augmentation: bool = True,
    augmentation_profile: AugmentationProfile = "mild",
) -> transforms.Compose:
    transform_steps: list[transforms.Transform] = []

    uses_damage_safe_randaugment = augmentation_profile in {
        "damage_safe_ra_low",
        "damage_safe_ra_medium",
        "damage_safe_ra_high",
    }

    if use_augmentation and augmentation_profile == "strong":
        transform_steps.append(
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.82, 1.0),
                ratio=(0.9, 1.1),
            )
        )
    elif include_resize:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    if use_augmentation and augmentation_profile == "mild":
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.05,
                    hue=0.02,
                ),
            ]
        )
    elif use_augmentation and augmentation_profile == "strong":
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.RandomPerspective(distortion_scale=0.08, p=0.25),
                transforms.ColorJitter(
                    brightness=0.18,
                    contrast=0.18,
                    saturation=0.08,
                    hue=0.02,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.15,
                ),
            ]
        )
    elif use_augmentation and augmentation_profile == "damage_safe":
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.035, 0.035),
                    scale=(0.94, 1.06),
                    shear=(-2, 2),
                    fill=245,
                ),
                transforms.RandomPerspective(distortion_scale=0.05, p=0.2),
                transforms.ColorJitter(
                    brightness=0.14,
                    contrast=0.16,
                    saturation=0.07,
                    hue=0.015,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.55))],
                    p=0.12,
                ),
            ]
        )
    elif use_augmentation and uses_damage_safe_randaugment:
        randaugment, _noise_std = build_damage_safe_randaugment_profile(
            augmentation_profile
        )
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                randaugment,
            ]
        )
    elif use_augmentation and augmentation_profile != "none":
        raise ValueError(f"Unsupported augmentation_profile: {augmentation_profile}")

    transform_steps.append(transforms.ToTensor())
    if use_augmentation and augmentation_profile == "damage_safe":
        transform_steps.extend(
            [
                AddGaussianNoise(std=0.012, p=0.15),
                transforms.RandomErasing(
                    p=0.08,
                    scale=(0.004, 0.018),
                    ratio=(0.4, 2.5),
                    value="random",
                ),
            ]
        )
    elif use_augmentation and uses_damage_safe_randaugment:
        _randaugment, noise_std = build_damage_safe_randaugment_profile(
            augmentation_profile
        )
        transform_steps.append(
            AddGaussianNoise(std=noise_std, p=0.18)
        )

    transform_steps.append(
        transforms.Normalize(
            mean=RGB_NORMALIZATION_MEAN,
            std=RGB_NORMALIZATION_STD,
        )
    )

    return transforms.Compose(transform_steps)


def build_eval_transform(
    image_size: int = 224,
    *,
    include_resize: bool = True,
) -> transforms.Compose:
    transform_steps: list[transforms.Transform] = []

    if include_resize:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=RGB_NORMALIZATION_MEAN,
                std=RGB_NORMALIZATION_STD,
            ),
        ]
    )

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
            score_band = row["training_score_band"]
            if score_band not in SCORE_BAND_TO_INDEX:
                raise ValueError(
                    f"Row {row_index} has invalid training_score_band={score_band!r}"
                )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]

        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        if self.target_mode in {"coarse", "coarse_ordinal"}:
            target = torch.tensor(
                COARSE_CLASS_TO_INDEX[row["training_coarse_class"]],
                dtype=torch.long,
            )
        elif self.target_mode == "score_band":
            target = torch.tensor(
                SCORE_BAND_TO_INDEX[row["training_score_band"]],
                dtype=torch.long,
            )
        elif self.target_mode == "multitask":
            target = {
                "coarse": torch.tensor(
                    COARSE_CLASS_TO_INDEX[row["training_coarse_class"]],
                    dtype=torch.long,
                ),
                "score_band": torch.tensor(
                    SCORE_BAND_TO_INDEX[row["training_score_band"]],
                    dtype=torch.long,
                ),
            }
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
