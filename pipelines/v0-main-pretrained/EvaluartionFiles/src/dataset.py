"""
BoxDamageDataset
================
Loads images + labels from manifest.csv and applies a rich augmentation
pipeline designed to maximise variation from a small synthetic base.

Label modes
-----------
- "score"   : regression on training_representative_score  (0–100)
- "coarse"  : 4-class classification  (intact / minor / moderate / severe)
- "band"    : 10-band classification  (0-10 … 86-100)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── label maps ───────────────────────────────────────────────────────────────
COARSE_CLASSES = ["intact", "minor", "moderate", "severe"]
COARSE_TO_IDX  = {c: i for i, c in enumerate(COARSE_CLASSES)}

BAND_CLASSES = ["0-10","11-20","21-30","31-35","36-45",
                "46-55","56-65","66-75","76-85","86-100"]
BAND_TO_IDX  = {b: i for i, b in enumerate(BAND_CLASSES)}

# ── augmentation pipelines ───────────────────────────────────────────────────
IMG_MEAN = (0.485, 0.456, 0.406)   # ImageNet
IMG_STD  = (0.229, 0.224, 0.225)

def build_train_transform(img_size: int = 384) -> A.Compose:
    """
    Heavy augmentation to synthetically multiply the dataset.
    Covers: geometric, colour, noise, blur, compression, occlusion.
    """
    return A.Compose([
        # ── geometry ────────────────────────────────────────────────────────
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.65, 1.0),
                            ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15,
                           rotate_limit=20, border_mode=0, p=0.7),
        A.Perspective(scale=(0.03, 0.09), p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        # ── colour / brightness ─────────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.30,
                                   contrast_limit=0.30, p=0.8),
        A.HueSaturationValue(hue_shift_limit=12,
                             sat_shift_limit=30,
                             val_shift_limit=20, p=0.6),
        A.RandomGamma(gamma_limit=(70, 140), p=0.4),
        A.CLAHE(clip_limit=3.0, p=0.3),
        A.ToGray(p=0.05),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.05, p=0.4),
        # ── noise / texture ─────────────────────────────────────────────────
        A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
        # ── blur / sharpness ────────────────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.3),
        # ── compression / degradation ────────────────────────────────────────
        A.ImageCompression(quality_range=(60, 95), p=0.35),
        A.Downscale(scale_range=(0.55, 0.85), p=0.2),
        # ── occlusion ────────────────────────────────────────────────────────
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(16, 48),
                        hole_width_range=(16, 48),
                        p=0.35),
        # ── shadow / lighting ────────────────────────────────────────────────
        A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                       num_shadows_limit=(1, 3),
                       shadow_dimension=4, p=0.25),
        # ── normalise ────────────────────────────────────────────────────────
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def build_val_transform(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


# ── dataset class ─────────────────────────────────────────────────────────────
class BoxDamageDataset(Dataset):
    """
    Parameters
    ----------
    manifest_path : path to manifest.csv
    images_root   : directory that contains the images/ subfolder
                    (i.e. manifest image_path = images/<filename>.png)
    label_mode    : "score" | "coarse" | "band"
    transform     : albumentations Compose pipeline
    split_ids     : optional list of prompt_id ints to filter rows
    """

    def __init__(
        self,
        manifest_path: str,
        images_root: str,
        label_mode: str = "coarse",
        transform=None,
        split_ids=None,
    ):
        self.label_mode = label_mode
        self.transform = transform
        self.images_root = Path(images_root)

        df = pd.read_csv(manifest_path)
        # keep only QC-accepted rows
        df = df[df["qc_status"] == "accepted_as_labeled"].copy()
        if split_ids is not None:
            df = df[df["prompt_id"].isin(split_ids)].copy()
        df = df.reset_index(drop=True)
        self.df = df

    # ── helpers ──────────────────────────────────────────────────────────────
    def num_classes(self) -> int:
        if self.label_mode == "coarse":
            return len(COARSE_CLASSES)
        elif self.label_mode == "band":
            return len(BAND_CLASSES)
        else:
            return 1   # regression

    def class_names(self):
        if self.label_mode == "coarse":
            return COARSE_CLASSES
        elif self.label_mode == "band":
            return BAND_CLASSES
        else:
            return ["score"]

    def get_sampler_weights(self) -> WeightedRandomSampler:
        """Return a balanced sampler to counter class imbalance."""
        if self.label_mode == "score":
            raise ValueError("Weighted sampler not meaningful for regression mode.")
        col = "training_coarse_class" if self.label_mode == "coarse" \
              else "training_score_band"
        counts = self.df[col].value_counts()
        weights = self.df[col].map(lambda c: 1.0 / counts[c]).values
        return WeightedRandomSampler(
            weights=torch.FloatTensor(weights),
            num_samples=len(weights),
            replacement=True,
        )

    # ── core interface ────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["image_path"]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        # build label
        if self.label_mode == "score":
            label = torch.tensor(
                float(row["training_representative_score"]) / 100.0,
                dtype=torch.float32,
            )
        elif self.label_mode == "coarse":
            label = torch.tensor(
                COARSE_TO_IDX[row["training_coarse_class"]], dtype=torch.long
            )
        else:  # band
            label = torch.tensor(
                BAND_TO_IDX[row["training_score_band"]], dtype=torch.long
            )

        meta = {
            "prompt_id": int(row["prompt_id"]),
            "image_path": str(row["image_path"]),
            "score": float(row["training_representative_score"]),
            "coarse": row["training_coarse_class"],
            "band": row["training_score_band"],
        }
        return image, label, meta
