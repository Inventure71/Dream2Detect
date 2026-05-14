from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_pil_image

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.training.dataset import (  # noqa: E402
    RGB_NORMALIZATION_MEAN,
    RGB_NORMALIZATION_STD,
    build_train_transform,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create before/after contact sheets for training augmentation QC."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--augmentation-profile", default="damage_safe")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--augmentations-per-image", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def tensor_to_display_image(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(RGB_NORMALIZATION_MEAN).view(3, 1, 1)
    std = torch.tensor(RGB_NORMALIZATION_STD).view(3, 1, 1)
    denormalized = torch.clamp(tensor.cpu() * std + mean, 0.0, 1.0)
    return to_pil_image(denormalized)


def resize_for_sheet(image: Image.Image, size: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), (245, 245, 245))
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def main() -> None:
    args = build_argument_parser().parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    frame = pd.read_csv(args.manifest)
    if len(frame) == 0:
        raise ValueError("Manifest is empty.")

    sample_count = min(args.sample_size, len(frame))
    sampled_indices = random.sample(range(len(frame)), sample_count)
    sampled = frame.iloc[sampled_indices].reset_index(drop=True)

    transform = build_train_transform(
        image_size=args.image_size,
        include_resize=False,
        use_augmentation=True,
        augmentation_profile=args.augmentation_profile,
    )

    tile_size = 160
    label_height = 58
    padding = 10
    columns = 1 + args.augmentations_per_image
    sheet_width = columns * tile_size + (columns + 1) * padding
    sheet_height = sample_count * (tile_size + label_height) + (sample_count + 1) * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(13)
    small_font = load_font(11)

    for row_position, row in sampled.iterrows():
        y = padding + row_position * (tile_size + label_height + padding)
        source_path = Path(str(row["image_path"]))
        with Image.open(source_path) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            original_tile = resize_for_sheet(source, tile_size)

            x = padding
            sheet.paste(original_tile, (x, y))
            draw.rectangle((x, y, x + tile_size, y + tile_size), outline=(40, 40, 40))
            draw.text((x, y + tile_size + 4), "original", font=font, fill=(0, 0, 0))

            for aug_index in range(args.augmentations_per_image):
                augmented = tensor_to_display_image(transform(source.copy()))
                tile = resize_for_sheet(augmented, tile_size)
                x = padding + (aug_index + 1) * (tile_size + padding)
                sheet.paste(tile, (x, y))
                draw.rectangle((x, y, x + tile_size, y + tile_size), outline=(40, 40, 40))
                draw.text(
                    (x, y + tile_size + 4),
                    f"{args.augmentation_profile} #{aug_index + 1}",
                    font=font,
                    fill=(0, 0, 0),
                )

        label = (
            f"row={sampled_indices[row_position]} "
            f"{row['training_score_band']} {row['training_coarse_class']}"
        )
        draw.text(
            (padding, y + tile_size + 24),
            label,
            font=small_font,
            fill=(30, 30, 30),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    sampled.to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Wrote augmentation preview: {args.output}")
    print(f"Wrote sampled rows: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
