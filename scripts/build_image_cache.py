from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic processed-image cache and derived manifest."
    )

    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=REPO_ROOT / "data/datasets/synthetic_starting_dataset_phase1_round1.csv",
        help="Path to the source manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/synthetic_all_processed_128",
        help="Directory where processed images will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=REPO_ROOT / "data/datasets/synthetic_starting_dataset_phase1_round1_processed_128.csv",
        help="Path to the derived manifest CSV.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square target image size.",
    )

    return parser


def process_one_image(
    source_path: Path,
    destination_path: Path,
    *,
    image_size: int,
) -> None:
    """
    Deterministic preprocessing only:
    - open source image
    - convert to RGB
    - resize to the target square resolution
    - save as PNG

    This function intentionally does not perform random augmentation.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(source_path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    image.save(destination_path, format="PNG")


def build_processed_manifest(
    source_manifest_path: Path,
    output_dir: Path,
    output_manifest_path: Path,
    *,
    image_size: int,
) -> None:
    frame = pd.read_csv(source_manifest_path)

    if len(frame) == 0:
        raise ValueError("Source manifest is empty.")

    if "image_path" not in frame.columns:
        raise ValueError("Source manifest must contain an image_path column.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    processed_paths: list[str] = []

    for row_index, row in frame.iterrows():
        source_path = Path(row["image_path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(
                f"Row {row_index} points to a missing source image: {source_path}"
            )

        destination_name = source_path.name
        destination_path = output_dir / destination_name

        process_one_image(
            source_path,
            destination_path,
            image_size=image_size,
        )

        processed_paths.append(str(destination_path))

    derived_frame = frame.copy()
    derived_frame["source_image_path"] = derived_frame["image_path"]
    derived_frame["image_path"] = processed_paths
    derived_frame["preprocessing_profile"] = f"rgb_resize_{image_size}"
    derived_frame["preprocessing_source_manifest"] = str(source_manifest_path.resolve())

    derived_frame.to_csv(output_manifest_path, index=False)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    source_manifest_path = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_manifest_path = args.output_manifest.resolve()

    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {source_manifest_path}")

    print("Building processed image cache")
    print(f"Source manifest: {source_manifest_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output manifest: {output_manifest_path}")
    print(f"Image size: {args.image_size}")
    print()

    build_processed_manifest(
        source_manifest_path,
        output_dir,
        output_manifest_path,
        image_size=args.image_size,
    )

    print("Processed cache completed successfully.")


if __name__ == "__main__":
    main()
