from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic processed-image cache and derived manifest."
    )

    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv"
        ),
        help="Path to the source manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "data/cache/synthetic_combined_phase1_plus_scaleup_200_processed_224"
        ),
        help="Directory where processed images will be written.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv"
        ),
        help="Path to the derived manifest CSV.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square target image size.",
    )
    parser.add_argument(
        "--preprocessing-mode",
        choices=["resize", "pad_resize"],
        default="resize",
        help=(
            "resize stretches the image to the square target. pad_resize preserves "
            "aspect ratio and pads to the square target."
        ),
    )
    parser.add_argument(
        "--source-image-column",
        default="source_image_path",
        help=(
            "Manifest column containing source image paths. Use source_image_path "
            "when rebuilding a new cache from an existing processed manifest."
        ),
    )

    return parser


def process_one_image(
    source_path: Path,
    destination_path: Path,
    *,
    image_size: int,
    preprocessing_mode: str,
) -> None:
    """
    Deterministic preprocessing only:
    - open source image
    - convert to RGB
    - resize to the target square resolution, optionally preserving aspect ratio
    - save as PNG

    This function intentionally does not perform random augmentation.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if preprocessing_mode == "resize":
            image = image.resize(
                (image_size, image_size),
                resample=Image.Resampling.LANCZOS,
            )
        elif preprocessing_mode == "pad_resize":
            image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
            padded = Image.new("RGB", (image_size, image_size), (245, 245, 245))
            offset = (
                (image_size - image.width) // 2,
                (image_size - image.height) // 2,
            )
            padded.paste(image, offset)
            image = padded
        else:
            raise ValueError(f"Unsupported preprocessing_mode: {preprocessing_mode}")
        image.save(destination_path, format="PNG")


def build_processed_manifest(
    source_manifest_path: Path,
    output_dir: Path,
    output_manifest_path: Path,
    *,
    image_size: int,
    source_image_column: str = "image_path",
    preprocessing_mode: str = "resize",
) -> None:
    frame = pd.read_csv(source_manifest_path)

    if len(frame) == 0:
        raise ValueError("Source manifest is empty.")

    if source_image_column not in frame.columns:
        raise ValueError(
            f"Source manifest must contain a {source_image_column!r} column."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    processed_paths: list[str] = []

    for row_index, row in frame.iterrows():
        source_path = Path(row[source_image_column]).resolve()
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
            preprocessing_mode=preprocessing_mode,
        )

        processed_paths.append(str(destination_path))

    derived_frame = frame.copy()
    derived_frame["source_image_path"] = [
        str(Path(value).resolve()) for value in frame[source_image_column].tolist()
    ]
    derived_frame["image_path"] = processed_paths
    profile_prefix = "rgb_resize" if preprocessing_mode == "resize" else "rgb_pad_resize"
    derived_frame["preprocessing_profile"] = f"{profile_prefix}_{image_size}"
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
    print(f"Preprocessing mode: {args.preprocessing_mode}")
    print(f"Source image column: {args.source_image_column}")
    print()

    build_processed_manifest(
        source_manifest_path,
        output_dir,
        output_manifest_path,
        image_size=args.image_size,
        source_image_column=args.source_image_column,
        preprocessing_mode=args.preprocessing_mode,
    )

    print("Processed cache completed successfully.")


if __name__ == "__main__":
    main()
