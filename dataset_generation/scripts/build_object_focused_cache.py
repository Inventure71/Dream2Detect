from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.preprocessing.object_crop import (
    DEFAULT_SAM3_PROMPTS,
    build_crop_contact_sheet,
    build_crop_geometry_contact_sheet,
    build_object_focused_manifest,
)


def parse_prompts(raw: str) -> tuple[str, ...]:
    prompts = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not prompts:
        raise ValueError("At least one prompt is required.")
    return prompts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an object-focused image cache by cropping to the package/box."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--source-image-column", default="image_path")
    parser.add_argument(
        "--crop-backend",
        choices=["hybrid", "sam3", "center"],
        default="sam3",
        help=(
            "sam3 requires SAM 3 detections; hybrid tries SAM 3 then falls back to "
            "center-prior cropping; center disables SAM 3."
        ),
    )
    parser.add_argument(
        "--sam3-prompts",
        default=",".join(DEFAULT_SAM3_PROMPTS),
        help="Comma-separated SAM 3 text prompts.",
    )
    parser.add_argument(
        "--sam3-device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device passed to SAM3 when the installed SAM3 package supports a device argument.",
    )
    parser.add_argument("--crop-padding-fraction", type=float, default=0.01)
    parser.add_argument("--center-crop-fraction", type=float, default=0.9)
    parser.add_argument("--sam3-min-area-ratio", type=float, default=0.05)
    parser.add_argument("--sam3-max-area-ratio", type=float, default=0.98)
    parser.add_argument(
        "--resize-mode",
        choices=["square_crop", "stretch", "pad"],
        default="square_crop",
        help=(
            "square_crop converts the SAM box to a square before resizing; stretch "
            "distorts the detected crop; pad preserves aspect ratio and letterboxes."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N rows. Use 0 to disable progress messages.",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=None,
        help="Optional side-by-side source/crop contact sheet path.",
    )
    parser.add_argument(
        "--geometry-contact-sheet",
        type=Path,
        default=None,
        help=(
            "Optional four-column sheet: original, SAM bbox overlay, square crop "
            "overlay, and resulting image."
        ),
    )
    parser.add_argument("--contact-sheet-max-rows", type=int, default=40)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    source_manifest = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_manifest = args.output_manifest.resolve()

    if not source_manifest.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {source_manifest}")

    print("Building object-focused image cache")
    print(f"Source manifest: {source_manifest}")
    print(f"Output directory: {output_dir}")
    print(f"Output manifest: {output_manifest}")
    print(f"Image size: {args.image_size}")
    print(f"Crop backend: {args.crop_backend}")
    print(f"SAM 3 device: {args.sam3_device}")
    print(f"SAM 3 prompts: {args.sam3_prompts}")
    print(f"Source image column: {args.source_image_column}")
    print(f"Resize mode: {args.resize_mode}")
    if args.limit is not None:
        print(f"Limit: {args.limit}")
    print()

    sam3_available = importlib.util.find_spec("sam3") is not None
    if args.crop_backend == "sam3" and not sam3_available:
        raise RuntimeError(
            "SAM 3 is not installed in this Python environment. Install and configure "
            "facebookresearch/sam3 before running with --crop-backend sam3. Use "
            "--crop-backend hybrid only if center fallback is acceptable."
        )
    if args.crop_backend == "hybrid" and not sam3_available:
        print(
            "WARNING: SAM 3 is not installed. Hybrid mode will use center-prior "
            "fallback for every image.",
            file=sys.stderr,
        )

    derived = build_object_focused_manifest(
        source_manifest,
        output_dir,
        output_manifest,
        image_size=args.image_size,
        source_image_column=args.source_image_column,
        crop_backend=args.crop_backend,
        sam3_prompts=parse_prompts(args.sam3_prompts),
        sam3_device=args.sam3_device,
        resize_mode=args.resize_mode,
        crop_padding_fraction=args.crop_padding_fraction,
        center_crop_fraction=args.center_crop_fraction,
        sam3_min_area_ratio=args.sam3_min_area_ratio,
        sam3_max_area_ratio=args.sam3_max_area_ratio,
        limit=args.limit,
        progress_every=args.progress_every,
    )

    if args.contact_sheet is not None:
        contact_sheet = args.contact_sheet.resolve()
        build_crop_contact_sheet(
            output_manifest,
            contact_sheet,
            max_rows=args.contact_sheet_max_rows,
        )
        print(f"Contact sheet: {contact_sheet}")
    if args.geometry_contact_sheet is not None:
        geometry_contact_sheet = args.geometry_contact_sheet.resolve()
        build_crop_geometry_contact_sheet(
            output_manifest,
            geometry_contact_sheet,
            max_rows=args.contact_sheet_max_rows,
        )
        print(f"Geometry contact sheet: {geometry_contact_sheet}")

    print("Object-focused cache completed successfully.")
    print(f"Rows written: {len(derived)}")
    print(derived["object_crop_status"].value_counts().to_string())
    print(derived["object_crop_method"].value_counts().to_string())


if __name__ == "__main__":
    main()
