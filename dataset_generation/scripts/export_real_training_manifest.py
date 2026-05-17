from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.bands import get_band_definition


FEATURE_COLUMNS = (
    "damage_profile_primary",
    "damage_profile_secondary",
    "damage_location_primary",
    "box_form_factor",
    "box_pattern",
    "label_presence",
    "tape_profile",
    "background_context",
    "camera_angle",
    "lighting_style",
)

FIELDNAMES = [
    "prompt_id",
    "prompt_uid",
    "image_path",
    "prompt_status",
    "image_status",
    "qc_status",
    "intended_score_band",
    "intended_coarse_class",
    "intended_representative_score",
    "reviewed_score_band",
    "reviewed_coarse_class",
    "final_score_band",
    "final_coarse_class",
    "training_score_band",
    "training_coarse_class",
    "training_representative_score",
    "training_label_source",
    "prompt_title",
    "prompt_model",
    "band_spec_file",
    "qc_notes",
    *FEATURE_COLUMNS,
    "source_dataset",
    "original_label",
    "source_registry",
    "review_status",
    "disagreement_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export labeled real images into the standard Dream2Detect training manifest format."
    )
    parser.add_argument(
        "--registry",
        default=str(REPO_ROOT / "data/registries/real_relabel_registry.csv"),
        help="Path to the real relabel registry CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/datasets/real_labeled_dataset_current.csv"),
        help="Destination path for the training manifest.",
    )
    parser.add_argument(
        "--relative-image-paths",
        action="store_true",
        help="Keep image paths relative to the repo instead of writing absolute paths.",
    )
    return parser.parse_args()


def _stable_prompt_id(image_id: str, fallback: int) -> int:
    match = re.search(r"_(\d+)$", image_id)
    if match:
        return int(match.group(1))
    return fallback


def _resolve_image_path(raw_path: str) -> Path:
    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    return image_path


def _require_labeled_row(row: dict[str, str], row_number: int) -> None:
    score_band = row["final_score_band"]
    coarse_class = row["final_coarse_class"]
    representative_score = row["final_representative_score"]
    band = get_band_definition(score_band)

    if coarse_class != band.coarse_class:
        raise ValueError(
            f"Row {row_number} has final_coarse_class={coarse_class!r}, "
            f"but band {score_band!r} maps to {band.coarse_class!r}."
        )
    if int(representative_score) != band.representative_score:
        raise ValueError(
            f"Row {row_number} has final_representative_score={representative_score!r}, "
            f"but band {score_band!r} maps to {band.representative_score}."
        )

    image_path = _resolve_image_path(row["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(
            f"Row {row_number} points to a missing image: {image_path}"
        )


def _manifest_row(
    row: dict[str, str],
    *,
    row_number: int,
    registry_path: Path,
    relative_image_paths: bool,
) -> dict[str, str | int]:
    band = get_band_definition(row["final_score_band"])
    image_path = _resolve_image_path(row["image_path"])
    manifest_image_path = row["image_path"] if relative_image_paths else str(image_path)

    manifest = {
        "prompt_id": _stable_prompt_id(row["image_id"], row_number),
        "prompt_uid": row["image_id"],
        "image_path": manifest_image_path,
        "prompt_status": "not_applicable",
        "image_status": "available",
        "qc_status": "labeled",
        "intended_score_band": "",
        "intended_coarse_class": "",
        "intended_representative_score": "",
        "reviewed_score_band": row["human_score_band"],
        "reviewed_coarse_class": row["human_coarse_class"],
        "final_score_band": row["final_score_band"],
        "final_coarse_class": row["final_coarse_class"],
        "training_score_band": row["final_score_band"],
        "training_coarse_class": row["final_coarse_class"],
        "training_representative_score": band.representative_score,
        "training_label_source": "real_human_final",
        "prompt_title": f"Real labeled image {row['image_id']}",
        "prompt_model": "",
        "band_spec_file": band.spec_filename,
        "qc_notes": row["review_notes"],
        "source_dataset": row["source_dataset"],
        "original_label": row["original_label"],
        "source_registry": str(registry_path),
        "review_status": row["review_status"],
        "disagreement_flag": row["disagreement_flag"],
    }
    for column_name in FEATURE_COLUMNS:
        manifest[column_name] = ""
    return manifest


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with registry_path.open(encoding="utf-8", newline="") as handle:
        registry_rows = list(csv.DictReader(handle))

    rows_to_write: list[dict[str, str | int]] = []
    status_counts: dict[str, int] = {}
    for row_number, row in enumerate(registry_rows, start=1):
        review_status = row["review_status"]
        status_counts[review_status] = status_counts.get(review_status, 0) + 1
        if review_status != "labeled":
            continue
        if not row["final_score_band"]:
            raise ValueError(f"Row {row_number} is labeled but has no final_score_band.")
        _require_labeled_row(row, row_number)
        rows_to_write.append(
            _manifest_row(
                row,
                row_number=row_number,
                registry_path=registry_path,
                relative_image_paths=args.relative_image_paths,
            )
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"registry_rows={len(registry_rows)}")
    print(f"status_counts={status_counts}")
    print(f"training_rows={len(rows_to_write)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
