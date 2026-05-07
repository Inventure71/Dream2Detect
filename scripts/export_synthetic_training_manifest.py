from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.bands import get_band_definition
from dream2detect.storage.repository import Dream2DetectRepository


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a synthetic training manifest from generated images with best-available labels."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV path for the exported training manifest.",
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Export only reviewed images accepted as labeled or relabeled.",
    )
    return parser.parse_args()


def _resolve_training_label(prompt_row) -> tuple[str, str, int, str]:
    qc_status = prompt_row["qc_status"]
    if qc_status == "accepted_relabel":
        score_band = prompt_row["final_score_band"]
        coarse_class = prompt_row["final_coarse_class"]
        label_source = "final_relabel"
    elif qc_status == "accepted_as_labeled":
        score_band = prompt_row["score_band"]
        coarse_class = prompt_row["coarse_class"]
        label_source = "intended_reviewed_match"
    else:
        score_band = prompt_row["score_band"]
        coarse_class = prompt_row["coarse_class"]
        label_source = "intended_unreviewed"
    representative_score = get_band_definition(score_band).representative_score
    return score_band, coarse_class, representative_score, label_source


def _include_row(prompt_row, *, reviewed_only: bool) -> bool:
    if prompt_row["image_status"] != "generated":
        return False
    if prompt_row["qc_status"] == "rejected":
        return False
    if reviewed_only and prompt_row["qc_status"] not in {"accepted_as_labeled", "accepted_relabel"}:
        return False
    return True


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repository = Dream2DetectRepository()
    repository.initialize()
    prompts = repository.list_prompts()

    rows_to_write: list[dict[str, str | int]] = []
    for prompt_row in prompts:
        if not _include_row(prompt_row, reviewed_only=args.reviewed_only):
            continue
        training_score_band, training_coarse_class, training_representative_score, training_label_source = (
            _resolve_training_label(prompt_row)
        )
        row = {
            "prompt_id": int(prompt_row["id"]),
            "prompt_uid": prompt_row["prompt_uid"],
            "image_path": prompt_row["image_ref"],
            "prompt_status": prompt_row["prompt_status"],
            "image_status": prompt_row["image_status"],
            "qc_status": prompt_row["qc_status"],
            "intended_score_band": prompt_row["score_band"],
            "intended_coarse_class": prompt_row["coarse_class"],
            "intended_representative_score": int(prompt_row["representative_score"]),
            "reviewed_score_band": prompt_row["reviewed_score_band"] or "",
            "reviewed_coarse_class": prompt_row["reviewed_coarse_class"] or "",
            "final_score_band": prompt_row["final_score_band"] or "",
            "final_coarse_class": prompt_row["final_coarse_class"] or "",
            "training_score_band": training_score_band,
            "training_coarse_class": training_coarse_class,
            "training_representative_score": training_representative_score,
            "training_label_source": training_label_source,
            "prompt_title": prompt_row["prompt_title"],
            "prompt_model": prompt_row["prompt_model"],
            "band_spec_file": prompt_row["band_spec_file"],
            "qc_notes": prompt_row["qc_notes"] or "",
        }
        for column_name in FEATURE_COLUMNS:
            row[column_name] = prompt_row[column_name] or ""
        rows_to_write.append(row)

    fieldnames = list(rows_to_write[0].keys()) if rows_to_write else [
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
        *FEATURE_COLUMNS,
        "qc_notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(f"Wrote {len(rows_to_write)} rows to {output_path}")


if __name__ == "__main__":
    main()
