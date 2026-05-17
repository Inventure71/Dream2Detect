from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.storage.repository import Dream2DetectRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the full stored prompt for one prompt id.")
    parser.add_argument("--id", required=True, type=int, help="Prompt row id in the SQLite database")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Dream2DetectRepository()
    repository.initialize()
    row = repository.get_prompt(args.id)
    if row is None:
        raise SystemExit(f"Prompt id={args.id} not found.")

    print(f"id: {row['id']}")
    print(f"uid: {row['prompt_uid']}")
    print(f"band: {row['score_band']}")
    print(f"coarse_class: {row['coarse_class']}")
    print(f"representative_score: {row['representative_score']}")
    print(f"prompt_status: {row['prompt_status']}")
    print(f"image_status: {row['image_status']}")
    print(f"qc_status: {row['qc_status']}")
    print(f"title: {row['prompt_title']}")
    print("\nfeature_assignment:\n")
    feature_fields = [
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
    ]
    for field_name in feature_fields:
        print(f"{field_name}: {row[field_name] or ''}")
    print("\nprompt_text:\n")
    print(row["prompt_text"])
    print("\nprompt_rationale:\n")
    print(row["prompt_rationale"] or "")
    print("\nuniqueness_notes:\n")
    print(row["uniqueness_notes"] or "")
    if row["review_notes"]:
        print("\nreview_notes:\n")
        print(row["review_notes"])
    if (
        row["reviewed_score_band"]
        or row["reviewed_coarse_class"]
        or row["final_score_band"]
        or row["final_coarse_class"]
        or row["qc_notes"]
    ):
        print("\nimage_qc:\n")
        print(f"reviewed_score_band: {row['reviewed_score_band'] or ''}")
        print(f"reviewed_coarse_class: {row['reviewed_coarse_class'] or ''}")
        print(f"final_score_band: {row['final_score_band'] or ''}")
        print(f"final_coarse_class: {row['final_coarse_class'] or ''}")
        print(f"qc_notes: {row['qc_notes'] or ''}")
    if row["image_error"]:
        print("\nimage_error:\n")
        print(row["image_error"])


if __name__ == "__main__":
    main()
