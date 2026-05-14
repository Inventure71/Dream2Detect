from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/datasets/label_audit_candidates.csv"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a prioritized CSV for manual label audit."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-all-intact",
        action="store_true",
        default=True,
        help="Include all intact examples because this class is currently weak.",
    )
    return parser


def build_audit_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "image_path",
        "training_score_band",
        "training_coarse_class",
        "training_label_source",
        "qc_status",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    boundary_bands = {
        "21-30",
        "31-35",
        "36-45",
        "56-65",
        "66-75",
    }
    audit_reasons: list[str] = []

    for _, row in frame.iterrows():
        reasons: list[str] = []
        if row["training_coarse_class"] == "intact":
            reasons.append("intact_failure_class")
        if row["training_score_band"] in boundary_bands:
            reasons.append("class_boundary_band")
        if str(row["training_label_source"]).endswith("unreviewed"):
            reasons.append("unreviewed_label")
        audit_reasons.append(";".join(reasons))

    candidates = frame.copy()
    candidates["audit_reason"] = audit_reasons
    candidates = candidates[candidates["audit_reason"] != ""].copy()

    priority_order = {
        "intact": 0,
        "minor": 1,
        "moderate": 2,
        "severe": 3,
    }
    candidates["_class_priority"] = candidates["training_coarse_class"].map(
        priority_order
    )
    candidates = candidates.sort_values(
        by=["_class_priority", "training_score_band", "image_path"],
        kind="stable",
    ).drop(columns=["_class_priority"])

    preferred_columns = [
        "audit_reason",
        "image_path",
        "source_image_path",
        "training_score_band",
        "training_coarse_class",
        "training_representative_score",
        "training_label_source",
        "qc_status",
        "prompt_id",
        "prompt_title",
        "damage_profile_primary",
        "damage_profile_secondary",
        "damage_location_primary",
    ]
    columns = [column for column in preferred_columns if column in candidates.columns]
    return candidates[columns]


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    candidates = build_audit_candidates(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)

    print(f"Audit candidates: {len(candidates)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
