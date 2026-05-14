from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply manual synthetic QC decisions to a training manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--review-round",
        required=True,
        help="Value written to qc_review_round for rows touched by this decision file.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    frame = pd.read_csv(args.manifest)
    decisions = pd.read_csv(args.decisions)

    required_decision_columns = {
        "row_index",
        "qc_status",
        "reviewed_score_band",
        "reviewed_coarse_class",
        "final_score_band",
        "final_coarse_class",
        "training_representative_score",
        "training_label_source",
        "qc_notes",
    }
    missing_columns = required_decision_columns - set(decisions.columns)
    if missing_columns:
        raise ValueError(
            "Decision CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if decisions["row_index"].duplicated().any():
        duplicate_ids = decisions.loc[
            decisions["row_index"].duplicated(),
            "row_index",
        ].tolist()
        raise ValueError(f"Decision CSV has duplicate row_index values: {duplicate_ids}")

    touched = 0
    for decision in decisions.to_dict(orient="records"):
        row_index = int(decision["row_index"])
        if row_index < 0 or row_index >= len(frame):
            raise IndexError(f"Decision row_index={row_index} is outside manifest bounds.")

        frame.at[row_index, "qc_status"] = decision["qc_status"]
        frame.at[row_index, "reviewed_score_band"] = decision["reviewed_score_band"]
        frame.at[row_index, "reviewed_coarse_class"] = decision["reviewed_coarse_class"]
        frame.at[row_index, "final_score_band"] = decision["final_score_band"]
        frame.at[row_index, "final_coarse_class"] = decision["final_coarse_class"]
        frame.at[row_index, "training_score_band"] = decision["final_score_band"]
        frame.at[row_index, "training_coarse_class"] = decision["final_coarse_class"]
        frame.at[row_index, "training_representative_score"] = int(
            decision["training_representative_score"]
        )
        frame.at[row_index, "training_label_source"] = decision["training_label_source"]
        frame.at[row_index, "qc_notes"] = decision["qc_notes"]
        frame.at[row_index, "qc_review_round"] = args.review_round
        touched += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"Applied {touched} QC decisions.")
    print(f"Wrote manifest: {args.output}")


if __name__ == "__main__":
    main()
