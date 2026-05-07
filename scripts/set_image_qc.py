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
    parser = argparse.ArgumentParser(description="Record QC review for one generated image prompt row.")
    parser.add_argument("--id", required=True, type=int, help="Prompt row id in the SQLite database")
    parser.add_argument(
        "--qc-status",
        required=True,
        choices=("accepted_as_labeled", "accepted_relabel", "rejected"),
        help="QC decision for the generated image",
    )
    parser.add_argument("--reviewed-band", default=None, help="Band judged from the generated image")
    parser.add_argument("--reviewed-class", default=None, help="Coarse class judged from the generated image")
    parser.add_argument("--final-band", default=None, help="Final accepted band after QC")
    parser.add_argument("--final-class", default=None, help="Final accepted coarse class after QC")
    parser.add_argument("--notes", default=None, help="Optional QC notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Dream2DetectRepository()
    repository.initialize()
    repository.update_generated_image_qc(
        args.id,
        qc_status=args.qc_status,
        reviewed_score_band=args.reviewed_band,
        reviewed_coarse_class=args.reviewed_class,
        final_score_band=args.final_band,
        final_coarse_class=args.final_class,
        qc_notes=args.notes,
    )
    print(f"Updated prompt {args.id} QC to {args.qc_status}")


if __name__ == "__main__":
    main()
