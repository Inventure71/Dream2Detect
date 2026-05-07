from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.features import sample_feature_assignments
from dream2detect.storage.repository import Dream2DetectRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview balanced feature assignments for one severity band.")
    parser.add_argument("--band", required=True, help="Score band, for example 36-45")
    parser.add_argument("--count", required=True, type=int, help="Number of assignments to sample")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampler seed for reproducible previews")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Dream2DetectRepository()
    repository.initialize()
    assignments = sample_feature_assignments(
        score_band=args.band,
        count=args.count,
        repository=repository,
        seed=args.seed,
    )
    for index, assignment in enumerate(assignments, start=1):
        print(f"assignment_{index}:")
        for key, value in assignment.as_dict().items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
