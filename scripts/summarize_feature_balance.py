from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.features.catalog import FEATURE_AXES
from dream2detect.storage.repository import Dream2DetectRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize feature-value counts for one band from the prompt database.")
    parser.add_argument("--band", required=True, help="Score band, for example 36-45")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Dream2DetectRepository()
    repository.initialize()

    for axis in FEATURE_AXES:
        counts = repository.get_feature_value_counts(score_band=args.band, axis_name=axis.name)
        print(f"{axis.name}:")
        allowed_values = [option.value for option in axis.allowed_options(args.band)]
        if not allowed_values:
            print("  <no allowed values>")
            continue
        for value in allowed_values:
            print(f"  {value}: {counts[value]}")
        print()


if __name__ == "__main__":
    main()
