from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.services.prompt_drafting import draft_prompts_for_band, save_drafted_prompts
from dream2detect.storage.repository import Dream2DetectRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft and store prompts for one severity band.")
    parser.add_argument("--band", required=True, help="Score band, for example 36-45")
    parser.add_argument("--count", required=True, type=int, help="Number of prompts to draft")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampler seed for reproducible feature assignment selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Dream2DetectRepository()
    repository.initialize()
    drafted = draft_prompts_for_band(args.band, args.count, repository, seed=args.seed)
    save_drafted_prompts(args.band, drafted, repository)
    print(f"Stored {len(drafted)} drafted prompts for band {args.band}")


if __name__ == "__main__":
    main()
