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
    parser = argparse.ArgumentParser(description="Update prompt review status after manual review.")
    parser.add_argument("--id", required=True, type=int, help="Prompt row id in the SQLite database")
    parser.add_argument("--status", required=True, help="New prompt status, for example approved or rejected")
    parser.add_argument("--notes", default=None, help="Optional review notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Dream2DetectRepository()
    repository.initialize()
    repository.update_prompt_review(args.id, prompt_status=args.status, review_notes=args.notes)
    print(f"Updated prompt {args.id} to status={args.status}")


if __name__ == "__main__":
    main()
