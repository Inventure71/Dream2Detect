from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.services.image_generation import generate_images_for_pending_prompts
from dream2detect.storage.repository import Dream2DetectRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images for prompts waiting on image generation.")
    parser.add_argument(
        "--prompt-status",
        default="approved",
        help="Only generate images for prompts with this prompt_status. Default: approved",
    )
    parser.add_argument(
        "--output-subdir",
        default="pilot",
        help="Subdirectory under generated_images/ where images should be saved. Default: pilot",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of prompts to generate in this run. Recommended for smoke tests.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of image generations to run in parallel. Keep this small for controlled spend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Dream2DetectRepository()
    repository.initialize()
    written = generate_images_for_pending_prompts(
        repository,
        prompt_status=args.prompt_status,
        output_subdir=args.output_subdir,
        limit=args.limit,
        max_concurrency=args.max_concurrency,
    )
    print(f"Generated {len(written)} image(s)")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
