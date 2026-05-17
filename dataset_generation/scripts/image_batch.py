from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.pipelines.v1.image_batch import (
    ingest_completed_image_batch,
    prepare_image_generation_batch,
    refresh_image_batch_status,
    submit_prepared_image_batch,
)
from dream2detect.storage.repository import Dream2DetectRepository


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, submit, check, and ingest OpenAI Batch image jobs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Write a Batch API JSONL file from pending prompt rows. Does not submit.",
    )
    prepare.add_argument("--batch-name", default=None)
    prepare.add_argument("--prompt-status", default="drafted")
    prepare.add_argument("--image-status", default="pending")
    prepare.add_argument("--limit", type=int, default=None)
    prepare.add_argument("--model", default="gpt-image-2")
    prepare.add_argument("--quality", default="medium")
    prepare.add_argument("--size", default="1024x1024")
    prepare.add_argument("--output-subdir", default="targeted_555_round1")
    prepare.add_argument(
        "--batch-root",
        type=Path,
        default=REPO_ROOT / "data/batches/image_generation",
    )

    submit = subparsers.add_parser(
        "submit",
        help="Upload a prepared JSONL file and create the OpenAI batch.",
    )
    submit.add_argument("--batch-dir", type=Path, required=True)

    status = subparsers.add_parser(
        "status",
        help="Refresh and print the OpenAI batch status.",
    )
    status.add_argument("--batch-dir", type=Path, required=True)

    ingest = subparsers.add_parser(
        "ingest",
        help="Download completed batch results, save PNGs, and update SQLite.",
    )
    ingest.add_argument("--batch-dir", type=Path, required=True)
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest prompts even if they are already marked generated.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        repository = Dream2DetectRepository()
        repository.initialize()
        prepared = prepare_image_generation_batch(
            repository,
            batch_name=args.batch_name,
            prompt_status=args.prompt_status,
            image_status=args.image_status,
            limit=args.limit,
            model=args.model,
            quality=args.quality,
            size=args.size,
            output_subdir=args.output_subdir,
            batch_root=args.batch_root,
        )
        print("Prepared image batch.")
        print(f"Requests: {prepared.request_count}")
        print(f"Batch dir: {prepared.batch_dir}")
        print(f"Input JSONL: {prepared.input_jsonl_path}")
        print(f"Manifest JSON: {prepared.manifest_json_path}")
        print(f"Metadata JSON: {prepared.metadata_path}")
        print()
        print("Submit later with:")
        print(f"python3 scripts/image_batch.py submit --batch-dir {prepared.batch_dir}")
        return

    if args.command == "submit":
        metadata = submit_prepared_image_batch(args.batch_dir)
        print("Submitted image batch.")
        print(f"Batch id: {metadata['batch_id']}")
        print(f"Input file id: {metadata['input_file_id']}")
        print(f"Status: {metadata['status']}")
        return

    if args.command == "status":
        metadata = refresh_image_batch_status(args.batch_dir)
        print(f"Batch id: {metadata['batch_id']}")
        print(f"Status: {metadata['status']}")
        print(f"Request counts: {metadata.get('request_counts')}")
        print(f"Output file id: {metadata.get('output_file_id')}")
        print(f"Error file id: {metadata.get('error_file_id')}")
        return

    if args.command == "ingest":
        repository = Dream2DetectRepository()
        repository.initialize()
        counts = ingest_completed_image_batch(
            repository,
            batch_dir=args.batch_dir,
            force=args.force,
        )
        print("Ingested image batch results.")
        print(f"Generated: {counts['generated']}")
        print(f"Failed: {counts['failed']}")
        print(f"Skipped: {counts['skipped']}")
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
