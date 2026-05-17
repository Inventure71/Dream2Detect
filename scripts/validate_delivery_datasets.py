from __future__ import annotations

import argparse
import csv
from pathlib import Path


def validate_dataset(dataset_root: Path) -> tuple[int, int]:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    missing = 0
    rows = 0
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(f"{manifest_path} is missing image_path")

        for row in reader:
            rows += 1
            image_path = dataset_root / row["image_path"]
            if not image_path.exists():
                missing += 1

    return rows, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate delivery dataset manifests.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root containing synthetic/ and real/ delivery dataset folders.",
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "real"],
        help="Validate one dataset instead of both.",
    )
    args = parser.parse_args()

    names = [args.dataset] if args.dataset else ["synthetic", "real"]
    failures = []
    for name in names:
        rows, missing = validate_dataset(args.dataset_root / name)
        print(f"{name}: rows={rows} missing_images={missing}")
        if rows == 0 or missing:
            failures.append(name)

    if failures:
        raise SystemExit(f"Invalid delivery datasets: {', '.join(failures)}")


if __name__ == "__main__":
    main()

