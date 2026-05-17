from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


DATASETS = {
    "synthetic": {
        "source_manifest": "data/datasets/synthetic_full_qc_plus_v2_scale_processed_384.csv",
        "description": "Full-QC synthetic delivery dataset with V2 scale-up examples.",
    },
    "real": {
        "source_manifest": "data/datasets/real_labeled_dataset_current_padded_384_full.csv",
        "description": "Current labeled real-image delivery dataset, padded to 384 px.",
    },
}


def _resolve_source_image(source_root: Path, image_path_value: str) -> Path:
    image_path = Path(image_path_value)
    if image_path.is_absolute():
        return image_path
    return source_root / image_path


def _unique_name(used_names: set[str], row_index: int, source_path: Path) -> str:
    candidate = source_path.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    suffix = source_path.suffix
    stem = source_path.stem
    candidate = f"{row_index:04d}_{stem}{suffix}"
    while candidate in used_names:
        candidate = f"{row_index:04d}_{len(used_names)}_{stem}{suffix}"
    used_names.add(candidate)
    return candidate


def build_dataset(name: str, source_root: Path, output_root: Path) -> None:
    config = DATASETS[name]
    source_manifest = source_root / config["source_manifest"]
    dataset_root = output_root / name
    images_root = dataset_root / "images"

    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest}")

    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    images_root.mkdir(parents=True)

    used_names: set[str] = set()
    missing: list[str] = []
    copied_rows: list[dict[str, str]] = []

    with source_manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {source_manifest}")

        fieldnames = list(reader.fieldnames)
        for row_index, row in enumerate(reader):
            source_image = _resolve_source_image(source_root, row["image_path"])
            if not source_image.exists():
                missing.append(str(source_image))
                continue

            image_name = _unique_name(used_names, row_index, source_image)
            destination = images_root / image_name
            shutil.copy2(source_image, destination)

            copied_row = dict(row)
            copied_row["image_path"] = f"images/{image_name}"
            copied_rows.append(copied_row)

    if missing:
        examples = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{name} has {len(missing)} missing source images. First missing files:\n{examples}"
        )

    output_manifest = dataset_root / "manifest.csv"
    with output_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(copied_rows)

    readme = dataset_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {name.title()} Delivery Dataset",
                "",
                config["description"],
                "",
                f"- Source manifest: `{config['source_manifest']}`",
                f"- Rows: `{len(copied_rows)}`",
                "- Image paths in `manifest.csv` are relative to this dataset folder.",
                "- Use this folder as the image root for the restored main-branch scripts.",
                "",
                "Example:",
                "",
                "```bash",
                f"python3 scripts/validate_delivery_datasets.py --dataset {name}",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact Dream2Detect delivery datasets.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="Repo checkout containing the full ignored data/ tree.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset"),
        help="Output directory for compact delivery datasets.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    for name in DATASETS:
        build_dataset(name, source_root, output_root)
        print(f"built {name}: {output_root / name}")


if __name__ == "__main__":
    main()

