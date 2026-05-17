from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.pipelines.v2 import (  # noqa: E402
    ImageGenerationSpec,
    V2PipelineConfig,
    build_v2_pipeline_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the v2 real-failure-targeted synthetic image pipeline artifacts. "
            "This writes prompts and Batch JSONL but does not submit an API job."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--count-per-band", type=int, default=12)
    parser.add_argument(
        "--band-count",
        action="append",
        default=[],
        metavar="BAND=COUNT",
        help=(
            "Override one score band's prompt count. May be repeated, e.g. "
            "--band-count 86-100=4 --band-count 11-20=12."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/pipelines/v2",
    )
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--size", default="1024x1024")
    return parser.parse_args()


def parse_band_counts(values: list[str]) -> dict[str, int] | None:
    if not values:
        return None
    band_counts: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Band count override must use BAND=COUNT format: {value!r}")
        band, count_text = value.split("=", 1)
        band_counts[band] = int(count_text)
    return band_counts


def main() -> None:
    args = parse_args()
    artifacts = build_v2_pipeline_artifacts(
        V2PipelineConfig(
            run_name=args.run_name,
            count_per_band=args.count_per_band,
            band_counts=parse_band_counts(args.band_count),
            seed=args.seed,
            output_dir=args.output_dir,
            image_spec=ImageGenerationSpec(
                model=args.model,
                quality=args.quality,
                size=args.size,
            ),
        )
    )
    print("Prepared v2 image pipeline artifacts.")
    print(f"Prompts: {artifacts.prompt_count}")
    print(f"Run dir: {artifacts.run_dir}")
    print(f"Prompt manifest: {artifacts.prompt_manifest_path}")
    print(f"Batch JSONL: {artifacts.input_jsonl_path}")
    print(f"Batch metadata: {artifacts.metadata_path}")
    print(f"QC rubric: {artifacts.qc_rubric_path}")
    print()
    print("No API call was made. Review the manifest and QC rubric before submitting any batch.")


if __name__ == "__main__":
    main()
