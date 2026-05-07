from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.training.train_classifier import train_synthetic_classifier


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the first Dream2Detect synthetic coarse classifier."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/datasets/synthetic_starting_dataset_phase1_round1.csv",
        help="Path to the training manifest CSV.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square image size used for resizing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for train/validation/test dataloaders.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility and training setup.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    print("Starting Dream2Detect synthetic coarse-classifier training")
    print(f"Manifest: {manifest_path}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Random seed: {args.random_seed}")
    print()

    result = train_synthetic_classifier(
        manifest_path=manifest_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )

    print("\nTraining run completed.")
    print(f"Best validation epoch: {result.best_val_epoch}")
    print(f"Final test accuracy: {result.test_metrics.accuracy:.4f}")
    print(f"Final test macro F1: {result.test_metrics.macro_f1:.4f}")


if __name__ == "__main__":
    main()
