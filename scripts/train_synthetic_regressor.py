from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.training.train_regressor import train_synthetic_regressor


DEFAULT_MANIFEST = (
    REPO_ROOT / "data/datasets/synthetic_combined_phase1_plus_scaleup_200_processed_224.csv"
)
RUNS_ROOT = REPO_ROOT / "data/training_runs/synthetic_regressor"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Dream2Detect synthetic representative-score regressor."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-profile",
        choices=["mild", "strong", "damage_safe"],
        default="mild",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            RUNS_ROOT
            / f"{manifest_path.stem}_{args.image_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ).resolve()
    )

    print("Starting Dream2Detect synthetic representative-score regression")
    print(f"Manifest: {manifest_path}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Augmentation: {not args.no_augmentation}")
    print(f"Output directory: {output_dir}")
    print()

    result = train_synthetic_regressor(
        manifest_path=manifest_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_p=args.dropout,
        random_seed=args.random_seed,
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_augmentation=not args.no_augmentation,
        augmentation_profile=args.augmentation_profile,
    )

    print("\nRegression run completed.")
    print(f"Best validation epoch: {result.best_val_epoch}")
    if result.early_stopped:
        print(f"Early stopped at epoch: {result.stopped_epoch}")
    print(f"Device: {result.device}")
    print(f"Final test MAE: {result.test_metrics.mae:.3f}")
    print(f"Final test bucket accuracy: {result.test_metrics.bucket_accuracy:.4f}")
    print(f"Final test bucket macro F1: {result.test_metrics.bucket_macro_f1:.4f}")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
