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


DEFAULT_MANIFEST = REPO_ROOT / "dataset/synthetic/manifest.csv"
RUNS_ROOT = REPO_ROOT / "outputs/synthetic_regressor"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Dream2Detect synthetic representative-score regressor."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam"],
        default="adamw",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "reduce_on_plateau"],
        default="reduce_on_plateau",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-profile",
        choices=["mild", "strong", "damage_safe"],
        default="damage_safe",
    )
    parser.add_argument(
        "--model-variant",
        choices=[
            "simple_cnn_regressor",
            "residual_cnn_regressor",
            "residual_cnn_groupnorm_regressor",
        ],
        default="residual_cnn_groupnorm_regressor",
    )
    parser.add_argument(
        "--target-mode",
        choices=["fine_normalized"],
        default="fine_normalized",
    )
    parser.add_argument("--overfit-subset-size", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=["stratified_random", "metadata_family_holdout"],
        default="metadata_family_holdout",
    )
    parser.add_argument("--split-group-column", type=str, default=None)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=1)
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
    print(f"Optimizer: {args.optimizer}")
    print(f"LR scheduler: {args.lr_scheduler}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Augmentation: {not args.no_augmentation}")
    print(f"Augmentation profile: {args.augmentation_profile}")
    print(f"Model variant: {args.model_variant}")
    print(f"Target mode: {args.target_mode}")
    print(f"Split strategy: {args.split_strategy}")
    if args.resume_from is not None:
        print(f"Resume from: {args.resume_from.resolve()}")
    print(f"Output directory: {output_dir}")
    print()

    result = train_synthetic_regressor(
        manifest_path=manifest_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        lr_scheduler_name=args.lr_scheduler,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        dropout_p=args.dropout,
        random_seed=args.random_seed,
        output_dir=output_dir,
        resume_from=args.resume_from,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_augmentation=not args.no_augmentation,
        augmentation_profile=args.augmentation_profile,
        model_variant=args.model_variant,
        target_mode=args.target_mode,
        overfit_subset_size=args.overfit_subset_size,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_strategy=args.split_strategy,
        split_group_column=args.split_group_column,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
    )

    print("\nRegression run completed.")
    print(f"Best validation epoch: {result.best_val_epoch}")
    if result.early_stopped:
        print(f"Early stopped at epoch: {result.stopped_epoch}")
    print(f"Device: {result.device}")
    print(f"Final test MAE: {result.test_metrics.mae:.3f}")
    print(f"Final test 10-band accuracy: {result.test_metrics.accuracy:.4f}")
    print(f"Final test 10-band macro F1: {result.test_metrics.macro_f1:.4f}")
    print(f"Final test mean band error: {result.test_metrics.mean_band_error:.4f}")
    print(f"Final test +-1 band accuracy: {result.test_metrics.within_one_band_accuracy:.4f}")
    print(f"Final test collapsed coarse macro F1: {result.test_metrics.bucket_macro_f1:.4f}")
    print(f"Artifacts written to: {output_dir}")
    print(f"Epoch metrics CSV: {output_dir / 'epoch_metrics.csv'}")
    print(f"Training curves: {output_dir / 'training_curves.png'}")
    print(f"Latest checkpoint: {output_dir / 'checkpoints/latest.pt'}")


if __name__ == "__main__":
    main()
