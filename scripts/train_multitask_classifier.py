from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.training.train_multitask_classifier import train_multitask_classifier


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Dream2Detect multitask coarse + score-band classifier."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "reduce_on_plateau"],
        default="reduce_on_plateau",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=8)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--augmentation-profile",
        choices=["mild", "strong", "damage_safe"],
        default="damage_safe",
    )
    parser.add_argument("--auxiliary-band-loss-weight", type=float, default=0.3)
    parser.add_argument("--band-ordinal-loss-weight", type=float, default=0.2)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=10)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    result = train_multitask_classifier(
        manifest_path=args.manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        lr_scheduler_name=args.lr_scheduler,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        min_learning_rate=args.min_learning_rate,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        use_balanced_sampler=not args.no_balanced_sampler,
        weight_decay=args.weight_decay,
        dropout_p=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_augmentation=not args.no_augmentation,
        augmentation_profile=args.augmentation_profile,
        auxiliary_band_loss_weight=args.auxiliary_band_loss_weight,
        band_ordinal_loss_weight=args.band_ordinal_loss_weight,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
    )

    print("\nMultitask training completed.")
    print(f"Best validation epoch: {result.best_val_epoch}")
    print(f"Device: {result.device}")
    print(f"Final coarse accuracy: {result.test_metrics.coarse_metrics.accuracy:.4f}")
    print(f"Final coarse macro F1: {result.test_metrics.coarse_metrics.macro_f1:.4f}")
    print(f"Final band accuracy: {result.test_metrics.score_band_metrics.accuracy:.4f}")
    print(f"Final band macro F1: {result.test_metrics.score_band_metrics.macro_f1:.4f}")
    print(f"Artifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
