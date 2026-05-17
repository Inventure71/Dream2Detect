from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.training.train_classifier import train_synthetic_classifier


DEFAULT_RAW_MANIFEST = REPO_ROOT / "dataset/synthetic/manifest.csv"
RUNS_ROOT = REPO_ROOT / "outputs/synthetic_classifier"


def choose_default_manifest(image_size: int) -> Path:
    return DEFAULT_RAW_MANIFEST


def build_default_output_dir(manifest_path: Path, image_size: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_ROOT / f"{manifest_path.stem}_{image_size}_{timestamp}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Dream2Detect synthetic classifier."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RAW_MANIFEST,
        help="Path to the training manifest CSV.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for training.",
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
        default=120,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam"],
        default="adamw",
        help="Optimizer. AdamW is the default because it decouples weight decay.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay for regularization.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability in the classifier head.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "reduce_on_plateau"],
        default="reduce_on_plateau",
        help="Adaptive learning-rate strategy.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Factor for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=10,
        help="Validation selection-metric plateau patience before reducing LR.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
        help="Lower bound for adaptive learning-rate scheduling.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without validation macro-F1 improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation macro-F1 improvement required to reset patience.",
    )
    parser.add_argument(
        "--overfit-subset-size",
        type=int,
        default=None,
        help="Debug mode: train/validate/test on the same small balanced subset.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility and training setup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for checkpoints, metrics, and split files.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from, usually checkpoints/latest.pt.",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize model weights only for fine-tuning.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="Fraction of rows used for training.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of rows used for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of rows held out for testing.",
    )
    parser.add_argument(
        "--balanced-sampler-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Training sampler policy. auto keeps the old coarse default but disables the sampler for score-band runs.",
    )
    parser.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Deprecated alias for --balanced-sampler-mode off.",
    )
    parser.add_argument(
        "--score-band-soft-label-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for soft ordinal score-band targets.",
    )
    parser.add_argument(
        "--score-band-emd-weight",
        type=float,
        default=0.0,
        help="Weight for the cumulative squared EMD distance term on score-band training.",
    )
    parser.add_argument(
        "--score-band-class-weight-strategy",
        choices=["balanced", "effective"],
        default="effective",
        help="Score-band class weighting strategy when training without the balanced sampler.",
    )
    parser.add_argument(
        "--score-band-effective-beta",
        type=float,
        default=0.999,
        help="Effective-number beta used when score-band class weighting is set to effective.",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable random training augmentation.",
    )
    parser.add_argument(
        "--augmentation-profile",
        choices=[
            "mild",
            "strong",
            "damage_safe",
            "damage_safe_ra_low",
            "damage_safe_ra_medium",
            "damage_safe_ra_high",
        ],
        default="mild",
        help="Training augmentation strength when augmentation is enabled.",
    )
    parser.add_argument(
        "--model-variant",
        choices=["simple_cnn", "residual_cnn", "residual_cnn_groupnorm", "resnet18"],
        default="simple_cnn",
        help="Classifier architecture to train.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights when the selected model supports them.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone parameters when using a pretrained model.",
    )
    parser.add_argument(
        "--ordinal-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional smooth ordinal penalty on expected class index. "
            "0 disables ordinal-aware loss."
        ),
    )
    parser.add_argument(
        "--target-label-mode",
        choices=["coarse", "coarse_ordinal", "score_band"],
        default="coarse",
        help="Classifier target: four coarse classes or ten official score bands.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["stratified_random", "metadata_family_holdout"],
        default="stratified_random",
        help="Dataset split policy. metadata_family_holdout keeps derived prompt families disjoint across splits.",
    )
    parser.add_argument(
        "--split-group-column",
        default=None,
        help="Optional manifest column used as the group id for metadata_family_holdout.",
    )
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=1,
        help="Write non-best epoch checkpoints at this interval.",
    )
    parser.add_argument(
        "--plot-every-n-epochs",
        type=int,
        default=1,
        help="Redraw training plots at this interval. 0 means final plots only.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else choose_default_manifest(args.image_size).resolve()
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.resume_from is not None:
        output_dir = args.resume_from.resolve().parents[1]
    else:
        output_dir = build_default_output_dir(manifest_path, args.image_size).resolve()

    print("Starting Dream2Detect synthetic classifier training")
    print(f"Manifest: {manifest_path}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"LR scheduler: {args.lr_scheduler}")
    if args.lr_scheduler != "none":
        print(
            "LR scheduler settings: "
            f"factor={args.lr_scheduler_factor}, "
            f"patience={args.lr_scheduler_patience}, "
            f"min_lr={args.min_learning_rate}"
        )
    print(f"Random seed: {args.random_seed}")
    print(
        "Split fractions: "
        f"train={args.train_fraction}, "
        f"val={args.val_fraction}, "
        f"test={args.test_fraction}"
    )
    print(f"Split strategy: {args.split_strategy}")
    if args.split_group_column is not None:
        print(f"Split group column: {args.split_group_column}")
    if args.no_balanced_sampler and args.balanced_sampler_mode != "auto":
        raise ValueError(
            "Use either --no-balanced-sampler or --balanced-sampler-mode, not both."
        )
    if args.no_balanced_sampler:
        balanced_sampler_mode = "off"
    else:
        balanced_sampler_mode = args.balanced_sampler_mode
    print(f"Balanced sampler mode: {balanced_sampler_mode}")
    print(f"Augmentation: {not args.no_augmentation}")
    if not args.no_augmentation:
        print(f"Augmentation profile: {args.augmentation_profile}")
    print(f"Model variant: {args.model_variant}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Ordinal loss weight: {args.ordinal_loss_weight}")
    print(f"Target label mode: {args.target_label_mode}")
    if args.target_label_mode == "score_band":
        print(f"Score-band soft-label sigma: {args.score_band_soft_label_sigma}")
        print(f"Score-band EMD weight: {args.score_band_emd_weight}")
        print(
            "Score-band class weighting: "
            f"{args.score_band_class_weight_strategy} "
            f"(beta={args.score_band_effective_beta})"
        )
    print(f"Checkpoint every N epochs: {args.checkpoint_every_n_epochs}")
    print(f"Plot every N epochs: {args.plot_every_n_epochs}")
    if args.early_stopping_patience is not None:
        print(
            "Early stopping: "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta}"
        )
    if args.overfit_subset_size is not None:
        print(f"Overfit subset size: {args.overfit_subset_size}")
    print(f"Output directory: {output_dir}")
    if args.resume_from is not None:
        print(f"Resume checkpoint: {args.resume_from.resolve()}")
    if args.init_from_checkpoint is not None:
        print(f"Initial model checkpoint: {args.init_from_checkpoint.resolve()}")
    print()

    result = train_synthetic_classifier(
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
        resume_from=args.resume_from.resolve() if args.resume_from is not None else None,
        use_balanced_sampler=(
            None
            if balanced_sampler_mode == "auto"
            else balanced_sampler_mode == "on"
        ),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        overfit_subset_size=args.overfit_subset_size,
        use_augmentation=not args.no_augmentation,
        augmentation_profile=args.augmentation_profile,
        model_variant=args.model_variant,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        ordinal_loss_weight=args.ordinal_loss_weight,
        score_band_soft_label_sigma=args.score_band_soft_label_sigma,
        score_band_emd_weight=args.score_band_emd_weight,
        score_band_class_weight_strategy=args.score_band_class_weight_strategy,
        score_band_effective_beta=args.score_band_effective_beta,
        target_label_mode=args.target_label_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_strategy=args.split_strategy,
        split_group_column=args.split_group_column,
        init_from_checkpoint=(
            args.init_from_checkpoint.resolve()
            if args.init_from_checkpoint is not None
            else None
        ),
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        plot_every_n_epochs=args.plot_every_n_epochs,
    )

    print("\nTraining run completed.")
    print(f"Best validation epoch: {result.best_val_epoch}")
    if result.early_stopped:
        print(f"Early stopped at epoch: {result.stopped_epoch}")
    print(f"Device: {result.device}")
    print(f"Used resize in transforms: {result.used_resize_in_transforms}")
    print(f"Final test accuracy: {result.test_metrics.accuracy:.4f}")
    print(f"Final test macro F1: {result.test_metrics.macro_f1:.4f}")
    print(f"Selection metric: {result.selection_metric_name}={result.best_val_metric:.4f}")
    if args.target_label_mode == "score_band":
        print(f"Final test mean band error: {result.test_metrics.mean_band_error:.4f}")
        print(
            "Final test +/-1 band accuracy: "
            f"{result.test_metrics.within_one_band_accuracy:.4f}"
        )
        print(
            "Collapsed 4-class test macro F1: "
            f"{result.test_metrics.collapsed_coarse_macro_f1:.4f}"
        )
    print(f"Artifacts written to: {output_dir}")
    print(f"Epoch metrics CSV: {output_dir / 'epoch_metrics.csv'}")
    print(f"Training curves: {output_dir / 'training_curves.png'}")
    print(f"Class monitoring: {output_dir / 'class_monitoring.png'}")
    print(f"Latest checkpoint: {output_dir / 'checkpoints/latest.pt'}")


if __name__ == "__main__":
    main()
