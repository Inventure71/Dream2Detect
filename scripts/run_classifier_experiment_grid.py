from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.training.train_classifier import train_synthetic_classifier


T = TypeVar("T")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_variant: str
    image_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    use_balanced_sampler: bool
    use_augmentation: bool
    augmentation_profile: str
    ordinal_loss_weight: float
    optimizer_name: str
    lr_scheduler_name: str
    random_seed: int


def parse_csv_values(raw: str, cast: Callable[[str], T]) -> list[T]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return [cast(value) for value in values]


def slug_float(value: float) -> str:
    return str(value).replace("-", "neg").replace(".", "p")


def build_experiment_configs(
    *,
    image_sizes: list[int],
    learning_rates: list[float],
    weight_decays: list[float],
    dropouts: list[float],
    balanced_sampler_options: list[bool],
    use_augmentation_options: list[bool],
    augmentation_profiles: list[str],
    model_variants: list[str],
    ordinal_loss_weights: list[float],
    optimizer_names: list[str],
    lr_scheduler_names: list[str],
    random_seeds: list[int],
) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    for image_size in image_sizes:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                for dropout in dropouts:
                    for use_balanced_sampler in balanced_sampler_options:
                        for use_augmentation in use_augmentation_options:
                            for augmentation_profile in augmentation_profiles:
                                if not use_augmentation and augmentation_profile != "mild":
                                    continue
                                for model_variant in model_variants:
                                    for ordinal_loss_weight in ordinal_loss_weights:
                                        for optimizer_name in optimizer_names:
                                            for lr_scheduler_name in lr_scheduler_names:
                                                for random_seed in random_seeds:
                                                    sampler_slug = (
                                                        "balanced"
                                                        if use_balanced_sampler
                                                        else "unbalanced"
                                                    )
                                                    augmentation_slug = (
                                                        augmentation_profile
                                                        if use_augmentation
                                                        else "noaug"
                                                    )
                                                    scheduler_slug = (
                                                        "sched"
                                                        if lr_scheduler_name
                                                        == "reduce_on_plateau"
                                                        else "nosched"
                                                    )
                                                    name = (
                                                        f"img{image_size}_"
                                                        f"{model_variant}_"
                                                        f"ord{slug_float(ordinal_loss_weight)}_"
                                                        f"lr{slug_float(learning_rate)}_"
                                                        f"wd{slug_float(weight_decay)}_"
                                                        f"do{slug_float(dropout)}_"
                                                        f"{optimizer_name}_"
                                                        f"{scheduler_slug}_"
                                                        f"{sampler_slug}_"
                                                        f"{augmentation_slug}_"
                                                        f"seed{random_seed}"
                                                    )
                                                    configs.append(
                                                        ExperimentConfig(
                                                            name=name,
                                                            model_variant=model_variant,
                                                            image_size=image_size,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            dropout=dropout,
                                                            use_balanced_sampler=(
                                                                use_balanced_sampler
                                                            ),
                                                            use_augmentation=use_augmentation,
                                                            augmentation_profile=(
                                                                augmentation_profile
                                                            ),
                                                            ordinal_loss_weight=(
                                                                ordinal_loss_weight
                                                            ),
                                                            optimizer_name=optimizer_name,
                                                            lr_scheduler_name=(
                                                                lr_scheduler_name
                                                            ),
                                                            random_seed=random_seed,
                                                        )
                                                    )
    return configs


def parse_bool_options(raw: str) -> list[bool]:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "on": True,
        "false": False,
        "0": False,
        "no": False,
        "off": False,
    }
    options: list[bool] = []
    for value in parse_csv_values(raw, str):
        key = value.lower()
        if key not in mapping:
            raise ValueError(f"Invalid boolean grid value: {value!r}")
        options.append(mapping[key])
    return options


def resolve_manifest(
    *,
    manifest: Path | None,
    manifest_template: str | None,
    image_size: int,
) -> Path:
    if manifest_template is not None:
        return Path(manifest_template.format(image_size=image_size)).resolve()
    if manifest is None:
        raise ValueError("Either --manifest or --manifest-template is required.")
    return manifest.resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small grid of Dream2Detect classifier experiments."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--manifest-template",
        default=None,
        help="Format string with {image_size}, for example data/datasets/name_{image_size}.csv.",
    )
    parser.add_argument("--image-sizes", default="224")
    parser.add_argument("--learning-rates", default="0.0003")
    parser.add_argument("--weight-decays", default="0.0001")
    parser.add_argument("--dropouts", default="0.1")
    parser.add_argument("--optimizers", default="adamw")
    parser.add_argument("--lr-schedulers", default="reduce_on_plateau")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--balanced-samplers", default="true")
    parser.add_argument("--augmentations", default="true")
    parser.add_argument("--augmentation-profiles", default="mild")
    parser.add_argument("--model-variants", default="simple_cnn")
    parser.add_argument("--ordinal-loss-weights", default="0.0")
    parser.add_argument("--num-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-seeds", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--split-strategy",
        choices=["stratified_random", "metadata_family_holdout"],
        default="stratified_random",
    )
    parser.add_argument("--split-group-column", default=None)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=1)
    parser.add_argument("--plot-every-n-epochs", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for this grid run. Defaults under data/training_runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    configs = build_experiment_configs(
        image_sizes=parse_csv_values(args.image_sizes, int),
        learning_rates=parse_csv_values(args.learning_rates, float),
        weight_decays=parse_csv_values(args.weight_decays, float),
        dropouts=parse_csv_values(args.dropouts, float),
        balanced_sampler_options=parse_bool_options(args.balanced_samplers),
        use_augmentation_options=parse_bool_options(args.augmentations),
        augmentation_profiles=parse_csv_values(args.augmentation_profiles, str),
        model_variants=parse_csv_values(args.model_variants, str),
        ordinal_loss_weights=parse_csv_values(args.ordinal_loss_weights, float),
        optimizer_names=parse_csv_values(args.optimizers, str),
        lr_scheduler_names=parse_csv_values(args.lr_schedulers, str),
        random_seeds=(
            parse_csv_values(args.random_seeds, int)
            if args.random_seeds is not None
            else [args.random_seed]
        ),
    )
    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (
            REPO_ROOT
            / "data/training_runs/synthetic_classifier_grid"
            / f"grid_{timestamp}"
        ).resolve()
    )

    print(f"Experiment count: {len(configs)}")
    print(f"Output root: {output_root}")

    summary_rows: list[dict[str, object]] = []
    for config in configs:
        manifest_path = resolve_manifest(
            manifest=args.manifest,
            manifest_template=args.manifest_template,
            image_size=config.image_size,
        )
        output_dir = output_root / config.name

        print()
        print(f"=== {config.name} ===")
        print(f"Manifest: {manifest_path}")
        print(f"Output: {output_dir}")

        if args.dry_run:
            continue

        result = train_synthetic_classifier(
            manifest_path=manifest_path,
            image_size=config.image_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer_name,
            lr_scheduler_name=config.lr_scheduler_name,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            min_learning_rate=args.min_learning_rate,
            weight_decay=config.weight_decay,
            dropout_p=config.dropout,
            random_seed=config.random_seed,
            output_dir=output_dir,
            use_balanced_sampler=config.use_balanced_sampler,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            use_augmentation=config.use_augmentation,
            augmentation_profile=config.augmentation_profile,
            model_variant=config.model_variant,
            ordinal_loss_weight=config.ordinal_loss_weight,
            split_strategy=args.split_strategy,
            split_group_column=args.split_group_column,
            checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
            plot_every_n_epochs=args.plot_every_n_epochs,
        )
        summary_rows.append(
            {
                "name": config.name,
                "manifest_path": str(manifest_path),
                "output_dir": str(output_dir),
                "image_size": config.image_size,
                "model_variant": config.model_variant,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "dropout": config.dropout,
                "optimizer_name": config.optimizer_name,
                "lr_scheduler_name": config.lr_scheduler_name,
                "use_balanced_sampler": config.use_balanced_sampler,
                "use_augmentation": config.use_augmentation,
                "augmentation_profile": config.augmentation_profile,
                "ordinal_loss_weight": config.ordinal_loss_weight,
                "random_seed": config.random_seed,
                "split_strategy": args.split_strategy,
                "split_group_column": args.split_group_column,
                "best_val_epoch": result.best_val_epoch,
                "early_stopped": result.early_stopped,
                "stopped_epoch": result.stopped_epoch,
                "test_accuracy": result.test_metrics.accuracy,
                "test_macro_f1": result.test_metrics.macro_f1,
            }
        )

    if summary_rows:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "grid_summary.csv"
        with summary_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print()
        print(f"Grid summary written to: {summary_path}")


if __name__ == "__main__":
    main()
