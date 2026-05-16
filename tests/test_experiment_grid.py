from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_classifier_experiment_grid import (
    build_experiment_configs,
    build_summary_row,
    parse_csv_values,
    resolve_balanced_sampler_value,
    sort_summary_rows,
)


class ExperimentGridTests(unittest.TestCase):
    def test_parse_csv_values_casts_values(self) -> None:
        self.assertEqual(parse_csv_values("0.001,0.0003", float), [0.001, 0.0003])
        self.assertEqual(parse_csv_values("128,224", int), [128, 224])

    def test_build_experiment_configs_expands_grid_with_stable_names(self) -> None:
        configs = build_experiment_configs(
            image_sizes=[128],
            learning_rates=[0.001, 0.0003],
            weight_decays=[0.0],
            dropouts=[0.1, 0.3],
            balanced_sampler_options=[True, False],
            use_augmentation_options=[True],
            augmentation_profiles=["mild"],
            model_variants=["simple_cnn"],
            ordinal_loss_weights=[0.0],
            score_band_soft_label_sigmas=[1.0],
            score_band_emd_weights=[0.0],
            score_band_effective_betas=[0.999],
            optimizer_names=["adamw"],
            lr_scheduler_names=["reduce_on_plateau"],
            random_seeds=[42],
        )

        self.assertEqual(len(configs), 8)
        self.assertEqual(
            configs[0].name,
            "img128_simple_cnn_ord0p0_sig1p0_emd0p0_beta0p999_lr0p001_wd0p0_do0p1_adamw_sched_balanced_mild_seed42",
        )
        self.assertFalse(configs[-1].use_balanced_sampler)
        self.assertEqual(configs[0].score_band_soft_label_sigma, 1.0)
        self.assertEqual(configs[0].score_band_emd_weight, 0.0)
        self.assertEqual(configs[0].score_band_effective_beta, 0.999)

    def test_build_experiment_configs_expands_random_seeds(self) -> None:
        configs = build_experiment_configs(
            image_sizes=[128],
            learning_rates=[0.001],
            weight_decays=[0.0],
            dropouts=[0.1],
            balanced_sampler_options=[True],
            use_augmentation_options=[True],
            augmentation_profiles=["mild"],
            model_variants=["simple_cnn"],
            ordinal_loss_weights=[0.0],
            score_band_soft_label_sigmas=[1.0],
            score_band_emd_weights=[0.0],
            score_band_effective_betas=[0.999],
            optimizer_names=["adamw"],
            lr_scheduler_names=["reduce_on_plateau"],
            random_seeds=[1, 2, 3],
        )

        self.assertEqual([config.random_seed for config in configs], [1, 2, 3])

    def test_build_experiment_configs_expands_sigma_and_beta(self) -> None:
        configs = build_experiment_configs(
            image_sizes=[128],
            learning_rates=[0.001],
            weight_decays=[0.0],
            dropouts=[0.1],
            balanced_sampler_options=[False],
            use_augmentation_options=[True],
            augmentation_profiles=["mild"],
            model_variants=["simple_cnn"],
            ordinal_loss_weights=[0.0],
            score_band_soft_label_sigmas=[0.75, 1.0],
            score_band_emd_weights=[0.0],
            score_band_effective_betas=[0.995, 0.999],
            optimizer_names=["adamw"],
            lr_scheduler_names=["reduce_on_plateau"],
            random_seeds=[42],
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            {(config.score_band_soft_label_sigma, config.score_band_effective_beta) for config in configs},
            {(0.75, 0.995), (0.75, 0.999), (1.0, 0.995), (1.0, 0.999)},
        )

    def test_resolve_balanced_sampler_value_uses_auto_score_band_default(self) -> None:
        self.assertIsNone(
            resolve_balanced_sampler_value(
                target_label_mode="score_band",
                use_balanced_sampler=True,
                balanced_sampler_mode="auto",
            )
        )
        self.assertTrue(
            resolve_balanced_sampler_value(
                target_label_mode="coarse",
                use_balanced_sampler=True,
                balanced_sampler_mode="auto",
            )
        )

    def test_build_summary_row_includes_score_band_metrics(self) -> None:
        config = build_experiment_configs(
            image_sizes=[384],
            learning_rates=[0.0003],
            weight_decays=[0.0001],
            dropouts=[0.1],
            balanced_sampler_options=[False],
            use_augmentation_options=[True],
            augmentation_profiles=["damage_safe"],
            model_variants=["residual_cnn"],
            ordinal_loss_weights=[0.2],
            score_band_soft_label_sigmas=[1.0],
            score_band_emd_weights=[0.5],
            score_band_effective_betas=[0.999],
            optimizer_names=["adamw"],
            lr_scheduler_names=["reduce_on_plateau"],
            random_seeds=[42],
        )[0]
        result = SimpleNamespace(
            selection_metric_name="val_mean_band_error",
            best_val_metric=1.75,
            best_val_epoch=12,
            early_stopped=False,
            stopped_epoch=None,
            test_metrics=SimpleNamespace(
                accuracy=0.3,
                macro_f1=0.25,
                mean_band_error=1.2,
                within_one_band_accuracy=0.8,
                severe_band_error_rate=0.2,
                collapsed_coarse_accuracy=0.55,
                collapsed_coarse_macro_f1=0.5,
            ),
        )

        row = build_summary_row(
            config=config,
            manifest_path=Path("/tmp/manifest.csv"),
            output_dir=Path("/tmp/run"),
            result=result,
            target_label_mode="score_band",
            split_strategy="metadata_family_holdout",
            split_group_column=None,
        )

        self.assertEqual(row["selection_metric_name"], "val_mean_band_error")
        self.assertEqual(row["best_val_metric"], 1.75)
        self.assertEqual(row["test_mean_band_error"], 1.2)
        self.assertEqual(row["test_collapsed_coarse_macro_f1"], 0.5)
        self.assertEqual(row["score_band_emd_weight"], 0.5)

    def test_sort_summary_rows_uses_ascending_for_mean_band_error(self) -> None:
        rows = [
            {"selection_metric_name": "val_mean_band_error", "best_val_metric": 2.0},
            {"selection_metric_name": "val_mean_band_error", "best_val_metric": 1.0},
        ]

        ranked = sort_summary_rows(rows)

        self.assertEqual(ranked[0]["best_val_metric"], 1.0)


if __name__ == "__main__":
    unittest.main()
