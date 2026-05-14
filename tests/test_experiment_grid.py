from __future__ import annotations

import unittest

from scripts.run_classifier_experiment_grid import build_experiment_configs, parse_csv_values


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
            optimizer_names=["adamw"],
            lr_scheduler_names=["reduce_on_plateau"],
            random_seeds=[42],
        )

        self.assertEqual(len(configs), 8)
        self.assertEqual(
            configs[0].name,
            "img128_simple_cnn_ord0p0_lr0p001_wd0p0_do0p1_adamw_sched_balanced_mild_seed42",
        )
        self.assertFalse(configs[-1].use_balanced_sampler)

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
            optimizer_names=["adamw"],
            lr_scheduler_names=["reduce_on_plateau"],
            random_seeds=[1, 2, 3],
        )

        self.assertEqual([config.random_seed for config in configs], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
