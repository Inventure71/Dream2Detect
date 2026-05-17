from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from dream2detect.training.dataset import (
    SCORE_BAND_NAMES,
    DamageSafeRandAugment,
    SyntheticManifestDataset,
    build_train_transform,
)
from dream2detect.training.models import (
    ResidualCNNClassifier,
    ResidualGroupNormCNNClassifier,
    ResidualGroupNormCNNRegressor,
    SimpleCNNClassifier,
)
from dream2detect.training.models import ResidualCNNMultiTaskClassifier
from dream2detect.training.metrics import (
    band_indices_to_coarse_indices,
    collapse_score_band_probabilities_to_coarse,
    compute_scalar_score_band_metrics,
    scores_to_band_indices,
    summarize_ordinal_errors,
)
from dream2detect.training.splits import build_stratified_splits, derive_metadata_family_groups
from dream2detect.training.train_classifier import (
    build_classification_loss,
    build_cumulative_distribution,
    build_effective_number_class_weights,
    build_ordinal_classification_loss,
    compute_squared_emd_per_example,
    build_score_band_soft_label_loss,
    build_soft_ordinal_targets,
    build_overfit_indices,
    decode_cumulative_ordinal_logits,
    extract_epoch_selection_metric,
    EpochResult,
    validate_split_fractions,
)


class TrainingControlTests(unittest.TestCase):
    def test_model_uses_configurable_dropout_probability(self) -> None:
        model = SimpleCNNClassifier(num_classes=4, dropout_p=0.15)

        dropout_layers = [
            module for module in model.modules() if isinstance(module, torch.nn.Dropout)
        ]

        self.assertEqual(len(dropout_layers), 1)
        self.assertAlmostEqual(dropout_layers[0].p, 0.15)

    def test_train_transform_can_disable_random_augmentation(self) -> None:
        transform = build_train_transform(include_resize=False, use_augmentation=False)

        self.assertFalse(
            any(isinstance(step, transforms.RandomHorizontalFlip) for step in transform.transforms)
        )
        self.assertFalse(
            any(isinstance(step, transforms.ColorJitter) for step in transform.transforms)
        )
        self.assertTrue(
            any(isinstance(step, transforms.Normalize) for step in transform.transforms)
        )

    def test_damage_safe_transform_contains_label_safe_augmentations(self) -> None:
        transform = build_train_transform(
            include_resize=False,
            use_augmentation=True,
            augmentation_profile="damage_safe",
        )

        self.assertTrue(
            any(isinstance(step, transforms.RandomHorizontalFlip) for step in transform.transforms)
        )
        self.assertTrue(
            any(isinstance(step, transforms.RandomAffine) for step in transform.transforms)
        )
        self.assertTrue(
            any(isinstance(step, transforms.RandomPerspective) for step in transform.transforms)
        )
        self.assertTrue(
            any(isinstance(step, transforms.RandomErasing) for step in transform.transforms)
        )
        self.assertFalse(
            any(isinstance(step, transforms.RandomVerticalFlip) for step in transform.transforms)
        )

    def test_randaugment_damage_safe_profile_uses_constrained_operator(self) -> None:
        transform = build_train_transform(
            include_resize=False,
            use_augmentation=True,
            augmentation_profile="damage_safe_ra_medium",
        )

        self.assertTrue(
            any(isinstance(step, DamageSafeRandAugment) for step in transform.transforms)
        )
        self.assertFalse(
            any(isinstance(step, transforms.RandomErasing) for step in transform.transforms)
        )

    def test_residual_cnn_is_materially_larger_than_simple_cnn(self) -> None:
        simple = SimpleCNNClassifier(num_classes=4)
        residual = ResidualCNNClassifier(num_classes=4)

        simple_params = sum(parameter.numel() for parameter in simple.parameters())
        residual_params = sum(parameter.numel() for parameter in residual.parameters())

        self.assertLess(simple_params, 300_000)
        self.assertGreater(residual_params, 1_000_000)
        self.assertLess(residual_params, 4_000_000)

    def test_groupnorm_residual_model_uses_groupnorm_layers(self) -> None:
        model = ResidualGroupNormCNNClassifier(num_classes=4)

        self.assertTrue(
            any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
        )
        self.assertFalse(
            any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
        )

    def test_groupnorm_residual_regressor_outputs_bounded_scalar(self) -> None:
        model = ResidualGroupNormCNNRegressor(dropout_p=0.1)

        output = model(torch.zeros(2, 3, 128, 128))

        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))
        self.assertTrue(
            any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
        )
        self.assertFalse(
            any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
        )

    def test_ordinal_loss_adds_penalty_when_enabled(self) -> None:
        logits = torch.tensor([[3.0, 0.5, -1.0, -2.0], [-2.0, -1.0, 0.5, 3.0]])
        targets = torch.tensor([0, 0])
        weights = torch.ones(4)

        base_loss = build_classification_loss(
            class_weights=weights,
            use_balanced_sampler=True,
            ordinal_loss_weight=0.0,
            device=torch.device("cpu"),
        )(logits, targets)
        ordinal_loss = build_classification_loss(
            class_weights=weights,
            use_balanced_sampler=True,
            ordinal_loss_weight=0.5,
            device=torch.device("cpu"),
        )(logits, targets)

        self.assertGreater(float(ordinal_loss), float(base_loss))

    def test_cumulative_ordinal_loss_penalizes_wrong_thresholds(self) -> None:
        logits = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        targets = torch.tensor([3, 0])
        pos_weights = torch.ones(3)

        loss = build_ordinal_classification_loss(
            pos_weights=pos_weights,
            num_classes=4,
        )(logits, targets)

        self.assertGreater(float(loss), 0.0)

    def test_decode_cumulative_ordinal_logits_maps_threshold_counts_to_classes(self) -> None:
        logits = torch.tensor(
            [
                [-2.0, -2.0, -2.0],
                [2.0, -2.0, -2.0],
                [2.0, 2.0, -2.0],
                [2.0, 2.0, 2.0],
            ]
        )

        predictions = decode_cumulative_ordinal_logits(logits)

        self.assertEqual(predictions.tolist(), [0, 1, 2, 3])

    def test_soft_ordinal_targets_peak_at_true_band_and_sum_to_one(self) -> None:
        targets = torch.tensor([0, 5, 9])

        soft_targets = build_soft_ordinal_targets(
            targets,
            num_classes=10,
            sigma=1.0,
        )

        self.assertEqual(tuple(soft_targets.shape), (3, 10))
        self.assertTrue(torch.allclose(soft_targets.sum(dim=1), torch.ones(3)))
        self.assertEqual(torch.argmax(soft_targets[0]).item(), 0)
        self.assertEqual(torch.argmax(soft_targets[1]).item(), 5)
        self.assertEqual(torch.argmax(soft_targets[2]).item(), 9)

    def test_score_band_soft_label_loss_penalizes_far_errors(self) -> None:
        weights = torch.ones(10)
        targets = torch.tensor([5])
        loss_fn = build_score_band_soft_label_loss(
            class_weights=weights,
            use_balanced_sampler=False,
            ordinal_loss_weight=0.0,
            score_band_emd_weight=0.0,
            device=torch.device("cpu"),
            num_classes=10,
            soft_label_sigma=1.0,
        )

        near_logits = torch.tensor(
            [[-4.0, -4.0, -3.0, -2.0, 1.5, 3.0, 1.5, -2.0, -3.0, -4.0]]
        )
        far_logits = torch.tensor(
            [[3.0, 1.5, -1.0, -2.0, -3.0, -4.0, -3.0, -2.0, -1.0, 1.0]]
        )

        self.assertLess(float(loss_fn(near_logits, targets)), float(loss_fn(far_logits, targets)))

    def test_cumulative_distribution_accumulates_across_ordered_classes(self) -> None:
        probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        cumulative = build_cumulative_distribution(probabilities)

        self.assertTrue(
            torch.allclose(cumulative, torch.tensor([[0.1, 0.3, 0.6, 1.0]]))
        )

    def test_squared_emd_is_zero_for_identical_distributions(self) -> None:
        probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        emd = compute_squared_emd_per_example(probabilities, probabilities)

        self.assertTrue(torch.allclose(emd, torch.zeros(1)))

    def test_squared_emd_penalizes_farther_ordinal_distribution_errors_more(self) -> None:
        target = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        near = torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]])
        far = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])

        near_emd = compute_squared_emd_per_example(near, target)
        far_emd = compute_squared_emd_per_example(far, target)

        self.assertLess(float(near_emd.item()), float(far_emd.item()))

    def test_score_band_emd_weight_zero_preserves_existing_loss_path(self) -> None:
        weights = torch.ones(10)
        targets = torch.tensor([4])
        logits = torch.tensor([[-2.0, -1.0, 0.0, 2.0, 3.0, 1.5, 0.0, -1.0, -2.0, -3.0]])

        base_loss = build_score_band_soft_label_loss(
            class_weights=weights,
            use_balanced_sampler=False,
            ordinal_loss_weight=0.2,
            score_band_emd_weight=0.0,
            device=torch.device("cpu"),
            num_classes=10,
            soft_label_sigma=1.0,
        )
        same_loss = build_score_band_soft_label_loss(
            class_weights=weights,
            use_balanced_sampler=False,
            ordinal_loss_weight=0.2,
            score_band_emd_weight=0.0,
            device=torch.device("cpu"),
            num_classes=10,
            soft_label_sigma=1.0,
        )

        self.assertAlmostEqual(float(base_loss(logits, targets)), float(same_loss(logits, targets)))

    def test_effective_number_weights_upweight_rarer_classes(self) -> None:
        frame = pd.DataFrame(
            {
                "training_score_band": ["0-10"] * 5 + ["86-100"],
            }
        )

        weights = build_effective_number_class_weights(
            frame,
            train_indices=list(range(len(frame))),
            device=torch.device("cpu"),
            label_column="training_score_band",
            class_names=("0-10", "86-100"),
            beta=0.9,
        )

        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_score_band_selection_metric_uses_mean_band_error(self) -> None:
        score_band_result = EpochResult(
            loss=1.0,
            accuracy=0.4,
            macro_f1=0.35,
            confusion=[[1]],
            mean_band_error=0.8,
        )
        coarse_result = EpochResult(
            loss=1.0,
            accuracy=0.4,
            macro_f1=0.35,
            confusion=[[1]],
        )

        self.assertEqual(
            extract_epoch_selection_metric(score_band_result, target_label_mode="score_band"),
            0.8,
        )
        self.assertEqual(
            extract_epoch_selection_metric(coarse_result, target_label_mode="coarse"),
            0.35,
        )

    def test_score_band_probabilities_collapse_to_coarse_groups(self) -> None:
        probabilities = torch.tensor(
            [
                [0.6, 0.1, 0.1, 0.0, 0.05, 0.05, 0.0, 0.05, 0.03, 0.02],
                [0.0, 0.05, 0.10, 0.10, 0.10, 0.10, 0.05, 0.15, 0.15, 0.20],
            ]
        )

        collapsed = collapse_score_band_probabilities_to_coarse(probabilities)

        self.assertEqual(tuple(collapsed.shape), (2, 4))
        self.assertTrue(torch.allclose(collapsed.sum(dim=1), torch.ones(2)))
        self.assertEqual(torch.argmax(collapsed[0]).item(), 0)
        self.assertEqual(torch.argmax(collapsed[1]).item(), 3)

    def test_band_indices_map_to_coarse_indices(self) -> None:
        band_indices = torch.tensor([0, 1, 3, 4, 6, 7, 9])

        coarse_indices = band_indices_to_coarse_indices(band_indices)

        self.assertEqual(coarse_indices.tolist(), [0, 1, 1, 2, 2, 3, 3])

    def test_scores_to_band_indices_uses_official_upper_boundaries(self) -> None:
        scores = torch.tensor([-5.0, 5.0, 10.0, 10.01, 20.0, 35.1, 65.0, 85.1, 120.0])

        band_indices = scores_to_band_indices(scores)

        self.assertEqual(band_indices.tolist(), [0, 0, 0, 1, 1, 4, 6, 9, 9])

    def test_scalar_score_band_metrics_report_10_band_and_coarse_results(self) -> None:
        predictions = torch.tensor([0.05, 0.15, 0.55, 0.93])
        targets = torch.tensor([0.05, 0.33, 0.50, 0.80])

        metrics = compute_scalar_score_band_metrics(
            normalized_predictions=predictions,
            normalized_targets=targets,
        )

        self.assertAlmostEqual(metrics.score_mae, 9.0)
        self.assertEqual(metrics.band_metrics.accuracy, 0.5)
        self.assertAlmostEqual(metrics.band_ordinal_errors["mean_band_error"], 0.75)
        self.assertAlmostEqual(metrics.band_ordinal_errors["within_one_band_accuracy"], 0.75)
        self.assertEqual(metrics.coarse_metrics.accuracy, 1.0)

    def test_summarize_ordinal_errors_counts_near_misses(self) -> None:
        predictions = torch.tensor([0, 1, 3, 3])
        targets = torch.tensor([0, 2, 2, 0])

        summary = summarize_ordinal_errors(predictions, targets)

        self.assertEqual(summary["total"], 4)
        self.assertAlmostEqual(float(summary["exact_accuracy"]), 0.25)
        self.assertAlmostEqual(float(summary["within_one_band_accuracy"]), 0.75)
        self.assertAlmostEqual(float(summary["mean_band_error"]), 1.25)

    def test_overfit_indices_are_balanced_when_possible(self) -> None:
        frame = pd.DataFrame(
            {
                "training_coarse_class": [
                    "intact",
                    "intact",
                    "minor",
                    "minor",
                    "moderate",
                    "moderate",
                    "severe",
                    "severe",
                ]
            }
        )

        indices = build_overfit_indices(
            frame,
            subset_size=4,
            random_seed=7,
        )

        subset = frame.iloc[indices]
        self.assertEqual(len(indices), 4)
        self.assertEqual(
            set(subset["training_coarse_class"].tolist()),
            {"intact", "minor", "moderate", "severe"},
        )

    def test_dataset_can_target_official_score_bands(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "example.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            manifest_path = temp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "prompt_id": 1,
                        "image_path": str(image_path),
                        "training_score_band": "46-55",
                        "training_coarse_class": "moderate",
                        "training_representative_score": 50,
                        "training_label_source": "accepted_as_labeled",
                        "qc_status": "accepted_as_labeled",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = SyntheticManifestDataset(
                manifest_path,
                target_mode="score_band",
                transform=transforms.ToTensor(),
            )

            self.assertEqual(SCORE_BAND_NAMES[5], "46-55")
            self.assertEqual(int(dataset[0]["target"]), 5)

    def test_dataset_resolves_manifest_relative_image_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_path = temp_path / "images"
            images_path.mkdir()
            image_path = images_path / "example.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            manifest_path = temp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "prompt_id": 1,
                        "image_path": "images/example.png",
                        "training_score_band": "46-55",
                        "training_coarse_class": "moderate",
                        "training_representative_score": 50,
                        "training_label_source": "accepted_as_labeled",
                        "qc_status": "accepted_as_labeled",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = SyntheticManifestDataset(
                manifest_path,
                target_mode="score_band",
                transform=transforms.ToTensor(),
            )

            metadata = dataset[0]["metadata"]
            self.assertEqual(int(dataset[0]["target"]), 5)
            self.assertEqual(metadata["image_path"], str(image_path))

    def test_dataset_can_target_coarse_ordinal_with_scalar_class_indices(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "example.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            manifest_path = temp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "prompt_id": 1,
                        "image_path": str(image_path),
                        "training_score_band": "31-35",
                        "training_coarse_class": "minor",
                        "training_representative_score": 33,
                        "training_label_source": "accepted_as_labeled",
                        "qc_status": "accepted_as_labeled",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = SyntheticManifestDataset(
                manifest_path,
                target_mode="coarse_ordinal",
                transform=transforms.ToTensor(),
            )

            self.assertEqual(int(dataset[0]["target"]), 1)

    def test_dataset_can_target_normalized_fine_score(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "example.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            manifest_path = temp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "prompt_id": 1,
                        "image_path": str(image_path),
                        "training_score_band": "66-75",
                        "training_coarse_class": "severe",
                        "training_representative_score": 70,
                        "training_label_source": "accepted_as_labeled",
                        "qc_status": "accepted_as_labeled",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = SyntheticManifestDataset(
                manifest_path,
                target_mode="fine_normalized",
                transform=transforms.ToTensor(),
            )

            self.assertAlmostEqual(float(dataset[0]["target"]), 0.70)

    def test_metadata_family_holdout_split_keeps_groups_disjoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.csv"

            rows: list[dict[str, object]] = []
            class_names = ["intact", "minor", "moderate", "severe"]
            for class_name in class_names:
                for family_index in range(5):
                    for example_index in range(2):
                        rows.append(
                            {
                                "training_coarse_class": class_name,
                                "damage_profile_primary": f"{class_name}_damage_{family_index}",
                                "box_form_factor": "mailer" if family_index % 2 == 0 else "box",
                                "background_context": f"context_{family_index}",
                                "image_path": f"/tmp/{class_name}_{family_index}_{example_index}.png",
                            }
                        )
            frame = pd.DataFrame(rows)
            frame.to_csv(manifest_path, index=False)

            split = build_stratified_splits(
                manifest_path,
                random_seed=11,
                split_strategy="metadata_family_holdout",
            )
            family_ids = derive_metadata_family_groups(frame)

            train_groups = set(family_ids.iloc[split.train_indices].tolist())
            val_groups = set(family_ids.iloc[split.val_indices].tolist())
            test_groups = set(family_ids.iloc[split.test_indices].tolist())

            self.assertTrue(train_groups.isdisjoint(val_groups))
            self.assertTrue(train_groups.isdisjoint(test_groups))
            self.assertTrue(val_groups.isdisjoint(test_groups))

            for indices in (split.train_indices, split.val_indices, split.test_indices):
                labels = frame.iloc[indices]["training_coarse_class"].tolist()
                self.assertEqual(set(labels), set(class_names))

    def test_metadata_family_holdout_keeps_cross_band_families_disjoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "manifest.csv"

            bands = [
                ("11-20", "minor"),
                ("21-30", "minor"),
                ("31-35", "minor"),
                ("36-45", "moderate"),
                ("46-55", "moderate"),
                ("56-65", "moderate"),
            ]
            rows: list[dict[str, object]] = []
            for family_index in range(18):
                family_bands = list(bands)
                if family_index % 4 == 0:
                    family_bands = family_bands[:-1]
                if family_index % 5 == 0:
                    family_bands = family_bands[1:]
                for score_band, coarse_class in family_bands:
                    rows.append(
                        {
                            "training_score_band": score_band,
                            "training_coarse_class": coarse_class,
                            "damage_profile_primary": f"shared_damage_{family_index}",
                            "box_form_factor": "medium_standard",
                            "background_context": f"context_{family_index % 3}",
                            "image_path": f"/tmp/{family_index}_{score_band}.png",
                        }
                    )
            frame = pd.DataFrame(rows)
            frame.to_csv(manifest_path, index=False)

            split = build_stratified_splits(
                manifest_path,
                random_seed=17,
                label_column="training_score_band",
                split_strategy="metadata_family_holdout",
            )
            family_ids = derive_metadata_family_groups(frame)

            train_groups = set(family_ids.iloc[split.train_indices].tolist())
            val_groups = set(family_ids.iloc[split.val_indices].tolist())
            test_groups = set(family_ids.iloc[split.test_indices].tolist())

            self.assertTrue(train_groups.isdisjoint(val_groups))
            self.assertTrue(train_groups.isdisjoint(test_groups))
            self.assertTrue(val_groups.isdisjoint(test_groups))

    def test_dataset_can_return_multitask_targets(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "example.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            manifest_path = temp_path / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "prompt_id": 1,
                        "image_path": str(image_path),
                        "training_score_band": "66-75",
                        "training_coarse_class": "severe",
                        "training_representative_score": 70,
                        "training_label_source": "accepted_as_labeled",
                        "qc_status": "accepted_as_labeled",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            dataset = SyntheticManifestDataset(
                manifest_path,
                target_mode="multitask",
                transform=transforms.ToTensor(),
            )

            target = dataset[0]["target"]
            self.assertEqual(int(target["coarse"]), 3)
            self.assertEqual(int(target["score_band"]), 7)
            self.assertAlmostEqual(float(target["score"]), 0.7)

    def test_multitask_residual_model_returns_scalar_and_classification_heads(self) -> None:
        model = ResidualCNNMultiTaskClassifier(dropout_p=0.2)

        output = model(torch.zeros(2, 3, 128, 128))

        self.assertEqual(set(output.keys()), {"coarse_logits", "score_band_logits", "score"})
        self.assertEqual(tuple(output["coarse_logits"].shape), (2, 4))
        self.assertEqual(tuple(output["score_band_logits"].shape), (2, 10))
        self.assertEqual(tuple(output["score"].shape), (2,))
        self.assertTrue(torch.all(output["score"] >= 0))
        self.assertTrue(torch.all(output["score"] <= 1))

    def test_validate_split_fractions_accepts_real_policy(self) -> None:
        validate_split_fractions(
            train_fraction=0.2,
            val_fraction=0.2,
            test_fraction=0.6,
        )

    def test_validate_split_fractions_rejects_invalid_total(self) -> None:
        with self.assertRaises(ValueError):
            validate_split_fractions(
                train_fraction=0.6,
                val_fraction=0.2,
                test_fraction=0.3,
            )


if __name__ == "__main__":
    unittest.main()
