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
    SimpleCNNClassifier,
)
from dream2detect.training.models import ResidualCNNMultiTaskClassifier
from dream2detect.training.metrics import (
    band_indices_to_coarse_indices,
    collapse_score_band_probabilities_to_coarse,
    summarize_ordinal_errors,
)
from dream2detect.training.splits import build_stratified_splits, derive_metadata_family_groups
from dream2detect.training.train_classifier import (
    build_classification_loss,
    build_effective_number_class_weights,
    build_ordinal_classification_loss,
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

    def test_multitask_residual_model_returns_both_heads(self) -> None:
        model = ResidualCNNMultiTaskClassifier(dropout_p=0.2)

        output = model(torch.zeros(2, 3, 128, 128))

        self.assertEqual(set(output.keys()), {"coarse_logits", "score_band_logits"})
        self.assertEqual(tuple(output["coarse_logits"].shape), (2, 4))
        self.assertEqual(tuple(output["score_band_logits"].shape), (2, 10))

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
