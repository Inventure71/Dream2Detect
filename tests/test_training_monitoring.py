from __future__ import annotations

import unittest

import pandas as pd
import torch
from torchvision import transforms

from dream2detect.training.dataset import (
    RGB_NORMALIZATION_MEAN,
    RGB_NORMALIZATION_STD,
    build_eval_transform,
    build_train_transform,
)
from dream2detect.training.metrics import compute_classification_metrics
from dream2detect.training.train_classifier import build_sample_weights


class TrainingMonitoringTests(unittest.TestCase):
    def test_classification_metrics_include_per_class_and_prediction_distribution(self) -> None:
        logits = torch.tensor(
            [
                [4.0, 1.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 6.0, 1.0, 0.0],
                [0.0, 0.0, 7.0, 1.0],
                [0.0, 0.0, 8.0, 1.0],
            ]
        )
        targets = torch.tensor([0, 1, 2, 2, 3])

        metrics = compute_classification_metrics(
            logits,
            targets,
            num_classes=4,
            class_names=["intact", "minor", "moderate", "severe"],
        )

        self.assertEqual(metrics.predicted_class_distribution["intact"], 1)
        self.assertEqual(metrics.predicted_class_distribution["minor"], 2)
        self.assertEqual(metrics.predicted_class_distribution["moderate"], 2)
        self.assertEqual(metrics.predicted_class_distribution["severe"], 0)
        self.assertEqual(metrics.target_class_distribution["severe"], 1)
        self.assertAlmostEqual(metrics.per_class["intact"].precision, 1.0)
        self.assertAlmostEqual(metrics.per_class["severe"].recall, 0.0)
        self.assertAlmostEqual(metrics.per_class["severe"].f1, 0.0)

    def test_transforms_apply_standard_rgb_normalization_after_to_tensor(self) -> None:
        for transform in (
            build_train_transform(include_resize=False),
            build_eval_transform(include_resize=False),
        ):
            normalize_steps = [
                step
                for step in transform.transforms
                if isinstance(step, transforms.Normalize)
            ]
            self.assertEqual(len(normalize_steps), 1)
            self.assertEqual(tuple(normalize_steps[0].mean), RGB_NORMALIZATION_MEAN)
            self.assertEqual(tuple(normalize_steps[0].std), RGB_NORMALIZATION_STD)
            normalize_index = transform.transforms.index(normalize_steps[0])
            to_tensor_index = next(
                index
                for index, step in enumerate(transform.transforms)
                if isinstance(step, transforms.ToTensor)
            )
            self.assertGreater(normalize_index, to_tensor_index)

    def test_sample_weights_give_underrepresented_classes_larger_weights(self) -> None:
        frame = pd.DataFrame(
            {
                "training_coarse_class": [
                    "intact",
                    "minor",
                    "minor",
                    "moderate",
                    "moderate",
                    "moderate",
                ]
            }
        )

        weights = build_sample_weights(frame, train_indices=[0, 1, 2, 3, 4, 5])

        self.assertEqual(len(weights), 6)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[3])


if __name__ == "__main__":
    unittest.main()
