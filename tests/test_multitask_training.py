from __future__ import annotations

import unittest

import torch

from dream2detect.training.train_multitask_classifier import (
    build_multitask_loss,
    summarize_score_band_errors,
)


class MultiTaskTrainingTests(unittest.TestCase):
    def test_multitask_loss_uses_auxiliary_band_weight(self) -> None:
        outputs = {
            "coarse_logits": torch.tensor([[2.0, 0.0, -1.0, -2.0]]),
            "score_band_logits": torch.tensor(
                [[2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]]
            ),
        }
        targets = {
            "coarse": torch.tensor([0]),
            "score_band": torch.tensor([9]),
        }

        base_loss = build_multitask_loss(
            coarse_class_weights=torch.ones(4),
            score_band_weights=torch.ones(10),
            use_balanced_sampler=True,
            auxiliary_band_loss_weight=0.0,
            band_ordinal_loss_weight=0.0,
            device=torch.device("cpu"),
        )(outputs, targets)
        auxiliary_loss = build_multitask_loss(
            coarse_class_weights=torch.ones(4),
            score_band_weights=torch.ones(10),
            use_balanced_sampler=True,
            auxiliary_band_loss_weight=0.3,
            band_ordinal_loss_weight=0.2,
            device=torch.device("cpu"),
        )(outputs, targets)

        self.assertGreater(float(auxiliary_loss), float(base_loss))

    def test_score_band_error_summary_counts_near_misses(self) -> None:
        confusion = [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]

        summary = summarize_score_band_errors(confusion)

        self.assertEqual(summary["total"], 6)
        self.assertAlmostEqual(summary["exact_accuracy"], 0.5)
        self.assertAlmostEqual(summary["within_one_band_accuracy"], 5 / 6)
        self.assertAlmostEqual(summary["severe_band_error_rate"], 1 / 6)


if __name__ == "__main__":
    unittest.main()
