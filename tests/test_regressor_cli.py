from __future__ import annotations

import unittest

from scripts.train_synthetic_regressor import build_argument_parser


class RegressorCliTests(unittest.TestCase):
    def test_parser_accepts_v5_scalar_regression_controls(self) -> None:
        parser = build_argument_parser()

        args = parser.parse_args(
            [
                "--image-size",
                "384",
                "--optimizer",
                "adamw",
                "--lr-scheduler",
                "reduce_on_plateau",
                "--model-variant",
                "residual_cnn_groupnorm_regressor",
                "--target-mode",
                "fine_normalized",
                "--split-strategy",
                "metadata_family_holdout",
                "--overfit-subset-size",
                "40",
                "--checkpoint-every-n-epochs",
                "1",
            ]
        )

        self.assertEqual(args.image_size, 384)
        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.lr_scheduler, "reduce_on_plateau")
        self.assertEqual(args.model_variant, "residual_cnn_groupnorm_regressor")
        self.assertEqual(args.target_mode, "fine_normalized")
        self.assertEqual(args.split_strategy, "metadata_family_holdout")
        self.assertEqual(args.overfit_subset_size, 40)
        self.assertEqual(args.checkpoint_every_n_epochs, 1)


if __name__ == "__main__":
    unittest.main()
