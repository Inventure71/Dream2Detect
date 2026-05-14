from __future__ import annotations

import unittest
from pathlib import Path

from scripts.analyze_classifier_checkpoint import (
    ExamplePrediction,
    get_target_class_names,
    resolve_manifest_path,
    split_indices,
    summarize_split_for_target,
)


class AnalyzeClassifierCheckpointTests(unittest.TestCase):
    def test_get_target_class_names_uses_score_bands_for_score_band_mode(self) -> None:
        self.assertEqual(len(get_target_class_names("score_band")), 10)
        self.assertEqual(get_target_class_names("coarse"), ("intact", "minor", "moderate", "severe"))

    def test_manifest_override_takes_precedence_over_run_config(self) -> None:
        run_config = {"manifest_path": "/tmp/synthetic.csv"}
        manifest_path = Path("/tmp/real.csv")

        self.assertEqual(
            resolve_manifest_path(run_config, manifest_override=manifest_path),
            manifest_path,
        )

    def test_all_as_test_split_uses_every_row_for_test(self) -> None:
        indices = split_indices(
            manifest_path=Path("/tmp/not-used.csv"),
            random_seed=42,
            row_count=5,
            all_as_test=True,
        )

        self.assertEqual(indices["train"], [])
        self.assertEqual(indices["val"], [])
        self.assertEqual(indices["test"], [0, 1, 2, 3, 4])

    def test_all_as_test_does_not_require_split_fractions(self) -> None:
        indices = split_indices(
            manifest_path=Path("/tmp/not-used.csv"),
            random_seed=42,
            row_count=2,
            all_as_test=True,
            train_fraction=0.2,
            val_fraction=0.2,
            test_fraction=0.6,
        )

        self.assertEqual(indices["test"], [0, 1])

    def test_summarize_split_for_target_reports_collapsed_coarse_metrics(self) -> None:
        examples = [
            ExamplePrediction(
                split="test",
                row_index=0,
                image_path="/tmp/a.png",
                source_image_path="",
                source="synthetic",
                score_band="11-20",
                label_source="accepted",
                qc_status="accepted",
                true_label="11-20",
                predicted_label="21-30",
                true_coarse_class="minor",
                predicted_coarse_class="minor",
                correct=False,
                confidence=0.6,
                true_probability=0.4,
                margin=0.2,
                ordinal_error=1,
                probabilities={"11-20": 0.4, "21-30": 0.6},
                coarse_probabilities={"intact": 0.0, "minor": 1.0, "moderate": 0.0, "severe": 0.0},
            ),
            ExamplePrediction(
                split="test",
                row_index=1,
                image_path="/tmp/b.png",
                source_image_path="",
                source="synthetic",
                score_band="66-75",
                label_source="accepted",
                qc_status="accepted",
                true_label="66-75",
                predicted_label="76-85",
                true_coarse_class="severe",
                predicted_coarse_class="severe",
                correct=False,
                confidence=0.7,
                true_probability=0.35,
                margin=0.1,
                ordinal_error=1,
                probabilities={"66-75": 0.35, "76-85": 0.65},
                coarse_probabilities={"intact": 0.0, "minor": 0.0, "moderate": 0.0, "severe": 1.0},
            ),
        ]

        summary = summarize_split_for_target(examples, target_label_mode="score_band")

        self.assertEqual(summary["accuracy"], 0.0)
        self.assertEqual(summary["collapsed_coarse_accuracy"], 1.0)
        self.assertGreater(summary["collapsed_coarse_macro_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
