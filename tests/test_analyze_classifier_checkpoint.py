from __future__ import annotations

import unittest
from pathlib import Path

from scripts.analyze_classifier_checkpoint import resolve_manifest_path, split_indices


class AnalyzeClassifierCheckpointTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
