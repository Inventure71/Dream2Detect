from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from checkpoint_tools.test_checkpoints import (
    find_checkpoint_files,
    infer_checkpoint_kind,
    score_to_band_index,
)


class CheckpointToolTests(unittest.TestCase):
    def test_find_checkpoint_files_recurses_and_sorts_pt_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "z_latest.pt").touch()
            (root / "ignore.txt").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "a_best.pt").touch()

            found = find_checkpoint_files(root)

        self.assertEqual(
            [path.name for path in found],
            ["a_best.pt", "z_latest.pt"],
        )

    def test_infer_checkpoint_kind_from_training_config(self) -> None:
        self.assertEqual(
            infer_checkpoint_kind({"target_label_mode": "score_band"}),
            "classifier",
        )
        self.assertEqual(
            infer_checkpoint_kind({"target_label_mode": "multitask_scalar_score_band_coarse"}),
            "multitask",
        )
        self.assertEqual(
            infer_checkpoint_kind({"target_mode": "fine_normalized"}),
            "regressor",
        )

    def test_score_to_band_index_clamps_to_project_scale(self) -> None:
        self.assertEqual(score_to_band_index(-5.0), 0)
        self.assertEqual(score_to_band_index(0.0), 0)
        self.assertEqual(score_to_band_index(50.0), 5)
        self.assertEqual(score_to_band_index(100.0), 9)
        self.assertEqual(score_to_band_index(125.0), 9)


if __name__ == "__main__":
    unittest.main()
