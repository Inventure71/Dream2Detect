from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from dream2detect.training.train_classifier import (
    EarlyStoppingTracker,
    EpochResult,
    append_epoch_metrics,
    build_run_config,
    plot_class_monitoring,
    plot_epoch_metrics,
    save_epoch_checkpoint,
)


class TrainingArtifactTests(unittest.TestCase):
    def test_early_stopping_tracker_stops_after_patience_without_improvement(self) -> None:
        tracker = EarlyStoppingTracker(patience=2, min_delta=0.01)

        first = tracker.update(epoch_number=1, metric_value=0.20)
        second = tracker.update(epoch_number=2, metric_value=0.205)
        third = tracker.update(epoch_number=3, metric_value=0.206)

        self.assertTrue(first.is_best)
        self.assertFalse(second.is_best)
        self.assertFalse(third.is_best)
        self.assertTrue(third.should_stop)
        self.assertEqual(third.best_epoch, 1)

    def test_early_stopping_tracker_supports_min_mode(self) -> None:
        tracker = EarlyStoppingTracker(patience=2, min_delta=0.01, mode="min")

        first = tracker.update(epoch_number=1, metric_value=1.20)
        second = tracker.update(epoch_number=2, metric_value=1.195)
        third = tracker.update(epoch_number=3, metric_value=1.196)

        self.assertTrue(first.is_best)
        self.assertFalse(second.is_best)
        self.assertFalse(third.is_best)
        self.assertTrue(third.should_stop)
        self.assertEqual(third.best_epoch, 1)

    def test_build_run_config_records_training_controls(self) -> None:
        config = build_run_config(
            manifest_path=Path("/tmp/manifest.csv"),
            image_size=128,
            batch_size=8,
            num_epochs=20,
            learning_rate=0.001,
            optimizer_name="adamw",
            lr_scheduler_name="reduce_on_plateau",
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=10,
            min_learning_rate=1e-5,
            random_seed=42,
            device=torch.device("cpu"),
            include_resize=False,
            use_balanced_sampler=True,
            weight_decay=0.0001,
            dropout_p=0.2,
            early_stopping_patience=5,
            early_stopping_min_delta=0.01,
            overfit_subset_size=32,
            use_augmentation=False,
            augmentation_profile="mild",
            model_variant="simple_cnn",
            pretrained=False,
            freeze_backbone=False,
            ordinal_loss_weight=0.0,
            score_band_soft_label_sigma=1.0,
            score_band_emd_weight=0.5,
            score_band_class_weight_strategy="effective",
            score_band_effective_beta=0.999,
            target_label_mode="score_band",
            class_names=["0-10", "11-20"],
            train_fraction=0.2,
            val_fraction=0.2,
            test_fraction=0.6,
            init_from_checkpoint=Path("/tmp/initial.pt"),
            checkpoint_every_n_epochs=1,
            plot_every_n_epochs=1,
        )

        self.assertEqual(config["weight_decay"], 0.0001)
        self.assertEqual(config["dropout_p"], 0.2)
        self.assertEqual(config["optimizer_name"], "adamw")
        self.assertEqual(config["lr_scheduler_name"], "reduce_on_plateau")
        self.assertEqual(config["lr_scheduler_patience"], 10)
        self.assertEqual(config["early_stopping_patience"], 5)
        self.assertEqual(config["overfit_subset_size"], 32)
        self.assertFalse(config["use_augmentation"])
        self.assertEqual(config["ordinal_loss_weight"], 0.0)
        self.assertEqual(config["score_band_soft_label_sigma"], 1.0)
        self.assertEqual(config["score_band_emd_weight"], 0.5)
        self.assertEqual(config["score_band_class_weight_strategy"], "effective")
        self.assertEqual(config["score_band_effective_beta"], 0.999)
        self.assertEqual(config["target_label_mode"], "score_band")
        self.assertEqual(config["num_classes"], 2)
        self.assertEqual(config["class_names"], ["0-10", "11-20"])
        self.assertEqual(config["selection_metric_name"], "val_mean_band_error")
        self.assertEqual(config["selection_metric_mode"], "min")
        self.assertEqual(config["train_fraction"], 0.2)
        self.assertEqual(config["val_fraction"], 0.2)
        self.assertEqual(config["test_fraction"], 0.6)
        self.assertEqual(config["init_from_checkpoint"], "/tmp/initial.pt")
        self.assertEqual(config["checkpoint_every_n_epochs"], 1)
        self.assertEqual(config["plot_every_n_epochs"], 1)

    def test_append_epoch_metrics_writes_jsonl_and_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            train_metrics = EpochResult(
                loss=1.2,
                accuracy=0.25,
                macro_f1=0.18,
                confusion=[[1, 0], [0, 1]],
            )
            val_metrics = EpochResult(
                loss=1.1,
                accuracy=0.5,
                macro_f1=0.4,
                confusion=[[1, 0], [1, 0]],
            )

            append_epoch_metrics(
                output_dir=output_dir,
                epoch_number=1,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=True,
            )

            jsonl_path = output_dir / "epoch_metrics.jsonl"
            rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
            self.assertEqual(rows[0]["epoch"], 1)
            self.assertEqual(rows[0]["train"]["loss"], 1.2)
            self.assertEqual(rows[0]["val"]["macro_f1"], 0.4)
            self.assertTrue(rows[0]["is_best"])

            csv_path = output_dir / "epoch_metrics.csv"
            with csv_path.open(newline="") as csv_file:
                csv_rows = list(csv.DictReader(csv_file))

            self.assertEqual(csv_rows[0]["epoch"], "1")
            self.assertEqual(csv_rows[0]["learning_rate"], "")
            self.assertEqual(csv_rows[0]["train_loss"], "1.2")
            self.assertEqual(csv_rows[0]["val_macro_f1"], "0.4")
            self.assertEqual(csv_rows[0]["is_best"], "True")

    def test_save_epoch_checkpoint_writes_numbered_latest_and_best_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_metrics = EpochResult(
                loss=1.2,
                accuracy=0.25,
                macro_f1=0.18,
                confusion=[[1, 0], [0, 1]],
            )
            val_metrics = EpochResult(
                loss=1.1,
                accuracy=0.5,
                macro_f1=0.4,
                confusion=[[1, 0], [1, 0]],
            )

            save_epoch_checkpoint(
                output_dir=output_dir,
                epoch_number=1,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=True,
                config={"learning_rate": 0.001},
            )

            checkpoint_dir = output_dir / "checkpoints"
            self.assertTrue((checkpoint_dir / "epoch_001.pt").exists())
            self.assertTrue((checkpoint_dir / "latest.pt").exists())
            self.assertTrue((checkpoint_dir / "best.pt").exists())

            payload = torch.load(
                checkpoint_dir / "epoch_001.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["config"]["learning_rate"], 0.001)
            self.assertIn("model_state_dict", payload)
            self.assertIn("optimizer_state_dict", payload)

    def test_plot_epoch_metrics_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            history = [
                {
                    "epoch": 1,
                    "train_loss": 1.3,
                    "val_loss": 1.2,
                    "train_accuracy": 0.2,
                    "val_accuracy": 0.3,
                    "train_macro_f1": 0.15,
                    "val_macro_f1": 0.25,
                },
                {
                    "epoch": 2,
                    "train_loss": 1.1,
                    "val_loss": 1.0,
                    "train_accuracy": 0.4,
                    "val_accuracy": 0.5,
                    "train_macro_f1": 0.35,
                    "val_macro_f1": 0.45,
                },
            ]

            plot_path = plot_epoch_metrics(output_dir=output_dir, history=history)

            self.assertEqual(plot_path, output_dir / "training_curves.png")
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_plot_class_monitoring_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            history = [
                {
                    "epoch": 1,
                    "val_pred_intact": 10,
                    "val_pred_minor": 2,
                    "val_pred_moderate": 0,
                    "val_pred_severe": 0,
                    "val_intact_f1": 0.5,
                    "val_minor_f1": 0.2,
                    "val_moderate_f1": 0.0,
                    "val_severe_f1": 0.0,
                },
                {
                    "epoch": 2,
                    "val_pred_intact": 4,
                    "val_pred_minor": 4,
                    "val_pred_moderate": 3,
                    "val_pred_severe": 1,
                    "val_intact_f1": 0.7,
                    "val_minor_f1": 0.4,
                    "val_moderate_f1": 0.3,
                    "val_severe_f1": 0.1,
                },
            ]

            plot_path = plot_class_monitoring(output_dir=output_dir, history=history)

            self.assertEqual(plot_path, output_dir / "class_monitoring.png")
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
