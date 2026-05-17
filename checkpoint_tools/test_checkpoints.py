from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.training.dataset import (  # noqa: E402
    COARSE_CLASS_NAMES,
    SCORE_BAND_NAMES,
    SCORE_BAND_TO_COARSE_INDEX,
    build_eval_transform,
)
from dream2detect.training.models import (  # noqa: E402
    ResidualCNNMultiTaskClassifier,
    ResidualGroupNormCNNMultiTaskClassifier,
    build_classifier_model,
    build_regressor_model,
)
from dream2detect.training.train_classifier import (  # noqa: E402
    decode_cumulative_ordinal_logits,
    get_target_label_spec,
    model_output_dim_for_target_mode,
)


CheckpointKind = Literal["classifier", "regressor", "multitask"]


def find_checkpoint_files(checkpoints_dir: Path) -> list[Path]:
    return sorted(
        (path for path in checkpoints_dir.rglob("*.pt") if path.is_file()),
        key=lambda path: path.relative_to(checkpoints_dir).as_posix(),
    )


def infer_checkpoint_kind(config: dict[str, Any]) -> CheckpointKind:
    target_label_mode = str(config.get("target_label_mode", ""))
    target_mode = str(config.get("target_mode", ""))
    model_variant = str(config.get("model_variant", ""))

    if target_label_mode == "multitask_scalar_score_band_coarse" or model_variant.endswith(
        "_multitask"
    ):
        return "multitask"
    if target_mode == "fine_normalized" or model_variant.endswith("_regressor"):
        return "regressor"
    if target_label_mode in {"coarse", "coarse_ordinal", "score_band"}:
        return "classifier"

    raise ValueError(
        "Could not infer checkpoint kind from config. Expected target_label_mode, "
        "target_mode, or model_variant."
    )


def score_to_band_index(score: float) -> int:
    clamped = max(0.0, min(100.0, float(score)))
    ranges = [
        (0, 10),
        (11, 20),
        (21, 30),
        (31, 35),
        (36, 45),
        (46, 55),
        (56, 65),
        (66, 75),
        (76, 85),
        (86, 100),
    ]
    for index, (low, high) in enumerate(ranges):
        if low <= clamped <= high:
            return index
    return len(ranges) - 1


def band_index_to_representative_score(index: int) -> int:
    representative_scores = [5, 15, 25, 33, 40, 50, 60, 70, 80, 93]
    return representative_scores[index]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_checkpoint(
    checkpoint_path: Path,
    *,
    config_json: Path | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_json(config_json) if config_json is not None else None

    if isinstance(payload, dict) and "model_state_dict" in payload:
        checkpoint_config = payload.get("config")
        if config is None:
            if not isinstance(checkpoint_config, dict):
                raise ValueError(
                    f"{checkpoint_path} has model_state_dict but no usable config. "
                    "Pass --config-json."
                )
            config = checkpoint_config
        return payload["model_state_dict"], config

    if config is None:
        raise ValueError(
            f"{checkpoint_path} looks like a raw state_dict. Pass --config-json "
            "so the script knows which model architecture to build."
        )
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload).__name__}")
    return payload, config


def build_model_from_config(config: dict[str, Any]) -> tuple[nn.Module, CheckpointKind]:
    kind = infer_checkpoint_kind(config)
    dropout_p = float(config.get("dropout_p", 0.1))

    if kind == "classifier":
        target_label_mode = str(config.get("target_label_mode", "coarse"))
        _, class_names, _ = get_target_label_spec(target_label_mode)
        output_dim = model_output_dim_for_target_mode(
            target_label_mode,
            num_classes=len(class_names),
        )
        model = build_classifier_model(
            num_classes=output_dim,
            dropout_p=dropout_p,
            model_variant=str(config.get("model_variant", "simple_cnn")),
            pretrained=bool(config.get("pretrained", False)),
            freeze_backbone=bool(config.get("freeze_backbone", False)),
        )
        return model, kind

    if kind == "regressor":
        model = build_regressor_model(
            dropout_p=dropout_p,
            model_variant=str(config.get("model_variant", "simple_cnn_regressor")),
        )
        return model, kind

    model_variant = str(config.get("model_variant", "residual_cnn_groupnorm_multitask"))
    if model_variant == "residual_cnn_groupnorm_multitask":
        return ResidualGroupNormCNNMultiTaskClassifier(dropout_p=dropout_p), kind
    if model_variant == "residual_cnn_multitask":
        return ResidualCNNMultiTaskClassifier(dropout_p=dropout_p), kind
    raise ValueError(f"Unsupported multitask model_variant: {model_variant}")


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        image_size: int,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest_dir = manifest_path.parent
        frame = pd.read_csv(manifest_path)
        if limit is not None:
            frame = frame.head(limit)
        self.frame = frame.reset_index(drop=True)
        self.transform = build_eval_transform(image_size=image_size, include_resize=True)

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_image_path(self, image_path_value: object) -> Path:
        image_path = Path(str(image_path_value))
        if image_path.exists():
            return image_path
        return self.manifest_dir / image_path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        row = self.frame.iloc[index].to_dict()
        image_path = self._resolve_image_path(row["image_path"])
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))

        row["_resolved_image_path"] = str(image_path)
        return tensor, row


def choose_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict_batch(
    model: nn.Module,
    images: torch.Tensor,
    *,
    kind: CheckpointKind,
    config: dict[str, Any],
    multitask_head: str,
) -> tuple[list[int | None], list[int | None], list[float | None]]:
    with torch.no_grad():
        outputs = model(images)

    band_indices: list[int | None] = []
    coarse_indices: list[int | None] = []
    scores: list[float | None] = []

    if kind == "classifier":
        target_label_mode = str(config.get("target_label_mode", "coarse"))
        if target_label_mode == "coarse_ordinal":
            predicted = decode_cumulative_ordinal_logits(outputs).cpu().tolist()
        else:
            predicted = torch.argmax(outputs, dim=1).cpu().tolist()

        if target_label_mode == "score_band":
            for band_index in predicted:
                band_indices.append(int(band_index))
                coarse_indices.append(SCORE_BAND_TO_COARSE_INDEX[int(band_index)])
                scores.append(float(band_index_to_representative_score(int(band_index))))
        else:
            for coarse_index in predicted:
                band_indices.append(None)
                coarse_indices.append(int(coarse_index))
                scores.append(None)
        return band_indices, coarse_indices, scores

    if kind == "regressor":
        normalized_scores = outputs.detach().cpu().tolist()
        for normalized_score in normalized_scores:
            score = float(normalized_score) * 100.0
            band_index = score_to_band_index(score)
            band_indices.append(band_index)
            coarse_indices.append(SCORE_BAND_TO_COARSE_INDEX[band_index])
            scores.append(score)
        return band_indices, coarse_indices, scores

    if not isinstance(outputs, dict):
        raise ValueError("Multitask model returned a non-dict output.")

    if multitask_head == "score_band":
        predicted_bands = torch.argmax(outputs["score_band_logits"], dim=1).cpu().tolist()
        for band_index in predicted_bands:
            band_indices.append(int(band_index))
            coarse_indices.append(SCORE_BAND_TO_COARSE_INDEX[int(band_index)])
            scores.append(float(band_index_to_representative_score(int(band_index))))
        return band_indices, coarse_indices, scores

    if multitask_head == "coarse":
        predicted_coarse = torch.argmax(outputs["coarse_logits"], dim=1).cpu().tolist()
        for coarse_index in predicted_coarse:
            band_indices.append(None)
            coarse_indices.append(int(coarse_index))
            scores.append(None)
        return band_indices, coarse_indices, scores

    normalized_scores = outputs["score"].detach().cpu().tolist()
    for normalized_score in normalized_scores:
        score = float(normalized_score) * 100.0
        band_index = score_to_band_index(score)
        band_indices.append(band_index)
        coarse_indices.append(SCORE_BAND_TO_COARSE_INDEX[band_index])
        scores.append(score)
    return band_indices, coarse_indices, scores


def macro_f1(true_labels: list[str], predicted_labels: list[str], labels: tuple[str, ...]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(true_labels, predicted_labels, strict=True))
        fp = sum(t != label and p == label for t, p in zip(true_labels, predicted_labels, strict=True))
        fn = sum(t == label and p != label for t, p in zip(true_labels, predicted_labels, strict=True))
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def accuracy(true_labels: list[str], predicted_labels: list[str]) -> float:
    if not true_labels:
        return 0.0
    return sum(t == p for t, p in zip(true_labels, predicted_labels, strict=True)) / len(true_labels)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_coarse = [str(row["true_coarse_class"]) for row in rows]
    pred_coarse = [str(row["predicted_coarse_class"]) for row in rows if row["predicted_coarse_class"]]
    true_coarse_for_pred = [
        str(row["true_coarse_class"]) for row in rows if row["predicted_coarse_class"]
    ]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "coarse_accuracy": accuracy(true_coarse_for_pred, pred_coarse),
        "coarse_macro_f1": macro_f1(true_coarse_for_pred, pred_coarse, COARSE_CLASS_NAMES),
        "predicted_coarse_distribution": dict(Counter(pred_coarse)),
    }

    rows_with_bands = [
        row
        for row in rows
        if row["true_score_band"] and row["predicted_score_band"]
    ]
    if rows_with_bands:
        true_band_indices = [
            SCORE_BAND_NAMES.index(str(row["true_score_band"])) for row in rows_with_bands
        ]
        pred_band_indices = [
            SCORE_BAND_NAMES.index(str(row["predicted_score_band"])) for row in rows_with_bands
        ]
        abs_errors = [
            abs(true_index - pred_index)
            for true_index, pred_index in zip(true_band_indices, pred_band_indices, strict=True)
        ]
        summary.update(
            {
                "score_band_rows": len(rows_with_bands),
                "score_band_accuracy": sum(error == 0 for error in abs_errors) / len(abs_errors),
                "within_one_band_accuracy": sum(error <= 1 for error in abs_errors) / len(abs_errors),
                "mean_band_error": sum(abs_errors) / len(abs_errors),
                "predicted_score_band_distribution": dict(
                    Counter(str(row["predicted_score_band"]) for row in rows_with_bands)
                ),
            }
        )
    return summary


def safe_output_name(checkpoint_path: Path) -> str:
    parts = list(checkpoint_path.with_suffix("").parts)
    if "checkpoints" in parts:
        parts = parts[parts.index("checkpoints") + 1 :]
    return "__".join(part for part in parts if part)


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    manifest_path: Path,
    output_dir: Path,
    config_json: Path | None,
    batch_size: int,
    image_size: int,
    device: torch.device,
    multitask_head: str,
    limit: int | None,
) -> dict[str, Any]:
    state_dict, config = load_checkpoint(checkpoint_path, config_json=config_json)
    model, kind = build_model_from_config(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = ManifestImageDataset(manifest_path, image_size=image_size, limit=limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []

    for images, batch_rows in dataloader:
        images = images.to(device)
        pred_band_indices, pred_coarse_indices, pred_scores = predict_batch(
            model,
            images,
            kind=kind,
            config=config,
            multitask_head=multitask_head,
        )

        batch_size_actual = len(pred_coarse_indices)
        for index in range(batch_size_actual):
            true_score_band = str(batch_rows["training_score_band"][index])
            true_coarse_class = str(batch_rows["training_coarse_class"][index])
            pred_band_index = pred_band_indices[index]
            pred_coarse_index = pred_coarse_indices[index]
            rows.append(
                {
                    "image_path": str(batch_rows["image_path"][index]),
                    "true_score_band": true_score_band,
                    "predicted_score_band": (
                        SCORE_BAND_NAMES[pred_band_index]
                        if pred_band_index is not None
                        else ""
                    ),
                    "true_coarse_class": true_coarse_class,
                    "predicted_coarse_class": (
                        COARSE_CLASS_NAMES[pred_coarse_index]
                        if pred_coarse_index is not None
                        else ""
                    ),
                    "predicted_score": (
                        round(float(pred_scores[index]), 4)
                        if pred_scores[index] is not None
                        else ""
                    ),
                }
            )

    checkpoint_output = output_dir / safe_output_name(checkpoint_path)
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    predictions_path = checkpoint_output / "predictions.csv"
    with predictions_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "kind": kind,
            "model_variant": str(config.get("model_variant", "")),
            "target_label_mode": str(config.get("target_label_mode", config.get("target_mode", ""))),
            "multitask_head": multitask_head if kind == "multitask" else "",
            "predictions_csv": str(predictions_path),
        }
    )
    (checkpoint_output / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Dream2Detect checkpoints against a delivery manifest."
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("dataset/real/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint_results"))
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--multitask-head",
        choices=["scalar", "score_band", "coarse"],
        default="scalar",
        help="Which V5-B head to use as the primary prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None:
        checkpoint_files = [args.checkpoint]
    else:
        checkpoint_files = find_checkpoint_files(args.checkpoints_dir)

    if not checkpoint_files:
        raise SystemExit(
            f"No .pt checkpoints found. Put trained models under {args.checkpoints_dir} "
            "or pass --checkpoint."
        )
    if args.config_json is not None and len(checkpoint_files) > 1:
        raise SystemExit("--config-json can only be used with --checkpoint.")

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for checkpoint_path in checkpoint_files:
        print(f"[checkpoint] {checkpoint_path}")
        summary = evaluate_checkpoint(
            checkpoint_path,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            config_json=args.config_json,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=device,
            multitask_head=args.multitask_head,
            limit=args.limit,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
