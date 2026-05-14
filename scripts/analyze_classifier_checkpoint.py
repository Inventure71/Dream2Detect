from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from PIL import ImageDraw

from dream2detect.training.dataset import (
    COARSE_CLASS_NAMES,
    COARSE_CLASS_TO_INDEX,
    build_eval_transform,
)
from dream2detect.training.models import build_classifier_model
from dream2detect.training.splits import build_stratified_splits
from dream2detect.training.train_classifier import (
    decode_cumulative_ordinal_logits,
    infer_resize_policy,
    model_output_dim_for_target_mode,
)


@dataclass(frozen=True)
class ExamplePrediction:
    split: str
    row_index: int
    image_path: str
    source_image_path: str
    source: str
    score_band: str
    label_source: str
    qc_status: str
    true_class: str
    predicted_class: str
    correct: bool
    confidence: float
    true_probability: float
    margin: float
    ordinal_error: int
    probabilities: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a trained synthetic classifier checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing run_config.json and metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to analyze. Defaults to run-dir/checkpoints/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic outputs. Defaults to run-dir/diagnostics.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest override for out-of-run evaluation.",
    )
    parser.add_argument(
        "--all-as-test",
        action="store_true",
        help="Evaluate every row in the selected manifest as the test split.",
    )
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return json.loads(config_path.read_text())


def load_model(
    *,
    checkpoint_path: Path,
    run_config: dict[str, object],
    device: torch.device,
) -> torch.nn.Module:
    target_label_mode = str(run_config.get("target_label_mode", "coarse"))
    model = build_classifier_model(
        num_classes=model_output_dim_for_target_mode(
            target_label_mode,
            num_classes=len(COARSE_CLASS_NAMES),
        ),
        dropout_p=float(run_config.get("dropout_p", 0.1)),
        model_variant=str(run_config.get("model_variant", "simple_cnn")),
        pretrained=False,
        freeze_backbone=bool(run_config.get("freeze_backbone", False)),
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def resolve_manifest_path(
    run_config: dict[str, object],
    *,
    manifest_override: Path | None = None,
) -> Path:
    if manifest_override is not None:
        return manifest_override
    return Path(str(run_config["manifest_path"]))


def split_indices(
    manifest_path: Path,
    random_seed: int,
    *,
    row_count: int | None = None,
    all_as_test: bool = False,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    split_strategy: str = "stratified_random",
    split_group_column: str | None = None,
) -> dict[str, list[int]]:
    if all_as_test:
        if row_count is None:
            raise ValueError("row_count is required when all_as_test=True.")
        return {
            "train": [],
            "val": [],
            "test": list(range(row_count)),
        }

    split = build_stratified_splits(
        manifest_path,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
        split_strategy=split_strategy,
        split_group_column=split_group_column,
    )
    return {
        "train": split.train_indices,
        "val": split.val_indices,
        "test": split.test_indices,
    }


def predict_examples(
    *,
    model: torch.nn.Module,
    frame: pd.DataFrame,
    indices_by_split: dict[str, list[int]],
    image_size: int,
    include_resize: bool,
    device: torch.device,
    target_label_mode: str = "coarse",
) -> list[ExamplePrediction]:
    transform = build_eval_transform(
        image_size=image_size,
        include_resize=include_resize,
    )
    examples: list[ExamplePrediction] = []
    class_names = list(COARSE_CLASS_NAMES)

    with torch.no_grad():
        for split_name, indices in indices_by_split.items():
            for row_index in indices:
                row = frame.iloc[row_index]
                image_path = Path(row["image_path"])
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                logits = model(image_tensor)
                true_class = str(row["training_coarse_class"])
                true_index = COARSE_CLASS_TO_INDEX[true_class]
                if target_label_mode == "coarse_ordinal":
                    threshold_probabilities = torch.sigmoid(logits).squeeze(0).cpu()
                    predicted_index = int(
                        decode_cumulative_ordinal_logits(logits.cpu()).item()
                    )
                    class_probabilities = torch.zeros(len(class_names), dtype=torch.float32)
                    class_probabilities[0] = 1.0 - threshold_probabilities[0]
                    for class_index in range(1, len(class_names) - 1):
                        class_probabilities[class_index] = (
                            threshold_probabilities[class_index - 1]
                            - threshold_probabilities[class_index]
                        )
                    class_probabilities[-1] = threshold_probabilities[-1]
                    probabilities_tensor = torch.clamp(class_probabilities, min=0.0)
                    probability_total = probabilities_tensor.sum()
                    if probability_total > 0:
                        probabilities_tensor = probabilities_tensor / probability_total
                else:
                    probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()
                    predicted_index = int(torch.argmax(probabilities_tensor).item())

                probabilities = {
                    class_name: float(probabilities_tensor[class_index].item())
                    for class_index, class_name in enumerate(class_names)
                }
                predicted_class = class_names[predicted_index]
                sorted_probs = sorted(probabilities_tensor.tolist(), reverse=True)
                confidence = float(sorted_probs[0])
                margin = float(sorted_probs[0] - sorted_probs[1])

                examples.append(
                    ExamplePrediction(
                        split=split_name,
                        row_index=int(row_index),
                        image_path=str(image_path),
                        source_image_path=str(row.get("source_image_path", "")),
                        source=str(row.get("combined_dataset_source", "")),
                        score_band=str(row.get("training_score_band", "")),
                        label_source=str(row.get("training_label_source", "")),
                        qc_status=str(row.get("qc_status", "")),
                        true_class=true_class,
                        predicted_class=predicted_class,
                        correct=predicted_class == true_class,
                        confidence=confidence,
                        true_probability=float(probabilities_tensor[true_index].item()),
                        margin=margin,
                        ordinal_error=abs(predicted_index - true_index),
                        probabilities=probabilities,
                    )
                )

    return examples


def confusion_matrix(
    examples: list[ExamplePrediction],
) -> list[list[int]]:
    class_names = list(COARSE_CLASS_NAMES)
    matrix = [[0 for _ in class_names] for _ in class_names]
    for example in examples:
        true_index = COARSE_CLASS_TO_INDEX[example.true_class]
        predicted_index = COARSE_CLASS_TO_INDEX[example.predicted_class]
        matrix[true_index][predicted_index] += 1
    return matrix


def per_class_rows(
    examples: list[ExamplePrediction],
) -> list[dict[str, object]]:
    total = len(examples)
    matrix = confusion_matrix(examples)
    rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(COARSE_CLASS_NAMES):
        tp = matrix[class_index][class_index]
        fn = sum(matrix[class_index]) - tp
        fp = sum(row[class_index] for row in matrix) - tp
        tn = total - tp - fp - fn

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        fpr = safe_div(fp, fp + tn)
        fnr = safe_div(fn, fn + tp)
        npv = safe_div(tn, tn + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "class": class_name,
                "support": tp + fn,
                "predicted_count": tp + fp,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "negative_predictive_value": npv,
                "f1": f1,
            }
        )

    return rows


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def aggregate_group(
    examples: list[ExamplePrediction],
    group_attr: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[ExamplePrediction]] = {}
    for example in examples:
        key = str(getattr(example, group_attr))
        grouped.setdefault(key, []).append(example)

    rows: list[dict[str, object]] = []
    for key, group_examples in sorted(grouped.items()):
        correct = sum(1 for example in group_examples if example.correct)
        rows.append(
            {
                group_attr: key,
                "count": len(group_examples),
                "accuracy": safe_div(correct, len(group_examples)),
                "avg_confidence": average(example.confidence for example in group_examples),
                "avg_true_probability": average(
                    example.true_probability for example in group_examples
                ),
                "avg_ordinal_error": average(
                    example.ordinal_error for example in group_examples
                ),
            }
        )
    return rows


def feature_slice_rows(
    *,
    frame: pd.DataFrame,
    examples: list[ExamplePrediction],
) -> list[dict[str, object]]:
    slice_columns = [
        "combined_dataset_source",
        "training_score_band",
        "damage_profile_primary",
        "damage_profile_secondary",
        "damage_location_primary",
        "box_form_factor",
        "box_pattern",
        "label_presence",
        "tape_profile",
        "background_context",
        "camera_angle",
        "lighting_style",
    ]
    available_columns = [column for column in slice_columns if column in frame.columns]
    rows: list[dict[str, object]] = []

    for column in available_columns:
        grouped: dict[str, list[ExamplePrediction]] = {}
        for example in examples:
            value = str(frame.iloc[example.row_index].get(column, ""))
            grouped.setdefault(value, []).append(example)

        for value, group_examples in sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            correct = sum(1 for example in group_examples if example.correct)
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": value,
                    "count": len(group_examples),
                    "accuracy": safe_div(correct, len(group_examples)),
                    "avg_confidence": average(
                        example.confidence for example in group_examples
                    ),
                    "avg_true_probability": average(
                        example.true_probability for example in group_examples
                    ),
                    "avg_ordinal_error": average(
                        example.ordinal_error for example in group_examples
                    ),
                }
            )

    return rows


def confidence_bins(examples: list[ExamplePrediction]) -> list[dict[str, object]]:
    bins = [(0.25, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    rows: list[dict[str, object]] = []
    for lower, upper in bins:
        in_bin = [
            example
            for example in examples
            if lower <= example.confidence < upper
        ]
        if not in_bin:
            rows.append(
                {
                    "confidence_bin": f"[{lower:.2f},{upper:.2f})",
                    "count": 0,
                    "accuracy": "",
                    "avg_confidence": "",
                    "confidence_minus_accuracy": "",
                }
            )
            continue
        accuracy = safe_div(sum(1 for example in in_bin if example.correct), len(in_bin))
        avg_confidence = average(example.confidence for example in in_bin)
        rows.append(
            {
                "confidence_bin": f"[{lower:.2f},{upper:.2f})",
                "count": len(in_bin),
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "confidence_minus_accuracy": avg_confidence - accuracy,
            }
        )
    return rows


def average(values: object) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return float(sum(value_list)) / float(len(value_list))


def summarize_split(examples: list[ExamplePrediction]) -> dict[str, object]:
    if not examples:
        return {
            "count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mean_confidence": 0.0,
            "mean_true_probability": 0.0,
            "mean_ordinal_error": 0.0,
            "off_by_one_or_correct_rate": 0.0,
            "severe_ordinal_error_rate": 0.0,
        }

    class_rows = per_class_rows(examples)
    correct = sum(1 for example in examples if example.correct)
    off_by_one_or_correct = sum(
        1 for example in examples if example.ordinal_error <= 1
    )
    severe_errors = sum(1 for example in examples if example.ordinal_error >= 2)
    return {
        "count": len(examples),
        "accuracy": safe_div(correct, len(examples)),
        "macro_f1": average(row["f1"] for row in class_rows),
        "mean_confidence": average(example.confidence for example in examples),
        "mean_true_probability": average(example.true_probability for example in examples),
        "mean_ordinal_error": average(example.ordinal_error for example in examples),
        "off_by_one_or_correct_rate": safe_div(off_by_one_or_correct, len(examples)),
        "severe_ordinal_error_rate": safe_div(severe_errors, len(examples)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    rows = []
    for class_name, row in zip(COARSE_CLASS_NAMES, matrix):
        rows.append(
            {
                "true_class": class_name,
                **{
                    f"predicted_{predicted_class}": value
                    for predicted_class, value in zip(COARSE_CLASS_NAMES, row)
                },
            }
        )
    write_csv(path, rows)


def example_rows(examples: list[ExamplePrediction]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        rows.append(
            {
                "split": example.split,
                "row_index": example.row_index,
                "true_class": example.true_class,
                "predicted_class": example.predicted_class,
                "correct": example.correct,
                "confidence": example.confidence,
                "true_probability": example.true_probability,
                "margin": example.margin,
                "ordinal_error": example.ordinal_error,
                "score_band": example.score_band,
                "source": example.source,
                "label_source": example.label_source,
                "qc_status": example.qc_status,
                "image_path": example.image_path,
                "source_image_path": example.source_image_path,
                **{
                    f"prob_{class_name}": example.probabilities[class_name]
                    for class_name in COARSE_CLASS_NAMES
                },
            }
        )
    return rows


def top_error_rows(examples: list[ExamplePrediction], limit: int = 25) -> list[dict[str, object]]:
    wrong = [example for example in examples if not example.correct]
    wrong.sort(key=lambda example: (example.confidence, example.ordinal_error), reverse=True)
    return example_rows(wrong[:limit])


def write_error_contact_sheet(
    *,
    output_path: Path,
    examples: list[ExamplePrediction],
    limit: int = 25,
) -> None:
    wrong = [example for example in examples if not example.correct]
    wrong.sort(key=lambda example: (example.confidence, example.ordinal_error), reverse=True)
    selected = wrong[:limit]
    if not selected:
        return

    columns = 5
    image_size = 180
    caption_height = 70
    padding = 12
    tile_width = image_size
    tile_height = image_size + caption_height
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB",
        (
            columns * tile_width + (columns + 1) * padding,
            rows * tile_height + (rows + 1) * padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for item_index, example in enumerate(selected):
        column = item_index % columns
        row = item_index // columns
        x = padding + column * (tile_width + padding)
        y = padding + row * (tile_height + padding)
        image_path = Path(example.source_image_path or example.image_path)
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((image_size, image_size))
        image_x = x + (image_size - image.width) // 2
        image_y = y + (image_size - image.height) // 2
        sheet.paste(image, (image_x, image_y))
        caption = (
            f"true: {example.true_class}\n"
            f"pred: {example.predicted_class}\n"
            f"conf: {example.confidence:.2f} band: {example.score_band}"
        )
        draw.multiline_text((x, y + image_size + 4), caption, fill="black", spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_markdown_report(
    *,
    output_path: Path,
    run_dir: Path,
    checkpoint_path: Path,
    split_summaries: dict[str, dict[str, object]],
    test_examples: list[ExamplePrediction],
) -> None:
    test_matrix = confusion_matrix(test_examples)
    test_class_rows = per_class_rows(test_examples)
    prediction_distribution = Counter(example.predicted_class for example in test_examples)
    target_distribution = Counter(example.true_class for example in test_examples)
    high_conf_wrong = [
        example
        for example in test_examples
        if not example.correct and example.confidence >= 0.70
    ]
    high_conf_correct = [
        example
        for example in test_examples
        if example.correct and example.confidence >= 0.70
    ]

    lines = [
        "# Classifier Checkpoint Diagnostics",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Checkpoint: `{checkpoint_path}`",
        "",
        "## Split Summary",
        "",
        "| split | count | accuracy | macro F1 | mean confidence | mean true probability | mean ordinal error | off-by-one-or-correct | severe ordinal errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ["train", "val", "test"]:
        summary = split_summaries[split_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    split_name,
                    str(summary["count"]),
                    fmt(summary["accuracy"]),
                    fmt(summary["macro_f1"]),
                    fmt(summary["mean_confidence"]),
                    fmt(summary["mean_true_probability"]),
                    fmt(summary["mean_ordinal_error"]),
                    fmt(summary["off_by_one_or_correct_rate"]),
                    fmt(summary["severe_ordinal_error_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Test Confusion Matrix",
            "",
            "Rows are true classes; columns are predicted classes.",
            "",
            "| true \\ predicted | intact | minor | moderate | severe |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_name, row in zip(COARSE_CLASS_NAMES, test_matrix):
        lines.append(
            f"| {class_name} | " + " | ".join(str(value) for value in row) + " |"
        )

    lines.extend(
        [
            "",
            "## Test One-vs-Rest Class Metrics",
            "",
            "| class | support | predicted | TP | FP | TN | FN | precision | recall | specificity | FPR | FNR | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_class_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["class"]),
                    str(row["support"]),
                    str(row["predicted_count"]),
                    str(row["tp"]),
                    str(row["fp"]),
                    str(row["tn"]),
                    str(row["fn"]),
                    fmt(row["precision"]),
                    fmt(row["recall_sensitivity"]),
                    fmt(row["specificity"]),
                    fmt(row["false_positive_rate"]),
                    fmt(row["false_negative_rate"]),
                    fmt(row["f1"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Test Distribution",
            "",
            f"- Target distribution: `{dict(target_distribution)}`",
            f"- Prediction distribution: `{dict(prediction_distribution)}`",
            f"- High-confidence correct predictions (`confidence >= 0.70`): `{len(high_conf_correct)}`",
            f"- High-confidence wrong predictions (`confidence >= 0.70`): `{len(high_conf_wrong)}`",
            "",
            "## Main Diagnosis",
            "",
            "- Treat this section as diagnostic evidence, not a pass/fail claim.",
            "- Compare the prediction distribution against the target distribution to detect class collapse or domain-shift bias.",
            "- Use the one-vs-rest rows to identify which classes have low recall, low precision, or both.",
            "- High-confidence wrong predictions are the first examples to inspect visually because they show confident failure modes.",
            "- For synthetic-to-real evaluation, low real-domain accuracy is evidence of domain shift unless a real-trained baseline on the same split proves otherwise.",
            "",
            "## Output Files",
            "",
            "- `split_summary.csv`",
            "- `test_confusion_matrix.csv`",
            "- `test_per_class_tp_fp_tn_fn.csv`",
            "- `test_prediction_examples.csv`",
            "- `test_high_confidence_errors.csv`",
            "- `test_confidence_bins.csv`",
            "- `test_source_metrics.csv`",
            "- `test_score_band_metrics.csv`",
            "- `test_feature_slice_metrics.csv`",
            "- `test_high_confidence_errors_contact_sheet.jpg`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    checkpoint_path = args.checkpoint or run_dir / "checkpoints" / "best.pt"
    output_dir = args.output_dir or run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(run_dir)
    manifest_path = resolve_manifest_path(
        run_config,
        manifest_override=args.manifest,
    )
    image_size = int(run_config.get("image_size", 224))
    random_seed = int(run_config.get("random_seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = pd.read_csv(manifest_path)
    include_resize = infer_resize_policy(frame, image_size=image_size)
    indices_by_split = split_indices(
        manifest_path,
        random_seed,
        row_count=len(frame),
        all_as_test=args.all_as_test,
        train_fraction=float(run_config.get("train_fraction", 0.6)),
        val_fraction=float(run_config.get("val_fraction", 0.2)),
        test_fraction=float(run_config.get("test_fraction", 0.2)),
        split_strategy=str(run_config.get("split_strategy", "stratified_random")),
        split_group_column=(
            str(run_config["split_group_column"])
            if run_config.get("split_group_column") is not None
            else None
        ),
    )
    model = load_model(
        checkpoint_path=checkpoint_path,
        run_config=run_config,
        device=device,
    )

    all_examples = predict_examples(
        model=model,
        frame=frame,
        indices_by_split=indices_by_split,
        image_size=image_size,
        include_resize=include_resize,
        device=device,
        target_label_mode=str(run_config.get("target_label_mode", "coarse")),
    )
    examples_by_split = {
        split_name: [
            example for example in all_examples if example.split == split_name
        ]
        for split_name in indices_by_split
    }
    test_examples = examples_by_split["test"]
    split_summaries = {
        split_name: summarize_split(split_examples)
        for split_name, split_examples in examples_by_split.items()
    }

    write_csv(
        output_dir / "split_summary.csv",
        [
            {"split": split_name, **summary}
            for split_name, summary in split_summaries.items()
        ],
    )
    write_confusion_csv(
        output_dir / "test_confusion_matrix.csv",
        confusion_matrix(test_examples),
    )
    write_csv(
        output_dir / "test_per_class_tp_fp_tn_fn.csv",
        per_class_rows(test_examples),
    )
    write_csv(
        output_dir / "test_prediction_examples.csv",
        example_rows(test_examples),
    )
    write_csv(
        output_dir / "test_high_confidence_errors.csv",
        top_error_rows(test_examples),
    )
    write_csv(
        output_dir / "test_confidence_bins.csv",
        confidence_bins(test_examples),
    )
    write_csv(
        output_dir / "test_source_metrics.csv",
        aggregate_group(test_examples, "source"),
    )
    write_csv(
        output_dir / "test_score_band_metrics.csv",
        aggregate_group(test_examples, "score_band"),
    )
    write_csv(
        output_dir / "test_feature_slice_metrics.csv",
        feature_slice_rows(frame=frame, examples=test_examples),
    )
    write_error_contact_sheet(
        output_path=output_dir / "test_high_confidence_errors_contact_sheet.jpg",
        examples=test_examples,
    )
    write_markdown_report(
        output_path=output_dir / "diagnostic_report.md",
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        split_summaries=split_summaries,
        test_examples=test_examples,
    )

    print(f"Diagnostics written to: {output_dir}")
    print(json.dumps(split_summaries["test"], indent=2))


if __name__ == "__main__":
    main()
