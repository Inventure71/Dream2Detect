"""
evaluate.py
===========
Run the trained model against a labeled manifest and report accuracy metrics.
Reports both 4-class coarse accuracy AND 10-band accuracy (exact / ±1 / ±2 / ±3 / ±4)
directly comparable to the Dream2Detect benchmark table.

Usage
-----
python evaluate.py \
    --checkpoint outputs/v0_emd/best_model.pt \
    --manifest   dataset/real/manifest.csv \
    --images_dir dataset/real \
    --output_dir outputs/v0_emd_real_eval

Outputs
-------
- Console report: 4-class accuracy + 10-band benchmark table
- outputs/v0_emd_real_eval/results.csv          : per-image predictions vs ground truth
- outputs/v0_emd_real_eval/confusion_matrix.png : 4-class confusion matrix
- outputs/v0_emd_real_eval/band_error_dist.png  : band error distribution
- outputs/v0_emd_real_eval/score_scatter.png    : predicted vs actual score scatter
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).parent))
from src.model import BoxDamageModel
from src.dataset import IMG_MEAN, IMG_STD, COARSE_CLASSES


# ── band definitions ──────────────────────────────────────────────────────────
BAND_RANGES = [
    ("0-10",    0,   10),
    ("11-20",  11,   20),
    ("21-30",  21,   30),
    ("31-35",  31,   35),
    ("36-45",  36,   45),
    ("46-55",  46,   55),
    ("56-65",  56,   65),
    ("66-75",  66,   75),
    ("76-85",  76,   85),
    ("86-100", 86,  100),
]
BAND_NAMES = [b[0] for b in BAND_RANGES]

# ── band → coarse mapping ─────────────────────────────────────────────────────
# intact=0-10, minor=11-35, moderate=36-65, severe=66-100
BAND_TO_COARSE = [
    0,  # 0-10   → intact
    1,  # 11-20  → minor
    1,  # 21-30  → minor
    1,  # 31-35  → minor
    2,  # 36-45  → moderate
    2,  # 46-55  → moderate
    2,  # 56-65  → moderate
    3,  # 66-75  → severe
    3,  # 76-85  → severe
    3,  # 86-100 → severe
]


def band_idx_to_coarse(band_idx: int) -> str:
    return COARSE_CLASSES[BAND_TO_COARSE[band_idx]]


def score_to_band_idx(score: float) -> int:
    for i, (name, lo, hi) in enumerate(BAND_RANGES):
        if lo <= score <= hi:
            return i
    return len(BAND_RANGES) - 1


def band_name_to_idx(band_name: str) -> int:
    for i, (name, lo, hi) in enumerate(BAND_RANGES):
        if name == band_name:
            return i
    return -1


# ── transform ─────────────────────────────────────────────────────────────────
def build_eval_transform(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


# ── load model ────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> BoxDamageModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = BoxDamageModel(
        backbone_name=args["backbone"],
        label_mode=args["label_mode"],
        dropout=0.0,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] loaded from {checkpoint_path}")
    print(f"        trained for {ckpt['epoch']} epochs  |  "
          f"val_loss={ckpt['val_loss']:.4f}")
    return model


# ── predict single image with TTA ─────────────────────────────────────────────
@torch.no_grad()
def predict_image(model, image_path: str, transform, device):
    img_np = np.array(Image.open(image_path).convert("RGB"))

    flip_tfm = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])

    t1 = transform(image=img_np)["image"].unsqueeze(0).to(device)
    t2 = flip_tfm(image=img_np)["image"].unsqueeze(0).to(device)

    out1 = model(t1)
    out2 = model(t2)

    if isinstance(out1, dict):
        # dual mode — score + coarse heads
        avg_score    = (out1["score"].item() + out2["score"].item()) / 2
        damage_score = avg_score * 100
        avg_coarse   = (
            torch.softmax(out1["coarse"], dim=-1) +
            torch.softmax(out2["coarse"], dim=-1)
        ) / 2
        coarse_idx   = avg_coarse.argmax(dim=-1).item()
        coarse_probs = avg_coarse.squeeze().tolist()
        pred_band_idx = score_to_band_idx(damage_score)
    elif out1.shape[-1] == 10:
        # band mode — direct 10-class output
        avg_band      = (torch.softmax(out1, dim=-1) +
                         torch.softmax(out2, dim=-1)) / 2
        pred_band_idx = int(avg_band.argmax(dim=-1).item())
        lo            = BAND_RANGES[pred_band_idx][1]
        hi            = BAND_RANGES[pred_band_idx][2]
        damage_score  = (lo + hi) / 2
        coarse_idx    = BAND_TO_COARSE[pred_band_idx]
        coarse_probs  = [0.0] * 4
        coarse_probs[coarse_idx] = 1.0
    else:
        # score-only mode
        damage_score  = out1.item() * 100
        coarse_idx    = 0
        coarse_probs  = [0.0] * 4
        pred_band_idx = score_to_band_idx(damage_score)

    return {
        "damage_score":    round(damage_score, 1),
        "pred_coarse":     COARSE_CLASSES[coarse_idx],
        "pred_coarse_idx": coarse_idx,
        "pred_band":       BAND_NAMES[pred_band_idx],
        "pred_band_idx":   pred_band_idx,
        "probs":           {c: round(p, 4) for c, p in zip(COARSE_CLASSES, coarse_probs)},
    }


# ── band benchmark table ───────────────────────────────────────────────────────
def print_band_benchmark(true_band_idxs, pred_band_idxs, model_name="Your Model"):
    true_arr = np.array(true_band_idxs)
    pred_arr = np.array(pred_band_idxs)
    diff     = np.abs(true_arr - pred_arr)
    n        = len(diff)

    exact = (diff == 0).mean() * 100
    pm1   = (diff <= 1).mean() * 100
    pm2   = (diff <= 2).mean() * 100
    pm3   = (diff <= 3).mean() * 100
    pm4   = (diff <= 4).mean() * 100
    mbe   = diff.mean()

    print(f"\n{'='*72}")
    print(f"  10-BAND BENCHMARK  (n={n} images)")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Exact':>8} {'±1':>8} {'±2':>8} {'±3':>8} {'±4':>8} {'MBE':>9}")
    print(f"  {'-'*70}")
    print(f"  {model_name:<22} {exact:>7.2f}% {pm1:>7.2f}% {pm2:>7.2f}% {pm3:>7.2f}% {pm4:>7.2f}% {mbe:>9.4f}")
    print(f"  {'─'*70}")
    print(f"  Friend's models (real images):")
    comparisons = [
        ("V4 locked",     10.65, 33.25, 51.95, 69.87, 82.34, 2.6649),
        ("V4.5 EMD",      12.73, 35.32, 53.51, 70.65, 85.71, 2.5351),
        ("V4.5 SAM crop",  7.79, 24.68, 45.19, 63.64, 79.22, 2.9558),
        ("V5-A scalar",   13.51, 39.74, 58.96, 75.84, 89.35, 2.2727),
    ]
    for name, e, p1, p2, p3, p4, mb in comparisons:
        print(f"  {name:<22} {e:>7.2f}% {p1:>7.2f}% {p2:>7.2f}% {p3:>7.2f}% {p4:>7.2f}% {mb:>9.4f}")
    print(f"{'='*72}")

    return {"exact": exact, "pm1": pm1, "pm2": pm2,
            "pm3": pm3, "pm4": pm4, "mbe": mbe}


# ── main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--manifest",    required=True)
    p.add_argument("--images_dir",  required=True)
    p.add_argument("--output_dir",  default="outputs/v0_eval")
    p.add_argument("--img_size",    type=int, default=384)
    p.add_argument("--max_images",  type=int, default=None)
    p.add_argument("--model_name",  default="Your Model",
                   help="Name shown in the benchmark table e.g. 'ConvNeXt-S dual'")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[device] {device}")

    model     = load_model(args.checkpoint, device)
    transform = build_eval_transform(args.img_size)

    # ── load manifest ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    images_root = Path(args.images_dir)

    if "image_path" in df.columns:
        df["full_path"] = df["image_path"].apply(
            lambda p: str(images_root / p)
        )
    else:
        df["full_path"] = df.apply(
            lambda r: str(images_root / "images" / r.get("filename", "")),
            axis=1
        )

    df = df[df["full_path"].apply(lambda p: Path(p).exists())].copy()
    df = df.reset_index(drop=True)

    if args.max_images:
        df = df.head(args.max_images)

    print(f"\n[eval] {len(df)} images to evaluate")
    if len(df) == 0:
        print("[error] No images found — check your paths")
        return

    # ── run predictions ────────────────────────────────────────────────────────
    results = []
    for i, row in df.iterrows():
        try:
            pred = predict_image(model, row["full_path"], transform, device)
        except Exception as e:
            print(f"  [skip] {row['full_path']} — {e}")
            continue

        true_coarse   = row.get("training_coarse_class", "unknown")
        true_score    = float(row.get("training_representative_score", -1))
        true_band     = row.get("training_score_band", "")
        true_band_idx = band_name_to_idx(true_band) if true_band else \
                        score_to_band_idx(true_score) if true_score >= 0 else -1

        coarse_correct = true_coarse == pred["pred_coarse"]
        band_diff      = abs(true_band_idx - pred["pred_band_idx"]) \
                         if true_band_idx >= 0 else -1

        results.append({
            "image":          row["full_path"],
            "true_coarse":    true_coarse,
            "true_score":     true_score,
            "true_band":      true_band,
            "true_band_idx":  true_band_idx,
            "pred_coarse":    pred["pred_coarse"],
            "pred_score":     pred["damage_score"],
            "pred_band":      pred["pred_band"],
            "pred_band_idx":  pred["pred_band_idx"],
            "coarse_correct": coarse_correct,
            "band_diff":      band_diff,
            **{f"prob_{c}": pred["probs"].get(c, 0.0) for c in COARSE_CLASSES},
        })

        status = "✓" if coarse_correct else "✗"
        print(f"  [{i+1:3d}/{len(df)}] {status}  "
              f"true={true_coarse:8s}({true_band:6s})  "
              f"pred={pred['pred_coarse']:8s}({pred['pred_band']:6s})  "
              f"score={pred['damage_score']:5.1f}  "
              f"±{band_diff}bands  "
              f"{Path(row['full_path']).name}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    valid = results_df[
        (results_df["true_coarse"] != "unknown") &
        (results_df["true_band_idx"] >= 0)
    ].copy()

    # ── 4-class accuracy ───────────────────────────────────────────────────────
    acc = valid["coarse_correct"].mean()
    print(f"\n{'='*72}")
    print(f"  4-CLASS COARSE ACCURACY")
    print(f"{'='*72}")
    print(f"  Overall : {acc:.1%}  ({valid['coarse_correct'].sum()}/{len(valid)} correct)")
    print(f"\n  Per-Class:")
    for cls in COARSE_CLASSES:
        subset = valid[valid["true_coarse"] == cls]
        if len(subset) == 0:
            continue
        cls_acc = subset["coarse_correct"].mean()
        print(f"    {cls:10s} : {cls_acc:.1%}  "
              f"({int(subset['coarse_correct'].sum())}/{len(subset)})")

    print(f"\n  Classification Report:")
    print(classification_report(
        valid["true_coarse"], valid["pred_coarse"],
        labels=COARSE_CLASSES, zero_division=0,
    ))

    # ── 10-band benchmark ──────────────────────────────────────────────────────
    band_valid   = valid[valid["true_band_idx"] >= 0]
    band_metrics = print_band_benchmark(
        band_valid["true_band_idx"].tolist(),
        band_valid["pred_band_idx"].tolist(),
        model_name=args.model_name,
    )

    # ── score MAE ─────────────────────────────────────────────────────────────
    if valid["true_score"].notna().any():
        mae = np.abs(valid["pred_score"] - valid["true_score"]).mean()
        print(f"\n  Score MAE: {mae:.1f} points on 0-100 scale")

    # ── confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(
        valid["true_coarse"], valid["pred_coarse"], labels=COARSE_CLASSES
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=COARSE_CLASSES).plot(
        ax=ax, colorbar=True, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {acc:.1%} Coarse Accuracy", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close()

    # ── band error distribution ────────────────────────────────────────────────
    band_diffs = band_valid["band_diff"].values
    counts     = [(band_diffs == d).sum() for d in range(6)]
    fig, ax    = plt.subplots(figsize=(8, 4))
    colours    = ["#22c55e","#84cc16","#eab308","#f97316","#ef4444","#dc2626"]
    bars       = ax.bar(range(6), counts, color=colours)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Exact", "±1", "±2", "±3", "±4", "±5+"])
    ax.set_ylabel("Number of images")
    ax.set_title(f"Band Error Distribution  (MBE={band_metrics['mbe']:.4f})")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3, str(count),
                ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "band_error_dist.png", dpi=130)
    plt.close()

    # ── score scatter ──────────────────────────────────────────────────────────
    colour_map = {"intact": "#22c55e", "minor": "#eab308",
                  "moderate": "#f97316", "severe": "#ef4444"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in COARSE_CLASSES:
        subset = valid[valid["true_coarse"] == cls]
        ax.scatter(subset["true_score"], subset["pred_score"],
                   c=colour_map[cls], label=cls, alpha=0.7, s=40)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="perfect")
    ax.set_xlabel("True Score"); ax.set_ylabel("Predicted Score")
    ax.set_title("Predicted vs True Score")
    ax.legend(); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "score_scatter.png", dpi=130)
    plt.close()

    print(f"\n  Outputs saved to {out_dir}/")
    print(f"    confusion_matrix.png")
    print(f"    band_error_dist.png")
    print(f"    score_scatter.png")
    print(f"    results.csv")


if __name__ == "__main__":
    main()
