"""
evaluate_ensemble.py
====================
Combine two trained models and evaluate the ensemble on a manifest.
Averages predicted scores/probabilities from both models before
converting to bands — consistently beats any single model.

Best combinations to try:
- ConvNeXt-S dual  +  ConvNeXt-S band   (score + direct band)
- ConvNeXt-S dual  +  ConvNeXt-S EMD    (dual + EMD-calibrated)
- ConvNeXt-S band  +  ConvNeXt-Base dual (different sizes)

Usage
-----
python evaluate_ensemble.py \
    --checkpoint_a outputs/run_03/best_model.pt \
    --checkpoint_b outputs/band_01/best_model.pt \
    --manifest     outputs/run_03/manifest.csv \
    --images_dir   outputs/run_03/ \
    --output_dir   outputs/eval_ensemble_01 \
    --model_name   "Ensemble dual+band"
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
    ("0-10",    0,   10), ("11-20",  11,  20), ("21-30",  21,  30),
    ("31-35",  31,   35), ("36-45",  36,  45), ("46-55",  46,  55),
    ("56-65",  56,   65), ("66-75",  66,  75), ("76-85",  76,  85),
    ("86-100", 86,  100),
]
BAND_NAMES = [b[0] for b in BAND_RANGES]
NUM_BANDS  = len(BAND_RANGES)


def score_to_band_idx(score: float) -> int:
    for i, (_, lo, hi) in enumerate(BAND_RANGES):
        if lo <= score <= hi:
            return i
    return NUM_BANDS - 1


def band_name_to_idx(name: str) -> int:
    for i, (n, _, _) in enumerate(BAND_RANGES):
        if n == name:
            return i
    return -1


def build_transform(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def build_flip_transform(img_size=384):
    return A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    margs = ckpt["args"]
    model = BoxDamageModel(
        backbone_name=margs["backbone"],
        label_mode=margs["label_mode"],
        dropout=0.0,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  loaded: {checkpoint_path}  "
          f"mode={margs['label_mode']}  epoch={ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.4f}")
    return model, margs["label_mode"]


@torch.no_grad()
def get_band_probs(model, label_mode, image_np, transform, flip_tfm, device):
    """
    Return a (NUM_BANDS,) probability array for a single image.
    Works for dual (score→band) and band (direct) models.
    TTA: average of original + horizontal flip.
    """
    t1 = transform(image=image_np)["image"].unsqueeze(0).to(device)
    t2 = flip_tfm(image=image_np)["image"].unsqueeze(0).to(device)

    out1 = model(t1)
    out2 = model(t2)

    if label_mode == "dual":
        # convert continuous score to a soft band distribution
        score1 = out1["score"].item() * 100
        score2 = out2["score"].item() * 100
        avg_score = (score1 + score2) / 2

        # build soft distribution: peak at predicted band, decay adjacents
        band_probs = np.zeros(NUM_BANDS)
        pred_idx   = score_to_band_idx(avg_score)
        for i in range(NUM_BANDS):
            band_probs[i] = np.exp(-0.5 * (i - pred_idx) ** 2)
        band_probs /= band_probs.sum()

    elif label_mode == "band":
        p1 = torch.softmax(out1, dim=-1).squeeze().cpu().numpy()
        p2 = torch.softmax(out2, dim=-1).squeeze().cpu().numpy()
        band_probs = (p1 + p2) / 2

    else:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    return band_probs


def print_benchmark(true_idxs, pred_idxs, model_name, n):
    diff  = np.abs(np.array(true_idxs) - np.array(pred_idxs))
    exact = (diff == 0).mean() * 100
    pm1   = (diff <= 1).mean() * 100
    pm2   = (diff <= 2).mean() * 100
    pm3   = (diff <= 3).mean() * 100
    pm4   = (diff <= 4).mean() * 100
    mbe   = diff.mean()

    print(f"\n{'='*72}")
    print(f"  10-BAND BENCHMARK  (n={n})")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Exact':>8} {'±1':>8} {'±2':>8} {'±3':>8} {'±4':>8} {'MBE':>9}")
    print(f"  {'-'*70}")
    print(f"  {model_name:<22} {exact:>7.2f}% {pm1:>7.2f}% {pm2:>7.2f}% {pm3:>7.2f}% {pm4:>7.2f}% {mbe:>9.4f}")
    print(f"  {'─'*70}")
    print(f"  Friend's models (real images):")
    for name, e, p1, p2, p3, p4, mb in [
        ("V4 locked",     10.65, 33.25, 51.95, 69.87, 82.34, 2.6649),
        ("V4.5 EMD",      12.73, 35.32, 53.51, 70.65, 85.71, 2.5351),
        ("V4.5 SAM crop",  7.79, 24.68, 45.19, 63.64, 79.22, 2.9558),
        ("V5-A scalar",   13.51, 39.74, 58.96, 75.84, 89.35, 2.2727),
        ("ConvNeXt-S dual",15.58, 52.47, 78.70, 90.65, 95.06, 1.7039),
    ]:
        print(f"  {name:<22} {e:>7.2f}% {p1:>7.2f}% {p2:>7.2f}% {p3:>7.2f}% {p4:>7.2f}% {mb:>9.4f}")
    print(f"{'='*72}")
    return {"exact": exact, "pm1": pm1, "pm2": pm2,
            "pm3": pm3, "pm4": pm4, "mbe": mbe}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_a", required=True, help="First model checkpoint")
    p.add_argument("--checkpoint_b", required=True, help="Second model checkpoint")
    p.add_argument("--manifest",     required=True)
    p.add_argument("--images_dir",   required=True)
    p.add_argument("--output_dir",   default="outputs/eval_ensemble_01")
    p.add_argument("--model_name",   default="Ensemble")
    p.add_argument("--weight_a",     type=float, default=0.5,
                   help="Weight for model A (model B gets 1-weight_a)")
    p.add_argument("--img_size",     type=int, default=384)
    p.add_argument("--images_dir_a", default=None,
                   help="Images dir for model A. Defaults to --images_dir if not set.")
    p.add_argument("--images_dir_b", default=None,
                   help="Images dir for model B. Defaults to --images_dir if not set.")
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
    print(f"[models]")
    model_a, mode_a = load_model(args.checkpoint_a, device)
    model_b, mode_b = load_model(args.checkpoint_b, device)
    print(f"[weights] A={args.weight_a:.2f}  B={1-args.weight_a:.2f}")

    # resolve images dirs — fall back to --images_dir if not specified
    images_root_a = Path(args.images_dir_a) if args.images_dir_a else Path(args.images_dir)
    images_root_b = Path(args.images_dir_b) if args.images_dir_b else Path(args.images_dir)
    print(f"[images] A={images_root_a}  B={images_root_b}")

    transform  = build_transform(args.img_size)
    flip_tfm   = build_flip_transform(args.img_size)

    df = pd.read_csv(args.manifest)
    images_root = Path(args.images_dir)
    if "image_path" in df.columns:
        df["full_path"] = df["image_path"].apply(lambda p: str(images_root / p))
        df["full_path_a"] = df["image_path"].apply(lambda p: str(images_root_a / p))
        df["full_path_b"] = df["image_path"].apply(lambda p: str(images_root_b / p))
    df = df[df["full_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"\n[eval] {len(df)} images")

    results = []
    for i, row in df.iterrows():
        try:

            img_np_a = np.array(Image.open(row["full_path_a"]).convert("RGB"))
            img_np_b = np.array(Image.open(row["full_path_b"]).convert("RGB"))

            probs_a = get_band_probs(model_a, mode_a, img_np_a,
                                     transform, flip_tfm, device)
            probs_b = get_band_probs(model_b, mode_b, img_np_b,
                                     transform, flip_tfm, device)

            # weighted ensemble
            ensemble_probs = args.weight_a * probs_a + (1 - args.weight_a) * probs_b
            pred_band_idx  = int(np.argmax(ensemble_probs))
            pred_score     = (BAND_RANGES[pred_band_idx][1] +
                              BAND_RANGES[pred_band_idx][2]) / 2

        except Exception as e:
            print(f"  [skip] {row['full_path']} — {e}")
            continue

        true_coarse   = row.get("training_coarse_class", "unknown")
        true_score    = float(row.get("training_representative_score", -1))
        true_band     = row.get("training_score_band", "")
        true_band_idx = band_name_to_idx(true_band) if true_band else \
                        score_to_band_idx(true_score) if true_score >= 0 else -1
        band_diff     = abs(true_band_idx - pred_band_idx) if true_band_idx >= 0 else -1

        results.append({
            "image":         row["full_path"],
            "true_coarse":   true_coarse,
            "true_score":    true_score,
            "true_band":     true_band,
            "true_band_idx": true_band_idx,
            "pred_band":     BAND_NAMES[pred_band_idx],
            "pred_band_idx": pred_band_idx,
            "pred_score":    pred_score,
            "band_diff":     band_diff,
        })

        print(f"  [{i+1:3d}/{len(df)}]  "
              f"true={true_band:6s}  pred={BAND_NAMES[pred_band_idx]:6s}  "
              f"±{band_diff}  {Path(row['full_path']).name}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    valid = results_df[results_df["true_band_idx"] >= 0].copy()
    metrics = print_benchmark(
        valid["true_band_idx"].tolist(),
        valid["pred_band_idx"].tolist(),
        model_name=args.model_name,
        n=len(valid),
    )

    if valid["true_score"].notna().any():
        mae = np.abs(valid["pred_score"] - valid["true_score"]).mean()
        print(f"\n  Score MAE: {mae:.1f} points")

    # band error distribution
    band_diffs = valid["band_diff"].values
    counts     = [(band_diffs == d).sum() for d in range(6)]
    fig, ax    = plt.subplots(figsize=(8, 4))
    colours    = ["#22c55e","#84cc16","#eab308","#f97316","#ef4444","#dc2626"]
    bars       = ax.bar(range(6), counts, color=colours)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Exact", "±1", "±2", "±3", "±4", "±5+"])
    ax.set_ylabel("Images")
    ax.set_title(f"Ensemble Band Error  (MBE={metrics['mbe']:.4f})")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3, str(c), ha="center")
    plt.tight_layout()
    plt.savefig(out_dir / "band_error_dist.png", dpi=130)
    plt.close()

    print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
