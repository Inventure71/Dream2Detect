"""
predict.py
==========
Run any trained model on one or more images.
Automatically detects model type (dual or band) from the checkpoint.

Works with all Dream2Detect models:
  - EMD          (band mode)
  - SAM          (band mode) — pre-crop images with SAM before passing
  - BaseModel    (dual mode)
  - Dual         (dual mode)

Usage
-----
# Single image:
python predict.py --checkpoint outputs/EMD/best_model.pt \
                  --image /path/to/photo.jpg

# Directory of images:
python predict.py --checkpoint outputs/EMD/best_model.pt \
                  --image_dir /path/to/photos/ \
                  --output_csv results.csv

# With visualisation:
python predict.py --checkpoint outputs/EMD/best_model.pt \
                  --image /path/to/photo.jpg \
                  --visualize
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent))
from src.model import BoxDamageModel
from src.dataset import IMG_MEAN, IMG_STD, COARSE_CLASSES

# ── Band definitions (mirrors dataset.py) ────────────────────────────────────
BAND_RANGES = [
    ("0-10",    0,   10), ("11-20",  11,  20), ("21-30",  21,  30),
    ("31-35",  31,   35), ("36-45",  36,  45), ("46-55",  46,  55),
    ("56-65",  56,   65), ("66-75",  66,  75), ("76-85",  76,  85),
    ("86-100", 86,  100),
]
BAND_NAMES = [b[0] for b in BAND_RANGES]
NUM_BANDS  = len(BAND_RANGES)

# Coarse class mapping from band index
BAND_TO_COARSE = {
    0: "intact",                          # 0-10
    1: "minor",  2: "minor",             # 11-20, 21-30
    3: "minor",                           # 31-35
    4: "moderate", 5: "moderate",        # 36-45, 46-55
    6: "moderate",                        # 56-65
    7: "severe", 8: "severe", 9: "severe" # 66-75, 76-85, 86-100
}

SEVERITY_MAP = {
    "intact":   "✅ Intact — no visible damage",
    "minor":    "🟡 Minor — cosmetic damage only",
    "moderate": "🟠 Moderate — noticeable structural damage",
    "severe":   "🔴 Severe — significant damage / likely contents affected",
}


# ── Transform ─────────────────────────────────────────────────────────────────
def build_transform(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


def build_flip_transform(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device):
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
    label_mode = args["label_mode"]
    print(f"[model] loaded: {checkpoint_path}")
    print(f"        mode={label_mode}  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model, label_mode


# ── Core prediction ───────────────────────────────────────────────────────────
@torch.no_grad()
def predict_image(model, label_mode, img_np, transform, flip_tfm, device):
    """
    Predict a single image. Handles both dual and band mode models.
    TTA: averages original + horizontal flip.
    Returns a result dict with band, score, coarse class and severity.
    """
    t1 = transform(image=img_np)["image"].unsqueeze(0).to(device)
    t2 = flip_tfm(image=img_np)["image"].unsqueeze(0).to(device)

    out1 = model(t1)
    out2 = model(t2)

    if label_mode == "dual":
        # dual mode: output is dict with "score" (0-1) and "coarse" logits
        score1 = out1["score"].item() * 100
        score2 = out2["score"].item() * 100
        avg_score = (score1 + score2) / 2

        avg_coarse = (
            torch.softmax(out1["coarse"], dim=-1) +
            torch.softmax(out2["coarse"], dim=-1)
        ) / 2
        coarse_idx   = avg_coarse.argmax(dim=-1).item()
        coarse_probs = avg_coarse.squeeze().tolist()
        coarse_class = COARSE_CLASSES[coarse_idx]

        # convert score to band
        pred_band_idx = 0
        for i, (_, lo, hi) in enumerate(BAND_RANGES):
            if lo <= avg_score <= hi:
                pred_band_idx = i
                break

    elif label_mode == "band":
        # band mode: output is raw logits over 10 bands
        p1 = torch.softmax(out1, dim=-1).squeeze().cpu().numpy()
        p2 = torch.softmax(out2, dim=-1).squeeze().cpu().numpy()
        band_probs    = (p1 + p2) / 2
        pred_band_idx = int(np.argmax(band_probs))

        # derive score from band midpoint
        _, lo, hi  = BAND_RANGES[pred_band_idx]
        avg_score  = (lo + hi) / 2

        # derive coarse from band
        coarse_class = BAND_TO_COARSE[pred_band_idx]
        coarse_idx   = COARSE_CLASSES.index(coarse_class)
        coarse_probs = [0.0] * 4
        coarse_probs[coarse_idx] = 1.0

    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    return {
        "damage_score":   round(avg_score, 1),
        "band":           BAND_NAMES[pred_band_idx],
        "band_idx":       pred_band_idx,
        "coarse_class":   coarse_class,
        "coarse_probs":   {c: round(p, 4) for c, p in zip(COARSE_CLASSES, coarse_probs)},
        "severity_label": SEVERITY_MAP[coarse_class],
    }


# ── Visualisation ─────────────────────────────────────────────────────────────
def visualize_prediction(image_path: str, result: dict, save_path: str = None):
    import matplotlib.pyplot as plt

    colour_map = {
        "intact":   "#22c55e",
        "minor":    "#eab308",
        "moderate": "#f97316",
        "severe":   "#ef4444",
    }
    col = colour_map[result["coarse_class"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    img = Image.open(image_path).convert("RGB")
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(
        f'{Path(image_path).name}\n{result["severity_label"]}',
        fontsize=10
    )

    classes = list(result["coarse_probs"].keys())
    probs   = list(result["coarse_probs"].values())
    colours = [colour_map[c] for c in classes]
    axes[1].barh(classes, probs, color=colours)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title(
        f'Damage Score: {result["damage_score"]:.1f} / 100\n'
        f'Band: {result["band"]}  |  Class: {result["coarse_class"].upper()}',
        fontsize=11, fontweight="bold", color=col
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[viz] saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run any Dream2Detect model on images"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint e.g. outputs/EMD/best_model.pt")
    p.add_argument("--image",      default=None,
                   help="Single image path")
    p.add_argument("--image_dir",  default=None,
                   help="Directory of images")
    p.add_argument("--output_csv", default=None,
                   help="Save results to CSV")
    p.add_argument("--visualize",  action="store_true",
                   help="Show/save prediction visualisation")
    p.add_argument("--no_tta",     action="store_true",
                   help="Disable test-time augmentation (faster but less accurate)")
    p.add_argument("--img_size",   type=int, default=384)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[device] {device}")

    model, label_mode = load_model(args.checkpoint, device)
    transform  = build_transform(args.img_size)
    flip_tfm   = build_flip_transform(args.img_size)

    # if TTA disabled use same transform for both passes (no flip)
    if args.no_tta:
        flip_tfm = transform

    # collect image paths
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        d = Path(args.image_dir)
        image_paths = sorted(
            str(f) for f in d.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

    if not image_paths:
        print("[error] Provide --image or --image_dir")
        return

    results = []
    for img_path in image_paths:
        try:
            img_np = np.array(Image.open(img_path).convert("RGB"))
            result = predict_image(
                model, label_mode, img_np, transform, flip_tfm, device
            )
            result["image"] = img_path
            results.append(result)

            print(f"\n{'─'*55}")
            print(f"  Image    : {Path(img_path).name}")
            print(f"  Band     : {result['band']}")
            print(f"  Score    : {result['damage_score']:.1f} / 100")
            print(f"  Class    : {result['coarse_class'].upper()}")
            print(f"  {result['severity_label']}")
            print(f"  Probs    : " +
                  "  ".join(f"{k}={v:.3f}" for k, v in result["coarse_probs"].items()))

            if args.visualize:
                viz_path = Path(img_path).stem + "_prediction.png"
                visualize_prediction(img_path, result, save_path=viz_path)

        except Exception as e:
            print(f"[skip] {img_path} — {e}")
            continue

    if args.output_csv and results:
        with open(args.output_csv, "w", newline="") as f:
            fieldnames = [
                "image", "band", "damage_score",
                "coarse_class", "severity_label"
            ] + [f"prob_{c}" for c in COARSE_CLASSES]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "image":          r["image"],
                    "band":           r["band"],
                    "damage_score":   r["damage_score"],
                    "coarse_class":   r["coarse_class"],
                    "severity_label": r["severity_label"],
                }
                for c in COARSE_CLASSES:
                    row[f"prob_{c}"] = r["coarse_probs"].get(c, 0.0)
                writer.writerow(row)
        print(f"\n[csv] results saved → {args.output_csv}")


if __name__ == "__main__":
    main()
