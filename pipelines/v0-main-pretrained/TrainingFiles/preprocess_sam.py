"""
preprocess_sam.py
=================
Use Segment Anything Model (SAM) to automatically detect and crop the
box/package out of each image, removing distracting backgrounds.

Your model currently sees the full image including floors, walls, conveyor
belts etc. SAM cropping removes all of that and gives the model a clean
box-only view — reducing background noise and improving generalisation.

This creates a NEW images folder with cropped versions. You then train
on the cropped images using any of the train_*.py scripts.

Requirements
------------
pip install segment-anything
pip install opencv-python

Also download the SAM checkpoint (ViT-H, ~2.4GB):
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
Place it in the project root or specify with --sam_checkpoint.

Usage
-----
# Step 1: preprocess
python3 pipelines/v0-main-pretrained/TrainingFiles/preprocess_sam.py \
    --manifest dataset/synthetic/manifest.csv \
    --images_dir dataset/synthetic \
    --output_dir outputs/v0_sam_cropped \
    --sam_checkpoint sam_vit_h_4b8939.pth

# Step 2: train on SAM-cropped images (manifest stays the same,
#          just point images_dir to the sam_cropped folder)
python3 pipelines/v0-main-pretrained/TrainingFiles/train_band.py \
    --manifest outputs/v0_sam_cropped/manifest.csv \
    --images_dir outputs/v0_sam_cropped \
    --backbone convnext_small \
    --epochs 60 \
    --output_dir outputs/v0_sam_band \
    --fp16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="SAM preprocessing for box images")
    p.add_argument("--manifest",       required=True)
    p.add_argument("--images_dir",     required=True)
    p.add_argument("--output_dir",     required=True,
                   help="Where to save cropped images (same subfolder structure)")
    p.add_argument("--sam_checkpoint", default="sam_vit_h_4b8939.pth")
    p.add_argument("--model_type",     default="vit_h",
                   choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--pad",            type=float, default=0.05,
                   help="Padding fraction around detected box (0.05 = 5%%)")
    p.add_argument("--img_size",       type=int,   default=384)
    p.add_argument("--fallback_center_crop", action="store_true",
                   help="If SAM fails, use center crop instead of skipping")
    return p.parse_args()


def load_sam(checkpoint_path: str, model_type: str, device: torch.device):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("[error] segment-anything not installed.")
        print("        Run: pip install segment-anything")
        sys.exit(1)

    if not Path(checkpoint_path).exists():
        print(f"[error] SAM checkpoint not found: {checkpoint_path}")
        print(f"        Download from:")
        print(f"        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        sys.exit(1)

    print(f"[sam] loading {model_type} from {checkpoint_path} ...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=5000,
    )
    print(f"[sam] loaded OK")
    return mask_generator


def get_largest_central_mask(masks, img_h, img_w, min_area_frac=0.05):
    """
    From SAM masks, pick the one most likely to be the box:
    - Area > min_area_frac of image
    - Closest to image center
    - Largest among candidates
    """
    cx, cy = img_w / 2, img_h / 2
    img_area = img_h * img_w

    candidates = []
    for mask in masks:
        area = mask["area"]
        if area < img_area * min_area_frac:
            continue
        bbox = mask["bbox"]  # x, y, w, h
        mask_cx = bbox[0] + bbox[2] / 2
        mask_cy = bbox[1] + bbox[3] / 2
        dist = ((mask_cx - cx) ** 2 + (mask_cy - cy) ** 2) ** 0.5
        candidates.append((area, dist, mask))

    if not candidates:
        return None

    # prefer large + central: score = area / (dist + 1)
    candidates.sort(key=lambda x: x[0] / (x[1] + 1), reverse=True)
    return candidates[0][2]


def crop_to_mask(image_np, mask_data, pad_frac, target_size):
    """Crop image to mask bounding box with padding, resize to target_size."""
    x, y, w, h = mask_data["bbox"]
    img_h, img_w = image_np.shape[:2]

    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    cropped = image_np[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)
    return resized


def center_crop_fallback(image_np, target_size):
    """Square center crop as fallback when SAM fails."""
    h, w = image_np.shape[:2]
    size  = min(h, w)
    y1    = (h - size) // 2
    x1    = (w - size) // 2
    cropped = image_np[y1:y1+size, x1:x1+size]
    return cv2.resize(cropped, (target_size, target_size),
                      interpolation=cv2.INTER_LANCZOS4)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    mask_generator = load_sam(args.sam_checkpoint, args.model_type, device)

    df          = pd.read_csv(args.manifest)
    images_root = Path(args.images_dir)
    out_root    = Path(args.output_dir)

    # create output images subfolder mirroring input structure
    out_images  = out_root / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # copy manifest to output dir (unchanged — same labels)
    import shutil
    shutil.copy(args.manifest, out_root / "manifest.csv")
    print(f"[manifest] copied to {out_root}/manifest.csv")

    success = 0
    fallback = 0
    failed  = 0

    for i, row in df.iterrows():
        img_path = images_root / row["image_path"]
        out_path = out_images / Path(row["image_path"]).name

        if not img_path.exists():
            print(f"  [skip] not found: {img_path}")
            failed += 1
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = image_np.shape[:2]

        try:
            masks = mask_generator.generate(image_np)
            best  = get_largest_central_mask(masks, img_h, img_w)

            if best is not None:
                cropped = crop_to_mask(image_np, best, args.pad, args.img_size)
                success += 1
                status  = "✓ sam"
            else:
                raise ValueError("no suitable mask found")

        except Exception as e:
            if args.fallback_center_crop:
                cropped  = center_crop_fallback(image_np, args.img_size)
                fallback += 1
                status   = "~ fallback"
            else:
                print(f"  [fail] {img_path.name}: {e}")
                failed += 1
                continue

        Image.fromarray(cropped).save(out_path)
        print(f"  [{i+1:3d}/{len(df)}] {status}  {img_path.name} → {out_path.name}")

    print(f"\n[done]")
    print(f"  SAM crop success : {success}")
    print(f"  Center fallback  : {fallback}")
    print(f"  Failed/skipped   : {failed}")
    print(f"\nCropped images saved to: {out_images}/")
    print(f"Now train with:")
    print(f"  python train_band.py --manifest {out_root}/manifest.csv "
          f"--images_dir {out_root}/ --epochs 60 "
          f"--output_dir outputs/v0_sam_band --fp16")


if __name__ == "__main__":
    main()
