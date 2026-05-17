"""
train_emd.py
============
Train with Earth Mover's Distance (EMD) loss for 10-band classification.

EMD penalizes predictions proportionally to how far off they are.
Predicting band 5 when truth is band 6 = small penalty.
Predicting band 1 when truth is band 9 = large penalty.

Standard CrossEntropy treats all wrong predictions equally which leads
to high MBE. EMD directly optimises for lower MBE.

Difference from train_band.py:
- Uses EMD loss instead of CrossEntropy
- Directly optimises Mean Band Error

Usage
-----
python train_emd.py \
    --manifest data/manifest.csv \
    --images_dir data/ \
    --backbone convnext_small \
    --epochs 60 \
    --batch_size 16 \
    --output_dir outputs/emd_01 \
    --fp16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import (
    BoxDamageDataset,
    build_train_transform, build_val_transform,
)
from src.model import BoxDamageModel


# ── EMD Loss ──────────────────────────────────────────────────────────────────
class EMDLoss(nn.Module):
    """
    Earth Mover's Distance loss for ordinal classification.
    Converts logits to CDFs and minimises the area between predicted
    and true CDFs — naturally penalises predictions far from the truth.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)

        # build one-hot target
        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        # cumulative distribution functions
        pred_cdf   = torch.cumsum(probs,          dim=1)
        target_cdf = torch.cumsum(target_onehot,  dim=1)

        # EMD = mean squared CDF difference
        emd = torch.mean((pred_cdf - target_cdf) ** 2)
        return emd


def parse_args():
    p = argparse.ArgumentParser(description="Train with EMD loss")
    p.add_argument("--manifest",      required=True)
    p.add_argument("--images_dir",    required=True)
    p.add_argument("--backbone",      default="convnext_small")
    p.add_argument("--img_size",      type=int,   default=384)
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--dropout",       type=float, default=0.35)
    p.add_argument("--val_split",     type=float, default=0.15)
    p.add_argument("--freeze_epochs", type=int,   default=5)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--output_dir",    default="outputs/emd_01")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--fp16",          action="store_true")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = True


def run_epoch(model, loader, optimizer, scheduler, criterion,
              device, scaler, mode="train"):
    is_train = mode == "train"
    model.train(is_train)

    total_loss = 0.0
    all_pred, all_true = [], []

    for images, labels, meta in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=(scaler is not None)
            ):
                preds = model(images)
                loss  = criterion(preds, labels)

        if is_train:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * images.size(0)
        all_pred.extend(preds.argmax(dim=1).cpu().tolist())
        all_true.extend(labels.cpu().tolist())

    n    = len(loader.dataset)
    diff = np.abs(np.array(all_pred) - np.array(all_true))
    return {
        "loss": total_loss / n,
        "acc":  (diff == 0).mean(),
        "pm1":  (diff <= 1).mean(),
        "mbe":  diff.mean(),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[device] {device}")
    print(f"[mode]   10-band classification with EMD loss")

    # ── datasets ──────────────────────────────────────────────────────────────
    full_ds = BoxDamageDataset(
        manifest_path=args.manifest,
        images_root=args.images_dir,
        label_mode="band",
        transform=None,
    )
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=gen)

    train_ds = BoxDamageDataset(
        manifest_path=args.manifest,
        images_root=args.images_dir,
        label_mode="band",
        transform=build_train_transform(args.img_size),
        split_ids=[full_ds.df.iloc[i]["prompt_id"] for i in train_subset.indices],
    )
    val_ds = BoxDamageDataset(
        manifest_path=args.manifest,
        images_root=args.images_dir,
        label_mode="band",
        transform=build_val_transform(args.img_size),
        split_ids=[full_ds.df.iloc[i]["prompt_id"] for i in val_subset.indices],
    )

    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    sampler = train_ds.get_sampler_weights()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=(device_str == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device_str == "cuda"),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = BoxDamageModel(
        backbone_name=args.backbone,
        label_mode="band",
        num_bands=10,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.backbone}  params={total:,}")

    criterion = EMDLoss(num_classes=10)

    # ── optimiser ─────────────────────────────────────────────────────────────
    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda") \
             if (args.fp16 and device_str == "cuda") else None

    # ── training loop ──────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [],
               "train_mbe":  [], "val_mbe":  []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch == args.freeze_epochs + 1:
            print(f"[epoch {epoch}] Unfreezing backbone …")
            unfreeze_backbone(model)
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.lr * 0.05},
                {"params": model.neck.parameters(),     "lr": args.lr * 0.5},
                {"params": model.band_head.parameters(),"lr": args.lr * 0.5},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(args.epochs - epoch + 1) * len(train_loader),
                eta_min=1e-7,
            )

        train_m = run_epoch(model, train_loader, optimizer, scheduler,
                            criterion, device, scaler, mode="train")
        val_m   = run_epoch(model, val_loader, None, None,
                            criterion, device, None,   mode="val")

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(train_m["acc"])
        history["val_acc"].append(val_m["acc"])
        history["train_mbe"].append(train_m["mbe"])
        history["val_mbe"].append(val_m["mbe"])

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  "
              f"val_exact={val_m['acc']:.3f}  val_±1={val_m['pm1']:.3f}  "
              f"val_mbe={val_m['mbe']:.3f}  [{time.time()-t0:.1f}s]")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args) | {"label_mode": "band",
                                      "backbone": args.backbone},
            }, out_dir / "best_model.pt")
            print(f"  ↳ saved best checkpoint → {out_dir}/best_model.pt")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_m["loss"],
        "args": vars(args) | {"label_mode": "band", "backbone": args.backbone},
    }, out_dir / "final_model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("EMD Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Exact Band Accuracy"); axes[1].legend()
    axes[2].plot(history["train_mbe"], label="train")
    axes[2].plot(history["val_mbe"],   label="val")
    axes[2].set_title("Mean Band Error"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=120)
    plt.close()

    print(f"\n[done] best val_loss={best_val_loss:.4f}")
    print(f"Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
