"""
train_convnext_base.py
======================
Train with ConvNeXt Base — the next size up from ConvNeXt Small.
89M params vs 50M. More capacity to learn fine-grained damage patterns.
Uses dual head (score + coarse) same as the base model.

Difference from train.py:
- backbone forced to convnext_base
- batch_size default 8 (larger model needs more VRAM)
- slightly lower lr for stability with larger model

Usage
-----
python3 pipelines/v0-main-pretrained/TrainingFiles/train_convnext_base.py \
    --manifest dataset/synthetic/manifest.csv \
    --images_dir dataset/synthetic \
    --epochs 60 \
    --output_dir outputs/v0_convnext_base \
    --fp16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

V0_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V0_ROOT / "EvaluationFiles"))

from src.dataset import (
    BoxDamageDataset, COARSE_TO_IDX,
    build_train_transform, build_val_transform,
)
from src.model import BoxDamageModel, DualLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train with ConvNeXt Base backbone")
    p.add_argument("--manifest",      required=True)
    p.add_argument("--images_dir",    required=True)
    p.add_argument("--img_size",      type=int,   default=384)
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--val_split",     type=float, default=0.15)
    p.add_argument("--freeze_epochs", type=int,   default=5)
    p.add_argument("--alpha",         type=float, default=0.5)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--output_dir",    default="outputs/v0_convnext_base")
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
    all_coarse_pred, all_coarse_true = [], []
    all_scores_pred, all_scores_true = [], []

    for images, labels, meta in loader:
        images = images.to(device)
        score_targets  = torch.tensor(
            [s / 100.0 for s in meta["score"]], dtype=torch.float32
        ).to(device)
        coarse_targets = torch.tensor(
            [COARSE_TO_IDX[c] for c in meta["coarse"]], dtype=torch.long
        ).to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=(scaler is not None)
            ):
                preds = model(images)
                loss, _, _ = criterion(preds, score_targets, coarse_targets)

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
        all_coarse_pred.extend(preds["coarse"].argmax(dim=1).cpu().tolist())
        all_coarse_true.extend(coarse_targets.cpu().tolist())
        all_scores_pred.extend((preds["score"] * 100).cpu().tolist())
        all_scores_true.extend((score_targets * 100).cpu().tolist())

    n   = len(loader.dataset)
    acc = sum(p == t for p, t in zip(all_coarse_pred, all_coarse_true)) / len(all_coarse_true)
    mae = np.mean(np.abs(np.array(all_scores_pred) - np.array(all_scores_true)))
    return {"loss": total_loss / n, "acc": acc, "mae": mae}


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[device] {device}")
    print(f"[mode]   ConvNeXt Base — dual head (score + coarse)")

    # ── datasets ──────────────────────────────────────────────────────────────
    full_ds = BoxDamageDataset(
        manifest_path=args.manifest,
        images_root=args.images_dir,
        label_mode="dual",
        transform=None,
    )
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=gen)

    train_ds = BoxDamageDataset(
        manifest_path=args.manifest,
        images_root=args.images_dir,
        label_mode="dual",
        transform=build_train_transform(args.img_size),
        split_ids=[full_ds.df.iloc[i]["prompt_id"] for i in train_subset.indices],
    )
    val_ds = BoxDamageDataset(
        manifest_path=args.manifest,
        images_root=args.images_dir,
        label_mode="dual",
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

    # ── model — ConvNeXt Base ─────────────────────────────────────────────────
    model = BoxDamageModel(
        backbone_name="convnext_base",
        label_mode="dual",
        num_coarse=4,
        num_bands=10,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"[model] convnext_base  params={total:,}")

    criterion = DualLoss(alpha=args.alpha)

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
               "train_mae":  [], "val_mae":  []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch == args.freeze_epochs + 1:
            print(f"[epoch {epoch}] Unfreezing backbone …")
            unfreeze_backbone(model)
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.lr * 0.05},
                {"params": model.neck.parameters(),     "lr": args.lr * 0.5},
                {"params": model.score_head.parameters(),  "lr": args.lr * 0.5},
                {"params": model.coarse_head.parameters(), "lr": args.lr * 0.5},
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
        history["train_mae"].append(train_m["mae"])
        history["val_mae"].append(val_m["mae"])

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  "
              f"val_acc={val_m['acc']:.3f}  val_mae={val_m['mae']:.2f}  "
              f"[{time.time()-t0:.1f}s]")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args) | {"label_mode": "dual",
                                      "backbone": "convnext_base"},
            }, out_dir / "best_model.pt")
            print(f"  ↳ saved best checkpoint → {out_dir}/best_model.pt")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_m["loss"],
        "args": vars(args) | {"label_mode": "dual", "backbone": "convnext_base"},
    }, out_dir / "final_model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train"); axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Coarse Accuracy"); axes[1].legend()
    axes[2].plot(history["train_mae"], label="train"); axes[2].plot(history["val_mae"], label="val")
    axes[2].set_title("Score MAE"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curve.png", dpi=120)
    plt.close()

    print(f"\n[done] best val_loss={best_val_loss:.4f}")
    print(f"Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
