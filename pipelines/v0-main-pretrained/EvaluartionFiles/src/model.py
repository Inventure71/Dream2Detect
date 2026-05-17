"""
BoxDamageModel
==============
EfficientNet-B3 (or configurable timm backbone) fine-tuned for package
damage assessment.

Supports three output heads:
  - regression  : single sigmoid output → damage score 0-100
  - coarse      : 4-class softmax
  - band        : 10-class softmax

The dual-head variant (score + coarse) trains simultaneously and is the
recommended mode — it produces both a numeric score and a coarse class,
which is more useful for real-world deployment.
"""

import torch
import torch.nn as nn
import timm


# ── model ──────────────────────────────────────────────────────────────────────
class BoxDamageModel(nn.Module):
    """
    Parameters
    ----------
    backbone_name : timm model name, e.g. "efficientnet_b3" or "convnext_small"
    label_mode    : "score" | "coarse" | "band" | "dual"
    num_coarse    : number of coarse classes (4 by default)
    num_bands     : number of band classes (10 by default)
    dropout       : dropout rate before heads
    pretrained    : load ImageNet weights
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        label_mode: str = "dual",
        num_coarse: int = 4,
        num_bands: int = 10,
        dropout: float = 0.35,
        pretrained: bool = True,
    ):
        super().__init__()
        self.label_mode = label_mode

        # ── backbone ─────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,      # remove classifier
            global_pool="avg",  # global average pooling
        )
        feat_dim = self.backbone.num_features

        # ── shared neck ──────────────────────────────────────────────────────
        self.neck = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── heads ─────────────────────────────────────────────────────────────
        if label_mode in ("score", "dual"):
            self.score_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        if label_mode in ("coarse", "dual"):
            self.coarse_head = nn.Linear(512, num_coarse)
        if label_mode == "band":
            self.band_head = nn.Linear(512, num_bands)

    def forward(self, x):
        feat = self.backbone(x)
        neck = self.neck(feat)

        if self.label_mode == "dual":
            score  = self.score_head(neck).squeeze(-1)   # (B,)
            coarse = self.coarse_head(neck)               # (B, 4)
            return {"score": score, "coarse": coarse}
        elif self.label_mode == "score":
            return self.score_head(neck).squeeze(-1)
        elif self.label_mode == "coarse":
            return self.coarse_head(neck)
        else:  # band
            return self.band_head(neck)


# ── loss ───────────────────────────────────────────────────────────────────────
class DualLoss(nn.Module):
    """
    Combined loss for dual-head (score + coarse):
        L = α * MSE(score) + (1-α) * CrossEntropy(coarse)

    score targets should be in [0, 1] (i.e. raw score / 100).
    """

    def __init__(self, alpha: float = 0.5, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.ce    = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, preds: dict, score_targets, coarse_targets):
        loss_score  = self.mse(preds["score"], score_targets)
        loss_coarse = self.ce(preds["coarse"], coarse_targets)
        return self.alpha * loss_score + (1 - self.alpha) * loss_coarse, \
               loss_score.item(), loss_coarse.item()


# ── inference helper ────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: BoxDamageModel, image_tensor: torch.Tensor,
            device: str = "cpu"):
    """
    Run inference on a single pre-processed image tensor (C, H, W).

    Returns
    -------
    dict with keys:
        damage_score   : float 0-100
        coarse_class   : str
        coarse_probs   : list[float] – softmax probabilities per coarse class
        severity_label : str  (human-friendly)
    """
    from src.dataset import COARSE_CLASSES

    model.eval()
    x = image_tensor.unsqueeze(0).to(device)
    out = model(x)

    if isinstance(out, dict):
        score_raw = out["score"].item()
        coarse_logits = out["coarse"].cpu()
    else:
        # fallback for single-head
        score_raw = 0.0
        coarse_logits = out.cpu()

    damage_score = round(score_raw * 100, 1)
    coarse_probs = torch.softmax(coarse_logits, dim=-1).squeeze().tolist()
    coarse_idx   = int(torch.argmax(coarse_logits))
    coarse_class = COARSE_CLASSES[coarse_idx]

    severity_map = {
        "intact":   "✅ Intact — no visible damage",
        "minor":    "🟡 Minor — cosmetic damage only",
        "moderate": "🟠 Moderate — noticeable structural damage",
        "severe":   "🔴 Severe — significant damage / likely contents affected",
    }

    return {
        "damage_score":   damage_score,
        "coarse_class":   coarse_class,
        "coarse_probs":   {c: round(p, 4) for c, p in zip(COARSE_CLASSES, coarse_probs)},
        "severity_label": severity_map[coarse_class],
    }
