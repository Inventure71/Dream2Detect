from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _group_count_for_channels(num_channels: int, *, max_groups: int = 8) -> int:
    for group_count in range(min(max_groups, num_channels), 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


def _build_norm_2d(num_channels: int, *, norm_kind: str) -> nn.Module:
    if norm_kind == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_kind == "group":
        return nn.GroupNorm(_group_count_for_channels(num_channels), num_channels)
    raise ValueError(f"Unsupported norm_kind: {norm_kind}")


class SimpleCNNClassifier(nn.Module):
    """
    Small CNN for 4-class package defect severity classification.

    Input:
    - RGB image tensor of shape [batch, 3, H, W]

    Output:
    - logits of shape [batch, 4]
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.1) -> None:
        super().__init__()

        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # On MPS, adaptive average pooling currently requires the input spatial
        # size to be divisible by the requested output size. After four 2x2
        # pooling steps:
        # - 128x128 -> 8x8
        # - 224x224 -> 14x14
        # Both are divisible by 2, so (2, 2) is a safe shared target size.
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        norm_kind: str = "batch",
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _build_norm_2d(out_channels, norm_kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _build_norm_2d(out_channels, norm_kind=norm_kind),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _build_norm_2d(out_channels, norm_kind=norm_kind),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class ResidualCNNClassifier(nn.Module):
    """
    Stronger from-scratch CNN for subtle package-defect severity.

    This is intentionally not pretrained. It adds batch normalization,
    residual blocks, and higher channel capacity while staying much smaller
    than an ImageNet-scale model.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_p: float = 0.2,
        *,
        norm_kind: str = "batch",
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            _build_norm_2d(32, norm_kind=norm_kind),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32, norm_kind=norm_kind),
            ResidualBlock(32, 64, stride=2, norm_kind=norm_kind),
            ResidualBlock(64, 64, norm_kind=norm_kind),
            ResidualBlock(64, 128, stride=2, norm_kind=norm_kind),
            ResidualBlock(128, 128, norm_kind=norm_kind),
            ResidualBlock(128, 256, stride=2, norm_kind=norm_kind),
            ResidualBlock(256, 256, norm_kind=norm_kind),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResidualGroupNormCNNClassifier(ResidualCNNClassifier):
    """
    Residual CNN variant that replaces batch-normalization with group-normalization.
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.2) -> None:
        super().__init__(
            num_classes=num_classes,
            dropout_p=dropout_p,
            norm_kind="group",
        )


class ResidualCNNMultiTaskClassifier(nn.Module):
    """
    From-scratch residual CNN with a primary coarse head and an auxiliary
    score-band head.
    """

    def __init__(
        self,
        *,
        num_coarse_classes: int = 4,
        num_score_bands: int = 10,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.coarse_head = nn.Linear(256, num_coarse_classes)
        self.score_band_head = nn.Linear(256, num_score_bands)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return {
            "coarse_logits": self.coarse_head(x),
            "score_band_logits": self.score_band_head(x),
        }


class SimpleCNNRegressor(nn.Module):
    """
    Small CNN for package defect severity regression.

    Output is a single unbounded score; training code clamps predictions to
    the project severity range when deriving class labels for evaluation.
    """

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()

        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x.squeeze(dim=1)


def build_classifier_model(
    *,
    num_classes: int = 4,
    dropout_p: float = 0.1,
    model_variant: str = "simple_cnn",
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    if model_variant == "simple_cnn":
        return SimpleCNNClassifier(num_classes=num_classes, dropout_p=dropout_p)

    if model_variant == "residual_cnn":
        return ResidualCNNClassifier(num_classes=num_classes, dropout_p=dropout_p)

    if model_variant == "residual_cnn_groupnorm":
        return ResidualGroupNormCNNClassifier(
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    if model_variant == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported model_variant: {model_variant}")
