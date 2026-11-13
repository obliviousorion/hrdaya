"""Ultra-lightweight 1D-ResNet for ECG Biometrics.

Architecture:
- Initial Conv1D projection (1 -> 16)
- 3 Residual Blocks with Skip Connections & Depthwise-Separable Convolutions
- Global Adaptive Average Pooling
- Embedding dimension: 48
- Total parameters: ~18k (~72 KB FP32, ~22 KB INT8)
"""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        # Depthwise Separable Conv
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.pointwise1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.pointwise2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.pointwise1(self.conv1(x))))
        out = self.bn2(self.pointwise2(self.conv2(out)))
        out = self.relu(out + residual)
        return out


class ECG_ResNet(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=15, stride=2, padding=7),  # (B, 24, 150)
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                        # (B, 24, 75)
        )

        self.block1 = ResBlock1D(24, kernel_size=7)

        self.transition1 = nn.Sequential(
            nn.Conv1d(24, 48, kernel_size=1),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                        # (B, 48, 37)
        )

        self.block2 = ResBlock1D(48, kernel_size=5)

        self.pool = nn.AdaptiveAvgPool1d(1)                         # (B, 48, 1)
        self.emb_dim = emb_dim
        self.head = nn.Linear(48, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        return self.head(emb)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)                                      # (B, 1, 300)
        x = self.stem(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        emb = self.pool(x).squeeze(-1)                              # (B, 48)
        return emb
