"""Ultra-lightweight 1D-ResNet with Dual-Stream inputs and Squeeze-and-Excitation (SE) Attention.

Architecture:
- Conv1D projection stem (in_channels=2 -> 24)
- Stage 1: SE-ResBlock1D (24 channels, k=7)
- Transition: Conv1D (24 -> 48) + MaxPool1D
- Stage 2: SE-ResBlock1D (48 channels, k=5)
- Global Adaptive Average Pooling -> 48-dim embedding -> LayerNorm -> Linear(48 -> n_classes)
Total parameters: ~19k (~78 KB FP32, ~25 KB INT8)
"""
import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=-1)          # (B, C)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y


class SEResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        # Depthwise Separable Conv 1
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.pointwise1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise Separable Conv 2
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.pointwise2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = SEBlock1D(channels, reduction=4)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.pointwise1(self.conv1(x))))
        out = self.bn2(self.pointwise2(self.conv2(out)))
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class ECG_ResNet_DualStream(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 2, emb_dim: int = 48):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=15, stride=2, padding=7),  # (B, 24, 150)
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                                  # (B, 24, 75)
        )

        self.block1 = SEResBlock1D(24, kernel_size=7)

        self.transition1 = nn.Sequential(
            nn.Conv1d(24, 48, kernel_size=1),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                                  # (B, 48, 37)
        )

        self.block2 = SEResBlock1D(48, kernel_size=5)

        self.pool = nn.AdaptiveAvgPool1d(1)                                   # (B, 48, 1)
        self.norm = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        return self.head(emb)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, 2, 300) or (B, 300)
        if x.dim() == 2:
            # Generate dx on the fly if single channel provided
            dx = torch.diff(x, prepend=x[:, 0:1], dim=1)
            x = torch.stack([x, dx], dim=1)                                  # (B, 2, 300)
        elif x.dim() == 3 and x.size(1) == 1 and self.in_channels == 2:
            x_raw = x.squeeze(1)
            dx = torch.diff(x_raw, prepend=x_raw[:, 0:1], dim=1)
            x = torch.stack([x_raw, dx], dim=1)

        x = self.stem(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        emb = self.pool(x).squeeze(-1)                                        # (B, 48)
        emb = self.norm(emb)
        return emb
