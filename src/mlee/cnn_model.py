"""Compact 1D-CNN for ECG biometric identification.

Input: raw normalised beat windows (300 samples).
~15K parameters → ~60 KB FP32 → ~15 KB INT8.
"""
import torch
import torch.nn as nn


class ECG_CNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),                                # → (16, 150)

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                                # → (32, 75)

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                        # → (32, 1)
        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 300) raw normalised beat
        x = x.unsqueeze(1)                  # → (B, 1, 300)
        emb = self.encoder(x).squeeze(-1)   # → (B, 32)
        return self.head(emb)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 32-dim embedding (for cosine verification)."""
        x = x.unsqueeze(1)
        return self.encoder(x).squeeze(-1)
