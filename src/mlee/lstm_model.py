"""LSTM model for ECG biometric identification.

Input: AC/DCT(32) features (same as SVM baseline).
Treats the 32 coefficients as a sequence of length 32, 1 feature each.
Architecture: LSTM(64) → Linear(n_classes)
"""
import torch
import torch.nn as nn


class ECG_LSTM(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32) → (B, 32, 1) treat each DCT coeff as a timestep
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)   # h: (1, B, hidden)
        return self.head(h.squeeze(0))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden state embedding (for cosine verification)."""
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)
