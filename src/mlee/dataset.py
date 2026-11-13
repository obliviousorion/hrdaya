"""Data augmentation and Dataset classes for 1D ECG heartbeats."""
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        """
        X: (N, 300) raw beats
        y: (N,) integer class labels
        augment: bool, apply on-the-fly random temporal jitter, scaling, and noise
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # 1. Random temporal jitter (shift by +/- 8 samples)
            shift = np.random.randint(-8, 9)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

            # 2. Random amplitude scaling (+/- 10%)
            scale = np.random.uniform(0.90, 1.10)
            x = x * scale

            # 3. Add mild Gaussian noise
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        return x, y
