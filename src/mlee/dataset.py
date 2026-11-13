"""Data augmentation and Dataset classes for 1D ECG heartbeats (Dual-Stream support)."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, dual_stream: bool = True):
        """
        X: (N, 300) raw beats
        y: (N,) integer class labels
        augment: bool, apply on-the-fly random temporal jitter, scaling, noise, and resampling
        dual_stream: bool, return 2-channel tensor [x, dx] where dx = x[t] - x[t-1]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.dual_stream = dual_stream

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (300,)
        y = self.y[idx]

        if self.augment:
            # 1. Random temporal jitter (shift by +/- 8 samples)
            shift = np.random.randint(-8, 9)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

            # 2. Dynamic heart-rate resampling jitter (+/- 5% time dilation/compression)
            if np.random.rand() > 0.3:
                scale_factor = float(np.random.uniform(0.95, 1.05))
                new_len = int(300 * scale_factor)
                resampled = F.interpolate(
                    x.view(1, 1, 300), size=new_len, mode="linear", align_corners=False
                ).view(1, 1, -1)
                if new_len > 300:
                    start = (new_len - 300) // 2
                    x = resampled[0, 0, start : start + 300]
                else:
                    pad_left = (300 - new_len) // 2
                    pad_right = 300 - new_len - pad_left
                    x = F.pad(resampled, (pad_left, pad_right), mode="replicate").view(-1)

            # 3. Random amplitude scaling (+/- 10%)
            scale = float(np.random.uniform(0.90, 1.10))
            x = x * scale

            # 4. Add mild Gaussian noise
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        if self.dual_stream:
            # Channel 0: Normalized voltage x[t]
            # Channel 1: First-order discrete difference dx[t] = x[t] - x[t-1]
            dx = torch.diff(x, prepend=x[0:1])
            # Return shape: (2, 300)
            tensor_out = torch.stack([x, dx], dim=0)
        else:
            # Return shape: (1, 300)
            tensor_out = x.unsqueeze(0)

        return tensor_out, y
