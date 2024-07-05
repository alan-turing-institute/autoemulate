import numpy as np
import torch
from torch.utils.data import Dataset


class CNPDataset(Dataset):
    def __init__(self, X, y, n_context_points=10):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.n_context_points = n_context_points

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # select all points
        X = self.X[idx]
        y = self.y[idx]
        # print(f"X: {X.shape}, y: {y.shape}")

        # Randomly select context points
        context_idxs = torch.randperm(len(X))[: self.n_context_points]
        X_context = X[context_idxs]
        y_context = y[context_idxs]

        return {
            "X_context": X_context,
            "y_context": y_context,
            "X_target": X,
            "y_target": y,
        }
