import numpy as np
import torch
from skorch.dataset import Dataset


class CNPDataset(Dataset):
    def __init__(self, X, y, max_context_points, min_context_points=3):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.max_context_points = max_context_points
        self.min_context_points = min_context_points
        self.n_samples, self.x_dim = X.shape
        self.y_dim = y.shape[1] if len(y.shape) > 1 else 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # sample context points
        # num_context = np.random.randint(
        #     self.min_context_points, self.max_context_points
        # )
        num_context = 10

        # randomly select context
        perm = torch.randperm(self.n_samples)
        context_idxs = perm[:num_context]
        target_idxs = (
            perm  # including context points in targets seems to help with training
        )

        X_context = self.X[context_idxs]
        y_context = self.y[context_idxs]
        X_target = self.X[target_idxs]
        y = self.y[target_idxs]

        X = {"X_context": X_context, "y_context": y_context, "X_target": X_target}
        y = y
        return (X, y)
