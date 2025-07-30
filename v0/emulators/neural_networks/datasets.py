import numpy as np
import torch
from skorch.dataset import Dataset


class CNPDataset(Dataset):
    def __init__(self, X, y, max_context_points, min_context_points, n_episode):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if max_context_points >= n_episode:
            raise ValueError("max_context_points must be less than n_episode")
        self.max_context_points = max_context_points
        self.min_context_points = min_context_points
        self.n_samples, self.x_dim = X.shape
        self.n_episode = n_episode
        self.y_dim = y.shape[1] if len(y.shape) > 1 else 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # sample context points
        num_context = np.random.randint(
            self.min_context_points, self.max_context_points + 1
        )

        # randomly select context
        perm = torch.randperm(self.n_samples)
        context_idxs = perm[:num_context]
        target_idxs = perm[
            : self.n_episode
        ]  # including context points in targets helps with training

        X_context = self.X[context_idxs]
        y_context = self.y[context_idxs]
        X_target = self.X[target_idxs]
        y = self.y[target_idxs]

        X = {"X_context": X_context, "y_context": y_context, "X_target": X_target}
        y = y
        return (X, y)


def cnp_collate_fn(batch):
    """
    Collate function for CNP.

    Handles different dimensions in each sample due to different numbers of context points.

    TODO:
    this currently just adds zeros. These shouldn't have an impact on the output as long as we don't
    do layernorm or attention, but we should modify cnp etc. to handle this better.
    """
    X, y = zip(*batch)

    # Get the maximum number of context and target points in the batch
    max_context = max(x["X_context"].shape[0] for x in X)
    max_target = max(x["X_target"].shape[0] for x in X)

    # Initialize tensors to hold the batched data
    X_context_batched = torch.zeros(len(batch), max_context, X[0]["X_context"].shape[1])
    y_context_batched = torch.zeros(len(batch), max_context, X[0]["y_context"].shape[1])
    X_target_batched = torch.zeros(len(batch), max_target, X[0]["X_target"].shape[1])
    y_batched = torch.zeros(len(batch), max_target, y[0].shape[1])

    # Create masks for context and target
    context_mask = torch.zeros(len(batch), max_context, dtype=torch.bool)
    target_mask = torch.zeros(len(batch), max_target, dtype=torch.bool)

    # Fill in the batched tensors
    for i, (x, yi) in enumerate(zip(X, y)):
        n_context = x["X_context"].shape[0]
        n_target = x["X_target"].shape[0]

        X_context_batched[i, :n_context] = x["X_context"]
        y_context_batched[i, :n_context] = x["y_context"]
        X_target_batched[i, :n_target] = x["X_target"]
        y_batched[i, :n_target] = yi

        context_mask[i, :n_context] = True
        target_mask[i, :n_target] = True

    # Create a dictionary for X
    X_batched = {
        "X_context": X_context_batched,
        "y_context": y_context_batched,
        "X_target": X_target_batched,
        "context_mask": context_mask,
    }

    return X_batched, y_batched
