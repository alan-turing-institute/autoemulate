import numpy as np
import torch
from autoemulate.core.device import TorchDeviceMixin, get_torch_device
from autoemulate.core.types import DeviceLike, TensorLike
from torch.utils.data import Dataset


class CNPDataset(Dataset, TorchDeviceMixin):
    """
    Dataset for Conditional Neural Process (CNP).
    Samples context points and target points for
    each episode.
    """

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        min_context_points: int,
        max_context_points: int,
        n_episode: int,
        device: DeviceLike | None = None,
    ):
        """
        Parameters
        ----------
        x: (n_samples, x_dim)
            Input data
        y: (n_samples, y_dim)
            Output data
        min_context_points: int
            Minimum number of context points to sample
        max_context_points: int
            Maximum number of context points to sample
        n_episode: int
            Number of episodes to sample. Must be greater than max_context_points.
        """
        TorchDeviceMixin.__init__(self, device=device)
        self.x, self.y = self._move_tensors_to_device(x, y)
        if max_context_points >= n_episode:
            msg = "max_context_points must be less than n_episode"
            raise ValueError(msg)
        self.max_context_points = max_context_points
        self.min_context_points = min_context_points
        self.n_samples = len(self.x)
        # self.x_dim = self.x[0][0].shape[-1]
        self.n_episode = n_episode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.sample()

    def sample(self):
        """
        From the full dataset, sample context and target points.
        """
        x = self.x
        y = self.y

        # sample context points
        num_context = np.random.randint(
            self.min_context_points,
            self.max_context_points + 1,
        )

        # randomly select context
        perm = torch.randperm(self.n_samples)
        context_idxs = perm[:num_context]
        target_idxs = perm[
            : self.n_episode
        ]  # including context points in targets helps with training

        x_context = x[context_idxs]
        y_context = y[context_idxs]
        x_target = x[target_idxs]
        y_target = y[target_idxs]

        x = {
            "x_context": x_context,
            "y_context": y_context,
            "x_target": x_target,
        }

        return (x, y_target)


def cnp_collate_fn(batch, device: DeviceLike | None = None):
    """
    Collate function for CNP.

    Handles different dimensions in each sample due to different numbers of context
    points.

    TODO:
    this currently just adds zeros. These shouldn't have an impact on the output as long # noqa: E501
    as we don't do layernorm or attention, but we should modify cnp etc. to handle this # noqa: E501
    better.
    """
    X, y = zip(*batch, strict=False)
    device = get_torch_device(device)

    # Get the maximum number of context and target points in the batch
    max_context = max(x["x_context"].shape[0] for x in X)
    max_target = max(x["x_target"].shape[0] for x in X)

    # Initialize tensors to hold the batched data
    x_context_batched = torch.zeros(len(batch), max_context, X[0]["x_context"].shape[1])
    y_context_batched = torch.zeros(len(batch), max_context, X[0]["y_context"].shape[1])
    x_target_batched = torch.zeros(len(batch), max_target, X[0]["x_target"].shape[1])
    y_batched = torch.zeros(len(batch), max_target, y[0].shape[1])

    # Create masks for context and target
    context_mask = torch.zeros(len(batch), max_context, dtype=torch.bool)
    target_mask = torch.zeros(len(batch), max_target, dtype=torch.bool)

    # Fill in the batched tensors
    for i, (x, yi) in enumerate(zip(X, y, strict=False)):
        n_context = x["x_context"].shape[0]
        n_target = x["x_target"].shape[0]

        x_context_batched[i, :n_context] = x["x_context"]
        y_context_batched[i, :n_context] = x["y_context"]
        x_target_batched[i, :n_target] = x["x_target"]
        y_batched[i, :n_target] = yi

        context_mask[i, :n_context] = True
        target_mask[i, :n_target] = True

    # Create a dictionary for X
    x_batched = {
        "x_context": x_context_batched.to(device),
        "y_context": y_context_batched.to(device),
        "x_target": x_target_batched.to(device),
        "context_mask": context_mask.to(device),
    }

    return x_batched, y_batched.to(device)
