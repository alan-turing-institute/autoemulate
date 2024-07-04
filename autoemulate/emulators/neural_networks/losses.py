import torch
import torch.nn as nn
import torch.nn.functional as F


class CNPLoss(nn.Module):
    """
    Improved loss function for Conditional Neural Processes (CNP).

    This loss function creates a Normal distribution using the predicted mean and
    log standard deviation, then calculates the negative log-likelihood of the true
    values under this distribution.

    Methods:
    --------
    forward(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        Calculate the loss given predictions and true values.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        Calculate the loss given predictions and true values.

        Parameters:
        -----------
        y_pred : tuple
            Output of the CNP model, mean and log standard deviation.
            Both are tensors, Shape: [batch_size, output_dim]
        y_true : torch.Tensor
            True target values.
            Shape: [batch_size, ..., output_dim]

        Returns:
        --------
        torch.Tensor
            The calculated loss (negative log-likelihood), averaged over the batch.
        """
        # mean, log_sigma = torch.chunk(y_pred, 2, dim=-1)
        mean, log_sigma = y_pred

        # print(f"mean: {mean[:5]}, log_sigma: {log_sigma[:5]}")
        # print(log_sigma)
        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        # sigma = torch.exp(log_sigma)

        # Create a Normal distribution with the predicted mean and variance
        dist = torch.distributions.Normal(mean.squeeze(-1), sigma.squeeze(-1))

        # Calculate the negative log-likelihood
        nll = -dist.log_prob(y_true).sum(dim=-1)

        return nll.mean()
