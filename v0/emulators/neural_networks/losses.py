import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNPLoss(nn.Module):
    """
    Loss function for Conditional Neural Processes (CNP).

    This loss function calculates the negative log-likelihood of the true values
    under a Normal distribution parameterized by the predicted mean and log variance.

    Methods:
    --------
    forward(y_pred: tuple, y_true: torch.Tensor) -> torch.Tensor:
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
            Output of the CNP model, mean and log variance.
            Both are tensors, Shape: [batch_size, output_dim]
        y_true : torch.Tensor
            True target values.
            Shape: [batch_size, ..., output_dim]

        Returns:
        --------
        torch.Tensor
            The calculated loss (negative log-likelihood), averaged over the batch.
        """

        mean, logvar = y_pred
        variance = torch.exp(logvar.clamp(min=-20, max=20)) + 1e-6
        nll = 0.5 * torch.mean(
            logvar
            + torch.clamp((y_true - mean) ** 2 / variance, max=1e6)
            + torch.log(torch.tensor(2 * np.pi))
        )
        return nll
