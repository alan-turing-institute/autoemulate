from typing import Protocol

import torch
from gpytorch.kernels import Kernel
from gpytorch.means import Mean


class MeanModuleFn(Protocol):
    """Protocol for mean module functions."""

    def __call__(self, n_features: int | None, n_outputs: torch.Size | None) -> Mean:
        """Call the mean module function with the specified parameters."""
        ...


class CovarModuleFn(Protocol):
    """Protocol for covariance module functions."""

    # TODO: consider revising input API to be more flexible
    def __call__(self, n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
        """Call the covariance module function with the specified parameters."""
        ...
