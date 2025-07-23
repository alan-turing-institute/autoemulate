import torch
from gpytorch.means import ConstantMean, LinearMean, ZeroMean

from .poly_mean import PolyMean


def constant_mean(n_features: int | None, n_outputs: torch.Size | None) -> ConstantMean:
    _ = n_features  # Unused parameter
    return (
        ConstantMean(batch_shape=n_outputs) if n_outputs is not None else ConstantMean()
    )


def zero_mean(n_features: int, n_outputs: torch.Size | None) -> ZeroMean:
    _ = n_features  # Unused parameter
    return ZeroMean(batch_shape=n_outputs) if n_outputs is not None else ZeroMean()


def linear_mean(n_features: int, n_outputs: torch.Size | None) -> LinearMean:
    return (
        LinearMean(input_size=n_features, batch_shape=n_outputs)
        if n_outputs is not None
        else LinearMean(input_size=n_features)
    )


def poly_mean(n_features: int, n_outputs: torch.Size | None) -> PolyMean:
    return (
        PolyMean(degree=2, input_size=n_features, batch_shape=n_outputs)
        if n_outputs is not None
        else PolyMean(degree=2, input_size=n_features)
    )
