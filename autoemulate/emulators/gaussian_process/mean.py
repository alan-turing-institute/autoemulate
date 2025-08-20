import torch
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from typing import Callable
from .poly_mean import PolyMean
from .partially_learnable import PartiallyLearnableMean

def constant_mean(n_features: int | None, n_outputs: torch.Size | None) -> ConstantMean:
    """
    ConstantMean module.

    Parameters
    ----------
    n_features: int | None
        Number of input features. This parameter is not used in the ConstantMean module.
    n_outputs: torch.Size | None
        Batch shape of the mean. If None, the mean is not initialized with a batch
        shape.

    Returns
    -------
    ConstantMean
        The initialized ConstantMean module.
    """
    _ = n_features  # Unused parameter
    return (
        ConstantMean(batch_shape=n_outputs) if n_outputs is not None else ConstantMean()
    )


def zero_mean(n_features: int, n_outputs: torch.Size | None) -> ZeroMean:
    """ZeroMean module.

    Parameters
    ----------
    n_features: int
        Number of input features. This parameter is not used in the ZeroMean module.
    n_outputs: torch.Size | None
        Batch shape of the mean. If None, the mean is not initialized with a batch
        shape.

    Returns
    -------
    ZeroMean
        The initialized ZeroMean module.
    """
    _ = n_features  # Unused parameter
    return ZeroMean(batch_shape=n_outputs) if n_outputs is not None else ZeroMean()


def linear_mean(n_features: int, n_outputs: torch.Size | None) -> LinearMean:
    """
    LinearMean module.

    Parameters
    ----------
    n_features: int
        Number of input features.
    n_outputs: torch.Size | None
        Batch shape of the mean. If None, the mean is not initialized with a batch
        shape.

    Returns
    -------
    LinearMean
        The initialized LinearMean module.
    """
    return (
        LinearMean(input_size=n_features, batch_shape=n_outputs)
        if n_outputs is not None
        else LinearMean(input_size=n_features)
    )


def poly_mean(n_features: int, n_outputs: torch.Size | None) -> PolyMean:
    """
    PolyMean module (quadratic polynomial mean).

    Parameters
    ----------
    n_features: int
        Number of input features.
    n_outputs: torch.Size | None
        Batch shape of the mean. If None, the mean is not initialized with a batch
        shape.

    Returns
    -------
    PolyMean
        The initialized PolyMean module.
    """
    return (
        PolyMean(degree=2, input_size=n_features, batch_shape=n_outputs)
        if n_outputs is not None
        else PolyMean(degree=2, input_size=n_features)
    )

def partially_learnable_mean(
    n_features: int, 
    n_outputs: torch.Size | None, 
    mean_func: Callable = torch.sin,
    known_dim: int = 0
) -> PartiallyLearnableMean:
    """
    PartiallyLearnableMean module with known function for one dimension.

    Parameters
    ----------
    n_features: int
        Number of input features.
    n_outputs: torch.Size | None
        Batch shape of the mean. If None, the mean is not initialized with a batch
        shape.
    mean_func: callable
        Function to apply to the known dimension. Defaults to torch.sin.
    known_dim: int
        Dimension index for the known function. Defaults to 0.
        
    Returns
    -------
    PartiallyLearnableMean
        The initialized PartiallyLearnableMean module.
    """
    return (
        PartiallyLearnableMean(
            mean_func=mean_func,
            known_dim=known_dim,
            input_size=n_features,
            batch_shape=n_outputs
        )
        if n_outputs is not None
        else PartiallyLearnableMean(
            mean_func=mean_func,
            known_dim=known_dim,
            input_size=n_features
        )
    )