import torch
from gpytorch.kernels import (
    ConstantKernel,
    Kernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
    RQKernel,
)


def rbf(n_features: int | None, n_outputs: torch.Size | None) -> RBFKernel:
    """
    Radial Basis Function (RBF) kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    RBFKernel
        The initialized RBF kernel.
    """
    k = RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


def matern_5_2_kernel(
    n_features: int | None, n_outputs: torch.Size | None
) -> MaternKernel:
    """
    Matern 5/2 kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    MaternKernel
        The initialized Matern 5/2 kernel.
    """
    k = MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


def matern_3_2_kernel(
    n_features: int | None, n_outputs: torch.Size | None
) -> MaternKernel:
    """
    Matern 3/2 kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    MaternKernel
        The initialized Matern 3/2 kernel.

    """
    k = MaternKernel(
        nu=1.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


def rq_kernel(n_features: int | None, n_outputs: torch.Size | None) -> RQKernel:
    """
    Rational Quadratic (RQ) kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    RQKernel
        The initialized RQ kernel.
    """
    k = RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


def linear_kernel(n_features: int | None, n_outputs: torch.Size | None) -> LinearKernel:
    """
    Linear kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    LinearKernel
        The initialized Linear kernel.
    """
    return LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rbf_plus_constant(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    """
    Radial Basis Function (RBF) kernel plus a constant kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    Kernel
        The initialized RBF kernel plus a constant kernel.
    """
    rbf_kernel = RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    if n_features is not None:
        rbf_kernel.initialize(lengthscale=torch.ones(n_features) * 1.5)
    return rbf_kernel + ConstantKernel()


# combinations
def rbf_plus_linear(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    """
    Radial Basis Function (RBF) kernel plus a linear kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    Kernel
        The initialized RBF kernel plus a linear kernel.
    """
    rbf_kernel = RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    if n_features is not None:
        rbf_kernel.initialize(lengthscale=torch.ones(n_features) * 1.5)
    return rbf_kernel + LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def matern_5_2_plus_rq(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    """
    Matern 5/2 kernel plus a Rational Quadratic (RQ) kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    Kernel
        The initialized Matern 5/2 kernel plus a Rational Quadratic kernel.
    """
    matern_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    rq_kernel = RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    # Initialize lengthscales for both kernels if n_features is provided
    if n_features is not None:
        matern_kernel.initialize(lengthscale=torch.ones(n_features) * 1.5)
        rq_kernel.initialize(lengthscale=torch.ones(n_features) * 1.5)
    return matern_kernel + rq_kernel


def rbf_times_linear(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    """Radial Basis Function (RBF) kernel multiplied by a linear kernel.

    Parameters
    ----------
    n_features: int | None
        Number of input features. If None, the kernel is not initialized with a
        lengthscale.
    n_outputs: torch.Size | None
        Batch shape of the kernel. If None, the kernel is not initialized with a
        batch shape.

    Returns
    -------
    Kernel
        The initialized RBF kernel multiplied by a linear kernel.
    """
    rbf_kernel = RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    if n_features is not None:
        rbf_kernel.initialize(lengthscale=torch.ones(n_features) * 1.5)
    return rbf_kernel * LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
