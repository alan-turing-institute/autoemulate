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
    k = RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


def rbf_plus_constant(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    rbf_kernel = RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
    if n_features is not None:
        rbf_kernel.initialize(lengthscale=torch.ones(n_features) * 1.5)
    return rbf_kernel + ConstantKernel()


# combinations
def rbf_plus_linear(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
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
