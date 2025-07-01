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
    return MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def matern_3_2_kernel(
    n_features: int | None, n_outputs: torch.Size | None
) -> MaternKernel:
    return MaternKernel(
        nu=1.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rq_kernel(n_features: int | None, n_outputs: torch.Size | None) -> RQKernel:
    return RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rbf_plus_constant(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    k = (
        RBFKernel(
            ard_num_dims=n_features,
            batch_shape=n_outputs,
        )
    ) + ConstantKernel()
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


# combinations
def rbf_plus_linear(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    return RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) + LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def matern_5_2_plus_rq(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    return MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) + RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rbf_times_linear(n_features: int | None, n_outputs: torch.Size | None) -> Kernel:
    return RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) * LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
