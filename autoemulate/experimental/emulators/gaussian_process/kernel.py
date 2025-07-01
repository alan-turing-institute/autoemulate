import torch
from gpytorch.kernels import (
    ConstantKernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
    RQKernel,
)


# kernel functions for parameter search have to be outside the class so that pickle can
# find them
def rbf(n_features: int | None, n_outputs: torch.Size | None):
    k = (
        RBFKernel(
            ard_num_dims=n_features,
            batch_shape=n_outputs,
        )
        if n_outputs is not None
        else RBFKernel(ard_num_dims=n_features)
    )
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


def matern_5_2_kernel(n_features: int | None, n_outputs: torch.Size | None):
    return MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def matern_3_2_kernel(n_features: int | None, n_outputs: torch.Size | None):
    return MaternKernel(
        nu=1.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rq_kernel(n_features: int | None, n_outputs: torch.Size | None):
    return RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rbf_plus_constant(n_features: int | None, n_outputs: torch.Size | None):
    k = (
        RBFKernel(
            ard_num_dims=n_features,
            batch_shape=n_outputs,
        )
        if n_outputs is not None
        else RBFKernel(ard_num_dims=n_features)
    ) + ConstantKernel()
    return (
        k.initialize(lengthscale=torch.ones(n_features) * 1.5)
        if n_features is not None
        else k
    )


# combinations
def rbf_plus_linear(n_features: int | None, n_outputs: torch.Size | None):
    return RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) + LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def matern_5_2_plus_rq(n_features, n_outputs):
    return MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) + RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rbf_times_linear(n_features: int | None, n_outputs: torch.Size | None):
    return RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) * LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )
