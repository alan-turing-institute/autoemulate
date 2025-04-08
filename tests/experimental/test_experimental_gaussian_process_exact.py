import gpytorch
import gpytorch.kernels as gpyk
import gpytorch.means as gpym
import pytest
import torch
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.emulators.gaussian_process import constant_mean, rbf, rbf_times_linear
from autoemulate.experimental.types import DistributionLike
from sklearn.datasets import make_regression


@pytest.fixture
def sample_data_y1d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)  # type: ignore
    # TODO: consider how to convert to tensor within init, fit, predict
    y = y.reshape(-1, 1)
    return torch.Tensor(X), torch.Tensor(y)


@pytest.fixture
def new_data_y1d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=1)  # type: ignore
    return torch.Tensor(X), torch.Tensor(y)


@pytest.fixture
def sample_data_y2d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=0)  # type: ignore
    return torch.Tensor(X), torch.Tensor(y)


@pytest.fixture
def new_data_y2d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=1)  # type: ignore
    return torch.Tensor(X), torch.Tensor(y)


def mean() -> type[gpym.Mean]:
    return gpym.ConstantMean


def covar() -> list[type[gpyk.Kernel]]:
    return [gpytorch.kernels.ConstantKernel, gpytorch.kernels.RBFKernel]


# test multioutput GP
def test_multioutput_gp(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    gp = GaussianProcessExact(
        x,
        y,
        gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2),
        constant_mean,
        rbf_times_linear,
    )
    gp.fit(x, y)
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)


def test_predict_with_uncertainty_gp(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    print(x.shape, y.shape)
    gp = GaussianProcessExact(
        x,
        y,
        gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1),
        constant_mean,
        rbf,
    )
    gp.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == y.shape
    assert y_pred.variance.shape == y.shape


# TODO: update compare loop
# def test_gp_param_search(sample_data_y1d, new_data_y1d):
#     X, y = sample_data_y1d
#     X2, _ = new_data_y1d
#     em = AutoEmulate()
#     em.setup(X, y, models=["gp"], param_search_iters=3)
#     em.compare()
