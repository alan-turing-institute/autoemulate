import gpytorch
import pytest
import torch
from sklearn.datasets import make_regression

from autoemulate.experimental.config import FitConfig
from autoemulate.experimental.emulators.gpytorch_backend import GPExactRBF
from autoemulate.experimental.types import DistributionLike

# from autoemulate.emulators.gaussian_process_mt import GPExactRBFMT


@pytest.fixture
def sample_data_y1d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)  # type: ignore
    # TODO: consider how to convert to tensor within init, fit, predict
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


# test multioutput GP
def test_multioutput_gp(sample_data_y2d, new_data_y2d):
    X, y = sample_data_y2d
    X2, _ = new_data_y2d
    gp = GPExactRBF(X, y, gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2))
    gp.train()
    gp.fit(X, y, FitConfig())
    y_pred = gp.predict(X)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)


def test_predict_with_uncertainty_gp(sample_data_y1d, new_data_y1d):
    X, y = sample_data_y1d
    # y_shape = y.shape
    gp = GPExactRBF(X, y, gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1))
    gp.train()
    gp.fit(X, y, FitConfig())
    X2, _ = new_data_y1d
    y_pred = gp.predict(X2)
    assert isinstance(y_pred, DistributionLike)
    # TODO: fix shape assertion
    # assert y_pred.mean.shape == y_shape
    # assert y_pred.variance.shape == y_shape


# TODO: update compare loop
# def test_gp_param_search(sample_data_y1d, new_data_y1d):
#     X, y = sample_data_y1d
#     X2, _ = new_data_y1d
#     em = AutoEmulate()
#     em.setup(X, y, models=["gp"], param_search_iters=3)
#     em.compare()
