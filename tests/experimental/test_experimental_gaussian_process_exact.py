import gpytorch
import pytest
import torch
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.emulators.gaussian_process import constant_mean, rbf, rbf_times_linear
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DistributionLike
from sklearn.datasets import make_regression


@pytest.fixture
def sample_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)  # type: ignore
    # TODO: consider how to convert to tensor within init, fit, predict
    y = y.reshape(-1, 1)
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=1)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def sample_data_y2d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=0)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y2d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=1)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


def test_predict_with_uncertainty_gp(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
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


def test_tune_gp(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(GaussianProcessExact)
    assert len(scores) == 5
    assert len(configs) == 5
