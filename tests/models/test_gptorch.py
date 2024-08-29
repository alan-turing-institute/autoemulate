import gpytorch
import pytest
import torch
from sklearn.datasets import make_regression

from autoemulate.emulators.gaussian_process_torch import GaussianProcessTorch


@pytest.fixture
def sample_data_y1d():
    X, y = make_regression(n_samples=10, n_features=5, n_targets=1, random_state=0)
    return X, y


@pytest.fixture
def sample_data_y2d():
    X, y = make_regression(n_samples=10, n_features=5, n_targets=2, random_state=0)
    return X, y


def test_multi_output_gp(sample_data_y2d):
    X, y = sample_data_y2d
    gp = GaussianProcessTorch(random_state=42)
    gp.fit(X, y)
    assert gp.predict(X).shape == (10, 2)


def test_predict_with_uncertainty(sample_data_y1d):
    X, y = sample_data_y1d
    y_shape = y.shape
    gp = GaussianProcessTorch(random_state=42)
    gp.fit(X, y)
    y_pred, y_std = gp.predict(X, return_std=True)
    assert y_pred.shape == y_shape
    assert y_std.shape == y_shape


def test_multitask(sample_data_y2d):
    X, y = sample_data_y2d
    gp = GaussianProcessTorch(multitask=True, random_state=42)
    gp.fit(X, y)
    y_pred, y_std = gp.predict(X, return_std=True)
    assert y_pred.shape == y.shape
    assert y_std.shape == y.shape
