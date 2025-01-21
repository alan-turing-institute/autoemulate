import gpytorch
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV

from autoemulate.compare import AutoEmulate
from autoemulate.emulators.gaussian_process import GaussianProcess
from autoemulate.emulators.gaussian_process_mt import GaussianProcessMT

# tests for gpytorch GP's


@pytest.fixture
def sample_data_y1d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)
    return X, y


@pytest.fixture
def sample_data_y2d():
    X, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=0)
    return X, y


# test multitask GP
def test_multi_output_gpmt(sample_data_y2d):
    X, y = sample_data_y2d
    gp = GaussianProcessMT(random_state=42)
    gp.fit(X, y)
    assert gp.predict(X).shape == (20, 2)


def test_predict_with_uncertainty_gpmt(sample_data_y1d):
    X, y = sample_data_y1d
    y_shape = y.shape
    gp = GaussianProcessMT(random_state=42)
    gp.fit(X, y)
    y_pred, y_std = gp.predict(X, return_std=True)
    assert y_pred.shape == y_shape
    assert y_std.shape == y_shape


def test_multitask_gpmt(sample_data_y2d):
    X, y = sample_data_y2d
    gp = GaussianProcessMT(random_state=42)
    gp.fit(X, y)
    y_pred, y_std = gp.predict(X, return_std=True)
    assert y_pred.shape == y.shape
    assert y_std.shape == y.shape


def test_gpmt_param_search(sample_data_y1d):
    X, y = sample_data_y1d
    em = AutoEmulate()
    em.setup(X, y, models=["gp"], param_search_iters=3)
    em.compare()


# test multioutput GP
def test_multioutput_gp(sample_data_y2d):
    X, y = sample_data_y2d
    gp = GaussianProcess(random_state=42)
    gp.fit(X, y)
    assert gp.predict(X).shape == (20, 2)


def test_predict_with_uncertainty_gp(sample_data_y1d):
    X, y = sample_data_y1d
    y_shape = y.shape
    gp = GaussianProcess(random_state=42)
    gp.fit(X, y)
    y_pred, y_std = gp.predict(X, return_std=True)
    assert y_pred.shape == y_shape
    assert y_std.shape == y_shape


def test_multioutput_gp(sample_data_y2d):
    X, y = sample_data_y2d
    gp = GaussianProcess(random_state=42)
    gp.fit(X, y)
    y_pred, y_std = gp.predict(X, return_std=True)
    assert y_pred.shape == y.shape
    assert y_std.shape == y.shape


def test_gp_param_search(sample_data_y1d):
    X, y = sample_data_y1d
    em = AutoEmulate()
    em.setup(X, y, models=["gp"], param_search_iters=3)
    em.compare()
