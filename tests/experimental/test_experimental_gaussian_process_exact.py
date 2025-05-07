import gpytorch
import pytest
from autoemulate.emulators.gaussian_process import constant_mean, rbf, rbf_times_linear
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_model_device,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators.gaussian_process.exact import (
    GaussianProcessExact,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DistributionLike


def test_predict_with_uncertainty_gp(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    gp = GaussianProcessExact(
        x,
        y,
        gpytorch.likelihoods.MultitaskGaussianLikelihood,
        constant_mean,
        rbf,
    )
    gp.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == y.unsqueeze(1).shape
    assert y_pred.variance.shape == y.unsqueeze(1).shape


def test_multioutput_gp(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    gp = GaussianProcessExact(
        x,
        y,
        gpytorch.likelihoods.MultitaskGaussianLikelihood,
        constant_mean,
        rbf_times_linear,
    )
    gp.fit(x, y)
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_tune_gp(sample_data_y1d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5, device=device)
    scores, configs = tuner.run(GaussianProcessExact)
    assert len(scores) == 5
    assert len(configs) == 5


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_device(sample_data_y2d, new_data_y2d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    gp = GaussianProcessExact(x, y, device=device)
    assert check_model_device(gp, device)
    gp.fit(x, y)
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)
