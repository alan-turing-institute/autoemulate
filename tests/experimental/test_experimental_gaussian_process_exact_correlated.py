import gpytorch
import pytest
import torch
from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_model_device,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators.gaussian_process.exact_correlated import (
    CorrGPModule as GaussianProcessExact,
)
from autoemulate.experimental.emulators.gaussian_process.kernel import (
    rbf,
    rbf_times_linear,
)
from autoemulate.experimental.emulators.gaussian_process.mean import constant_mean
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


def test_gp_deterministic_with_seed(sample_data_y1d, new_data_y1d):
    """
    Gaussian Processes are deterministic given the same data and hyperparameters.
    Check that the random seed does not affect the output.
    """
    x, y = sample_data_y1d
    x2, _ = new_data_y1d

    # Create 2 models that should have the same output
    seed = 42
    set_random_seed(seed)
    model1 = GaussianProcessExact(
        x, y, gpytorch.likelihoods.MultitaskGaussianLikelihood, constant_mean, rbf
    )
    new_seed = 43
    set_random_seed(new_seed)
    model2 = GaussianProcessExact(
        x, y, gpytorch.likelihoods.MultitaskGaussianLikelihood, constant_mean, rbf
    )
    model1.fit(x, y)
    model2.fit(x, y)
    pred1 = model1.predict(x2)
    pred2 = model2.predict(x2)

    assert isinstance(pred1, DistributionLike)
    assert isinstance(pred2, DistributionLike)
    assert torch.allclose(pred1.mean, pred2.mean)
    assert torch.allclose(pred1.variance, pred2.variance)
