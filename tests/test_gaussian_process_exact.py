import itertools

import pytest
import torch
from autoemulate.core.device import (
    SUPPORTED_DEVICES,
    check_model_device,
    check_torch_device_is_available,
)
from autoemulate.core.tuner import Tuner
from autoemulate.core.types import DistributionLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.gaussian_process.exact import (
    GaussianProcess,
    GaussianProcessCorrelated,
)

GPS = [GaussianProcess, GaussianProcessCorrelated]


@pytest.mark.parametrize("emulator", GPS)
def test_predict_with_uncertainty_gp(sample_data_y1d, new_data_y1d, emulator):
    x, y = sample_data_y1d
    gp = emulator(x, y)
    gp.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == y.unsqueeze(1).shape
    assert y_pred.variance.shape == y.unsqueeze(1).shape
    assert not y_pred.mean.requires_grad

    y_pred_grad = gp.predict(x2, with_grad=True)
    assert y_pred_grad.mean.requires_grad


@pytest.mark.parametrize("emulator", GPS)
def test_multioutput_gp(sample_data_y2d, new_data_y2d, emulator):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    gp = emulator(x, y)
    gp.fit(x, y)
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)


@pytest.mark.parametrize(
    ("device", "emulator"), itertools.product(SUPPORTED_DEVICES, GPS)
)
def test_tune_gp(sample_data_y1d, device, emulator):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5, device=device)
    scores, params_list = tuner.run(emulator)
    assert len(scores) == 5
    assert len(params_list) == 5


@pytest.mark.parametrize(
    ("device", "emulator"), itertools.product(SUPPORTED_DEVICES, GPS)
)
def test_device(sample_data_y2d, new_data_y2d, device, emulator):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    gp = emulator(x, y, device=device)
    assert check_model_device(gp, device)
    gp.fit(x, y)
    y_pred = gp.predict(x2)
    assert isinstance(y_pred, DistributionLike)
    assert y_pred.mean.shape == (20, 2)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_gp_deterministic_with_seed(sample_data_y1d, new_data_y1d, device):
    """
    Gaussian Processes are deterministic given the same data and hyperparameters.
    Check that the random seed does not affect the output.
    """
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y1d
    x2, _ = new_data_y1d

    # Create 2 models that should have the same output
    seed = 42
    set_random_seed(seed)
    model1 = GaussianProcess(x, y, device=device)
    new_seed = 43
    set_random_seed(new_seed)
    model2 = GaussianProcess(x, y, device=device)
    set_random_seed(new_seed)
    model3 = GaussianProcess(x, y, device=device)
    model1.fit(x, y)
    model2.fit(x, y)
    model3.fit(x, y)
    pred1 = model1.predict(x2)
    pred2 = model2.predict(x2)
    pred3 = model2.predict(x2)

    assert isinstance(pred1, DistributionLike)
    assert isinstance(pred2, DistributionLike)
    assert isinstance(pred3, DistributionLike)
    assert torch.allclose(pred1.mean, pred2.mean)
    assert torch.allclose(pred1.variance, pred2.variance)
    assert torch.allclose(pred1.mean, pred3.mean)
    assert torch.allclose(pred1.variance, pred3.variance)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_gp_corr_deterministic_with_seed(sample_data_y1d, new_data_y1d, device):
    """Correlated GPs not currently deterministic due to task kernel initialization"""
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y1d
    x2, _ = new_data_y1d
    seed = 42
    new_seed = 43
    model1 = GaussianProcessCorrelated(x, y, device=device, seed=seed)
    model2 = GaussianProcessCorrelated(x, y, device=device, seed=new_seed)
    model3 = GaussianProcessCorrelated(x, y, device=device, seed=seed)
    model1.fit(x, y)
    pred1 = model1.predict(x2)
    model2.fit(x, y)
    pred2 = model2.predict(x2)
    model3.fit(x, y)
    pred3 = model3.predict(x2)

    assert isinstance(pred1, DistributionLike)
    assert isinstance(pred2, DistributionLike)
    assert isinstance(pred3, DistributionLike)
    assert not torch.allclose(pred1.mean, pred2.mean)
    assert not torch.allclose(pred1.variance, pred2.variance)
    assert torch.allclose(pred1.mean, pred3.mean)
    assert torch.allclose(pred1.variance, pred3.variance)
