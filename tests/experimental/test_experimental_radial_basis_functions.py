import numpy as np
import pytest
import torch
from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators.radial_basis_functions import (
    RadialBasisFunctions,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike
from scipy.interpolate import RBFInterpolator


def test_predict_rbf(sample_data_for_ae_compare, new_data_rbf):
    x, y = sample_data_for_ae_compare
    rbf = RadialBasisFunctions(x, y)
    rbf.fit(x, y)
    x2, _ = new_data_rbf
    y_pred = rbf.predict(x2)

    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == (56, 2)

    RBFscipy = RBFInterpolator(
        x,
        y,
        smoothing=rbf.smoothing,
        kernel=rbf.kernel,
        epsilon=rbf.epsilon,
        degree=rbf.degree,
    )
    y_pred_scipy = RBFscipy(x2)
    assert np.allclose(y_pred, y_pred_scipy, atol=1e-4)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_tune_rbf(sample_data_for_ae_compare, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_for_ae_compare
    n_iter = 5
    tuner = Tuner(x, y, n_iter=n_iter, device=device)
    scores, configs = tuner.run(RadialBasisFunctions)
    assert len(scores) == n_iter
    assert len(configs) == n_iter


def test_rbf_predict_deterministic_with_seed(sample_data_for_ae_compare, new_data_rbf):
    """
    RBFInterpolator should be deterministic given the same data
    and parameters so we do not expect different outputs for different seeds.
    """
    x, y = sample_data_for_ae_compare
    x2, _ = new_data_rbf

    # Set a random seed
    seed = 42
    set_random_seed(seed)
    model1 = RadialBasisFunctions(x, y)
    model1.fit(x, y)
    pred1 = model1.predict(x2)

    # Use a different seed
    new_seed = 43
    set_random_seed(new_seed)
    model2 = RadialBasisFunctions(x, y)
    model2.fit(x, y)
    pred2 = model2.predict(x2)

    assert isinstance(pred1, torch.Tensor)
    assert isinstance(pred2, torch.Tensor)
    assert torch.allclose(pred1, pred2)
