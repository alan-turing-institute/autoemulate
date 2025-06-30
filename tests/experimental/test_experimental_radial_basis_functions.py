import pytest
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators.radial_basis_functions import (
    RadialBasisFunctions,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_rbf(sample_data_rbf, new_data_rbf):
    x, y = sample_data_rbf
    rbf = RadialBasisFunctions(x, y)
    rbf.fit(x, y)
    x2, _ = new_data_rbf
    y_pred = rbf.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == (56, 2)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_tune_rbf(sample_data_rbf, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_rbf
    n_iter = 5
    tuner = Tuner(x, y, n_iter=n_iter, device=device)
    scores, configs = tuner.run(RadialBasisFunctions)
    assert len(scores) == n_iter
    assert len(configs) == n_iter
