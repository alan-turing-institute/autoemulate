import pytest
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators.nn.mlp import MLP
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import TensorLike


def test_predict_mlp(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    mlp = MLP(x, y)
    mlp.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = mlp.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == y.unsqueeze(1).shape


def test_multioutput_mlp(sample_data_y2d, new_data_y2d):
    x, y = sample_data_y2d
    x2, _ = new_data_y2d
    mlp = MLP(x, y)
    mlp.fit(x, y)
    y_pred = mlp.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == (20, 2)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_tune_mlp(sample_data_y1d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5, device=device)
    scores, configs = tuner.run(MLP)
    assert len(scores) == 5
    assert len(configs) == 5
