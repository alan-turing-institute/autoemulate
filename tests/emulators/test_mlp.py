import pytest
import torch
from autoemulate.core.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.core.tuner import Tuner
from autoemulate.core.types import TensorLike
from autoemulate.emulators.nn.mlp import MLP


def test_predict_mlp(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    mlp = MLP(x, y)
    mlp.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = mlp.predict(x2)
    assert isinstance(y_pred, TensorLike)
    assert y_pred.shape == y.unsqueeze(1).shape
    assert not y_pred.requires_grad

    y_pred_grad = mlp.predict(x2, with_grad=True)
    assert isinstance(y_pred_grad, TensorLike)
    assert y_pred_grad.requires_grad


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
    n_iter = 5
    tuner = Tuner(x, y, n_iter=n_iter, device=device)
    scores, params_list = tuner.run(MLP)
    assert len(scores) == n_iter
    assert len(params_list) == n_iter


def test_mlp_predict_deterministic_with_seed(sample_data_y2d, new_data_y2d):
    """
    Test that fitting two models with the same seed and data
    produces identical predictions.
    """
    x, y = sample_data_y2d
    x2, _ = new_data_y2d

    # Set a random seed for reproducibility
    seed = 42
    model1 = MLP(x, y, random_seed=seed)
    model1.fit(x, y)
    pred1 = model1.predict(x2)

    # Use the same seed to ensure deterministic behavior
    model2 = MLP(x, y, random_seed=seed)
    model2.fit(x, y)
    pred2 = model2.predict(x2)

    # Use a different seed to ensure deterministic behavior
    new_seed = 43
    model3 = MLP(x, y, random_seed=new_seed)
    model3.fit(x, y)
    pred3 = model3.predict(x2)

    assert isinstance(pred1, TensorLike)
    assert isinstance(pred2, TensorLike)
    assert isinstance(pred3, TensorLike)
    assert torch.allclose(pred1, pred2)
    msg = "Predictions should differ with different seeds."
    assert not torch.allclose(pred1, pred3), msg
