import pytest
import torch
from autoemulate.core.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.core.tuner import Tuner
from autoemulate.core.types import TensorLike
from autoemulate.emulators.nn.mlp import MLP, _generate_mlp_docstring


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

    assert isinstance(pred1, torch.Tensor)
    assert isinstance(pred2, torch.Tensor)
    assert isinstance(pred3, torch.Tensor)
    assert torch.allclose(pred1, pred2)
    msg = "Predictions should differ with different seeds."
    assert not torch.allclose(pred1, pred3), msg


def test_mlp_docstring_no_additional():
    expected_docstring = """
    Parameters
    ----------
    x: TensorLike
        Input features.
    y: TensorLike
        Target values.
    activation_cls: type[nn.Module]
        Activation function to use in the hidden layers. Defaults to `nn.ReLU`.
    layer_dims: list[int] | None
        Dimensions of the hidden layers. If None, defaults to [32, 16].
        Defaults to None.
    weight_init: str
        Weight initialization method. Options are "default", "normal", "uniform",
        "zeros", "ones", "xavier_uniform", "xavier_normal", "kaiming_uniform",
        "kaiming_normal". Defaults to "default".
    scale: float
        Scale parameter for weight initialization methods. Used as:
        - gain for Xavier methods
        - std for normal distribution
        - bound for uniform distribution (range: [-scale, scale])
        - ignored for Kaiming methods (uses optimal scaling)
        Defaults to 1.0.
    bias_init: str
        Bias initialization method. Options: "zeros", "default":
            - "zeros" initializes biases to zero
            - "default" uses PyTorch's default uniform initialization
    dropout_prob: float | None
        Dropout probability for regularization. If None, no dropout is applied.
        Defaults to None.
    lr: float
        Learning rate for the optimizer. Defaults to 1e-2.
    params_size: int
        Number of parameters to predict per output dimension. Defaults to 1.
    random_seed: int | None
        Random seed for reproducibility. If None, no seed is set. Defaults to None.
    device: DeviceLike | None
        Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to None.
    scheduler_cls: type[LRScheduler] | None
        Learning rate scheduler class. If None, no scheduler is used. Defaults to
        None.
    scheduler_params: dict | None
        Additional keyword arguments related to the scheduler.

    Raises
    ------
    ValueError
        If the input dimensions of `x` and `y` are not matrices.
    """

    assert _generate_mlp_docstring("", None) == expected_docstring


def test_mlp_docstring_with_args():
    expected_docstring = """
    Parameters
    ----------
    x: TensorLike
        Input features.
    y: TensorLike
        Target values.
    activation_cls: type[nn.Module]
        Activation function to use in the hidden layers. Defaults to `nn.ReLU`.
    layer_dims: list[int] | None
        Dimensions of the hidden layers. If None, defaults to [32, 16].
        Defaults to None.
    weight_init: str
        Weight initialization method. Options are "default", "normal", "uniform",
        "zeros", "ones", "xavier_uniform", "xavier_normal", "kaiming_uniform",
        "kaiming_normal". Defaults to "default".
    scale: float
        Scale parameter for weight initialization methods. Used as:
        - gain for Xavier methods
        - std for normal distribution
        - bound for uniform distribution (range: [-scale, scale])
        - ignored for Kaiming methods (uses optimal scaling)
        Defaults to 1.0.
    bias_init: str
        Bias initialization method. Options: "zeros", "default":
            - "zeros" initializes biases to zero
            - "default" uses PyTorch's default uniform initialization
    dropout_prob: float | None
        Dropout probability for regularization. If None, no dropout is applied.
        Defaults to 0.4.
    lr: float
        Learning rate for the optimizer. Defaults to 1e-2.
    params_size: int
        Number of parameters to predict per output dimension. Defaults to 1.
    random_seed: int | None
        Random seed for reproducibility. If None, no seed is set. Defaults to None.
    device: DeviceLike | None
        Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to None.
    scheduler_cls: type[LRScheduler] | None
        Learning rate scheduler class. If None, no scheduler is used. Defaults to
        None.
    scheduler_params: dict | None
        Additional keyword arguments related to the scheduler.
    another_arg: str
        Description for another_arg.

    Raises
    ------
    ValueError
        If the input dimensions of `x` and `y` are not matrices.
    """
    another_arg = """
    another_arg: str
        Description for another_arg.
    """
    assert _generate_mlp_docstring(another_arg, 0.4) == expected_docstring
