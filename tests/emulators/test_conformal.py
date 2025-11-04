import torch
from autoemulate.core.types import DistributionLike, TensorLike
from autoemulate.emulators.conformal import ConformalMLP


def test_conformal_mlp():
    def f(x):
        return torch.sin(x)

    # Training data
    x_train = torch.rand(100, 3) * 10
    y_train = f(x_train)

    # Calibration data
    x_cal, y_cal = torch.rand(100, 3) * 10, f(torch.rand(100, 3) * 10)

    emulator = ConformalMLP(x_train, y_train, layer_dims=[100, 100], lr=1e-2)
    emulator.fit(x_train, y_train, validation_data=(x_cal, y_cal))

    # Test
    x_test = torch.linspace(0.0, 15.0, steps=1000).repeat(1, 3).reshape(-1, 3)
    y_test_hat = emulator.predict(x_test)
    assert isinstance(y_test_hat, DistributionLike)
    assert isinstance(y_test_hat.mean, TensorLike)
    assert isinstance(y_test_hat.variance, TensorLike)
    assert y_test_hat.mean.shape == (1000, 3)
    assert y_test_hat.variance.shape == (1000, 3)
    assert not y_test_hat.mean.requires_grad

    y_test_hat_grad = emulator.predict(x_test, with_grad=True)
    assert y_test_hat_grad.mean.requires_grad  # type: ignore  # noqa: PGH003
