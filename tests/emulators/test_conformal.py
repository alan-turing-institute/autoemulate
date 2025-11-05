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


def test_conformal_mlp_quantile_method():
    """Test Conformalized Quantile Regression (CQR) method."""

    def f(x):
        return torch.sin(x)

    # Training data
    x_train = torch.rand(100, 3) * 10
    y_train = f(x_train)

    # Calibration data
    x_cal, y_cal = torch.rand(100, 3) * 10, f(torch.rand(100, 3) * 10)

    emulator = ConformalMLP(
        x_train,
        y_train,
        method="quantile",
        layer_dims=[100, 100],
        lr=1e-2,
        epochs=50,
        quantile_emulator_kwargs={"epochs": 50, "lr": 1e-2},
    )
    emulator.fit(x_train, y_train, validation_data=(x_cal, y_cal))

    # Test
    x_test = torch.linspace(0.0, 15.0, steps=100).repeat(1, 3).reshape(-1, 3)
    y_test_hat = emulator.predict(x_test)
    assert isinstance(y_test_hat, DistributionLike)
    assert isinstance(y_test_hat.mean, TensorLike)
    assert isinstance(y_test_hat.variance, TensorLike)
    assert y_test_hat.mean.shape == (100, 3)
    assert y_test_hat.variance.shape == (100, 3)
    assert not y_test_hat.mean.requires_grad

    # Check that intervals vary across input space (not constant width)
    interval_widths = y_test_hat.variance.sqrt() * 2  # approximate width
    # Variance should differ across inputs for quantile method
    assert interval_widths.std() > 0, "Intervals should vary across input space"


def test_conformal_methods_comparison():
    """Compare split vs quantile conformal methods on heteroscedastic data."""

    def heteroscedastic_function(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Function with heteroscedastic noise (variance depends on x)."""
        mean = torch.sin(2 * x)
        # Variance increases with x
        noise_std = 0.1 + 0.3 * (x / 10.0).abs()
        noise = torch.randn_like(mean) * noise_std
        return mean, mean + noise

    # Generate training data with heteroscedastic noise
    torch.manual_seed(42)
    x_train = torch.rand(100, 1) * 10 - 5
    _, y_train = heteroscedastic_function(x_train)

    # Generate calibration data
    x_cal = torch.rand(50, 1) * 10 - 5
    _, y_cal = heteroscedastic_function(x_cal)

    # Generate test data
    x_test = torch.linspace(-5, 5, 50).reshape(-1, 1)

    # Test split method
    model_split = ConformalMLP(
        x_train,
        y_train,
        method="split",
        alpha=0.90,
        layer_dims=[32, 16],
        epochs=50,
        lr=1e-2,
    )
    model_split.fit(x_train, y_train, validation_data=(x_cal, y_cal))
    pred_split = model_split.predict(x_test)

    # Test quantile method
    model_quantile = ConformalMLP(
        x_train,
        y_train,
        method="quantile",
        alpha=0.90,
        layer_dims=[32, 16],
        epochs=50,
        lr=1e-2,
        quantile_emulator_kwargs={"epochs": 50, "lr": 1e-2},
    )
    model_quantile.fit(x_train, y_train, validation_data=(x_cal, y_cal))
    pred_quantile = model_quantile.predict(x_test, with_grad=False)

    # Compare interval widths
    with torch.no_grad():
        # Uniform distribution bounds provide interval limits directly
        split_base = pred_split.base_dist  # type: ignore[attr-defined]
        quantile_base = pred_quantile.base_dist  # type: ignore[attr-defined]

        width_split = (split_base.high - split_base.low).squeeze()
        width_quantile = (quantile_base.high - quantile_base.low).squeeze()

    # Split conformal should have more uniform widths
    # Quantile conformal should have more variable widths
    assert width_quantile.std() >= width_split.std(), (
        "Quantile method should have more variable interval widths"
    )

    # Both methods should produce valid predictions
    assert pred_split.mean.shape == x_test.shape  # type: ignore[attr-defined]
    assert pred_quantile.mean.shape == x_test.shape  # type: ignore[attr-defined]
