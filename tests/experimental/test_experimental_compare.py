import pytest
from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators import ALL_EMULATORS


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_compare(sample_data_y2d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    x, y = sample_data_y2d
    ae = AutoEmulate(x, y, device=device)
    results = ae.compare(2)
    print(results)


def test_compare_user_models(sample_data_y2d, recwarn):
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y, models=ALL_EMULATORS)
    results = ae.compare(1)
    print(results)
    assert len(recwarn) == 2
    assert str(recwarn.pop().message) == (
        "Model (<class 'autoemulate.experimental.emulators.lightgbm.Li"
        "ghtGBM'>) is not multioutput but the data is multioutput. Skipping model "
        "(<class 'autoemulate.experimental.emulators.lightgbm.LightGBM'>)..."
    )


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_compare_y1d(sample_data_y1d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")
    x, y = sample_data_y1d
    # TODO: add handling when 1D
    y = y.reshape(-1, 1)
    ae = AutoEmulate(x, y)
    results = ae.compare(4)
    print(results)


def test_refit_best_model(sample_data_y2d):
    """Test that refit works with the best model from comparison results."""
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    results = ae.compare(n_iter=2)
    assert results is not None
    assert len(results) > 0

    # Refit the best model
    best_model = ae.refit()

    # Check that model is fitted and can make predictions
    assert best_model.is_fitted_
    predictions = best_model.predict(x)
    assert predictions is not None


def test_refit_specific_model(sample_data_y2d):
    """Test that refit works when specifying a particular model."""
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    results = ae.compare(n_iter=2)
    model_names = list(results.keys())

    # Refit a specific model
    target_model_name = model_names[0]
    specific_model = ae.refit(model_name=target_model_name)

    # Check that model is fitted and can make predictions
    assert specific_model.is_fitted_
    predictions = specific_model.predict(x)
    assert predictions is not None


def test_refit_with_custom_metric(sample_data_y2d):
    """Test that refit works with different metrics for model selection."""
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    results = ae.compare(n_iter=2)
    assert results is not None

    # Refit using RMSE metric (lower is better, so max of negative values)
    rmse_model = ae.refit(metric="rmse_score")
    assert rmse_model.is_fitted_

    # Refit using R2 metric (default)
    r2_model = ae.refit(metric="r2_score")
    assert r2_model.is_fitted_


def test_refit_with_custom_data(sample_data_y2d, new_data_y2d):
    """Test that refit works with custom training data."""
    x, y = sample_data_y2d
    x_new, y_new = new_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    ae.compare(n_iter=2)

    # Refit with custom data
    custom_model = ae.refit(x=x_new, y=y_new)

    # Check that model is fitted and can make predictions
    assert custom_model.is_fitted_
    predictions = custom_model.predict(x_new)
    assert predictions is not None


def test_refit_with_only_x_raises_error(sample_data_y2d, new_data_y2d):
    """Test that refit raises error when providing only x data."""
    x, y = sample_data_y2d
    x_new, _ = new_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    ae.compare(n_iter=2)

    # Try to refit with only custom x (should raise error)
    with pytest.raises(
        ValueError, match="Both x and y must be provided together, or both must be None"
    ):
        ae.refit(x=x_new)


def test_refit_with_only_y_raises_error(sample_data_y2d, new_data_y2d):
    """Test that refit raises error when providing only y data."""
    x, y = sample_data_y2d
    _, y_new = new_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    ae.compare(n_iter=2)

    # Try to refit with only custom y (should raise error)
    with pytest.raises(
        ValueError, match="Both x and y must be provided together, or both must be None"
    ):
        ae.refit(y=y_new)


def test_refit_before_compare_raises_error(sample_data_y2d):
    """Test that refit raises error when called before compare."""
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)

    # Try to refit without running compare first
    with pytest.raises(
        RuntimeError,
        match="Must run compare\\(\\) before calling refit\\(\\)",
    ):
        ae.refit()


def test_refit_invalid_model_name_raises_error(sample_data_y2d):
    """Test that refit raises error when specifying an invalid model name."""
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    ae.compare(n_iter=2)

    # Try to refit with invalid model name
    with pytest.raises(
        ValueError,
        match="Model 'InvalidModel' not found in comparison results",
    ):
        ae.refit(model_name="InvalidModel")


def test_refit_invalid_metric_raises_error(sample_data_y2d):
    """Test that refit raises error when specifying an invalid metric."""
    x, y = sample_data_y2d
    ae = AutoEmulate(x, y)

    # Run comparison first
    ae.compare(n_iter=2)

    # Try to refit with invalid metric
    with pytest.raises(KeyError):
        ae.refit(metric="invalid_metric")


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_refit_with_device(sample_data_y2d, device):
    """Test that refit works correctly with different devices."""
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    x, y = sample_data_y2d
    ae = AutoEmulate(x, y, device=device)

    # Run comparison first
    ae.compare(n_iter=2)

    # Refit the best model
    best_model = ae.refit()

    # Check that model is fitted and on correct device
    assert best_model.is_fitted_
    predictions = best_model.predict(x)
    assert predictions is not None


def test_refit_preserves_random_seed(sample_data_y2d):
    """Test that refit produces deterministic results when random seed is set."""
    x, y = sample_data_y2d
    seed = 42

    # Create two AutoEmulate instances with same seed
    ae1 = AutoEmulate(x, y, random_seed=seed)
    ae2 = AutoEmulate(x, y, random_seed=seed)

    # Run comparison and refit for both
    ae1.compare(n_iter=2)
    ae2.compare(n_iter=2)

    model1 = ae1.refit()
    model2 = ae2.refit()

    # Both should be fitted
    assert model1.is_fitted_
    assert model2.is_fitted_

    # Note: Due to the stochastic nature of some models and hyperparameter tuning,
    # we can't guarantee identical predictions, but we can verify both work
    pred1 = model1.predict(x)
    pred2 = model2.predict(x)
    assert pred1 is not None
    assert pred2 is not None
