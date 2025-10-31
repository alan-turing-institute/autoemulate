"""Tests for metrics configuration and utilities."""

from functools import partial

import pytest
import torch
import torchmetrics
from autoemulate.core.metrics import (
    AVAILABLE_METRICS,
    CRPS,
    MAE,
    MSE,
    R2,
    RMSE,
    CRPSMetric,
    Metric,
    TorchMetrics,
    get_metric_config,
    get_metric_configs,
)
from torch.distributions import Normal

# Tests for the base Metric class


def test_metric_repr():
    """Test the __repr__ method of Metric."""
    # Create a mock metric instance
    metric = Metric()
    metric.name = "test_metric"
    metric.maximize = True

    repr_str = repr(metric)
    assert "Metric" in repr_str
    assert "test_metric" in repr_str
    assert "True" in repr_str


# Tests for the TorchMetrics class


def test_torchmetrics_initialization():
    """Test TorchMetrics initialization."""
    metric = TorchMetrics(metric=torchmetrics.R2Score, name="r2", maximize=True)

    assert metric.name == "r2"
    assert metric.maximize is True
    assert metric.metric == torchmetrics.R2Score


def test_torchmetrics_call():
    """Test calling a TorchMetrics instance."""
    metric_config = TorchMetrics(
        metric=torchmetrics.MeanSquaredError, name="mse", maximize=False
    )

    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.0, 2.0, 3.0])

    # The __call__ method should instantiate the metric
    # Then we need to use it properly with the torchmetrics API
    metric_instance = metric_config.metric()
    metric_instance.update(y_pred, y_true)
    result = metric_instance.compute()

    # For perfect predictions, MSE should be 0
    assert torch.isclose(result, torch.tensor(0.0))


# Tests for predefined metric instances


def test_r2_metric_configuration():
    """Test R2 metric configuration."""
    assert R2.name == "r2"
    assert R2.maximize is True
    assert R2.metric == torchmetrics.R2Score


def test_rmse_metric_configuration():
    """Test RMSE metric configuration."""
    assert RMSE.name == "rmse"
    assert RMSE.maximize is False
    assert isinstance(RMSE.metric, partial)


def test_mse_metric_configuration():
    """Test MSE metric configuration."""
    assert MSE.name == "mse"
    assert MSE.maximize is False
    assert MSE.metric in {
        torchmetrics.MeanAbsoluteError,
        torchmetrics.MeanSquaredError,
    }


def test_mae_metric_configuration():
    """Test MAE metric configuration."""
    assert MAE.name == "mae"
    assert MAE.maximize is False
    assert MAE.metric == torchmetrics.MeanAbsoluteError


def test_available_metrics_dict():
    """Test AVAILABLE_METRICS dictionary."""
    assert "r2" in AVAILABLE_METRICS
    assert "rmse" in AVAILABLE_METRICS
    assert "mse" in AVAILABLE_METRICS
    assert "mae" in AVAILABLE_METRICS

    assert AVAILABLE_METRICS["r2"] == R2
    assert AVAILABLE_METRICS["rmse"] == RMSE
    assert AVAILABLE_METRICS["mse"] == MSE
    assert AVAILABLE_METRICS["mae"] == MAE


def test_r2_computation():
    """Test R2 metric computation."""
    y_pred = torch.tensor([3.0, -0.5, 2.0, 7.0])
    y_true = torch.tensor([2.5, 0.0, 2.0, 8.0])

    # Instantiate, update, and compute
    metric = R2.metric()
    metric.update(y_pred, y_true)
    result = metric.compute()

    # R2 should be between -inf and 1
    assert result <= 1.0
    assert isinstance(result, torch.Tensor)


def test_rmse_computation():
    """Test RMSE metric computation."""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.0, 2.0, 3.0])

    # Instantiate, update, and compute
    metric = RMSE.metric()
    metric.update(y_pred, y_true)
    result = metric.compute()

    # For perfect predictions, RMSE should be 0
    assert torch.isclose(result, torch.tensor(0.0))


def test_mse_computation():
    """Test MSE metric computation."""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    # Instantiate, update, and compute
    metric = MSE.metric()
    metric.update(y_pred, y_true)
    result = metric.compute()

    # MSE should be 1.0 for these predictions
    assert torch.isclose(result, torch.tensor(1.0))


def test_mae_computation():
    """Test MAE metric computation."""
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 3.0, 4.0])

    # Instantiate, update, and compute
    metric = MAE.metric()
    metric.update(y_pred, y_true)
    result = metric.compute()

    # MAE should be 1.0 for these predictions
    assert torch.isclose(result, torch.tensor(1.0))


# Tests for get_metric_config function


def test_get_metric_config_with_string_r2():
    """Test get_metric_config with 'r2' string."""
    config = get_metric_config("r2")

    assert config == R2
    assert config.name == "r2"
    assert config.maximize is True


def test_get_metric_config_with_string_rmse():
    """Test get_metric_config with 'rmse' string."""
    config = get_metric_config("rmse")

    assert config == RMSE
    assert config.name == "rmse"
    assert config.maximize is False


def test_get_metric_config_with_string_mse():
    """Test get_metric_config with 'mse' string."""
    config = get_metric_config("mse")

    assert config == MSE
    assert config.name == "mse"
    assert config.maximize is False


def test_get_metric_config_with_string_mae():
    """Test get_metric_config with 'mae' string."""
    config = get_metric_config("mae")

    assert config == MAE
    assert config.name == "mae"
    assert config.maximize is False


def test_get_metric_config_case_insensitive():
    """Test get_metric_config is case insensitive."""
    config_upper = get_metric_config("R2")
    config_lower = get_metric_config("r2")
    config_mixed = get_metric_config("R2")

    assert config_upper == config_lower == config_mixed == R2


def test_get_metric_config_with_torchmetrics_instance():
    """Test get_metric_config with TorchMetrics instance."""
    custom_metric = TorchMetrics(
        metric=torchmetrics.R2Score, name="custom_r2", maximize=True
    )

    config = get_metric_config(custom_metric)

    assert config == custom_metric
    assert config.name == "custom_r2"


def test_get_metric_config_invalid_string():
    """Test get_metric_config with invalid string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown metric shortcut") as excinfo:
        get_metric_config("invalid_metric")

    assert "Unknown metric shortcut" in str(excinfo.value)
    assert "invalid_metric" in str(excinfo.value)
    assert "Available options" in str(excinfo.value)


def test_get_metric_config_unsupported_type():
    """Test get_metric_config with unsupported type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric type") as excinfo:
        get_metric_config(123)  # type: ignore[arg-type]

    assert "Unsupported metric type" in str(excinfo.value)


def test_get_metric_config_with_none():
    """Test get_metric_config with None raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric type") as excinfo:
        get_metric_config(None)  # type: ignore[arg-type]

    assert "Unsupported metric type" in str(excinfo.value)


# Tests for get_metric_configs function


def test_get_metric_configs_with_strings():
    """Test get_metric_configs with list of strings."""
    metrics = ["r2", "rmse", "mse"]
    configs = get_metric_configs(metrics)

    assert len(configs) == 3
    assert configs[0] == R2
    assert configs[1] == RMSE
    assert configs[2] == MSE


def test_get_metric_configs_with_mixed_types():
    """Test get_metric_configs with mixed types."""
    custom_metric = TorchMetrics(
        metric=torchmetrics.R2Score, name="custom_r2", maximize=True
    )

    metrics = ["r2", custom_metric, "mse"]
    configs = get_metric_configs(metrics)

    assert len(configs) == 3
    assert configs[0] == R2
    assert configs[1] == custom_metric
    assert configs[2] == MSE


def test_get_metric_configs_with_empty_list():
    """Test get_metric_configs with empty list."""
    configs = get_metric_configs([])

    assert len(configs) == 0
    assert configs == []


def test_get_metric_configs_with_single_metric():
    """Test get_metric_configs with single metric."""
    configs = get_metric_configs(["r2"])

    assert len(configs) == 1
    assert configs[0] == R2


def test_get_metric_configs_with_all_available_metrics():
    """Test get_metric_configs with all available metrics."""
    metrics = list(AVAILABLE_METRICS.keys())
    configs = get_metric_configs(metrics)

    assert len(configs) == len(AVAILABLE_METRICS)

    for i, metric_name in enumerate(metrics):
        assert configs[i] == AVAILABLE_METRICS[metric_name]


def test_get_metric_configs_with_torchmetrics_instances():
    """Test get_metric_configs with TorchMetrics instances."""
    metric1 = TorchMetrics(metric=torchmetrics.R2Score, name="r2_1", maximize=True)
    metric2 = TorchMetrics(
        metric=torchmetrics.MeanSquaredError, name="mse_1", maximize=False
    )

    configs = get_metric_configs([metric1, metric2])

    assert len(configs) == 2
    assert configs[0] == metric1
    assert configs[1] == metric2


def test_get_metric_configs_case_insensitive():
    """Test get_metric_configs is case insensitive for strings."""
    metrics = ["R2", "RMSE", "mse", "MaE", "Crps"]
    configs = get_metric_configs(metrics)

    assert len(configs) == 5
    assert configs[0] == R2
    assert configs[1] == RMSE
    assert configs[2] == MSE
    assert configs[3] == MAE
    assert configs[4] == CRPS


# Integration tests for metrics with actual computation


def test_multiple_metrics_on_same_data():
    """Test multiple metrics on the same data."""
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([1.1, 2.2, 2.9, 4.3])

    # Instantiate and compute each metric
    r2_metric = R2.metric()
    r2_metric.update(y_pred, y_true)
    r2_result = r2_metric.compute()

    rmse_metric = RMSE.metric()
    rmse_metric.update(y_pred, y_true)
    rmse_result = rmse_metric.compute()

    mse_metric = MSE.metric()
    mse_metric.update(y_pred, y_true)
    mse_result = mse_metric.compute()

    mae_metric = MAE.metric()
    mae_metric.update(y_pred, y_true)
    mae_result = mae_metric.compute()

    # All should return tensors
    assert isinstance(r2_result, torch.Tensor)
    assert isinstance(rmse_result, torch.Tensor)
    assert isinstance(mse_result, torch.Tensor)
    assert isinstance(mae_result, torch.Tensor)

    # RMSE should be sqrt of MSE
    assert torch.isclose(rmse_result, torch.sqrt(mse_result))


def test_metric_with_multidimensional_tensors():
    """Test metrics with multidimensional tensors."""
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.1, 2.1], [3.1, 4.1]])

    # Metrics should handle multidimensional inputs
    r2_metric = R2.metric()
    r2_metric.update(y_pred, y_true)
    r2_result = r2_metric.compute()

    mse_metric = MSE.metric()
    mse_metric.update(y_pred, y_true)
    mse_result = mse_metric.compute()

    assert isinstance(r2_result, torch.Tensor)
    assert isinstance(mse_result, torch.Tensor)


def test_metric_configs_workflow():
    """Test complete workflow of getting and using metric configs."""
    # Get configs from strings
    configs = get_metric_configs(["r2", "rmse"])

    # Use configs to compute metrics
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.0, 2.0, 3.0])

    results = {}
    for config in configs:
        metric = config.metric()
        metric.update(y_pred, y_true)
        results[config.name] = metric.compute()

    assert "r2" in results
    assert "rmse" in results
    assert torch.isclose(results["r2"], torch.tensor(1.0))  # Perfect R2
    assert torch.isclose(results["rmse"], torch.tensor(0.0))  # Perfect RMSE


# Tests for CRPS metric


def test_crps_in_available_metrics():
    """Test CRPS is in AVAILABLE_METRICS."""
    assert "crps" in AVAILABLE_METRICS
    assert AVAILABLE_METRICS["crps"] == CRPS


def test_crps_deterministic_reduces_to_mae():
    """Test CRPS with deterministic predictions reduces to MAE."""
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

    crps_result = CRPS(y_pred, y_true)

    mae = torch.nn.functional.l1_loss(y_pred, y_true)
    assert torch.isclose(crps_result, mae, rtol=1e-4), "CRPS should equal MAE"


@pytest.mark.parametrize(
    ("batch_size", "n_targets", "n_samples"),
    [
        (10, 1, 5),  # Single target
        (10, 3, 5),  # Multiple targets
        (5, 2, 10),  # More samples than batch
        (100, 5, 20),  # Large batch
        (1, 1, 100),  # Single sample, many ensemble members
    ],
)
def test_crps_with_various_shapes(batch_size, n_targets, n_samples):
    """Test CRPS with various tensor shapes."""
    y_pred = torch.randn(batch_size, n_targets, n_samples)
    y_true = torch.randn(batch_size, n_targets)

    result = CRPS(y_pred, y_true)

    assert result.ndim == 0, "Result should be a scalar tensor"
    assert isinstance(result, torch.Tensor)
    assert result >= 0, "CRPS should be non-negative"


def test_crps_with_distribution():
    """Test CRPS with distribution input."""
    batch_size, n_targets = 10, 3
    y_true = torch.randn(batch_size, n_targets)

    # Create a distribution
    mean = torch.randn(batch_size, n_targets)
    std = torch.ones(batch_size, n_targets) * 0.5
    y_pred_dist = Normal(mean, std)

    result = CRPS(y_pred_dist, y_true, n_samples=500)

    assert result.ndim == 0, "Result should be a scalar tensor"
    assert isinstance(result, torch.Tensor)
    assert result >= 0, "CRPS should be non-negative"


def test_crps_shape_mismatch_raises_error():
    """Test CRPS raises error for shape mismatch."""
    y_pred = torch.randn(10, 3, 5)
    y_true = torch.randn(10, 5)

    with pytest.raises(ValueError, match="does not match"):
        CRPS(y_pred, y_true)


def test_crps_invalid_dimensions_raises_error():
    """Test CRPS raises error for invalid dimensions."""
    y_pred = torch.randn(10, 3, 5, 7)
    y_true = torch.randn(10, 3)

    with pytest.raises(ValueError, match="incompatible"):
        CRPS(y_pred, y_true)


def test_crps_aggregation_across_batch():
    """Test CRPS aggregates correctly across batch dimension."""
    # Create predictions with very different errors in different batch elements
    y_pred = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    y_true = torch.tensor([[1.0, 2.0], [10.0, 20.0]])

    result_perfect = CRPS(y_pred, y_true)

    # Add error to second batch element
    y_pred_error = torch.tensor([[1.0, 2.0], [15.0, 25.0]])
    result_with_error = CRPS(y_pred_error, y_true)

    assert torch.isclose(result_perfect, torch.tensor(0.0))
    assert result_with_error > result_perfect, "Result with error should be larger"


def test_get_metric_config_crps():
    """Test get_metric_config with 'crps' string."""
    config = get_metric_config("crps")

    assert config == CRPS
    assert isinstance(config, CRPSMetric)
    assert config.name == "crps"
    assert config.maximize is False


def test_crps_with_1d_targets():
    """Test CRPS handles 1D targets correctly."""
    # When targets are 1D, predictions should have both target and sample dimensions
    y_pred = torch.randn(10, 1, 5)  # (batch, targets=1, samples)
    y_true = torch.randn(10)  # (batch,) - will be unsqueezed to (batch, 1)

    result = CRPS(y_pred, y_true)

    assert result.ndim == 0, "Result should be a scalar tensor"
    assert isinstance(result, torch.Tensor)
    assert result >= 0, "CRPS should be non-negative"


# Tests for OutputLike support in TorchMetrics


def test_torchmetrics_with_distribution_vs_mean():
    """Test TorchMetrics with distribution gives same result as using mean."""
    batch_size, n_targets = 10, 3
    y_true = torch.randn(batch_size, n_targets)

    # Create a Normal distribution
    mean = torch.randn(batch_size, n_targets)
    std = torch.ones(batch_size, n_targets) * 0.5
    y_pred_dist = Normal(mean, std)

    # Get result with distribution
    result_dist = MSE(y_pred_dist, y_true)

    # Get result with mean tensor
    result_mean = MSE(mean, y_true)

    assert torch.isclose(result_dist, result_mean, rtol=1e-4), "Should be close"


@pytest.mark.parametrize(
    "metric_instance",
    [
        metric
        for metric in AVAILABLE_METRICS.values()
        if isinstance(metric, TorchMetrics)
    ],
)
def test_all_torchmetrics_support_distributions(metric_instance):
    """Test all TorchMetrics instances support distribution inputs."""
    batch_size = 20
    y_true = torch.randn(batch_size, 2)

    # Create a distribution
    mean = torch.randn(batch_size, 2)
    std = torch.ones(batch_size, 2) * 0.3
    y_pred_dist = Normal(mean, std)

    # Should work without error
    result = metric_instance(y_pred_dist, y_true)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert torch.isfinite(result), "Result should be finite"


def test_torchmetrics_distribution_multioutput():
    """Test TorchMetrics with distribution for multioutput case."""
    batch_size, n_outputs = 50, 5
    y_true = torch.randn(batch_size, n_outputs)

    # Create distribution with different means for different outputs
    mean = torch.randn(batch_size, n_outputs)
    std = torch.rand(batch_size, n_outputs) * 0.5 + 0.1  # Avoid zero std
    y_pred_dist = Normal(mean, std)

    # Test with MAE
    result = MAE(y_pred_dist, y_true)

    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result >= 0, "MAE should be non-negative"
