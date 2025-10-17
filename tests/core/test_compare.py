import os
import tempfile

import pytest
import torch
from autoemulate.core.compare import AutoEmulate
from autoemulate.core.device import SUPPORTED_DEVICES, check_torch_device_is_available
from autoemulate.emulators import DEFAULT_EMULATORS
from autoemulate.emulators.base import Emulator
from torch.distributions import Transform


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_ae(sample_data_for_ae_compare, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    x, y = sample_data_for_ae_compare
    ae = AutoEmulate(x, y, device=device, n_iter=2, n_splits=2)
    best_result = ae.best_result()
    assert best_result is not None
    # Save the best model to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir)
        saved_model_full_path = ae.save(best_result, save_path)
        # Load the model back
        loaded_model = ae.load(saved_model_full_path)
        assert loaded_model is not None


def test_ae_with_str_models_and_dict_transforms(sample_data_for_ae_compare):
    """Test AutoEmulate with models passed as strings and transforms as dictionaries."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest", "GaussianProcessRBF"]
    x_transforms_list: list[list[Transform | dict]] = [
        [{"standardize": {}}],
        [{"pca": {"n_components": 3}}],
    ]
    y_transforms_list: list[list[Transform | dict]] = [[{"standardize": {}}]]

    ae = AutoEmulate(
        x,
        y,
        models=models,
        x_transforms_list=x_transforms_list,
        y_transforms_list=y_transforms_list,
        n_iter=2,
    )

    assert len(ae.results) > 0

    # Check that the models were properly converted from strings
    result_model_names = [result.model_name for result in ae.results]

    assert "MLP" in result_model_names
    assert "RandomForest" in result_model_names
    assert "GaussianProcessRBF" in result_model_names


def test_ae_no_tuning(sample_data_for_ae_compare):
    """Test AutoEmulate with model tuning disabled."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest", "GaussianProcessRBF"]

    ae = AutoEmulate(x, y, models=models, model_params={})

    assert len(ae.results) > 0

    # Check that the models were properly converted from strings
    result_model_names = [result.model_name for result in ae.results]

    assert "MLP" in result_model_names
    assert "RandomForest" in result_model_names
    assert "GaussianProcessRBF" in result_model_names

    mlp_params = ae.get_result(0).params
    assert mlp_params != {}
    assert "epochs" in mlp_params
    assert "layer_dims" in mlp_params
    assert "lr" in mlp_params
    assert "batch_size" in mlp_params
    assert "weight_init" in mlp_params
    assert "scale" in mlp_params
    assert "bias_init" in mlp_params
    assert "dropout_prob" in mlp_params

    rf_params = ae.get_result(1).params
    assert rf_params != {}
    assert "n_estimators" in rf_params
    assert "min_samples_split" in rf_params
    assert "min_samples_leaf" in rf_params
    assert "max_features" in rf_params
    assert "bootstrap" in rf_params
    assert "oob_score" in rf_params
    assert "max_depth" in rf_params
    assert "max_samples" in rf_params

    gp_params = ae.get_result(2).params
    assert gp_params == {}
    # TODO: update test as part of #867
    # assert gp_params != {}
    # assert "mean_module_fn" in gp_params
    # assert "covar_module_fn" in gp_params
    # assert "epochs" in gp_params
    # assert "lr" in gp_params
    # assert "likelihood_cls" in gp_params


def test_get_model_subset():
    """Test getting a subset of models based on pytroch and probabilistic flags."""

    x, y = torch.rand(10, 2), torch.rand(10)
    probabilistic_subset = {e for e in DEFAULT_EMULATORS if e.supports_uq}

    ae = AutoEmulate(x, y, only_probabilistic=True, model_params={})
    assert set(ae.models) == probabilistic_subset


@pytest.mark.parametrize(
    "tuning_metric",
    ["r2", "rmse", "mse", "mae"],
)
def test_ae_with_different_tuning_metrics(sample_data_for_ae_compare, tuning_metric):
    """Test AutoEmulate with different tuning metrics."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest"]

    ae = AutoEmulate(
        x,
        y,
        models=models,
        tuning_metric=tuning_metric,
        n_iter=2,
        n_splits=2,
        model_params={},  # Skip tuning for speed
    )

    assert len(ae.results) > 0
    # Verify that the tuning metric was set correctly
    assert ae.tuning_metric.name == tuning_metric
    # Verify results have test_metrics
    for result in ae.results:
        assert result.test_metrics is not None
        assert len(result.test_metrics) > 0


@pytest.mark.parametrize(
    "evaluation_metrics",
    [
        ["r2"],
        ["rmse"],
        ["mse"],
        ["mae"],
        ["r2", "rmse"],
        ["r2", "mse", "mae"],
        ["rmse", "mae"],
    ],
)
def test_ae_with_different_evaluation_metrics(
    sample_data_for_ae_compare, evaluation_metrics
):
    """Test AutoEmulate with different evaluation metrics."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest"]

    ae = AutoEmulate(
        x,
        y,
        models=models,
        evaluation_metrics=evaluation_metrics,
        n_iter=2,
        n_splits=2,
        model_params={},  # Skip tuning for speed
    )

    assert len(ae.results) > 0
    # Verify that evaluation metrics were set correctly
    assert len(ae.evaluation_metrics) == len(evaluation_metrics)
    metric_names = [m.name for m in ae.evaluation_metrics]
    assert metric_names == evaluation_metrics

    # Verify that all specified metrics are in test_metrics for each result
    for result in ae.results:
        assert result.test_metrics is not None
        for metric_name in evaluation_metrics:
            assert metric_name in result.test_metrics
            # Verify the metric value is a tuple of (mean, std)
            assert isinstance(result.test_metrics[metric_name], tuple)
            assert len(result.test_metrics[metric_name]) == 2


@pytest.mark.parametrize(
    ("tuning_metric", "evaluation_metrics"),
    [
        ("r2", ["r2", "rmse"]),
        ("rmse", ["r2", "rmse"]),
        ("mse", ["mse", "mae"]),
        ("mae", ["r2", "mae"]),
        ("r2", ["rmse", "mse", "mae"]),
        ("rmse", ["r2"]),
    ],
)
def test_ae_with_tuning_and_evaluation_metric_combinations(
    sample_data_for_ae_compare, tuning_metric, evaluation_metrics
):
    """Test AutoEmulate with various combinations of tuning and evaluation metrics."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest"]

    ae = AutoEmulate(
        x,
        y,
        models=models,
        tuning_metric=tuning_metric,
        evaluation_metrics=evaluation_metrics,
        n_iter=2,
        n_splits=2,
        model_params={},  # Skip tuning for speed
    )

    assert len(ae.results) > 0
    # Verify tuning metric
    assert ae.tuning_metric.name == tuning_metric
    # Verify evaluation metrics
    assert len(ae.evaluation_metrics) == len(evaluation_metrics)
    metric_names = [m.name for m in ae.evaluation_metrics]
    assert metric_names == evaluation_metrics

    # Verify all metrics are computed in results
    for result in ae.results:
        assert result.test_metrics is not None
        for metric_name in evaluation_metrics:
            assert metric_name in result.test_metrics


def test_ae_with_same_tuning_and_evaluation_metric(sample_data_for_ae_compare):
    """Test AutoEmulate when the same metric is used for tuning and evaluation."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest"]

    ae = AutoEmulate(
        x,
        y,
        models=models,
        tuning_metric="rmse",
        evaluation_metrics=["rmse"],
        n_iter=2,
        n_splits=2,
        model_params={},  # Skip tuning for speed
    )

    assert len(ae.results) > 0
    assert ae.tuning_metric.name == "rmse"
    assert len(ae.evaluation_metrics) == 1
    assert ae.evaluation_metrics[0].name == "rmse"

    for result in ae.results:
        assert "rmse" in result.test_metrics


def test_ae_with_maximizing_and_minimizing_metrics(sample_data_for_ae_compare):
    """Test AutoEmulate with a mix of maximizing and minimizing metrics."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest"]

    # r2 is maximizing, rmse/mse/mae are minimizing
    ae = AutoEmulate(
        x,
        y,
        models=models,
        tuning_metric="r2",
        evaluation_metrics=["r2", "rmse", "mse", "mae"],
        n_iter=2,
        n_splits=2,
        model_params={},  # Skip tuning for speed
    )

    assert len(ae.results) > 0
    # Verify metric properties
    assert ae.tuning_metric.maximize is True
    assert ae.evaluation_metrics[0].maximize is True  # r2
    assert ae.evaluation_metrics[1].maximize is False  # rmse
    assert ae.evaluation_metrics[2].maximize is False  # mse
    assert ae.evaluation_metrics[3].maximize is False  # mae

    # Verify all metrics are computed
    for result in ae.results:
        assert len(result.test_metrics) == 4
        assert all(
            metric in result.test_metrics for metric in ["r2", "rmse", "mse", "mae"]
        )
