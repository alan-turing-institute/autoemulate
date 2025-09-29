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


def test_get_model_subset():
    """Test getting a subset of models based on pytroch and probabilistic flags."""

    x, y = torch.rand(10, 2), torch.rand(10)
    probabilistic_subset = {e for e in DEFAULT_EMULATORS if e.supports_uq}

    ae = AutoEmulate(x, y, only_probabilistic=True, model_params={})
    assert set(ae.models) == probabilistic_subset
