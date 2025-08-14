import os
import tempfile

import pytest
from autoemulate.core.compare import AutoEmulate
from autoemulate.core.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.emulators.base import Emulator
from autoemulate.transforms.base import AutoEmulateTransform


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
    models: list[str | type[Emulator]] = ["mlp", "RandomForest", "gp"]
    x_transforms_list: list[list[AutoEmulateTransform | dict]] = [
        [{"standardize": {}}],
        [{"pca": {"n_components": 3}}],
    ]
    y_transforms_list: list[list[AutoEmulateTransform | dict]] = [[{"standardize": {}}]]

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
    print(ae.results)
    assert "MLP" in result_model_names
    assert "RandomForest" in result_model_names
    assert "GaussianProcess" in result_model_names
