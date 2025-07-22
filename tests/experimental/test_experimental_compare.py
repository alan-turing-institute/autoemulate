import pytest
from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.transforms.base import AutoEmulateTransform


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_ae(sample_data_for_ae_compare, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    x, y = sample_data_for_ae_compare
    AutoEmulate(x, y, device=device)


def test_ae_with_str_models_and_dict_transforms(sample_data_for_ae_compare):
    """Test AutoEmulate with models passed as strings and transforms as dictionaries."""
    x, y = sample_data_for_ae_compare
    models: list[str | type[Emulator]] = ["mlp", "RandomForest", "gpe"]
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
    assert "GaussianProcessExact" in result_model_names
