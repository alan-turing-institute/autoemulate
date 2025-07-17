import os
import tempfile

import pytest
from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_ae(sample_data_for_ae_compare, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    x, y = sample_data_for_ae_compare
    ae = AutoEmulate(x, y, device=device)
    best_result = ae.best_result()
    assert best_result is not None
    # Save the best model to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "best_model.pt")
        saved_model_full_path = ae.save(best_result, save_path)
        # Load the model back
        loaded_model = ae.load(saved_model_full_path)
        assert loaded_model is not None
