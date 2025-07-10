import pytest
from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.device import (
    SUPPORTED_DEVICES,
    check_torch_device_is_available,
)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_ae(sample_data_y2d, device):
    if not check_torch_device_is_available(device):
        pytest.skip(f"Device ({device}) is not available.")

    x, y = sample_data_y2d
    AutoEmulate(x, y, device=device)
