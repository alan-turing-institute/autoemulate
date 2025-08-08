from unittest.mock import MagicMock, patch

import pytest
import torch
from autoemulate.core.device import (
    SUPPORTED_DEVICES,
    TURN_OFF_MPS_IF_RUNNING_CI,
    TorchDeviceMixin,
)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_device(device):
    fake_xpu = MagicMock()
    fake_xpu.is_available.return_value = True

    # Change the expected MPS availability depending on
    # whether we are running in a CI environment or not.
    MPS_AVAILABLE = not TURN_OFF_MPS_IF_RUNNING_CI

    with (
        patch("torch.backends.mps.is_available", return_value=MPS_AVAILABLE),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch.object(torch, "xpu", fake_xpu),
    ):
        torch_device_mixin = TorchDeviceMixin(device)
        assert torch_device_mixin.device == torch.device(device)
