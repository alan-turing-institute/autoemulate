from unittest.mock import MagicMock, patch

import pytest
import torch
from autoemulate.core.device import (
    SUPPORTED_DEVICES,
    TorchDeviceMixin,
)


@pytest.mark.parametrize("device", SUPPORTED_DEVICES)
def test_device(device):
    fake_xpu = MagicMock()
    fake_xpu.is_available.return_value = True
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch.object(torch, "xpu", fake_xpu),
    ):
        torch_device_mixin = TorchDeviceMixin(device)
        assert torch_device_mixin.device == torch.device(device)
