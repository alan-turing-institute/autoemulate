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

    # Required to make test work if both being run locally or
    # run on CI.
    MPS_ON = not TURN_OFF_MPS_IF_RUNNING_CI
    if TURN_OFF_MPS_IF_RUNNING_CI and device == "mps":
        device = "cpu"

    with (
        patch("torch.backends.mps.is_available", return_value=MPS_ON),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch.object(torch, "xpu", fake_xpu),
    ):
        torch_device_mixin = TorchDeviceMixin(device)
        assert torch_device_mixin.device == torch.device(device)
