import torch
from torch import nn

from autoemulate.experimental.types import DeviceLike


class TorchDeviceError(NotImplementedError):
    """Exception raised when the device is not implemented in torch."""

    def __init__(self, device: str):
        msg = f"Backend ({device}) not implemented."
        super().__init__(msg)


def get_torch_device(device: DeviceLike | None) -> torch.device:
    """
    Gets the device returning the torch default device if None.

    Parameters
    ----------
    device : DeviceLike | None
        The device to get. If None, the default torch device is returned.
    Returns
    -------

    torch.device
        The device.

    Raises
    ------
    TorchDeviceError
        If the device is not a valid torch device.

    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.get_default_device()
    if device in ["cpu", "mps", "cuda"]:
        return torch.device(device)
    raise TorchDeviceError(device)


def check_torch_device_is_available(device: DeviceLike) -> bool:
    """
    Checks if the given device is available.

    Parameters
    ----------
    device : DeviceLike
        The device to check.

    Returns
    -------
    bool
        True if the device is available, False otherwise.

    Raises
    ------
    TorchDeviceError
        If the device is not a valid torch device.

    """
    if isinstance(device, torch.device) or device == "cpu":
        return True
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    raise TorchDeviceError(device)


def check_model_device(model: nn.Module, expected_device: DeviceLike) -> bool:
    """
    Checks if the model is on the expected device.

    Parameters
    ----------
    model : nn.Module
        The model to check.
    expected_device : str
        The expected device.

    Returns
    -------
    bool
        True if the model is on the expected device, False otherwise.
    """
    return (
        str(next(model.parameters()).device).split(":")[0]
        == str(expected_device).split(":")[0]
    )
