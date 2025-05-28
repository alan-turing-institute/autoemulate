import logging

import torch
from torch import nn

from autoemulate.experimental.types import DeviceLike, TensorLike

SUPPORTED_DEVICES: list[str] = ["cpu", "mps", "cuda", "xpu"]


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
    if device in SUPPORTED_DEVICES:
        return torch.device(device)
    raise TorchDeviceError(device)


def move_tensors_to_device(
    *args: TensorLike, device: torch.device
) -> tuple[TensorLike, ...]:
    """
    Moves the given tensor to the device.

    Parameters
    ----------
    *args : TensorLike
        The tensors to move.

    Returns
    -------
    tuple[TensorLike, ...]
        The tensors on the device.
    """
    return tuple(tensor.to(device) for tensor in args)


def check_torch_device_is_available(device: DeviceLike) -> bool:
    """
    Checks if the given device type is available.

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
    if device == "cpu" or (
        isinstance(device, torch.device) and device.type == torch.device("cpu").type
    ):
        return True
    if device == "mps" or (
        isinstance(device, torch.device) and device.type == torch.device("mps").type
    ):
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    if isinstance(device, torch.device) and device.type == "cuda":
        if device.index is not None:
            return device.index < torch.cuda.device_count()
        return True
    if device == "xpu" or (
        isinstance(device, torch.device) and device.type == torch.device("xpu").type
    ):
        return torch.xpu.is_available() if hasattr(torch, "xpu") else False
    raise TorchDeviceError(str(device))


def check_model_device(model: nn.Module, expected_device: str) -> bool:
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
        True if the model is on the expected device (ignoring device index), False
        otherwise.
    """
    return (
        str(next(model.parameters()).device).split(":")[0]
        == str(expected_device).split(":")[0]
    )


class TorchDeviceMixin:
    """
    Mixin class to add device management to a PyTorch model.

    Attributes
    ----------
    device : torch.device
        The device to use. If None, the default torch device is used.

    Raises
    ------
    TorchDeviceError
        If the device is not a valid torch device.

    """

    def __init__(self, device: DeviceLike | None = None, cpu_only: bool = False):
        # Warn if given device not CPU and cpu_only
        # TODO: check handling
        if cpu_only and (
            (isinstance(device, str) and device != "cpu")
            or (isinstance(device, torch.device) and torch.device("cpu") != device)
        ):
            msg = (
                f"The device ({device}) must be CPU for given model. Setting device as "
                "'cpu'."
            )
            # warnings.warn(msg, stacklevel=2)
            logging.warning(msg)

        self.device = get_torch_device(device)

        if not check_torch_device_is_available(self.device):
            raise TorchDeviceError(str(self.device))

    def _move_tensors_to_device(self, *args: TensorLike) -> tuple[TensorLike, ...]:
        """
        Moves the given tensor to the device.

        Parameters
        ----------
        *args : TensorLike
            The tensors to move.

        Returns
        -------
        tuple[TensorLike, ...]
            The tensors on the device.
        """
        return move_tensors_to_device(*args, device=self.device)
