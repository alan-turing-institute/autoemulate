import logging
import os

import gpytorch
import torch
from torch import nn
from typing_extensions import Self

from autoemulate.core.types import DeviceLike, TensorLike

SUPPORTED_DEVICES: list[str] = ["cpu", "mps", "cuda", "xpu"]

# Set this environment variable (to anything) to force mps to turn off.
# This is necessary because in github runners
# torch.backends.mps.is_available() returns true although it isn't.
if "TURN_OFF_MPS_IF_RUNNING_CI" in os.environ:
    TURN_OFF_MPS_IF_RUNNING_CI = True
else:
    TURN_OFF_MPS_IF_RUNNING_CI = False


class TorchDeviceError(NotImplementedError):
    """Exception raised when the device is not implemented in torch."""

    def __init__(self, device: str):
        msg = f"Backend ({device}) not implemented."
        super().__init__(msg)


def get_torch_device(device: DeviceLike | None) -> torch.device:
    """
    Get the device returning the torch default device if None.

    Parameters
    ----------
    device: DeviceLike | None
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
    Move the given tensor to the device.

    Parameters
    ----------
    *args: TensorLike
        The tensors to move.
    device: torch.device
        The device to move the tensors to.

    Returns
    -------
    tuple[TensorLike, ...]
        The tensors on the device.
    """
    return tuple(tensor.to(device) for tensor in args)


# ruff: noqa: PLR0911
def check_torch_device_is_available(device: DeviceLike) -> bool:
    """
    Check if the given device type is available.

    Parameters
    ----------
    device: DeviceLike
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
        if TURN_OFF_MPS_IF_RUNNING_CI:
            return False
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
    Check if the model is on the expected device.

    Parameters
    ----------
    model: nn.Module
        The model to check.
    expected_device: str
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
    device: torch.device
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
        Move the given tensor to the device.

        Parameters
        ----------
        *args: TensorLike
            The tensors to move.

        Returns
        -------
        tuple[TensorLike, ...]
            The tensors on the device.
        """
        return move_tensors_to_device(*args, device=self.device)

    def to(self, *args, **kwargs) -> Self:
        """
        Move to the given device (and optionally cast dtype).

        Mirrors the API of :meth:`torch.nn.Module.to` so subclasses that also
        inherit from :class:`nn.Module` do not produce conflicting overrides.

        Updates ``self.device`` when a device is supplied, walks instance
        attributes to move any owned tensors and device-aware children
        (recursing one level into list/tuple/dict containers), then delegates
        to :meth:`nn.Module.to` to move parameters and buffers when applicable.
        """
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]
        if device is not None:
            self.device = get_torch_device(device)

        # nn.Module params/buffers live in _parameters/_buffers (not vars(self))
        # so they are not double-moved by this walk.
        for name, val in list(vars(self).items()):
            moved = _move_value(val, args, kwargs)
            if moved is not val:
                setattr(self, name, moved)

        if isinstance(self, nn.Module):
            nn.Module.to(self, *args, **kwargs)

        # gpytorch caches device-bound tensors (e.g. ExactGP.prediction_strategy)
        # in plain attributes that nn.Module.to does not track. Invalidate them
        # so the next predict rebuilds them on the new device.
        if isinstance(self, gpytorch.Module):
            self._clear_cache()
        return self


def _move_value(val, args, kwargs):
    """Move a tensor/device-aware child, or recurse one level into containers."""
    if isinstance(val, torch.Tensor):
        return val.to(*args, **kwargs)
    if isinstance(val, nn.Module | TorchDeviceMixin):
        val.to(*args, **kwargs)
        return val
    if isinstance(val, list):
        for i, item in enumerate(val):
            val[i] = _move_value(item, args, kwargs)
        return val
    if isinstance(val, tuple):
        return tuple(_move_value(item, args, kwargs) for item in val)
    if isinstance(val, dict):
        for k, item in val.items():
            val[k] = _move_value(item, args, kwargs)
        return val
    return val
