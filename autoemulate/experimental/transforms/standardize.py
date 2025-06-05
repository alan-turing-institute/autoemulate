import torch
from torch.distributions import Transform, constraints

from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import TensorLike


class StandardizeTransform(AutoEmulateTransform):
    """Standardize transform for normalizing data.

    This transform is effectively a composition of two AffineTransforms with a
    translation by the mean and a scaling by the inverse of the standard deviation.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, cache_size=0):
        Transform.__init__(self, cache_size=cache_size)

    def fit(self, x: TensorLike):
        # TODO: add checks or shape of mean and std
        self.mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True)
        # Ensure std is not zero to avoid division by zero errors
        std[std < 10 * torch.finfo(std.dtype).eps] = 1.0
        self.std = std
        self._is_fitted = True

    def _call(self, x):
        self._check_is_fitted()
        return (x - self.mean) / self.std

    def _inverse(self, y):
        self._check_is_fitted()
        return y * self.std + self.mean

    def log_abs_det_jacobian(self, x, y):
        _, _ = x, y
        self._check_is_fitted()
        return torch.abs(self.std).log().sum()

    @property
    def _basis_matrix(self) -> TensorLike:
        return torch.eye(self.std.shape[1]) * self.std
