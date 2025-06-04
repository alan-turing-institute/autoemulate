import torch
from torch.distributions import Transform

from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import TensorLike


class StandardizeTransform(AutoEmulateTransform):
    """Standardize transform for normalizing data.

    This transform is effectively a composition of two AffineTransforms with a
    translation by the mean and a scaling by the inverse of the standard deviation.
    """

    bijective = True

    def __init__(self, event_dim=0, cache_size=0):
        Transform.__init__(self, cache_size=cache_size)
        self._event_dim = event_dim

    def fit(self, x: TensorLike):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, keepdim=True)
        self.is_fitted_ = True

    def _call(self, x):
        self._check_is_fitted()
        return (x - self.mean) / self.std

    def _inverse(self, y):
        self._check_is_fitted()
        return y * self.std + self.mean

    def log_abs_det_jacobian(self, x, y):
        _, _ = x, y
        self._check_is_fitted()
        return torch.abs(self.std).log().sum(dim=self._event_dim, keepdim=True)

    @property
    def _basis_matrix(self) -> TensorLike:
        return torch.eye(self.std.shape[1]) * self.std
