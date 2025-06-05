import torch
from torch.distributions import Transform, constraints

from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import TensorLike


class PCATransform(AutoEmulateTransform):
    """PCA transform for dimensionality reduction."""

    domain = constraints.real
    codomain = constraints.real
    bijective = False

    def __init__(self, n_components, cache_size: int = 0):
        Transform.__init__(self, cache_size=cache_size)
        self.n_components = n_components  # n_c

    def fit(self, x: TensorLike):
        self.mean = x.mean(0)
        _, _, v = torch.pca_lowrank(x, q=self.n_components)
        self.components = v[:, : self.n_components]  # (d, n_c)
        self._is_fitted = True

    def _call(self, x):
        self._check_is_fitted()
        return (x - self.mean) @ self.components

    def _inverse(self, y):
        self._check_is_fitted()
        # (n, n_c) x (n_c, d) + (n_c,)
        return y @ self.components.T + self.mean

    def log_abs_det_jacobian(self, x, y):
        _, _ = x, y
        msg = "log det Jacobian not computable for n_components < d as not bijective."
        raise RuntimeError(msg)

    @property
    def _basis_matrix(self) -> TensorLike:
        self._check_is_fitted()
        return self.components
