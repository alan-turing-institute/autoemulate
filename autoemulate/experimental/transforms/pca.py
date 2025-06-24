import torch
from torch.distributions import Transform, constraints

from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import TensorLike


class PCATransform(AutoEmulateTransform):
    """PCA transform for dimensionality reduction."""

    domain = constraints.real
    codomain = constraints.real
    bijective = False

    def __init__(self, n_components: int, niter: int = 1000):
        """
        Initialize the PCA transform.

        Parameters
        ----------
        n_components : int
            The number of principal components to use for the transformation.
        niter : int, default=1000
            The number of iterations for the PCA algorithm.
        """
        # Init the base class with no cache
        Transform.__init__(self, cache_size=0)
        self.n_components = n_components  # n_c
        self.niter = niter

    def fit(self, x: TensorLike):
        self.mean = x.mean(0)
        _, _, v = torch.pca_lowrank(x, q=self.n_components, niter=self.niter)
        # (d, n_c)
        self.components = v[:, : self.n_components]
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
