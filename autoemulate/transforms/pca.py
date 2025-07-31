import torch
from torch.distributions import Transform, constraints

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import TensorLike
from autoemulate.transforms.base import AutoEmulateTransform


class PCATransform(AutoEmulateTransform):
    """PCA transform for dimensionality reduction."""

    domain = constraints.real
    codomain = constraints.real
    bijective = False

    def __init__(self, n_components: int, niter: int = 1000, cache_size: int = 0):
        """
        Initialize the PCA transform.

        Parameters
        ----------
        n_components: int
            The number of principal components to use for the transformation.
        niter: int
            The number of iterations for the PCA algorithm. Defaults to 1000.
        cache_size: int
            Whether to cache previous transform. Set to 0 to disable caching. Set to
            1 to enable caching of the last single value. This might be useful for
            repeated expensive calls with the same input data but is by default
            disabled. See `PyTorch documentation <https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/distributions/transforms.py#L46-L89>`_
            for more details on caching. Defaults to 0.
        """
        Transform.__init__(self, cache_size=cache_size)
        self.n_components = n_components  # n_c
        self.niter = niter
        self.cache_size = cache_size  # Store for serialization

    def fit(self, x: TensorLike):
        """Fit the PCA transform to the data."""
        TorchDeviceMixin.__init__(self, device=x.device)
        self.check_matrix(x)
        self.mean = x.mean(0, keepdim=True)  # (1, d)
        _, _, v = torch.pca_lowrank(x, q=self.n_components, niter=self.niter)
        self.components = v[:, : self.n_components]  # (d, n_c)
        self._is_fitted = True

    def _call(self, x: TensorLike):
        self._check_is_fitted()
        return (x - self.mean) @ self.components

    def _inverse(self, y: TensorLike):
        self._check_is_fitted()
        return y @ self.components.T + self.mean  # (n, n_c) x (n_c, d) + (1, d)

    def log_abs_det_jacobian(self, x: TensorLike, y: TensorLike):
        """Log abs det Jacobian not computable for n_components < d as not bijective."""
        _, _ = x, y
        msg = (
            "log abs det Jacobian not computable for n_components < d as not bijective."
        )
        raise RuntimeError(msg)

    @property
    def _basis_matrix(self) -> TensorLike:
        self._check_is_fitted()
        return self.components
