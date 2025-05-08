import torch
from torch.distributions import Transform, constraints

from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import GaussianLike, TensorLike


class PCATransform(AutoEmulateTransform):
    """
    PCA transform for dimensionality reduction.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = False

    def __init__(self, n_components, cache_size: int = 0):
        Transform.__init__(self, cache_size=cache_size)
        self.n_components = n_components  # n_c
        self.fitted = False

    def fit(self, x: TensorLike):
        self.mean = x.mean(0)
        _, _, v = torch.pca_lowrank(x, q=self.n_components)
        self.components = v[:, : self.n_components]  # (d, n_c)
        self.fitted = True

    def _call(self, x):
        return (x - self.mean) @ self.components

    def _inverse(self, y):
        # (n, n_c) x (n_c, d) + (n_c,)
        return y @ self.components.T + self.mean

    def log_abs_det_jacobian(self, x, y):
        _, _ = x, y
        msg = "log det Jacobian not computable for n_components < d as not bijective."
        raise RuntimeError(msg)

    def _inverse_gaussian(self, x: GaussianLike) -> GaussianLike:
        mean_pca = x.mean
        cov_pca = x.covariance_matrix
        assert isinstance(mean_pca, TensorLike)
        assert isinstance(cov_pca, TensorLike)
        # Expand components to match covariance matrix size across samples and tasks
        components_expanded = torch.kron(torch.eye(mean_pca.shape[0]), self.components)
        # Transform covariance
        cov_orig = components_expanded @ cov_pca @ components_expanded.T
        # Ensure positive definiteness
        cov_orig = cov_orig + 1e-5 * torch.eye(cov_orig.shape[0])
        # Transform mean
        mean_orig = mean_pca @ self.components.T  # (n, n_c) x (n_c, d)

        return GaussianLike(mean_orig, cov_orig)
