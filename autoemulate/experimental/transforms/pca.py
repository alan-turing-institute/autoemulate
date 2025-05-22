import torch
from torch.distributions import Transform, constraints

from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.transforms.utils import make_positive_definite
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

    def fit(self, x: TensorLike):
        self.mean = x.mean(0)
        _, _, v = torch.pca_lowrank(x, q=self.n_components)
        self.components = v[:, : self.n_components]  # (d, n_c)
        self.is_fitted_ = True

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

    def _inverse_gaussian(self, x: GaussianLike) -> GaussianLike:
        self._check_is_fitted()
        mean_pca = x.mean
        cov_pca = x.covariance_matrix
        assert isinstance(mean_pca, TensorLike)
        assert isinstance(cov_pca, TensorLike)
        # Expand components to match covariance matrix size across samples and tasks
        components_expanded = torch.kron(torch.eye(mean_pca.shape[0]), self.components)
        # Transform covariance
        cov_orig = components_expanded @ cov_pca @ components_expanded.T
        # Ensure positive definiteness
        # TODO: check how to make this more robust
        cov_orig = cov_orig + 1e-4 * torch.eye(cov_orig.shape[0])
        # Transform mean
        mean_orig = mean_pca @ self.components.T  # (n, n_c) x (n_c, d)

        return GaussianLike(mean_orig, cov_orig)

    def _inverse_sample(self, x: GaussianLike, n_samples: int = 100) -> GaussianLike:
        self._check_is_fitted()
        mean_pca = x.mean
        cov_pca = x.covariance_matrix
        mean_orig = mean_pca @ self.components.T
        assert isinstance(mean_pca, TensorLike)
        assert isinstance(cov_pca, TensorLike)

        def sample_cov():
            sample_pca = x.sample()
            sample = sample_pca @ self.components.T + mean_orig
            mean_reshaped = mean_orig.view(-1, 1)
            return (
                (sample - mean_reshaped)
                @ (sample - mean_reshaped).T
                / (sample.shape[0] - 1)
            )

        # Generate samples and take the mean to estimate covariance in original space
        cov_orig = torch.stack([sample_cov() for _ in range(n_samples)]).mean(0)

        # Ensure positive definiteness
        cov_orig = make_positive_definite(cov_orig)

        return GaussianLike(mean_orig, cov_orig)
