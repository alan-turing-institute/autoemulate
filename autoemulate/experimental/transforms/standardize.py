# from torch.distributions import Transform, TransformedDistribution, constraints
import torch
from torch.distributions import Transform

# from torch.distributions import Transform
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import GaussianLike, TensorLike


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

    def _inverse_gaussian(self, x: GaussianLike) -> GaussianLike:
        self._check_is_fitted()
        mean = x.mean
        cov = x.covariance_matrix
        assert isinstance(mean, TensorLike)
        assert isinstance(cov, TensorLike)

        components = torch.eye(self.std.shape[1]) * self.std

        # Expand components to match covariance matrix size across samples and tasks
        components_expanded = torch.kron(torch.eye(mean.shape[0]), components)

        # Transform covariance
        cov_orig = components_expanded @ cov @ components_expanded.T
        cov_orig = make_positive_definite(cov_orig)

        # Transform mean
        mean_orig = mean @ components.T  # (n, n_c) x (n_c, d)

        return GaussianLike(mean_orig, cov_orig)

    def _inverse_sample(self, x: GaussianLike, n_samples: int = 100) -> GaussianLike:
        # return self._inverse_gaussian(x)
        self._check_is_fitted()
        mean = x.mean
        cov = x.covariance_matrix
        components = torch.eye(self.std.shape[1]) * self.std

        # TODO: refactor below into a utility function
        mean_orig = mean @ components.T
        assert isinstance(mean, TensorLike)
        assert isinstance(cov, TensorLike)

        def sample_cov():
            sample = x.sample()
            sample_orig = (sample @ components.T + mean_orig).view(-1, 1)
            mean_reshaped = mean_orig.view(-1, 1)
            return (
                (sample_orig - mean_reshaped)
                @ (sample_orig - mean_reshaped).T
                / (sample_orig.shape[0] - 1)
            )

        # Generate samples and take the mean to estimate covariance in original space
        cov_orig = torch.stack([sample_cov() for _ in range(n_samples)]).mean(0)

        print(cov_orig.round(), mean_orig.shape)
        # Ensure positive definiteness
        cov_orig = make_positive_definite(cov_orig)

        return GaussianLike(mean_orig, cov_orig)
