import torch

# from torch.distributions import Transform, TransformedDistribution, constraints
from torch.distributions import Transform, constraints

from autoemulate.experimental.transforms.base import AutoEmulateTransform

# from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import GaussianLike, TensorLike


class StandardizeTransform(AutoEmulateTransform):
    """
    Standardize transform for normalizing data.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, cache_size: int = 0):
        Transform.__init__(self, cache_size=cache_size)

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
        return -torch.log(self.std).sum()

    def _inverse_gaussian(self, x: GaussianLike) -> GaussianLike:
        _ = x
        self._check_is_fitted()
        # TODO: complete impl
        msg = "TODO"
        raise NotImplementedError(msg)
        # mean_z = x.mean
        # cov_z = x.covariance_matrix
        # mean_orig = x.mean + self.mean
        # assert isinstance(cov_z, TensorLike)
        # cov_orig = x.covariance_matrix * torch.diag(self.std.view(-1, 1))
        # return TransformedDistribution(x, [self])

    def _inverse_sample(self, x: GaussianLike, n_samples: int = 100) -> GaussianLike:
        msg = "Inverse sampling not implemented for StandardizeTransform."
        raise NotImplementedError(msg)
