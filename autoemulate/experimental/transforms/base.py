from abc import ABC, abstractmethod

import torch
from pyro.distributions import TransformModule
from torch.distributions import Transform

from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import GaussianLike, TensorLike


class AutoEmulateTransform(Transform, ABC):
    is_fitted_: bool = False

    # TODO: consider if the override also needs to consider DistributionLike case
    def __call__(self, x: TensorLike) -> TensorLike:
        output = super().__call__(x)
        assert isinstance(output, TensorLike)
        return output

    @abstractmethod
    def fit(self, x: TensorLike): ...

    def _check_is_fitted(self):
        if not self.is_fitted_:
            msg = f"Transform ({self.__name__}) has not been fitted yet."
            raise ValueError(msg)

    def _expanded_basis_matrix(self, x: TensorLike) -> TensorLike:
        """Get the basis matrix for the transformation."""
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def _inverse_sample(self, x: GaussianLike, n_samples: int = 100) -> GaussianLike:
        """Generate samples from a Gaussian distribution."""
        mean = x.mean
        cov = x.covariance_matrix
        mean_orig = self.inv(mean)
        assert isinstance(mean, TensorLike)
        assert isinstance(cov, TensorLike)
        assert isinstance(mean_orig, TensorLike)

        def sample_cov():
            sample = x.sample()
            sample_orig = self.inv(sample)
            assert isinstance(sample_orig, TensorLike)
            sample_orig = sample_orig.view(-1, 1)
            mean_reshaped = mean_orig.view(-1, 1)
            return (
                (sample_orig - mean_reshaped)
                @ (sample_orig - mean_reshaped).T
                / (sample_orig.shape[0] - 1)
            )

        # Generate samples and take mean to estimate covariance, make positive definite
        cov_orig = torch.stack([sample_cov() for _ in range(n_samples)]).mean(0)
        cov_orig = make_positive_definite(cov_orig)

        return GaussianLike(mean_orig, cov_orig)

    def _inverse_gaussian(self, x: GaussianLike) -> GaussianLike:
        mean = x.mean
        cov = x.covariance_matrix
        mean_orig = self.inv(mean)
        assert isinstance(mean, TensorLike)
        assert isinstance(cov, TensorLike)
        assert isinstance(mean_orig, TensorLike)

        expanded_basis_matrix = self._expanded_basis_matrix(mean)

        # Transform covariance and make it positive definite
        cov_orig = expanded_basis_matrix @ cov @ expanded_basis_matrix.T
        cov_orig = make_positive_definite(cov_orig)

        return GaussianLike(mean_orig, cov_orig)


class AutoEmulateTransformModule(TransformModule):
    @abstractmethod
    def fit(self, x): ...
    def _inverse_gaussian(self, x: GaussianLike) -> GaussianLike:
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def _inverse_sample(self, x: GaussianLike, n_samples: int = 100) -> GaussianLike:
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)


# TODO: conside adding AutoEmulateComposeTransform
