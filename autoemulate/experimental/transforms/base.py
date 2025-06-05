from abc import ABC, abstractmethod

import torch
from pyro.distributions import TransformModule
from torch.distributions import Transform

from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import GaussianLike, TensorLike


class AutoEmulateTransform(Transform, ABC):
    _is_fitted: bool = False

    # TODO: consider if the override also needs to consider DistributionLike case
    def __call__(self, x: TensorLike) -> TensorLike:
        output = super().__call__(x)
        assert isinstance(output, TensorLike)
        return output

    @abstractmethod
    def fit(self, x: TensorLike): ...

    def _check_is_fitted(self):
        if not self._is_fitted:
            msg = f"Transform ({self}) has not been fitted yet."
            raise ValueError(msg)

    @property
    def _basis_matrix(self) -> TensorLike:
        """Constant basis matrix for transforming matrices. Subclasses should implement
        this property (if possible) to return the appropriate basis matrix.
        """
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def _expanded_basis_matrix(self, x: TensorLike) -> TensorLike:
        """Expanded basis matrix for the transformation given x sample tensor.

        Given n samples in x, this returns a Kronecker product of the identity matrix
        with the basis matrix, effectively expanding the basis matrix to match the
        number of samples in x.

        Parameters
        ----------
            x (TensorLike): Input tensor to determine the number of samples.

        Returns
        -------
            TensorLike: Expanded basis matrix.
        """
        self._check_is_fitted()
        return torch.kron(torch.eye(x.shape[0]), self._basis_matrix)

    def _inverse_sample(self, x: GaussianLike, n_samples: int = 100) -> GaussianLike:
        """Generate samples from a Gaussian distribution."""
        mean = x.mean
        cov = x.covariance_matrix
        mean_orig = self.inv(mean)
        assert isinstance(mean, TensorLike)
        assert isinstance(cov, TensorLike)
        assert isinstance(mean_orig, TensorLike)

        def sample_cov():
            # Draws samples from gaussian in latent and transforms to original space
            sample = x.sample()
            sample_orig = self.inv(sample)
            assert isinstance(sample_orig, TensorLike)
            sample_orig = sample_orig.view(-1, 1)
            mean_reshaped = mean_orig.view(-1, 1)
            return (sample_orig - mean_reshaped) @ (sample_orig - mean_reshaped).T

        # Generate samples and take unbiased mean to estimate covariance
        cov_orig = torch.stack([sample_cov() for _ in range(n_samples)]).sum(0) / (
            n_samples - 1
        )
        # Ensure positive definite
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

        # Transform covariance matrix
        cov_orig = expanded_basis_matrix @ cov @ expanded_basis_matrix.T

        # Ensure positive definite
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
