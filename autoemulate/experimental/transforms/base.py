from abc import ABC, abstractmethod

import torch
from linear_operator.operators import DiagLinearOperator
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

    def _expanded_basis_matrix(self, y: TensorLike) -> TensorLike:
        """Expanded basis matrix for the transformation given codomain `y` sample
        tensor.

        Given `n` samples in `y`, this returns a Kronecker product of the identity
        matrix with the basis matrix, effectively expanding the basis matrix to match
        the number of samples in `y`.

        Parameters
        ----------
            x (TensorLike): Input tensor to determine the number of samples.

        Returns
        -------
            TensorLike: Expanded basis matrix.
        """
        self._check_is_fitted()
        return torch.kron(torch.eye(y.shape[0]), self._basis_matrix)

    def _inverse_sample(
        self, y: GaussianLike, n_samples: int = 1000, full_covariance: bool = True
    ) -> GaussianLike:
        """Transforms a `GaussianLike` in the codomain to a `GaussianLike` in the domain
        through generating samples from `y` in the codomain and mapping those back
        to the original space `x`.

        The empirical mean and covariance of the samples are computed, and a
        `GaussianLike` object in the domain is returned with these statistics.

        Parameters
        ----------
            y : GaussianLike
                The distribution in the codomain.
            n_samples : int, default=1000
                Number of samples to generate from the distribution `y`.
            full_covariance : bool, default=True
                If True, calculates a full covariance matrix from samples; otherwise,
                calculates only the diagonal of the covariance matrix. This is useful
                for a high-dimensional domain where full covariance might be
                computationally expensive.

        Returns
        -------
            GaussianLike
                A `GaussianLike` object representing the distribution in the domain,
                with mean and covariance derived from the samples.

        Raises
        ------
            RuntimeError
                If covariance matrix cannot be made positive definite.

        """
        samples = self.inv(torch.stack([y.sample() for _ in range(n_samples)], dim=0))
        assert isinstance(samples, TensorLike)
        mean = samples.mean(dim=0)
        cov = (
            make_positive_definite(samples.view(n_samples, -1).T.cov())
            if full_covariance
            else DiagLinearOperator(samples.view(n_samples, -1).var(dim=0))
        )
        return GaussianLike(mean, cov)

    def _inverse_gaussian(self, y: GaussianLike) -> GaussianLike:
        mean = y.mean
        cov = y.covariance_matrix
        mean_orig = self.inv(mean)
        assert isinstance(mean, TensorLike)
        assert isinstance(cov, TensorLike)
        assert isinstance(mean_orig, TensorLike)

        # Get the expanded basis matrix around the mean
        expanded_basis_matrix = self._expanded_basis_matrix(mean)

        # Transform covariance matrix
        cov_orig = expanded_basis_matrix @ cov @ expanded_basis_matrix.T

        # Ensure positive definite
        cov_orig = make_positive_definite(cov_orig)

        return GaussianLike(mean_orig, cov_orig)


# TODO: complete implementation in ...
# class AutoEmulateTransformModule(TransformModule):
#     @abstractmethod
#     def fit(self, x): ...
#     def _inverse_gaussian(self, y: GaussianLike) -> GaussianLike:
#         msg = "This method should be implemented in subclasses."
#         raise NotImplementedError(msg)

#     def _inverse_sample(self, y: GaussianLike, n_samples: int = 100) -> GaussianLike:
#         msg = "This method should be implemented in subclasses."
#         raise NotImplementedError(msg)
