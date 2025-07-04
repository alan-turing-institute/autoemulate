from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from linear_operator.operators import DiagLinearOperator
from torch.distributions import Transform

from autoemulate.experimental.data.utils import ConversionMixin, ValidationMixin
from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import (
    DistributionLike,
    GaussianLike,
    GaussianProcessLike,
    TensorLike,
)


class AutoEmulateTransform(Transform, ABC, ValidationMixin, ConversionMixin):
    """Base class for transforms in the AutoEmulate framework.

    This class subclasses the `torch.distributions.Transform` class and provides
    additional functionality for fitting transforms to data and transforming
    Gaussian distributions between the codomain and domain of the transform.

    """

    _is_fitted: bool = False

    # TODO: consider if the override also needs to consider DistributionLike case
    def __call__(self, x: TensorLike) -> TensorLike:
        output = super().__call__(x)
        assert isinstance(output, TensorLike)
        return output

    @abstractmethod
    def fit(self, x: TensorLike): ...

    def _check_is_fitted(self):
        """Check if the transform has been fitted and otherwise raise an error."""
        if not self._is_fitted:
            msg = f"Transform ({self}) has not been fitted yet."
            raise RuntimeError(msg)

    @property
    def _basis_matrix(self) -> TensorLike:
        """Constant basis matrix for transforming matrices. Subclasses should implement
        this property (if possible) to return the appropriate basis matrix.
        """
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def _expanded_basis_matrix(self, y: TensorLike) -> TensorLike:
        """Expanded basis matrix for the transformation of the number of samples in
        a given codomain `y` sample tensor.

        Given `n` samples in `y`, this returns a Kronecker product of the identity
        matrix with the basis matrix, effectively expanding the basis matrix to match
        the number of samples in `y`.

        The default implementation assumes that the transform has implemented the
        `_basis_matrix` property for a single sample. However, this can be overridden
        in subclasses to enable alternative approaches to generating the expanded basis
        matrix for a given set of samples.

        Parameters
        ----------
        y : TensorLike
            Input tensor of shape `(n, )` from which to compute the expanded
            basis matrix. The number of samples `n` is inferred from the shape of `y`.

        Returns
        -------
        TensorLike
            Expanded basis matrix.

        Raises
        ------
        RuntimeError
            If the transform has not been fitted yet.

        """
        self._check_is_fitted()
        return torch.kron(torch.eye(y.shape[0]), self._basis_matrix)

    def _inverse_sample_gaussian_like(
        self, y: GaussianLike, n_samples: int = 1000, full_covariance: bool = True
    ) -> GaussianLike:
        return _inverse_sample_gaussian_like(
            self.inv, y, n_samples=n_samples, full_covariance=full_covariance
        )

    def _inverse_sample_gaussian_process_like(
        self, y: GaussianLike, n_samples: int = 1000, full_covariance: bool = True
    ) -> GaussianLike:
        return _inverse_sample_gaussian_process_like(
            self.inv, y, n_samples=n_samples, full_covariance=full_covariance
        )

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
            A `GaussianLike` object representing the distribution in the domain with
            mean and covariance derived from the samples.

        Raises
        ------
        RuntimeError
            If covariance matrix cannot be made positive definite.

        """
        if isinstance(y, GaussianProcessLike):
            return self._inverse_sample_gaussian_process_like(
                y, n_samples=n_samples, full_covariance=full_covariance
            )
        # TODO (#579): remove raise once fully implemented
        msg = "Implementation to be complete in #579"
        raise NotImplementedError(msg)

        return self._inverse_sample_gaussian_like(
            y, n_samples=n_samples, full_covariance=full_covariance
        )

    def _inverse_gaussian(self, y: GaussianLike) -> GaussianLike:
        r"""Transforms a `GaussianLike` in the codomain to a `GaussianLike` in the
        domain by applying the inverse of the transform to the mean and covariance of
        `y` in the codomain.

        The default implementation computes the mean and covariance of `y` and:
        - applies the inverse transformation to the mean
        - transforms the covariance matrix using the expanded basis matrix

        Parameters
        ----------
        y : GaussianLike
            The distribution in the codomain to be transformed back to the domain.

        Returns
        -------
        GaussianLike
            A `GaussianLike` representing the distribution in the domain, with mean and
            covariance derived from the transformed statistics.

        Raises
        ------
        RuntimeError
            If the covariance matrix cannot be made positive definite.

        TypeError
            If the input `y` is not of type `GaussianLike` or `GaussianProcessLike`.


        Notes
        -----
        This method assumes that the transform is either an exactly linear or a
        well-defined approximately linear transformation between the codomain and the
        domain such that the transform can be expressed as follows:

        .. math::
            \mathcal{N}(\mu_y, \Sigma_y)
            \rightarrow
            \mathcal{N}(f^{-1}(\mu_y), \; A \Sigma_y A^\top)

        where :math:`f^{-1}` is the inverse of the transform, :math:`\mu_y` is the mean
        of the distribution in the codomain, :math:`\Sigma_y` is the covariance matrix
        of the distribution in the codomain, and :math:`A` is the linear transformation
        matrix (expanded basis matrix) derived from the transform.

        """
        mean = y.mean
        cov = y.covariance_matrix
        mean_orig = self.inv(mean)
        assert isinstance(mean, TensorLike)
        assert isinstance(cov, TensorLike)
        assert isinstance(mean_orig, TensorLike)

        if isinstance(y, GaussianProcessLike):
            # Get the expanded basis matrix around the mean
            expanded_basis_matrix = self._expanded_basis_matrix(mean)

            # Transform covariance matrix
            cov_orig = expanded_basis_matrix @ cov @ expanded_basis_matrix.T

            # Ensure positive definite
            cov_orig = make_positive_definite(cov_orig)

            return GaussianProcessLike(mean_orig, cov_orig)

        # TODO (#579): remove raise once fully implemented
        msg = "Implementation to be complete in #579"
        raise NotImplementedError(msg)

        if isinstance(y, GaussianLike):
            if len(y.batch_shape) > 1:
                msg = f"Batch shape ({y.batch_shape}) greater than ndim=1 not supported"
                raise NotImplementedError(msg)
            # Transform covariance matrix
            # TODO: update if _basis_matrix is not implemented
            cov_orig = self._basis_matrix @ cov @ self._basis_matrix.T

            # Ensure positive definite
            if cov_orig.ndim > 2:
                cov_orig = torch.stack([make_positive_definite(c) for c in cov_orig], 0)

            return GaussianLike(mean_orig, cov_orig)

        msg = f"Unsupported type: {type(y)}"
        raise TypeError(msg)


def _inverse_sample_gaussian_like(
    c: Callable,
    y: DistributionLike,
    n_samples: int = 1000,
    full_covariance: bool = True,
) -> GaussianLike:
    """Transforms a `DistributionLike` to a `GaussianLike` through sampling from `y`.

    Parameters
    ----------
    c : Callable
        A callable that applies a transformation to the generated samples.
    y : DistributionLike
        The distribution from which to sample.
    n_samples : int, default=1000
        Number of samples to generate from the distribution `y`.
    full_covariance : bool, default=True
        If True, calculates a full covariance matrix from samples; otherwise,
        calculates only the diagonal of the covariance matrix. This is useful
        for a high-dimensional domain where full covariance might be
        computationally expensive.

    Returns
    -------
    GaussianProcessLike
        A `GaussianProcessLike` object representing the distribution from the empirical
        samples.

    Raises
    ------
    NotImplementedError
        If the batch shape of `y` is greater than 1, as this implementation does
        not support multi-dimensional batch shapes.

    """
    if len(y.batch_shape) > 1:
        msg = f"Batch shape ({y.batch_shape}) greater than ndim=1 not supported"
        raise NotImplementedError(msg)
    samples = c(torch.stack([y.sample() for _ in range(n_samples)], dim=0))
    assert isinstance(samples, TensorLike)
    mean = samples.mean(dim=0)

    # TODO check the handling if no batch dim is present
    cov = (
        torch.stack(
            [make_positive_definite(s.T.cov()) for s in samples.transpose(0, 1)], 0
        )
        if full_covariance
        else DiagLinearOperator(samples.var(dim=0))
    )

    return GaussianLike(mean, cov)


def _inverse_sample_gaussian_process_like(
    c: Callable,
    y: GaussianProcessLike,
    n_samples: int = 1000,
    full_covariance: bool = True,
) -> GaussianProcessLike:
    """Transforms a `GaussianProcessLike` to another `GaussianProcessLike` through
    sampling from `y`.

    Parameters
    ----------
    c : Callable
        A callable that applies a transformation to the generated samples.
    y : GaussianProcessLike
        The Gaussian Process from which to sample.
    n_samples : int, default=1000
        Number of samples to generate from the distribution `y`.
    full_covariance : bool, default=True
        If True, calculates a full covariance matrix from samples; otherwise, calculates
        only the diagonal of the covariance matrix. This is useful for a
        high-dimensional domain where full covariance might be computationally
        expensive.

    Returns
    -------
    GaussianProcessLike
        A `GaussianProcessLike` object representing the distribution from the empirical
        samples.

    Raises
    ------
    RuntimeError
        If the covariance matrix cannot be made positive definite.

    """
    samples = c(torch.stack([y.sample() for _ in range(n_samples)], dim=0))
    assert isinstance(samples, TensorLike)
    mean = samples.mean(dim=0)
    cov = (
        make_positive_definite(samples.view(n_samples, -1).T.cov())
        if full_covariance
        else DiagLinearOperator(samples.view(n_samples, -1).var(dim=0))
    )
    return GaussianProcessLike(mean, cov)


# TODO (#536): complete implementation
# class AutoEmulateTransformModule(TransformModule):
#     @abstractmethod
#     def fit(self, x): ...
#     def _inverse_gaussian(self, y: GaussianLike) -> GaussianLike:
#         msg = "This method should be implemented in subclasses."
#         raise NotImplementedError(msg)

#     def _inverse_sample(self, y: GaussianLike, n_samples: int = 100) -> GaussianLike:
#         msg = "This method should be implemented in subclasses."
#         raise NotImplementedError(msg)
