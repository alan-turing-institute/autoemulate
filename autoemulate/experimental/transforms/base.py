import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from linear_operator.operators import DiagLinearOperator
from torch.distributions import Transform
from typing_extensions import Self

from autoemulate.experimental.data.utils import ConversionMixin, ValidationMixin
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import (
    DistributionLike,
    GaussianLike,
    GaussianProcessLike,
    TensorLike,
)


class AutoEmulateTransform(
    Transform, ABC, ValidationMixin, ConversionMixin, TorchDeviceMixin
):
    """
    Base class for transforms in the AutoEmulate framework.

    This class subclasses the `torch.distributions.Transform` class and provides
    additional functionality for fitting transforms to data and transforming
    Gaussian distributions between the codomain and domain of the transform.

    """

    _is_fitted: bool = False

    # TODO: consider if the override also needs to consider DistributionLike case
    def __call__(self, x: TensorLike) -> TensorLike:
        """Apply the transform to input tensor `x`."""
        output = super().__call__(x)
        assert isinstance(output, TensorLike)
        return output

    @abstractmethod
    def fit(self, x: TensorLike):
        """Fit the transform to the input tensor `x`."""
        ...

    def _check_is_fitted(self):
        """Check if the transform has been fitted and otherwise raise an error."""
        if not self._is_fitted:
            msg = f"Transform ({self}) has not been fitted yet."
            raise RuntimeError(msg)

    def to_dict(self) -> dict:
        """
        Serialize the transform to a dictionary.

        Returns
        -------
        dict
            A dictionary with transform name as key and initialization parameters
            as value. The format is:
            {
                "transform_name": {param1: value1, param2: value2, ...}
            }
        """
        # Get the transform name (convert class name to lowercase, remove 'Transform')
        class_name = self.__class__.__name__
        if class_name.endswith("Transform"):
            transform_name = class_name[:-9].lower()  # Remove 'Transform' suffix
        else:
            transform_name = class_name.lower()

        # Get initialization parameters from the __init__ signature
        init_signature = inspect.signature(self.__class__.__init__)
        init_params = {}

        for param_name, param in init_signature.parameters.items():
            # Skip 'self' parameter
            if param_name == "self":
                continue

            # Get the value from the instance if it exists
            if hasattr(self, param_name):
                value = getattr(self, param_name)

                # Handle special cases for serialization
                if isinstance(value, torch.Tensor):
                    # Convert tensors to lists for JSON serialization
                    init_params[param_name] = value.tolist()
                elif isinstance(value, torch.device):
                    # Convert device to string
                    init_params[param_name] = str(value)
                else:
                    init_params[param_name] = value
            elif param.default is not inspect.Parameter.empty:
                # Use default value if attribute doesn't exist
                init_params[param_name] = param.default

        return {transform_name: init_params}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """
        Deserialize a transform from a dictionary.

        Parameters
        ----------
        data: dict
            A dictionary with transform name as key and initialization parameters
            as value, as created by `to_dict()`.

        Returns
        -------
        AutoEmulateTransform
            An instance of the transform class with the specified parameters.

        Raises
        ------
        ValueError
            If the dictionary format is invalid or the transform cannot be found.

        """
        if len(data) != 1:
            msg = "Dictionary must contain exactly one transform"
            raise ValueError(msg)

        transform_name, init_params = next(iter(data.items()))

        # Get transform class from registry
        transform_class = cls._get_transform_class(transform_name)
        if transform_class is None:
            available = ", ".join(cls._get_available_transforms())
            msg = f"Unknown transform: {transform_name}. Available: {available}"
            raise ValueError(msg)

        # Handle special cases for deserialization
        processed_params = {}
        for param_name, value in init_params.items():
            if param_name == "device" and isinstance(value, str):
                # Convert device string back to torch.device
                if value != "None":
                    processed_params[param_name] = torch.device(value)
                else:
                    processed_params[param_name] = None
            else:
                processed_params[param_name] = value

        return transform_class(**processed_params)

    @classmethod
    def _get_transform_class(cls, transform_name: str):
        """
        Get transform class from the registry.

        Parameters
        ----------
        transform_name: str
            The name of the transform (e.g., 'pca', 'vae', 'standardize')

        Returns
        -------
        type[AutoEmulateTransform] | None
            The transform class if found, None otherwise
        """
        from . import TRANSFORM_REGISTRY  # Lazy import to avoid circular dependency

        return TRANSFORM_REGISTRY.get(transform_name)

    @classmethod
    def _get_available_transforms(cls):
        """
        Get list of available transform names.

        Returns
        -------
        list[str]
            List of available transform names
        """
        from . import TRANSFORM_REGISTRY  # Lazy import to avoid circular dependency

        return list(TRANSFORM_REGISTRY.keys())

    @property
    def _basis_matrix(self) -> TensorLike:
        """
        Constant basis matrix for transforming matrices.

        Subclasses should implement this property (if possible) to return the
        appropriate basis matrix.
        """
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def _expanded_basis_matrix(self, y: TensorLike) -> TensorLike:
        """
        Return the expanded basis matrix for the transformation.

        Expanded basis matrix for the transformation of the number of samples in
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
        y: TensorLike
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
        return torch.kron(torch.eye(y.shape[0], device=self.device), self._basis_matrix)

    def _batch_basis_matrix(self, y: TensorLike) -> TensorLike:
        """
        Return the batch basis matrix for the transformation.

        Batch basis matrix for the transformation of the number of samples in
        a given codomain `y` sample tensor.

        Given `n` samples in `y`, this returns a basis matrix for transforming each
        element of a batch.

        The default implementation assumes that the transform has implemented the
        `_basis_matrix` property for a single sample and is therefore simply repeats
        this along a new batch dim.

        However, the method takes a `y` tensor argument to allow the method to be
        overriden with local approximations such as through the delta method.

        Parameters
        ----------
        y: TensorLike
            Input tensor of shape `(n, )` from which to compute the batch
            basis matrix. The number of samples `n` is inferred from the shape of `y`.

        Returns
        -------
        TensorLike
            Batch basis matrix.

        Raises
        ------
        RuntimeError
            If the transform has not been fitted yet.
        """
        self._check_is_fitted()
        return self._basis_matrix.repeat(y.shape[0], 1, 1)

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
        """
        Invert a `GaussianLike` distribution by sampling from the codomain.

        Transforms a `GaussianLike` in the codomain to a `GaussianLike` in the domain
        through generating samples from `y` in the codomain and mapping those back
        to the original space `x`.

        The empirical mean and covariance of the samples are computed, and a
        `GaussianLike` object in the domain is returned with these statistics.

        Parameters
        ----------
        y: GaussianLike
            The distribution in the codomain.
        n_samples: int, default=1000
            Number of samples to generate from the distribution `y`.
        full_covariance: bool, default=True
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

        return self._inverse_sample_gaussian_like(
            y, n_samples=n_samples, full_covariance=full_covariance
        )

    def _inverse_gaussian(self, y: GaussianLike) -> GaussianLike:
        r"""
        Invert a `GaussianLike` distribution to the original space.

        Transforms a `GaussianLike` in the codomain to a `GaussianLike` in the
        domain by applying the inverse of the transform to the mean and covariance of
        `y` in the codomain.

        The default implementation computes the mean and covariance of `y` and:
        - applies the inverse transformation to the mean
        - transforms the covariance matrix using the expanded basis matrix

        Parameters
        ----------
        y: GaussianLike
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
            cov_orig = make_positive_definite(
                cov_orig, min_jitter=1e-6, max_tries=3, clamp_eigvals=True
            )

            return GaussianProcessLike(mean_orig, cov_orig)

        if isinstance(y, GaussianLike):
            if len(y.batch_shape) > 1:
                msg = f"Batch shape ({y.batch_shape}) greater than ndim=1 not supported"
                raise NotImplementedError(msg)
            # Transform covariance matrix
            cov_orig = (
                self._batch_basis_matrix(mean)
                @ cov
                @ self._batch_basis_matrix(mean).transpose(-1, -2)
            )

            # Ensure positive definite
            if cov_orig.ndim > 2:
                # TODO: consider revising whether to only use jitter
                cov_orig = torch.stack(
                    [
                        make_positive_definite(
                            c, min_jitter=1e-6, max_tries=3, clamp_eigvals=True
                        )
                        for c in cov_orig
                    ],
                    0,
                )

            return GaussianLike(mean_orig, cov_orig)

        msg = f"Unsupported type: {type(y)}"
        raise TypeError(msg)


def _inverse_sample_gaussian_like(
    c: Callable,
    y: DistributionLike,
    n_samples: int = 1000,
    full_covariance: bool = True,
) -> GaussianLike:
    """
    Invert a `DistributionLike` to a `GaussianLike` through sampling from `y`.

    Transforms a `DistributionLike` to a `GaussianLike` through sampling from `y`.

    Parameters
    ----------
    c: Callable
        A callable that applies a transformation to the generated samples.
    y: DistributionLike
        The distribution from which to sample.
    n_samples: int, default=1000
        Number of samples to generate from the distribution `y`.
    full_covariance: bool, default=True
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
    batch_shape = y.batch_shape
    if len(batch_shape) > 1:
        msg = f"Batch shape ({batch_shape}) greater than ndim=1 not supported"
        raise NotImplementedError(msg)

    # Sample from the distribution `y` and apply the transformation `c`
    samples = torch.vmap(c)(torch.stack([y.sample() for _ in range(n_samples)], dim=0))

    # Remove batch dim if batch_shape is empty
    if len(batch_shape) == 0:
        samples = samples.squeeze(1)

    # Ensure the samples are of type TensorLike
    assert isinstance(samples, TensorLike)

    # Compute the mean and covariance of the samples
    mean = samples.mean(dim=0)
    if not full_covariance:
        return GaussianLike(mean, DiagLinearOperator(samples.var(dim=0)))

    cov = (
        # Loop over the batch dimension and compute covariance for each sample
        torch.stack(
            [
                make_positive_definite(
                    s.T.cov(), min_jitter=1e-6, max_tries=3, clamp_eigvals=False
                )
                for s in samples.transpose(0, 1)
            ],
            0,
        )
        if len(batch_shape) > 0
        # If no batch shape, compute covariance for the entire sample set
        else make_positive_definite(
            samples.T.cov(), min_jitter=1e-6, max_tries=3, clamp_eigvals=False
        )
    )
    return GaussianLike(mean, cov)


def _inverse_sample_gaussian_process_like(
    c: Callable,
    y: GaussianProcessLike,
    n_samples: int = 1000,
    full_covariance: bool = True,
) -> GaussianProcessLike:
    """
    Invert a `GaussianProcessLike` distribution by sampling from the codomain.

    Transforms a `GaussianProcessLike` to another `GaussianProcessLike` through
    sampling from `y`.

    Parameters
    ----------
    c: Callable
        A callable that applies a transformation to the generated samples.
    y: GaussianProcessLike
        The Gaussian Process from which to sample.
    n_samples: int
        Number of samples to generate from the distribution `y`. Defaults to 1000.
    full_covariance: bool
        If True, calculates a full covariance matrix from samples; otherwise, calculates
        only the diagonal of the covariance matrix. This is useful for a
        high-dimensional domain where full covariance might be computationally
        expensive. Defaults to True.

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
        make_positive_definite(
            samples.reshape(n_samples, -1).T.cov(),
            min_jitter=1e-6,
            max_tries=3,
            clamp_eigvals=True,
        )
        if full_covariance
        else DiagLinearOperator(samples.reshape(n_samples, -1).var(dim=0))
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
