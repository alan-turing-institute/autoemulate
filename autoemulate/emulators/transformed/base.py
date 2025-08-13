import torch
from linear_operator.operators import DiagLinearOperator
from torch.distributions import ComposeTransform, Transform, TransformedDistribution

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import (
    DeviceLike,
    DistributionLike,
    GaussianLike,
    GaussianProcessLike,
    OutputLike,
    TensorLike,
)
from autoemulate.data.utils import ValidationMixin
from autoemulate.emulators.base import Emulator
from autoemulate.transforms.base import (
    AutoEmulateTransform,
    _inverse_sample_gaussian_like,
    _inverse_sample_gaussian_process_like,
)


class TransformedEmulator(Emulator, ValidationMixin):
    """
    A transformed emulator that applies transformations to input and target data.

    This class wraps an emulator model with configurable transformations applied to
    both input features (x) and target variables (y). The emulator is trained and
    makes predictions in the transformed space, with automatic inverse transformations
    applied to return results in the original data space.

    The transformation workflow for fitting:
    ```
    Original space:     x                                         y
                        │                                         │
                        ▼ x_transforms.fit()   y_transforms.fit() ▼
                        │                                         │
    Transformed space:  x_t ──────────► emulator.fit() ◄──────── y_t
    ```


    The transformation workflow for prediction:
    ```
    Original space:     x ────────────────────► y_pred
                        │                       ▲
                        │ x_transforms          │ y_transforms⁻¹
                        ▼                       │
    Transformed space:  x_t ──► emulator ──► y_t_pred
    ```

    Key features:
    - Sequential application of multiple transformations
    - Automatic handling of different prediction output types (tensors, distributions)
    - Support for both analytical and sampling-based inverse transformations
    - Configurable behavior for high-dimensional targets

    Attributes
    ----------
    x_transforms: list[Transform]
        List of transformations applied to input data (x) in sequential order.
    model: Emulator
        The underlying emulator model that operates on transformed data.
    y_transforms: list[Transform]
        List of transformations applied to target data (y) in sequential order.
    """

    x_transforms: list[Transform]
    model: Emulator
    y_transforms: list[Transform]

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        x_transforms: list[Transform] | None,
        y_transforms: list[Transform] | None,
        model: type[Emulator],
        output_from_samples: bool = True,
        n_samples: int = 100,
        full_covariance: bool = False,
        max_targets: int = 200,
        device: DeviceLike | None = None,
        **kwargs,
    ):
        """
        Initialize a transformed emulator.

        Parameters
        ----------
        x: TensorLike
            Input training data tensor of shape (n_samples, n_features).
        y: TensorLike
            Target training data tensor of shape (n_samples, n_targets).
        x_transforms: list[Transform] | None
            List of transforms to apply to input data in sequential order.
            If None, no transformations are applied to x.
        y_transforms: list[Transform] | None
            List of transforms to apply to target data in sequential order.
            If None, no transformations are applied to y.
        model: type[Emulator]
            The emulator class to instantiate and train on transformed data.
        output_from_samples: bool
            Whether to obtain predictions by sampling from the model's predictive
            distribution. Automatically set to True for high-dimensional targets
            (n_targets > max_targets). Defaults to False.
        n_samples: int
            Number of samples to draw when using sampling-based predictions.
            Only used when output_from_samples=True. Defauls to 100.
        full_covariance: bool
            Whether to use full covariance matrix for predictions. If False,
            uses diagonal covariance. Automatically set to False for
            high-dimensional targets (n_targets > max_targets). Defaults to False.
        max_targets: int
            Threshold for switching to approximate sampling-based predictions
            with diagonal covariance when dealing with high-dimensional targets.
            Defaults to 200.
        device: DeviceLike | None
            Device for tensor operations. If None, uses the default device.
            Defaults to None.
        **kwargs
            Additional keyword arguments passed to the emulator constructor.

        Notes
        -----
        - Transforms are fitted on the provided training data during initialization
        - The underlying emulator is trained on the transformed data
        - For targets with dimensionality > max_targets, the emulator automatically
          switches to sampling-based predictions with diagonal covariance for efficiency
        """
        self.x_transforms = x_transforms or []
        self.y_transforms = y_transforms or []
        self._fit_transforms(x, y)
        self.untransformed_model_name = model.model_name()
        self.model = model(
            self._transform_x(x), self._transform_y_tensor(y), device=device, **kwargs
        )
        self.output_from_samples = output_from_samples or y.shape[1] > max_targets
        if not output_from_samples and not all(
            isinstance(t, AutoEmulateTransform) for t in self.y_transforms
        ):
            msg = (
                "y_transforms must be a list of AutoEmulateTransform instances to "
                f"support outputs without sampling. y_transforms: {self.y_transforms}"
            )
            raise RuntimeError(msg)
        self.n_samples = n_samples
        self.full_covariance = full_covariance and y.shape[1] <= max_targets
        TorchDeviceMixin.__init__(self, device=device)
        self.supports_grad = self.model.supports_grad and all(
            t.bijective for t in self.x_transforms
        )
        # TODO: update to be derived from attribute of the underlying emulator
        # For now, just set as True
        self.supports_uq = True

    def _fit_transforms(self, x: TensorLike, y: TensorLike):
        """
        Fit the transforms on the provided training data.

        Parameters
        ----------
        x: TensorLike
            Input training data tensor of shape `(n_samples, n_features)`.
        y: TensorLike
            Target training data tensor of shape `(n_samples, n_targets)`.
        """
        # Fit transforms
        current_x = x
        for transform in self.x_transforms:
            if isinstance(transform, AutoEmulateTransform):
                transform.fit(current_x)
            current_x = transform(current_x)
            assert isinstance(current_x, TensorLike)
        # Fit target transforms
        current_y = y
        for transform in self.y_transforms:
            if isinstance(transform, AutoEmulateTransform):
                transform.fit(current_y)
            current_y = transform(current_y)
            assert isinstance(current_y, TensorLike)

    def refit(self, x: TensorLike, y: TensorLike, retrain_transforms: bool = False):
        """
        Refit the emulator with new data and optionally retrain transforms.

        Parameters
        ----------
        x: TensorLike
            New input training data tensor of shape `(n_samples, n_features)`.
        y: TensorLike
            New target training data tensor of shape `(n_samples, n_targets)`.
        retrain_transforms: bool
            Whether to retrain the transforms on the new data. If False,
            uses the existing fitted transforms from initialization. Defaults to False.

        Notes
        -----
        When retrain_transforms=False, the transforms fitted during initialization
        are applied to the new data. This assumes the new data comes from the same
        distribution as the original training data.
        """
        if retrain_transforms:
            self._fit_transforms(x, y)
        self.fit(x, y)

    def _transform_x(self, x: TensorLike) -> TensorLike:
        """
        Transform the input tensor `x` using `x_transforms` returning a `TensorLike`.

        Parameters
        ----------
        x: TensorLike
            Input tensor to be transformed.

        Returns
        -------
        TensorLike
            Transformed input tensor after applying all x_transforms.

        """
        x_t = ComposeTransform(self.x_transforms)(x)
        assert isinstance(x_t, TensorLike)
        return x_t

    def _transform_y_tensor(self, y: TensorLike) -> TensorLike:
        """
        Transform the target tensor `y` using `y_transforms` returning a `TensorLike`.

        Parameters
        ----------
        y: TensorLike
            Target tensor to be transformed.

        Returns
        -------
        TensorLike
            Transformed target tensor after applying all `y_transforms`.

        """
        y_t = ComposeTransform(self.y_transforms)(y)
        assert isinstance(y_t, TensorLike)
        return y_t

    def _inv_transform_y_tensor(self, y_t: TensorLike) -> TensorLike:
        """
        Invert the transformed target tensor `y_t` back to the original space.

        Parameters
        ----------
        y_t: TensorLike
            Transformed target tensor to be inverted.

        Returns
        -------
        TensorLike
            Inverted target tensor in the original data space after applying all
            inverse `y_transforms`.
        """
        y = ComposeTransform(self.y_transforms).inv(y_t)
        assert isinstance(y, TensorLike)
        return y

    def _inv_transform_y_gaussian(self, y_t: GaussianLike) -> GaussianLike:
        """
        Invert the transformed `GaussianLike` `y_t` back to the original space.

        The inversion is performed with calls to each transform's inverse_gaussian
        method that aims to use the analytical or approximate non-sampling inverse of
        the transformation for Gaussian distributions.

        Parameters
        ----------
        y_t: GaussianLike
            Transformed MultitaskMultivariateNormal target distribution to be inverted.

        Returns
        -------
        GaussianLike
            Inverted GaussianLike distribution in the original data space after applying
            all inverse `y_transforms` with the transforms `inverse_gaussian` methods.
        """
        # Invert the order since the combined transform is an inversion
        for transform in self.y_transforms[::-1]:
            if not isinstance(transform, AutoEmulateTransform):
                msg = (
                    "y_transforms must be a list of AutoEmulateTransform instances "
                    f"to support _inverse_gaussian method. Transform used: {transform}"
                )
                raise TypeError(msg)
            y_t = transform._inverse_gaussian(y_t)
        return y_t

    def _inv_transform_y_gaussian_sample(
        self, y_t: DistributionLike
    ) -> GaussianLike | GaussianProcessLike:
        """
        Invert the transformed distribution `y_t` by sampling.

        Invert the transformed distribution `y_t` by sampling and calculating
        empirical mean and covariance from the samples in the original space to
        parameterize a `GaussianLike` distribution.

        This method accepts any `DistributionLike` input but returns `GaussianLike` or
        `GaussianProcessLike` distributions.

        This method uses the number of samples specified in the initialization
        (`n_samples`) to draw samples from the transformed distribution `y_t` and
        returns a full covariance `GaussianLike` if specified (`full_covariance=True`
        in the initialization and fewer than `max_targets`) or a diagonal covariance
        `GaussianLike` otherwise for computational feasibility.

        Parameters
        ----------
        y_t: DistributionLike
            Transformed target distribution to be inverted by sampling.

        Returns
        -------
        GaussianLike | GaussianProcessLike
            A `GaussianProcessLike` distribution if the input was `GaussianProcessLike`,
            or a `GaussianLike` distribution if the input was any other
            `DistributionLike`. The distribution is parameterized by the empirical mean
            and covariance of the samples drawn from the transformed distribution in the
            original data space after applying all inverse `y_transforms`.

        Raises
        ------
        RuntimeError
            If the empirical covariance cannot be made positive definite.

        Notes
        -----
        This method can be used when the emulator's predictive distribution is not
        `GaussianLike` or when analytical or alternative approximate inversion is not
        possible.
        """
        # Handle GaussianProcessLike distinctly
        if isinstance(y_t, GaussianProcessLike):
            return _inverse_sample_gaussian_process_like(
                self._inv_transform_y_tensor, y_t, self.n_samples, self.full_covariance
            )

        # If `y_t` is not a `GaussianProcessLike`, sample from it and return a
        # `GaussianLike`
        return _inverse_sample_gaussian_like(
            self._inv_transform_y_tensor, y_t, self.n_samples, self.full_covariance
        )

    def _inv_transform_y_distribution(self, y_t: DistributionLike) -> DistributionLike:
        """
        Invert the transformed distribution `y_t` back to the original space `y`.

        This method applies the inverse transformations to the distribution `y_t`
        using the `inv` method of the `ComposeTransform` class, which is a composition
        of all inverse transforms in `y_transforms`.

        Parameters
        ----------
        y_t: DistributionLike

        Returns
        -------
        DistributionLike
            Inverted distribution `y` in the original data space after applying all
            inverse `y_transforms`.

        Raises
        ------
        RuntimeError
            If the distribution cannot be inverted due to non-bijective transforms.

        Notes
        -----
        The method requires that all transforms in `y_transforms` are bijective
        with log det Jacobian defined so that the returned transformed distribution is
        valid. As such, this method is:
        - is not used in dimensionality reduction contexts (e.g. with PCA, VAE, etc.)
        - does not return a distribution with mean and variance readily implemented
          (this would require further empirical estimation from samples from the
          returned transformed distribution `y`).
        """
        return TransformedDistribution(y_t, [ComposeTransform(self.y_transforms).inv])

    def _fit(self, x: TensorLike, y: TensorLike):
        # Transform x and y
        x_t = self._transform_x(x)
        y_t = self._transform_y_tensor(y)

        # Fit on transformed variables
        self.model.fit(x_t, y_t)

    def _predict(self, x: TensorLike, with_grad: bool) -> OutputLike:
        if with_grad and not self.supports_grad:
            msg = "Gradient calculation is not supported."
            raise ValueError(msg)

        # Transform and invert transform for prediction in original data space
        x_t = self._transform_x(x)
        y_t_pred = self.model.predict(x_t, with_grad)

        # If TensorLike, transform tensor back to original space
        if isinstance(y_t_pred, TensorLike):
            return self._inv_transform_y_tensor(y_t_pred)

        # Output derived by analytical/approximate transformations
        if not self.output_from_samples:
            if isinstance(y_t_pred, GaussianLike):
                return self._inv_transform_y_gaussian(y_t_pred)
            msg = (
                f"y_t_pred ({type(y_t_pred)}) is not `GaussianLike` and not currently "
                "supported for when inverting without sampling."
            )
            raise ValueError(msg)

        # Output derived by sampling and inverting to original space
        if isinstance(y_t_pred, DistributionLike):
            y_pred = self._inv_transform_y_distribution(y_t_pred)
            samples = y_pred.rsample(torch.Size([self.n_samples]))
            if not self.full_covariance:
                return GaussianLike(
                    samples.mean(dim=0),
                    DiagLinearOperator(samples.var(dim=0, unbiased=False)),
                )
            msg = (
                "Full covariance sampling is not currently implemented for "
                "`TransformedEmulator` with sampling-based predictions since a "
                "consistent structure for the covariance matrix cannot be guranteed "
                "for any set of composed `torch.distributions.Transform`."
            )
            raise NotImplementedError(msg)

        msg = (
            "Invalid output type from model prediction. Expected TensorLike,"
            "GaussianLike, or DistributionLike. Received: "
            f"{type(y_t_pred)}"
        )
        raise ValueError(msg)

    def predict_mean_and_variance(
        self, x: TensorLike, with_grad: bool = False
    ) -> tuple[TensorLike, TensorLike]:
        """
        Predict the mean and variance of the target variable for input `x`.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape `(n_samples, n_features)` for which to predict
            the mean and variance.
        with_grad: bool
            Whether to compute gradients with respect to the input. Defaults to False.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tuple containing:
            - Mean tensor of shape `(n_samples, n_targets)`.
            - Variance tensor of shape `(n_samples, n_targets)`.
        """
        if not self.model.supports_uq:
            msg = f"TransformedEmulator model ({self.model}) does not support UQ."
            raise RuntimeError(msg)
        y_pred = self._predict(x, with_grad)
        assert isinstance(y_pred, DistributionLike)
        samples = (
            y_pred.rsample(torch.Size([self.n_samples]))
            if with_grad
            else y_pred.sample(torch.Size([self.n_samples]))
        )
        return samples.mean(dim=0), samples.var(dim=0)

    @staticmethod
    def is_multioutput() -> bool:
        """Not implemented for TransformedEmulator.

        TransformedEmulator does not implement is_multioutput as a staticmethod
        since it depends on the emulator instance.
        """
        msg = (
            "TransformedEmulator does not implement is_multioutput as a staticmethod"
            "since it depends on the emulator instance."
        )
        raise NotImplementedError(msg)


# TODO: implement TransformedModuleEmulator with learnable parameters
# class TransformedModuleEmulator(Emulator, nn.Module, ValidationMixin):
#     def __init__(
#         self,
#         x: TensorLike,
#         y: TensorLike,
#         transforms: list[AutoEmulateTransformModule],
#         target_transform: list[AutoEmulateTransformModule],
#         model: nn.Module,
#     ):
#         _, _ = x, y
#         self.transforms = transforms
#         self.target_transform = target_transform
#         self.model = model

#     def forward(self, x):
#         x = self._transform_x(x)
#         return self._inv_transform_y_tensor(self.model(x))

#     # TODO: fix the types here
#     def _transform_x(self, x: TensorLike) -> TensorLike:
#         return ComposeTransformModule(self.transforms)(x)

#     def _transform_y_tensor(self, y: TensorLike) -> TensorLike:
#         return ComposeTransformModule(self.target_transforms)(y)

#     def _inv_transform_y_tensor(self, y: TensorLike) -> TensorLike:
#         return ComposeTransformModule(self.target_transforms).inv(y)

#     def _inv_transform_y_distribution(
#         self, y_dis: DistributionLike
#     ) -> DistributionLike:
#         return TransformedDistribution(
#             y_dis, [ComposeTransform(self.target_transform).inv]
#         )

#     def _fit(self, x, y): ...
