import torch
from linear_operator.operators import DiagLinearOperator
from torch.distributions import ComposeTransform, Transform, TransformedDistribution
from torch.func import jacrev

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import (
    DeviceLike,
    DistributionLike,
    GaussianLike,
    OutputLike,
    TensorLike,
)
from autoemulate.data.utils import ConversionMixin, ValidationMixin
from autoemulate.emulators.base import Emulator
from autoemulate.emulators.transformed.delta_method import (
    delta_method,
    delta_method_mean_only,
)
from autoemulate.transforms.base import AutoEmulateTransform, is_affine


class TransformedEmulator(Emulator, ValidationMixin, ConversionMixin):
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

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        x_transforms: list[Transform] | None,
        y_transforms: list[Transform] | None,
        model: type[Emulator],
        output_from_samples: bool = False,
        n_samples: int = 100,
        full_covariance: bool = False,
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
            Only used when output_from_samples=True. Defaults to 100.
        full_covariance: bool
            Whether to use full covariance matrix for predictions. If False,
            uses diagonal covariance. Automatically set to False for
            high-dimensional targets (n_targets > max_targets). Defaults to False.
        device: DeviceLike | None
            Device for tensor operations. If None, uses the default device.
            Defaults to None.
        **kwargs
            Additional keyword arguments passed to the emulator constructor.

        Notes
        -----
        - Transforms are fitted on the provided training data during initialization.
        - The underlying emulator is trained on the transformed data.
        - An empirical check tests whether all y_transforms behave approximately affine.
          If so, the mean is inverted directly; otherwise a mean-only delta method
          correction is used. This is enabled by default and is lightweight.
        """
        self.x_transforms = x_transforms or []
        self.y_transforms = y_transforms or []

        # Convert and move the new data to device
        TorchDeviceMixin.__init__(self, device=device)
        x, y = self._move_tensors_to_device(x, y)

        self._fit_transforms(x, y)
        self.untransformed_model_name = model.model_name()
        self.model = model(
            self._transform_x(x),
            self._transform_y_tensor(y),
            device=device,
            **kwargs,
        )
        # Cache for constant Jacobian of inverse y-transform when affine
        self._fixed_jacobian_y_inv = None
        self.output_from_samples = output_from_samples
        if (
            not output_from_samples
            and full_covariance
            and not all(isinstance(t, AutoEmulateTransform) for t in self.y_transforms)
        ):
            msg = (
                "y_transforms must be a list of AutoEmulateTransform instances to "
                f"support outputs without sampling. y_transforms: {self.y_transforms}"
            )
            raise RuntimeError(msg)
        self.n_samples = n_samples
        self.full_covariance = full_covariance

        # TODO: add API to indicate that pdf not valid when not all transforms bijective
        self.supports_grad = self.model.supports_grad
        self.supports_uq = self.model.supports_uq

        # Precompute and cache the Jacobian of the inverse y-transform if affine
        if not self.output_from_samples and self.all_y_transforms_affine:
            try:
                self._compute_and_cache_inv_y_jacobian(y)
            except Exception:
                self._fixed_jacobian_y_inv = None

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
        self._y_transforms_affine = []
        current_y = y
        for transform in self.y_transforms:
            if isinstance(transform, AutoEmulateTransform):
                transform.fit(current_y)

            # Check if transform is affine for later use in prediction
            self._y_transforms_affine.append(is_affine(transform, current_y))

            # Now update the running transformed target
            current_y = transform(current_y)
            assert isinstance(current_y, TensorLike)

        # Cache whether all y transforms are affine
        self.all_y_transforms_affine = (
            all(self._y_transforms_affine) if self._y_transforms_affine else False
        )

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
            # Invalidate and recompute cached Jacobian if transforms changed
            self._fixed_jacobian_y_inv = None
            if not self.output_from_samples and self.all_y_transforms_affine:
                try:
                    self._compute_and_cache_inv_y_jacobian(y)
                except Exception:
                    self._fixed_jacobian_y_inv = None
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

    def _fit(
        self,
        x: TensorLike,
        y: TensorLike,
        validation_data: tuple[TensorLike, TensorLike] | None = None,  # noqa: ARG002
    ):
        # Transform x and y
        x_t = self._transform_x(x)
        y_t = self._transform_y_tensor(y)

        # Detach transformed tensors to avoid retaining graphs from transforms
        # during training of the underlying model. Transforms are treated as fixed
        # feature engineering at fit time; gradients are only required at predict.
        if isinstance(x_t, torch.Tensor):
            x_t = x_t.detach()
        if isinstance(y_t, torch.Tensor):
            y_t = y_t.detach()

        # Fit on transformed variables
        self.model.fit(x_t, y_t)

    def predict_mean(
        self, x: TensorLike, with_grad: bool = False, n_samples: int | None = None
    ) -> TensorLike:
        """
        Predict the mean of the target variable for input `x`.

        Parameters
        ----------
        x: TensorLike
            Input tensor of shape `(n_batch, n_features)` for which to predict
            the mean.
        with_grad: bool
            Whether to compute gradients with respect to the input. Defaults to False.
        n_samples: int | None
            Number of samples to draw when using sampling-based predictions. If
            specified, overrides `n_samples` specified at initialization.
            Defaults to None.

        Returns
        -------
        TensorLike
            Mean tensor of shape `(n_batch, n_targets)`.
        """
        x = self._ensure_with_grad(x, with_grad)
        y_t_pred = self.model.predict(self._transform_x(x), with_grad)

        if isinstance(y_t_pred, TensorLike):
            return self._inv_transform_y_tensor(y_t_pred)

        if not self.output_from_samples:
            # Output with inverting mean
            if self.all_y_transforms_affine:
                mean = self._inv_transform_y_tensor(y_t_pred.mean)
                return mean.detach() if not with_grad else mean

            # Output with delta method (mean only)
            out = delta_method_mean_only(
                ComposeTransform(self.y_transforms).inv,
                y_t_pred.mean,
                (
                    y_t_pred.covariance_matrix
                    if isinstance(y_t_pred, GaussianLike)
                    else y_t_pred.variance
                ),
                True,
            )
            return out["mean_total"].detach() if not with_grad else out["mean_total"]

        # Output from samples
        y_pred = self._inv_transform_y_distribution(y_t_pred)
        n_samples_ = torch.Size([self.n_samples if n_samples is None else n_samples])
        samples = y_pred.rsample(n_samples_) if with_grad else y_pred.sample(n_samples_)
        mean = samples.mean(dim=0)
        return mean.detach() if not with_grad else mean

    def _predict(self, x: TensorLike, with_grad: bool) -> OutputLike:
        if with_grad and not self.supports_grad:
            msg = "Gradient calculation is not supported."
            raise ValueError(msg)

        # Ensure x has requires_grad if with_grad is True
        x = self._ensure_with_grad(x, with_grad)

        # Transform and invert transform for prediction in original data space
        x_t = self._transform_x(x)
        y_t_pred = self.model.predict(x_t, with_grad)

        # If TensorLike, transform tensor back to original space
        if isinstance(y_t_pred, TensorLike):
            return self._inv_transform_y_tensor(y_t_pred)

        # Output derived by analytical/approximate transformations
        if not self.output_from_samples:
            if isinstance(y_t_pred, GaussianLike):
                # Full covariance calculation
                if self.full_covariance:
                    return self._inv_transform_y_gaussian(y_t_pred)
                # Variance only
                output = delta_method(
                    ComposeTransform(self.y_transforms).inv,  # type: ignore  # noqa: PGH003
                    y_t_pred.mean,
                    y_t_pred.covariance_matrix,
                    # If all affine, mean transformation is exact
                    include_second_order=not self.all_y_transforms_affine,
                    fixed_jacobian=self._fixed_jacobian_y_inv,
                )
                mean, var = output["mean_total"], output["variance_approx"]
                if not with_grad:
                    mean = mean.detach()
                    var = var.detach()
                # Add small jitter to variance to ensure it's positive for Normal dist
                min_variance = 1e-6
                var = torch.clamp(var, min=min_variance)
                # Assume batch shape only dim
                return torch.distributions.Independent(
                    torch.distributions.Normal(mean, var.sqrt()),
                    reinterpreted_batch_ndims=mean.ndim - 1,
                )

            msg = (
                f"Inverse transform without sampling for y_t_pred ({type(y_t_pred)}) "
                "is not currently supported, expected GaussianLike."
            )
            raise ValueError(msg)

        # Output derived by sampling and inverting to original space
        if isinstance(y_t_pred, DistributionLike):
            y_pred = self._inv_transform_y_distribution(y_t_pred)
            n_samples = torch.Size([self.n_samples])
            samples = (
                y_pred.rsample(n_samples) if with_grad else y_pred.sample(n_samples)
            )
            # Return mean and variance
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

    def _compute_and_cache_inv_y_jacobian(self, y: TensorLike) -> None:
        """Compute and cache a constant Jacobian for inverse y-transform.

        The Jacobian J = d(inv_y_transform)/dy_t is constant when all y-transforms
        are affine, so we precompute it once at an arbitrary point and reuse it.
        """
        # Build a small representative input with batch=1 in transformed space
        y_t_example = self._transform_y_tensor(y[:1])
        # Flatten everything except batch
        batch_size = y_t_example.shape[0]
        assert batch_size == 1
        input_dim = y_t_example[0].numel()
        x0 = torch.zeros(
            (input_dim,), dtype=y_t_example.dtype, device=y_t_example.device
        )

        def forward_fn_flat(z_flat: TensorLike) -> TensorLike:
            # z_flat: (input_dim,)
            z = z_flat.view(y_t_example.shape)
            out = ComposeTransform(self.y_transforms).inv(z)
            assert isinstance(out, TensorLike)
            return out.reshape(-1)  # (output_dim,)

        # Jacobian of shape (output_dim, input_dim)
        jac = jacrev(forward_fn_flat)(x0)
        jac = jac[0] if isinstance(jac, tuple) else jac  # In case of tuple outputs

        # Cache
        self._fixed_jacobian_y_inv = jac.detach()


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
