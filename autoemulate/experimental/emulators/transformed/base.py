from typing import cast

import torch
from linear_operator.operators import DiagLinearOperator
from torch.distributions import (
    ComposeTransform,
    Transform,
    TransformedDistribution,
)

from autoemulate.experimental.data.utils import ValidationMixin
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import (
    DeviceLike,
    DistributionLike,
    GaussianLike,
    OutputLike,
    TensorLike,
)


class TransformedEmulator(Emulator, ValidationMixin):
    """A transformed emulator that applies transformations to input and target data.

    This class wraps an emulator model with configurable transformations applied to
    both input features (x) and target variables (y). The emulator is trained and
    makes predictions in the transformed space, with automatic inverse transformations
    applied to return results in the original data space.

    The transformation workflow:
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

    x_transforms : list[AutoEmulateTransform]
        List of transformations applied to input data (x) in sequential order.
    model : Emulator
        The underlying emulator model that operates on transformed data.
    y_transforms : list[AutoEmulateTransform]
        List of transformations applied to target data (y) in sequential order.

    """

    x_transforms: list[AutoEmulateTransform]
    model: Emulator
    y_transforms: list[AutoEmulateTransform]

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        x_transforms: list[AutoEmulateTransform] | None,
        y_transforms: list[AutoEmulateTransform] | None,
        model: type[Emulator],
        output_from_samples: bool = False,
        n_samples: int = 1000,
        full_covariance: bool = False,
        max_targets: int = 200,
        device: DeviceLike | None = None,
        **kwargs,
    ):
        """Initialize a transformed emulator.

        Parameters
        ----------
        x : TensorLike
            Input training data tensor of shape (n_samples, n_features).
        y : TensorLike
            Target training data tensor of shape (n_samples, n_targets).
        x_transforms : list[AutoEmulateTransform] | None
            List of transforms to apply to input data in sequential order.
            If None, no transformations are applied to x.
        y_transforms : list[AutoEmulateTransform] | None
            List of transforms to apply to target data in sequential order.
            If None, no transformations are applied to y.
        model : type[Emulator]
            The emulator class to instantiate and train on transformed data.
        output_from_samples : bool, default=False
            Whether to obtain predictions by sampling from the model's predictive
            distribution. Automatically set to True for high-dimensional targets
            (n_targets > max_targets).
        n_samples : int, default=1000
            Number of samples to draw when using sampling-based predictions.
            Only used when output_from_samples=True.
        full_covariance : bool, default=False
            Whether to use full covariance matrix for predictions. If False,
            uses diagonal covariance. Automatically set to False for
            high-dimensional targets (n_targets > max_targets).
        max_targets : int, default=200
            Threshold for switching to approximate sampling-based predictions
            with diagonal covariance when dealing with high-dimensional targets.
        device : DeviceLike | None, default=None
            Device for tensor operations. If None, uses the default device.
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
        self.model = model(self._transform_x(x), self._transform_y_tensor(y), **kwargs)
        self.output_from_samples = output_from_samples or y.shape[1] > max_targets
        self.n_samples = n_samples
        self.full_covariance = full_covariance and y.shape[1] <= max_targets
        TorchDeviceMixin.__init__(self, device=device)

    def _fit_transforms(self, x: TensorLike, y: TensorLike):
        # Fit transforms
        current_x = x
        for transform in self.x_transforms:
            transform.fit(current_x)
            current_x = transform(current_x)
        # Fit target transforms
        current_y = y
        for transform in self.y_transforms:
            transform.fit(current_y)
            current_y = transform(current_y)

    def refit(self, x: TensorLike, y: TensorLike, retrain_transforms: bool = False):
        """Refit the emulator with new data and optionally retrain transforms.

        Parameters
        ----------
        x : TensorLike
            New input training data tensor of shape (n_samples, n_features).
        y : TensorLike
            New target training data tensor of shape (n_samples, n_targets).
        retrain_transforms : bool, default=False
            Whether to retrain the transforms on the new data. If False,
            uses the existing fitted transforms from initialization.

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
        x_t = ComposeTransform(self._cast(self.x_transforms))(x)
        assert isinstance(x_t, TensorLike)
        return x_t

    def _transform_y_tensor(self, y: TensorLike) -> TensorLike:
        y_t = ComposeTransform(self._cast(self.y_transforms))(y)
        assert isinstance(y_t, TensorLike)
        return y_t

    @staticmethod
    def _cast(transforms: list[AutoEmulateTransform]) -> list[Transform]:
        return cast(list[Transform], transforms)

    def _inv_transform_y_tensor(self, y_t: TensorLike) -> TensorLike:
        target_transforms = self._cast(self.y_transforms)
        y = ComposeTransform(target_transforms).inv(y_t)
        assert isinstance(y, TensorLike)
        return y

    def _inv_transform_y_mvn(self, y_t: GaussianLike) -> GaussianLike:
        # Invert the order since the combined transform is an inversion
        for transform in self.y_transforms[::-1]:
            y_t = transform._inverse_gaussian(y_t)
        return y_t

    def _inv_transform_y_mvn_sample(self, y_t: DistributionLike) -> GaussianLike:
        samples = self._inv_transform_y_tensor(
            torch.stack([y_t.sample() for _ in range(self.n_samples)], dim=0)
        )
        assert isinstance(samples, TensorLike)
        mean = samples.mean(dim=0)
        cov = (
            make_positive_definite(samples.view(self.n_samples, -1).T.cov())
            if self.full_covariance
            # Efficient for large output shape but no covariance
            else DiagLinearOperator(samples.view(self.n_samples, -1).var(dim=0))
        )
        return GaussianLike(mean, cov)

    def _inv_transform_y_distribution(self, y_t: DistributionLike) -> DistributionLike:
        """Invert the distribution using the target transforms."""
        target_transforms = self._cast(self.y_transforms)
        return TransformedDistribution(y_t, [ComposeTransform(target_transforms).inv])

    def _fit(self, x: TensorLike, y: TensorLike):
        # Transform x and y
        x_t = self._transform_x(x)
        y_t = self._transform_y_tensor(y)

        # Fit on transformed variables
        self.model.fit(x_t, y_t)

    def _predict(self, x: TensorLike) -> OutputLike:
        # Transform and invert transform for prediction in original data space
        x_t = self._transform_x(x)
        y_t_pred = self.model.predict(x_t)

        # If TensorLike, transform tensor back to original space
        if isinstance(y_t_pred, TensorLike):
            return self._inv_transform_y_tensor(y_t_pred)

        # Output derived by analytical/approximate transformations
        if not self.output_from_samples:
            if isinstance(y_t_pred, GaussianLike):
                return self._inv_transform_y_mvn(y_t_pred)
            if isinstance(y_t_pred, DistributionLike):
                return self._inv_transform_y_distribution(y_t_pred)
            msg = "y_pred is not TensorLike, GaussianLike or DistributionLike"
            raise ValueError(msg)

        # Output derived by sampling and inverting to original space
        if isinstance(y_t_pred, DistributionLike):
            return self._inv_transform_y_mvn_sample(y_t_pred)
        msg = (
            "Invalid output type from model prediction. Expected TensorLike,"
            "GaussianLike, or DistributionLike. Received: "
            f"{type(y_t_pred)}"
        )
        raise ValueError(msg)

    # TODO: this requires self, should we update the base emulator?
    def is_multioutput(self) -> bool:  # type: ignore PGH003
        # Check if the model is multioutput
        return self.model.is_multioutput()


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
