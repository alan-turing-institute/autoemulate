from typing import cast

from torch.distributions import ComposeTransform, Transform, TransformedDistribution

from autoemulate.experimental.data.utils import ValidationMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.transforms.base import AutoEmulateTransform
from autoemulate.experimental.types import (
    DistributionLike,
    GaussianLike,
    OutputLike,
    TensorLike,
)


class TransformedEmulator(Emulator, ValidationMixin):
    transforms: list[AutoEmulateTransform]
    model: Emulator
    target_transforms: list[AutoEmulateTransform]

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        transforms: list[AutoEmulateTransform] | None,
        target_transforms: list[AutoEmulateTransform] | None,
        model: type[Emulator],
        **kwargs,
    ):
        """Initialize a transformed emulator.

        Parameters
        ----------
            x (TensorLike): Input data tensor.
            y (TensorLike): Target data tensor.
            transforms (list[AutoEmulateTransform] | None): List of transforms to apply
                to the input data. The transforms are applied to x in the order they
                appear in the list.
            target_transforms (list[AutoEmulateTransform] | None): List of transforms to
                apply to the target data. The transforms are applied to y in the order
                they appear in the list
            model (type[Emulator]): The emulator model class to use.
            **kwargs: Additional keyword arguments to pass to the model constructor.


        """
        self.transforms = transforms if transforms is not None else []
        self.target_transforms = (
            target_transforms if target_transforms is not None else []
        )
        self._fit_transforms(x, y)
        self.model = model(self._transform_x(x), self._transform_y_tensor(y), **kwargs)

    def _fit_transforms(self, x: TensorLike, y: TensorLike):
        # Fit transforms
        current_x = x
        for transform in self.transforms:
            transform.fit(current_x)
            current_x = transform(current_x)
        # Fit target transforms
        current_y = y
        for transform in self.target_transforms:
            transform.fit(current_y)
            current_y = transform(current_y)

    def refit(self, x: TensorLike, y: TensorLike, retrain_transforms: bool = False):
        """Refit the emulator with new data and optionally retrain transforms.

        Parameters
        ----------
            x (TensorLike): New input data tensor.
            y (TensorLike): New target data tensor.
            retrain_transforms (bool): If True, retrain the transforms on the new data.
                If False, use the existing transforms. Default is False.
        """
        if not retrain_transforms:
            self._fit_transforms(x, y)
        self.fit(x, y)

    def _transform_x(self, x: TensorLike) -> TensorLike:
        transformed_x = ComposeTransform(self._cast(self.transforms))(x)
        assert isinstance(transformed_x, TensorLike)
        return transformed_x

    def _transform_y_tensor(self, y: TensorLike) -> TensorLike:
        transformed_y = ComposeTransform(self._cast(self.target_transforms))(y)
        assert isinstance(transformed_y, TensorLike)
        return transformed_y

    @staticmethod
    def _cast(transforms: list[AutoEmulateTransform]) -> list[Transform]:
        return cast(list[Transform], transforms)

    def _inv_transform_y_tensor(self, y: TensorLike) -> TensorLike:
        target_transforms = self._cast(self.target_transforms)
        inv_transformed_y = ComposeTransform(target_transforms).inv(y)
        assert isinstance(inv_transformed_y, TensorLike)
        return inv_transformed_y

    def _inv_transform_y_mvn(self, y_dis: GaussianLike) -> GaussianLike:
        # Invert the order since the combined transform is an inversion
        for transform in self.target_transforms[::-1]:
            y_dis = transform._inverse_gaussian(y_dis)
        return y_dis

    def _inv_transform_y_distribution(
        self, y_dis: DistributionLike
    ) -> DistributionLike:
        """Invert the distribution using the target transforms."""
        target_transforms = self._cast(self.target_transforms)
        return TransformedDistribution(y_dis, [ComposeTransform(target_transforms).inv])

    def _fit(self, x: TensorLike, y: TensorLike):
        # Transform x and y
        x = self._transform_x(x)
        y = self._transform_y_tensor(y)
        # Fit on transformed variables
        self.model.fit(x, y)

    def _predict(self, x: TensorLike) -> OutputLike:
        # Transform and invert transform for prediction in original data space
        x = self._transform_x(x)
        y_pred = self.model.predict(x)
        if isinstance(y_pred, TensorLike):
            return self._inv_transform_y_tensor(y_pred)
        if isinstance(y_pred, GaussianLike):
            return self._inv_transform_y_mvn(y_pred)
        if isinstance(y_pred, DistributionLike):
            return self._inv_transform_y_distribution(y_pred)
        msg = "y_pred is not TensorLike, GaussianLike or DistributionLike"
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
