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

    def refit(self, x: TensorLike, y: TensorLike, retrain_transforms: bool = False):
        # Retrain transforms if requested
        if not retrain_transforms:
            # TODO: add fit method for composed transforms
            # self.transforms.fit(x)
            # self.target_transforms.fit(y)
            ...

        # Fit on transformed variables
        self.model.fit(x, y)
        # TODO: add implementation
        msg = "refit not implemented yet"
        raise NotImplementedError(msg)

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        transforms: list[AutoEmulateTransform] | None,
        target_transforms: list[AutoEmulateTransform] | None,
        model: type[Emulator],
        **kwargs,
    ):
        self.transforms = transforms if transforms is not None else []
        self.target_transforms = (
            target_transforms if target_transforms is not None else []
        )

        # Fit transforms and target_transform at init
        current_x = x
        for transform in self.transforms:
            transform.fit(current_x)
            current_x = transform(current_x)
        current_y = y
        for transform in self.target_transforms:
            transform.fit(current_y)
            current_y = transform(current_y)

        self.model = model(self._transform_x(x), self._transform_y_tensor(y), **kwargs)

    def _transform_x(self, x: TensorLike) -> TensorLike:
        # TODO: consider removing cast and either creating new class or not subclassing
        # to begin with
        transformed_x = ComposeTransform(cast(list[Transform], self.transforms))(x)
        assert isinstance(transformed_x, TensorLike)
        return transformed_x

    def _transform_y_tensor(self, y: TensorLike) -> TensorLike:
        transformed_y = ComposeTransform(cast(list[Transform], self.target_transforms))(
            y
        )
        assert isinstance(transformed_y, TensorLike)
        return transformed_y

    def _inv_transform_y_tensor(self, y: TensorLike) -> TensorLike:
        inv_transformed_y = ComposeTransform(
            cast(list[Transform], self.target_transforms)
        ).inv(y)
        assert isinstance(inv_transformed_y, TensorLike)
        return inv_transformed_y

    def _inv_transform_y_mvn(self, y_dis: GaussianLike) -> GaussianLike:
        for transform in self.target_transforms:
            y_dis = transform._inverse_gaussian(y_dis)
        return y_dis

    def _inv_transform_y_distribution(
        self, y_dis: DistributionLike
    ) -> DistributionLike:
        return TransformedDistribution(
            y_dis, [ComposeTransform(cast(list[Transform], self.target_transforms)).inv]
        )

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
        msg = "y_pred is not TensorLike or DistributionLike"
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
