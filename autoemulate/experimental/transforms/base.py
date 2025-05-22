from abc import ABC, abstractmethod

from pyro.distributions import TransformModule
from torch.distributions import Transform

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

    def _inverse_sample(self, x: GaussianLike, n_samples: int = 100) -> GaussianLike:
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def _inverse_gaussian(self, x: GaussianLike) -> GaussianLike:
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)


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
