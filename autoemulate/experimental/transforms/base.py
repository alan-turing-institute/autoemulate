from abc import ABC, abstractmethod

from pyro.distributions import TransformModule
from torch.distributions import Transform

from autoemulate.experimental.types import GaussianLike


class AutoEmulateTransform(Transform, ABC):
    @abstractmethod
    def fit(self, x): ...

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
