from abc import ABC, abstractmethod
import torch

from autoemulate.experimental.types import InputLike


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, x: InputLike) -> InputLike: ...


class StandardizerMixin(Preprocessor):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def preprocess(self, x):
        # TODO: check the expected dims being used here
        # TODO: handle case when self.std contains 0s
        return (x - self.mean) / self.std
