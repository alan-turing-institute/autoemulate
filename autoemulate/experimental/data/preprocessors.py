from abc import ABC, abstractmethod
import torch

from autoemulate.experimental.types import InputLike


class Preprocessor(ABC):
    @abstractmethod
    def __init__(*args, **kwargs): ...

    @abstractmethod
    def preprocess(self, x: InputLike) -> InputLike: ...


class Standardizer(Preprocessor):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        if len(mean.shape) != 2:
            raise ValueError(f"mean is expected to be 2D, shape passed ({mean.shape})")
        if len(std.shape) != 2:
            raise ValueError(f"std is expected to be 2D, shape passed ({std.shape})")

        # Set small values in std as 1.0 instead, see:
        # https://github.com/scikit-learn/scikit-learn/blob/812ff67e6725a8ca207a37f5ed4bfeafc5d1265d/sklearn/preprocessing/_data.py#L111
        std[std < 10 * torch.finfo(std.dtype).eps] = 1.0
        self.mean = mean
        self.std = std

    def preprocess(self, x):
        """
        Parameters
        ----------

        x : torch.Tensor
            The input tensor to be standardized.

        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected 2D torch.Tensor, actual type {type(x)}")
        if len(x.shape) != 2:
            raise ValueError(
                f"Expected 2D torch.Tensor, actual shape dim {len(x.shape)}"
            )
        return (x - self.mean) / self.std
