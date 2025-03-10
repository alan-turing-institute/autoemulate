import torch
import numpy as np
from abc import ABC, abstractmethod

from autoemulate.refactor.utils import convert_to_numpy, convert_to_tensor


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor):
        pass

    @abstractmethod
    def predict(
        self, X: np.ndarray | torch.Tensor
    ) -> torch.Tensor | torch.distributions.Distribution:
        pass

    # @abstractmethod
    # def tune(self, X, y: torch.Tensor):
    #     pass

    # @abstractmethod
    # def run_cv(self, X: torch.Tensor, y: torch.Tensor):
    #     pass


# Mixin for handling input types
class InputTypeMixin:
    def convert_to_tensor(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        return convert_to_tensor(X)

    def convert_to_numpy(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        return convert_to_numpy(X)
