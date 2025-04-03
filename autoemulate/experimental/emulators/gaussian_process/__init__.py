from gpytorch.means import Mean
from gpytorch.kernels import Kernel
import torch
from typing import Protocol


class MeanModuleFn(Protocol):
    def __call__(self, n_features: int, n_outputs: torch.Size) -> Mean: ...


class CovarModuleFn(Protocol):
    def __call__(self, n_features: int, n_outputs: torch.Size) -> Kernel: ...
