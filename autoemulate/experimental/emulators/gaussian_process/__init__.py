from typing import Protocol

import torch
from gpytorch.kernels import Kernel
from gpytorch.means import Mean


class MeanModuleFn(Protocol):
    def __call__(self, n_features: int, n_outputs: torch.Size) -> Mean: ...


class CovarModuleFn(Protocol):
    # TODO: consider revising input API to be more flexible
    def __call__(self, n_features: int, n_outputs: torch.Size) -> Kernel: ...
