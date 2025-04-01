from typing import Any
from typing import TypeAlias
import numpy as np

from torch.utils.data import DataLoader
import torch
import torch.utils
import torch.utils.data

NumpyLike: TypeAlias = np.ndarray
TensorLike: TypeAlias = torch.Tensor
DistributionLike: TypeAlias = torch.distributions.Distribution
InputLike: TypeAlias = NumpyLike | TensorLike | DataLoader | torch.utils.data.Dataset
OutputLike: TypeAlias = DistributionLike | TensorLike | tuple[TensorLike, TensorLike]
ValueLike: TypeAlias = Any
TuneConfig: TypeAlias = dict[str, list[ValueLike]]
ModelConfig: TypeAlias = dict[str, ValueLike]
