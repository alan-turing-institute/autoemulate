from typing import TypeAlias
import numpy as np
from torch.utils.data import DataLoader
import torch

NumpyLike: TypeAlias = np.ndarray
TensorLike: TypeAlias = torch.Tensor
DistributionLike: TypeAlias = torch.distributions.Distribution
InputLike: TypeAlias = NumpyLike | TensorLike | DataLoader
OutputLike: TypeAlias = DistributionLike | TensorLike | tuple[TensorLike, TensorLike]
