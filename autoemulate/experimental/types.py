from typing import Any, TypeAlias

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader

NumpyLike: TypeAlias = np.ndarray
TensorLike: TypeAlias = torch.Tensor
DistributionLike: TypeAlias = torch.distributions.Distribution
InputLike: TypeAlias = NumpyLike | TensorLike | DataLoader | torch.utils.data.Dataset
OutputLike: TypeAlias = DistributionLike | TensorLike | tuple[TensorLike, TensorLike]
ParamLike: TypeAlias = Any
TuneConfig: TypeAlias = dict[str, list[ParamLike]]
ModelConfig: TypeAlias = dict[str, ParamLike]
