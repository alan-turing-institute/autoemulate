from typing import Any
import numpy as np

import torch
import torch.utils
import torch.utils.data

NumpyLike = np.ndarray
TensorLike = torch.Tensor
DistributionLike = torch.distributions.Distribution
InputLike = (
    NumpyLike | TensorLike | torch.utils.data.DataLoader | torch.utils.data.Dataset
)
OutputLike = DistributionLike | TensorLike | tuple[TensorLike, TensorLike]
ParamLike = Any
TuneConfig = dict[str, list[ParamLike]]
ModelConfig = dict[str, ParamLike]
