from typing import Any, TypeAlias

import numpy as np
import torch
import torch.utils
import torch.utils.data
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from torch.utils.data import DataLoader

NumpyLike: TypeAlias = np.ndarray
TensorLike: TypeAlias = torch.Tensor
DistributionLike: TypeAlias = torch.distributions.Distribution
GaussianLike: TypeAlias = MultivariateNormal
GaussianProcessLike: TypeAlias = MultitaskMultivariateNormal
InputLike: TypeAlias = NumpyLike | TensorLike | DataLoader | torch.utils.data.Dataset
OutputLike: TypeAlias = DistributionLike | TensorLike
ParamLike: TypeAlias = Any
TuneParams: TypeAlias = dict[str, list[ParamLike]]
ModelParams: TypeAlias = dict[str, ParamLike]
TransformedEmulatorParams: TypeAlias = dict[str, ParamLike]
DeviceLike: TypeAlias = str | torch.device

# Torch dtype's
TorchScalarDType = (torch.float32, torch.float64, torch.int32, torch.int64)
