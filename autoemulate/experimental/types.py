import numpy as np
from torch.utils.data import DataLoader
import torch

NumpyLike = np.ndarray
TensorLike = torch.Tensor
DistributionLike = torch.distributions.Distribution
InputLike = NumpyLike | TensorLike | DataLoader
OutputLike = DistributionLike | TensorLike | tuple[TensorLike, TensorLike]
