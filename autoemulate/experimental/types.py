import numpy as np
import torch

NumpyLike = np.ndarray
TensorLike = torch.Tensor
DistributionLike = torch.distributions.Distribution
InputLike = NumpyLike | TensorLike | torch.utils.data.DataLoader
OutputLike = DistributionLike | TensorLike | tuple[TensorLike, TensorLike]
