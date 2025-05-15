from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class Gaussian(ABC):
    """Abstract base for a Gaussian emulator over n inputs and d outputs."""
    mean: Tensor  # shape [n, d]

    def __post_init__(self):
        if not isinstance(self.mean, Tensor) or self.mean.ndim != 2:
            raise ValueError("`mean` must be a 2D torch.Tensor of shape [n, d]")

    @property
    def n(self) -> int:
        return self.mean.shape[0]

    @property
    def d(self) -> int:
        return self.mean.shape[1]

    @abstractmethod
    def logdet(self) -> Tensor:
        """Log‐determinant of the full covariance."""
        ...

    @abstractmethod
    def trace(self) -> Tensor:
        """Trace of the full covariance."""
        ...

    @abstractmethod
    def max_eig(self) -> Tensor:
        """Largest eigenvalue (spectral norm) of the full covariance."""
        ...

@dataclass
class Dense(Gaussian):
    cov: Tensor  # shape [n*d, n*d]

    def __post_init__(self):
        super().__post_init__()
        nd = self.n * self.d
        if self.cov.shape != (nd, nd):
            raise ValueError(f"`cov` must be square of shape ({nd}, {nd})")

    def logdet(self) -> Tensor:
        return self.cov.logdet()

    def trace(self) -> Tensor:
        return self.cov.trace()

    def max_eig(self) -> Tensor:
        return torch.linalg.norm(self.cov, ord=2)

@dataclass
class BlockDiagonal(Gaussian):
    cov: Tensor  # shape [n, d, d]

    def __post_init__(self):
        super().__post_init__()
        if self.cov.shape != (self.n, self.d, self.d):
            raise ValueError(f"`cov` must have shape ({self.n}, {self.d}, {self.d})")
        
    def logdet(self) -> Tensor:
        return self.cov.logdet().sum()

    def trace(self) -> Tensor:
        return self.cov.diagonal(dim1=-2, dim2=-1).sum()

    def max_eig(self) -> Tensor:
        return torch.linalg.norm(self.cov, ord=2, dim=(-2, -1)).max()

@dataclass
class Diagonal(Gaussian):
    cov: Tensor  # shape [n, d]

    def __post_init__(self):
        super().__post_init__()
        if self.cov.shape != (self.n, self.d):
            raise ValueError(f"`cov` must have shape ({self.n}, {self.d})")

    def logdet(self) -> Tensor:
        return self.cov.log().sum()

    def trace(self) -> Tensor:
        return self.cov.sum()

    def max_eig(self) -> Tensor:
        return self.cov.max()

@dataclass
class Separable(Gaussian):
    cov_n: Tensor  # shape [n, n]
    cov_d: Tensor  # shape [d, d]

    def __post_init__(self):
        super().__post_init__()
        if self.cov_n.shape != (self.n, self.n) or self.cov_d.shape != (self.d, self.d):
            raise ValueError(f"`cov_n` must be ({self.n}, {self.n}) and `cov_d` must be ({self.d}, {self.d})")

    def logdet(self) -> Tensor:
        return self.d * self.cov_n.logdet() + self.n * self.cov_d.logdet()

    def trace(self) -> Tensor:
        return torch.trace(self.cov_n) * torch.trace(self.cov_d)

    def max_eig(self) -> Tensor:
        return (
            torch.linalg.norm(self.cov_n, ord=2)
            * torch.linalg.norm(self.cov_d, ord=2)
        )
