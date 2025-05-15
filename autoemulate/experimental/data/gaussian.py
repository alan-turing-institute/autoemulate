from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar

import torch
from torch import Tensor

T = TypeVar("T", bound="Gaussian")


class Gaussian(ABC):
    """Abstract base for a Gaussian emulator over n inputs and d outputs."""

    def __init__(self, mean: Tensor):
        if not isinstance(mean, Tensor) or mean.ndim != 2:
            raise ValueError("`mean` must be a 2D torch.Tensor of shape [n, d]")
        self.mean = mean

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

    @abstractmethod
    def to_dense(self) -> Tensor:
        """Converts covariance representation to full dense covariance."""
        ...


class Dense(Gaussian):
    def __init__(self, mean: Tensor, cov: Tensor):
        super().__init__(mean)
        nd = self.mean.shape[0] * self.mean.shape[1]
        if cov.shape != (nd, nd):
            raise ValueError(f"`cov` must be square of shape ({nd}, {nd})")
        self.cov = cov

    def logdet(self) -> Tensor:
        return self.cov.logdet()

    def trace(self) -> Tensor:
        return self.cov.trace()

    def max_eig(self) -> Tensor:
        return torch.linalg.norm(self.cov, ord=2)

    def to_dense(self) -> Tensor:
        return self.cov


class Block_Diagonal(Gaussian):
    def __init__(self, mean: Tensor, cov: Tensor):
        super().__init__(mean)
        n, d = mean.shape
        if cov.shape != (n, d, d):
            raise ValueError(f"`cov` must have shape ({n}, {d}, {d})")
        self.cov = cov

    def logdet(self) -> Tensor:
        return self.cov.logdet().sum()

    def trace(self) -> Tensor:
        return self.cov.diagonal(dim1=-2, dim2=-1).sum()

    def max_eig(self) -> Tensor:
        return torch.linalg.norm(self.cov, ord=2, dim=(-2, -1)).max()

    def to_dense(self) -> Tensor:
        return torch.block_diag(*self.cov)


class Diagonal(Gaussian):
    def __init__(self, mean: Tensor, cov: Tensor):
        super().__init__(mean)
        n, d = mean.shape
        if cov.shape != (n, d):
            raise ValueError(f"`cov` must have shape ({n}, {d})")
        self.cov = cov

    def logdet(self) -> Tensor:
        return self.cov.log().sum()

    def trace(self) -> Tensor:
        return self.cov.sum()

    def max_eig(self) -> Tensor:
        return self.cov.max()

    def to_dense(self) -> Tensor:
        return self.cov.reshape(-1).diag()


class Separable(Gaussian):
    def __init__(self, mean: Tensor, cov_n: Tensor, cov_d: Tensor):
        super().__init__(mean)
        n, d = mean.shape
        if cov_n.shape != (n, n) or cov_d.shape != (d, d):
            raise ValueError(
                f"`cov_n` must be ({n}, {n}) and `cov_d` must be ({d}, {d})"
            )
        self.cov_n, self.cov_d = cov_n, cov_d

    def logdet(self) -> Tensor:
        n, d = self.mean.shape
        return d * self.cov_n.logdet() + n * self.cov_d.logdet()

    def trace(self) -> Tensor:
        return torch.trace(self.cov_n) * torch.trace(self.cov_d)

    def max_eig(self) -> Tensor:
        norm_n = torch.linalg.norm(self.cov_n, ord=2)
        norm_d = torch.linalg.norm(self.cov_d, ord=2)
        norm = norm_n * norm_d
        return norm

    def to_dense(self):
        return torch.kron(self.cov_n, self.cov_d)


class Dirac(Gaussian):
    def __init__(self, mean: Tensor):
        super().__init__(mean)

    def logdet(self) -> Tensor:
        return torch.tensor(
            float("-inf"), device=self.mean.device, dtype=self.mean.dtype
        )

    def trace(self) -> Tensor:
        return torch.tensor(0.0, device=self.mean.device, dtype=self.mean.dtype)

    def max_eig(self) -> Tensor:
        return torch.tensor(0.0, device=self.mean.device, dtype=self.mean.dtype)

    def to_dense(self):
        n, d = self.mean.shape
        return torch.zeros(n * d, n * d, device=self.mean.device, dtype=self.mean.dtype)


class Empirical:
    def __init__(self, samples: Tensor):
        # Checks
        if samples.ndim != 3:
            raise ValueError("Samples must have 3 dimension: (k, n, d).")
        k, n, d = samples.shape

        # Mean and covaraince
        self.mean = samples.mean(dim=0)
        mu = (samples - self.mean).reshape(k, n * d)
        self.cov = (mu.T @ mu) / (k - 1)


class Ensemble(Empirical, Dense):
    def __init__(self, gaussians: list[Gaussian]):
        # Epistemic mean and covariance
        means = torch.stack([dist.mean for dist in gaussians])
        Empirical.__init__(self, means)

        # Add aleatoric covariance
        self.cov = self.cov + torch.stack([dist.to_dense() for dist in gaussians]).mean(
            dim=0
        )
        Dense.__init__(self, self.mean, self.cov)


class Empirical_Dense(Empirical, Dense):
    def __init__(self, samples: Tensor):
        Empirical.__init__(self, samples)
        Dense.__init__(self, self.mean, self.cov)


class Empirical_Block_Diagonal(Empirical, Block_Diagonal):
    def __init__(self, samples: Tensor):
        Empirical.__init__(self, samples)
        n, d = self.mean.shape
        self.cov = (
            self.cov.reshape(n, d, n, d).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        )
        Block_Diagonal.__init__(self, self.mean, self.cov)


class Empirical_Diagonal(Empirical, Diagonal):
    def __init__(self, samples: Tensor):
        Empirical.__init__(self, samples)
        n, d = self.mean.shape
        self.cov = self.cov.diagonal().reshape(n, d)
        Diagonal.__init__(self, self.mean, self.cov)
