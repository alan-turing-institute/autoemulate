from abc import ABC, abstractmethod

# from dataclasses import dataclass, field
from typing import Self

import torch
from torch import Tensor


class Gaussian(ABC):
    def __init__(self, mean: Tensor):
        if mean.ndim == 2:
            self.mean = mean
        else:
            s = "`mean` must have shape (n, d)."
            raise ValueError(s)

    @abstractmethod
    def logdet(self) -> Tensor: ...

    @abstractmethod
    def trace(self) -> Tensor: ...

    @abstractmethod
    def max_eig(self) -> Tensor: ...


class Dense(Gaussian):
    def __init__(self, mean: Tensor, covariance: Tensor):
        super().__init__(mean)
        n, d = self.mean.shape
        nd = n * d
        if covariance.shape == (nd, nd):
            self.covariance = covariance
        else:
            s = f"`covariance` must be ({nd}, {nd}); got {tuple(covariance.shape)}"
            raise ValueError(s)

    def logdet(self) -> Tensor:
        return self.covariance.logdet()

    def trace(self) -> Tensor:
        return self.covariance.trace()

    def max_eig(self) -> Tensor:
        return torch.linalg.norm(self.covariance, ord=2)


class Empirical(Dense):
    def __init__(self, samples: Tensor):
        # Checks
        if samples.ndim != 3:
            s = "`samples` must have shape (k, n, d)."
            raise ValueError(s)
        k, n, d = samples.shape

        # Mean and covaraince
        mean = samples.mean(dim=0)
        mu = (samples - mean).reshape(k, n * d)
        covariance = (mu.T @ mu) / (k - 1)
        super().__init__(mean, covariance)


class Structured(Gaussian):
    @abstractmethod
    def to_dense(self) -> Dense: ...

    @classmethod
    @abstractmethod
    def from_dense(cls, dense: Dense) -> Self: ...


class Ensemble(Empirical):
    def __init__(self, gaussians: list[Dense | Structured]):
        if all(isinstance(dist, (Dense, Structured)) for dist in gaussians):
            # Epistemic
            dists: list[Dense] = [
                dist.to_dense() if isinstance(dist, Structured) else dist
                for dist in gaussians
            ]
            means = torch.stack([dist.mean for dist in gaussians])
            super().__init__(means)

            # Aleatoric
            self.covariance += torch.stack([dist.covariance for dist in dists]).mean(
                dim=0
            )

        else:
            s = "gaussians must be dense or structured."
            raise ValueError(s)


class Block_Diagonal(Structured):
    def __init__(self, mean: Tensor, covariance: Tensor) -> None:
        super().__init__(mean)
        n, d = self.mean.shape
        if covariance.shape != (n, d, d):
            s = f"`covariance` must have shape ({n}, {d}, {d})."
            raise ValueError(s)
        self.covariance = covariance

    def logdet(self) -> Tensor:
        return self.covariance.logdet().sum()

    def trace(self) -> Tensor:
        return self.covariance.diagonal(dim1=-2, dim2=-1).sum()

    def max_eig(self) -> Tensor:
        return torch.linalg.norm(self.covariance, ord=2, dim=(-2, -1)).max()

    def to_dense(self) -> Dense:
        return Dense(self.mean, torch.block_diag(*self.covariance))

    @classmethod
    def from_dense(cls, dense: Dense) -> Self:
        mean, (n, d) = dense.mean, dense.mean.shape
        covariance = (
            dense.covariance.reshape(n, d, n, d)
            .diagonal(dim1=0, dim2=2)
            .permute(2, 0, 1)
        )
        return cls(mean, covariance)


class Diagonal(Structured):
    def __init__(self, mean: Tensor, covariance: Tensor) -> None:
        super().__init__(mean)
        n, d = self.mean.shape
        if covariance.shape == (n, d):
            self.covariance = covariance
        else:
            s = f"`covariance` must have shape ({n}, {d})."
            raise ValueError(s)

    def logdet(self) -> Tensor:
        return self.covariance.log().sum()

    def trace(self) -> Tensor:
        return self.covariance.sum()

    def max_eig(self) -> Tensor:
        return self.covariance.max()

    def to_dense(self) -> Dense:
        return Dense(self.mean, self.covariance.reshape(-1).diag())

    @classmethod
    def from_dense(cls, dense: Dense):
        mean, (n, d) = dense.mean, dense.mean.shape
        covariance = dense.covariance.diag().reshape(n, d)
        return cls(mean, covariance)


class Separable(Structured):
    def __init__(self, mean: Tensor, cov_n: Tensor, cov_d: Tensor):
        super().__init__(mean)
        n, d = self.mean.shape
        if cov_n.shape == (n, n) and cov_d.shape == (d, d):
            self.cov_n, self.cov_d = cov_n, cov_d
        else:
            s = f"`cov_n` must be ({n}, {n}) and `cov_d` must be ({d}, {d})"
            raise ValueError(s)

    def logdet(self) -> Tensor:
        n, d = self.mean.shape
        return d * self.cov_n.logdet() + n * self.cov_d.logdet()

    def trace(self) -> Tensor:
        return torch.trace(self.cov_n) * torch.trace(self.cov_d)

    def max_eig(self) -> Tensor:
        norm_n = torch.linalg.norm(self.cov_n, ord=2)
        norm_d = torch.linalg.norm(self.cov_d, ord=2)
        return norm_n * norm_d

    def to_dense(self) -> Dense:
        return Dense(self.mean, torch.kron(self.cov_n, self.cov_d))

    @classmethod
    def from_dense(cls, dense: Dense) -> Self:
        s = "Separable covariance not implemented yet."
        raise NotImplementedError(s)


class Dirac(Structured):
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

    def to_dense(self) -> Dense:
        n, d = self.mean.shape
        return Dense(
            self.mean,
            torch.zeros(n * d, n * d, device=self.mean.device, dtype=self.mean.dtype),
        )

    @classmethod
    def from_dense(cls, dense: Dense):
        return cls(dense.mean)
