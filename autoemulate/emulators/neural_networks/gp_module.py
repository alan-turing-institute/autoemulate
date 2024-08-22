import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPModule(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_inputs=None,
        train_targets=None,
        likelihood=None,
        mean=None,
        covar=None,
    ):
        """Multioutput GP module for correlated outputs, see Bonilla et al. 2008."""

        super().__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            # gpytorch.means.ConstantMean(),
            mean,
            num_tasks=likelihood.num_tasks,
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            # gpytorch.kernels.MaternKernel(),
            covar,
            num_tasks=likelihood.num_tasks,
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class BatchIndependentGP(gpytorch.models.ExactGP):
    """Multioutput GP for independent outputs."""

    def __init__(self, train_inputs=None, train_targets=None, likelihood=None):
        super().__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([likelihood.num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() * gpytorch.kernels.ConstantKernel(),
            batch_shape=torch.Size([likelihood.num_tasks]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
