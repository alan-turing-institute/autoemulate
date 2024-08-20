import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPModule(gpytorch.models.ExactGP):
    def __init__(self, train_inputs=None, train_targets=None, likelihood=None):
        # detail: We don't set train_inputs and train_targets here skorch because
        # will take care of that.
        super().__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=likelihood.num_tasks,
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=likelihood.num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
