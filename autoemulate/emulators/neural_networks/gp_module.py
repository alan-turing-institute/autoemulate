import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrGPModule(gpytorch.models.ExactGP):
    """
    Multioutput GP module for correlated outputs, see Bonilla et al. 2008.
    """

    def __init__(
        self,
        likelihood=None,
        mean=None,
        covar=None,
    ):
        super().__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            mean,
            num_tasks=likelihood.num_tasks,
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            covar,
            num_tasks=likelihood.num_tasks,
            rank=1,
        )

    def forward(self, x):
        """
        Forward pass of the GP module.

        Parameters
        ----------
        x: (batch_size, n_points, input_dim)

        Returns
        -------
        gpytorch.distributions.MultitaskMultivariateNormal
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class IndepGPModule(gpytorch.models.ExactGP):
    """
    Multioutput GP for independent outputs. Fits one GP per output.
    """

    def __init__(
        self,
        likelihood=None,
        mean=None,
        covar=None,
    ):
        super().__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
        num_tasks = likelihood.num_tasks
        # create multioutput through batch shape
        # TODO: this might need a ScaleKernel too
        self.mean_module = mean
        self.mean_module.batch_shape = torch.Size([num_tasks])
        self.covar_module = covar
        self.covar_module.batch_shape = torch.Size([num_tasks])

    def forward(self, x):
        """
        Forward pass of the GP module.

        Parameters
        ----------
        x: (batch_size, n_points, input_dim)

        Returns
        -------
        gpytorch.distributions.MultitaskMultivariateNormal
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
