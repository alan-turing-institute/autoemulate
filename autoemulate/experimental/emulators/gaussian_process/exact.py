import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import torch
from torch import nn
import numpy as np

from autoemulate.experimental.emulators.base import (
    Emulator,
    InputTypeMixin,
)
from autoemulate.utils import set_random_seed
from autoemulate.experimental.types import InputLike, OutputLike

from gpytorch.means import Mean
from gpytorch.kernels import Kernel


class GaussianProcessExact(Emulator, InputTypeMixin, gpytorch.models.ExactGP):
    likelihood: MultitaskGaussianLikelihood
    random_state: int | None = None
    epochs: int = 10

    def __init__(
        self,
        x: InputLike,
        y: OutputLike,
        likelihood: MultitaskGaussianLikelihood,
        mean_module: Mean,
        covar_module: Kernel,
        random_state: int | None = None,
    ):
        if random_state is not None:
            set_random_seed(random_state)
        assert isinstance(y, torch.Tensor) and isinstance(x, torch.Tensor)
        self.y_dim_ = y.ndim
        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1

        # wrapping in ScaleKernel is generally good, as it adds an output scale parameter
        if not isinstance(covar_module, gpytorch.kernels.ScaleKernel):
            covar_module = gpytorch.kernels.ScaleKernel(
                covar_module, batch_shape=torch.Size([self.n_outputs_])
            )

        super().__init__(x, y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: InputLike):
        assert isinstance(x, torch.Tensor)
        mean = self.mean_module(x)
        assert isinstance(mean, torch.Tensor)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean, covar)
        )

    def fit(self, x: InputLike, y: InputLike | None):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        for epoch in range(self.epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            output = self(x)
            loss = mll(output, y)
            assert isinstance(loss, torch.Tensor)
            loss = -loss
            loss.backward()

            print(
                "Iter %d/%d - Loss: %.3f  kernels: %s  noise: %s  likelihood: %s"
                % (
                    epoch + 1,
                    self.epochs,
                    loss,
                    "; ".join(
                        [
                            f"{name} | {sub_kernel.lengthscale}"
                            for (
                                name,
                                sub_kernel,
                            ) in self.covar_module.base_kernel.named_sub_kernels()
                        ]
                    ),
                    self.likelihood.noise,
                    self.likelihood,
                )
            )
            optimizer.step()

        self.is_fitted_ = True

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        return self(x)

    def cross_validate(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")

    @staticmethod
    def get_tune_config():
        return {
            "epochs": [100, 200, 300],
            "batch_size": [16, 32],
            "activation": [
                nn.ReLU,
                nn.GELU,
            ],
            "optimizer": [torch.optim.Adam],
            "lr": list(np.logspace(-6, -1)),
        }
