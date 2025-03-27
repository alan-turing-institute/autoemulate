import gpytorch
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
import torch
from torch import nn

from autoemulate.experimental.config import FitConfig
from autoemulate.experimental.emulators.base import (
    Emulator,
    InputTypeMixin,
)
from autoemulate.utils import set_random_seed
from autoemulate.experimental.types import InputLike, OutputLike

_default_fit_config = FitConfig(
    epochs=10,
    batch_size=16,
    shuffle=True,
    criterion=torch.nn.MSELoss,
    optimizer=torch.optim.Adam,
    device="cpu",
    verbose=False,
)


class GPyTorch(nn.Module, Emulator, InputTypeMixin):  # type: ignore
    """PyTorchBackend is a torch model and implements the base class.
    This provides default implementations to further subclasses.
    This means that models can subclass and only need to implement
    `.forward()` to have an emulator to be run in `AutoEmulate`"""

    likelihood: GaussianLikelihood

    def fit(  # type: ignore
        self, x: InputLike, y: OutputLike | None, config: FitConfig
    ):
        for epoch in range(config.epochs):
            print(epoch)
            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.1
            )  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(config.epochs):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(x)
                # Calc loss and backprop gradients
                loss = -mll(output, y)
                loss.backward()
                # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                #     i + 1, config.epochs, loss.item(),
                #     model.covar_module.base_kernel.lengthscale.item(),
                #     model.likelihood.noise.item()
                # ))
                print(loss)
                optimizer.step()

        self.is_fitted_ = True

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        return self(x)

    def cross_validate(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")

    def tune(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")


class GPExactRBF(gpytorch.models.ExactGP, GPyTorch):
    random_state: int | None = None

    def __init__(
        self, x: InputLike, y: OutputLike, likelihood: MultitaskGaussianLikelihood
    ):
        if self.random_state is not None:
            set_random_seed(self.random_state)
        print(type(y), y)
        assert isinstance(y, torch.Tensor)
        self.y_dim_ = y.ndim
        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1
        # y = y.astype(torch.float32)

        # GP's work better when the target values are normalized
        # if self.normalize_y:
        #     self._y_train_mean = torch.mean(y, axis=0)
        #     self._y_train_std = torch.std(y, axis=0)
        #     y = (y - self._y_train_mean) / self._y_train_std

        # default modules
        mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.n_outputs_])
        )

        # combined RBF + constant kernel works well in a lot of cases
        rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=self.n_features_in_,  # different lengthscale for each feature
            batch_shape=torch.Size([self.n_outputs_]),  # batched multioutput
            # seems to work better when we initialize the lengthscale
        ).initialize(lengthscale=torch.ones(self.n_features_in_) * 1.5)
        constant = gpytorch.kernels.ConstantKernel()
        combined = rbf + constant

        covar_module = gpytorch.kernels.ScaleKernel(
            combined, batch_shape=torch.Size([self.n_outputs_])
        )

        # wrapping in ScaleKernel is generally good, as it adds an outputscale parameter
        if not isinstance(covar_module, gpytorch.kernels.ScaleKernel):
            covar_module = gpytorch.kernels.ScaleKernel(
                covar_module, batch_shape=torch.Size([self.n_outputs_])
            )

        super().__init__(x, y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: InputLike) -> OutputLike:
        # assert x is torch.Tensor
        mean = self.mean_module(x)
        # assert mean is torch.Tensor
        covar = self.covar_module(x)  # type: ignore
        # return gpytorch.distributions.MultivariateNormal(mean, covar)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean, covar)
        )


if __name__ == "__main__":
    x = torch.rand(10, 2)
    y = torch.rand(10, 1)
    likelihood = MultitaskGaussianLikelihood(num_tasks=1)
    model = GPExactRBF(x, y, likelihood)
    model.fit(x, y, _default_fit_config)
    print(model(x))
