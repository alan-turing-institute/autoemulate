import gpytorch
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
import torch

from autoemulate.experimental.config import FitConfig
from autoemulate.experimental.emulators.base import (
    Emulator,
    InputTypeMixin,
)
from autoemulate.utils import set_random_seed
from autoemulate.experimental.types import InputLike, OutputLike


class GPyTorch(Emulator, InputTypeMixin, gpytorch.models.ExactGP):  # type: ignore
    likelihood: GaussianLikelihood

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError("Subclassing required.")

    def fit(  # type: ignore
        self, x: InputLike, y: OutputLike | None, config: FitConfig
    ):
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for epoch in range(config.epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            output = model(x)
            loss = mll(output, y)
            assert isinstance(loss, torch.Tensor)
            loss = -loss
            loss.backward()

            print(
                "Iter %d/%d - Loss: %.3f  kernels: %s  noise: %s  likelihood: %s"
                % (
                    epoch + 1,
                    config.epochs,
                    loss,
                    "; ".join(
                        [
                            f"{name} | {sub_kernel.lengthscale}"
                            for (
                                name,
                                sub_kernel,
                            ) in model.covar_module.base_kernel.named_sub_kernels()
                        ]
                    ),
                    model.likelihood.noise,
                    model.likelihood,
                )
            )
            optimizer.step()

        self.is_fitted_ = True

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        return self(x)

    def cross_validate(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")

    def tune(self, x: InputLike) -> None:
        raise NotImplementedError("This function is not yet implemented.")


class GPExactRBF(GPyTorch):
    random_state: int | None = None

    def __init__(
        self, x: InputLike, y: OutputLike, likelihood: MultitaskGaussianLikelihood
    ):
        if self.random_state is not None:
            set_random_seed(self.random_state)
        assert isinstance(y, torch.Tensor) and isinstance(x, torch.Tensor)
        self.y_dim_ = y.ndim
        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1

        mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.n_outputs_])
        )
        rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=self.n_features_in_,
            batch_shape=torch.Size([self.n_outputs_]),
            # seems to work better when we initialize the lengthscale
        ).initialize(lengthscale=torch.ones(self.n_features_in_) * 1.5)
        constant = gpytorch.kernels.ConstantKernel()
        combined = rbf + constant

        covar_module = gpytorch.kernels.ScaleKernel(
            combined, batch_shape=torch.Size([self.n_outputs_])
        )
        # wrapping in ScaleKernel is generally good, as it adds an output scale parameter
        if not isinstance(covar_module, gpytorch.kernels.ScaleKernel):
            covar_module = gpytorch.kernels.ScaleKernel(
                covar_module, batch_shape=torch.Size([self.n_outputs_])
            )
        super().__init__(x, y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        print(type(mean))
        assert isinstance(mean, torch.Tensor)
        covar = self.covar_module(x)  # type: ignore
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean, covar)
        )


if __name__ == "__main__":
    x = torch.rand(10, 2)
    y = torch.rand(10, 1)
    likelihood = MultitaskGaussianLikelihood(num_tasks=1)
    model = GPExactRBF(x, y, likelihood)
    model.fit(x, y, FitConfig())
    print(model(x))
