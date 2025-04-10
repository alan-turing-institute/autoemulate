import logging

import gpytorch
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.kernels import (
    ScaleKernel,
)
from torch import nn

from autoemulate.experimental.data.preprocessors import Preprocessor, Standardizer
from autoemulate.experimental.emulators.base import (
    Emulator,
    InputTypeMixin,
)
from autoemulate.emulators.gaussian_process import (
    zero_mean,
    constant_mean,
    linear_mean,
    poly_mean,
)
from autoemulate.emulators.gaussian_process import (
    rbf,
    matern_5_2_kernel,
    matern_3_2_kernel,
    rq_kernel,
    rbf_plus_constant,
    rbf_plus_linear,
    matern_5_2_plus_rq,
    rbf_times_linear,
)
from autoemulate.experimental.emulators.gaussian_process import (
    CovarModuleFn,
    MeanModuleFn,
)
from autoemulate.experimental.types import InputLike, OutputLike
from autoemulate.utils import set_random_seed


class GaussianProcessExact(
    Emulator, InputTypeMixin, gpytorch.models.ExactGP, Preprocessor
):
    """
    Gaussian Process Exact Emulator
    This class implements an exact Gaussian Process emulator using the GPyTorch library

    It supports:
    - multi-task Gaussian processes
    - custom mean and kernel specification

    """

    def __init__(
        self,
        x: InputLike,
        y: InputLike,
        likelihood: MultitaskGaussianLikelihood,
        mean_module_fn: MeanModuleFn,
        covar_module_fn: CovarModuleFn,
        preprocessor_cls: type[Preprocessor] | None = None,
        random_state: int | None = None,
        epochs: int = 10,
        batch_size: int = 16,
        activation: type[nn.Module] = nn.ReLU,
        lr: float = 0.01,
    ):
        if random_state is not None:
            set_random_seed(random_state)

        x, y = self._convert_to_tensors(self._convert_to_dataset(x, y))
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)

        # Initialize the mean and covariance modules
        # TODO: consider refactoring to only pass torch tensors x and y
        mean_module = mean_module_fn(tuple(x.shape)[1], torch.Size([y.shape[1]]))
        combined_kernel = covar_module_fn(tuple(x.shape)[1], torch.Size([y.shape[1]]))

        # If the combined kernel is not a ScaleKernel, wrap it in one
        covar_module = (
            combined_kernel
            if isinstance(combined_kernel, ScaleKernel)
            else ScaleKernel(
                combined_kernel,
                batch_shape=torch.Size([y.shape[1]]),
            )
        )

        assert isinstance(y, torch.Tensor) and isinstance(x, torch.Tensor)
        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1

        # Initialize preprocessor
        if preprocessor_cls is not None:
            if issubclass(preprocessor_cls, Standardizer):
                self.preprocessor = preprocessor_cls(
                    x.mean(0, keepdim=True), x.std(0, keepdim=True)
                )
            else:
                raise NotImplementedError(
                    f"Preprocessor ({preprocessor_cls}) not currently implemented."
                )
        else:
            self.preprocessor = None

        # Init must be called with preprocessed data
        super().__init__(self.preprocess(x), y, likelihood)
        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation

    def preprocess(self, x: InputLike) -> InputLike:
        """Preprocess the input data using the preprocessor."""
        if self.preprocessor is not None:
            x = self.preprocessor.preprocess(x)
        return x

    def forward(self, x: InputLike):
        assert isinstance(x, torch.Tensor)
        mean = self.mean_module(x)

        assert isinstance(mean, torch.Tensor)
        covar = self.covar_module(x)

        return MultitaskMultivariateNormal.from_batch_mvn(
            MultivariateNormal(mean, covar)
        )

    def log_epoch(self, epoch: int, loss: torch.Tensor):
        logger = logging.getLogger(__name__)
        assert self.likelihood.noise is not None
        logger.info(
            f"Epoch: {epoch + 1:{int(np.log10(self.epochs) + 1)}.0f}/{self.epochs}; "
            f"MLL: {-loss:4.3f}; noise: {self.likelihood.noise.item():4.3f}"
        )

    def fit(self, x: InputLike, y: InputLike | None):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        x = self.preprocess(x)
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = mll(output, y)
            assert isinstance(loss, torch.Tensor)
            loss = -loss
            loss.backward()
            self.log_epoch(epoch, loss)
            optimizer.step()

    def predict(self, x: InputLike) -> OutputLike:
        self.eval()
        x = self.preprocess(x)
        return self(x)

    @staticmethod
    def get_tune_config():
        return {
            "mean_module_fn": [
                constant_mean,
                zero_mean,
                linear_mean,
                poly_mean,
            ],
            "covar_module_fn": [
                rbf,
                matern_5_2_kernel,
                matern_3_2_kernel,
                rq_kernel,
                rbf_plus_constant,
                rbf_plus_linear,
                matern_5_2_plus_rq,
                rbf_times_linear,
            ],
            "epochs": [100, 200, 500, 1000],
            "batch_size": [16, 32],
            "activation": [
                nn.ReLU,
                nn.GELU,
            ],
            "lr": list(np.logspace(-3, -1)),
            "preprocessor_cls": [None, Standardizer],
        }
