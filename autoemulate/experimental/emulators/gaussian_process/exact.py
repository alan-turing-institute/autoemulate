import logging

import gpytorch
import numpy as np
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MultitaskKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import MultitaskMean
from torch import nn

from autoemulate.experimental.data.preprocessors import Preprocessor, Standardizer
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import GaussianProcessEmulator
from autoemulate.experimental.emulators.gaussian_process import (
    CovarModuleFn,
    MeanModuleFn,
)
from autoemulate.experimental.types import (
    DeviceLike,
    GaussianProcessLike,
    TensorLike,
)

from .kernel import (
    matern_3_2_kernel,
    matern_5_2_kernel,
    matern_5_2_plus_rq,
    rbf,
    rbf_plus_constant,
    rbf_plus_linear,
    rbf_times_linear,
    rq_kernel,
)
from .mean import (
    constant_mean,
    linear_mean,
    poly_mean,
    zero_mean,
)


class GaussianProcessExact(
    GaussianProcessEmulator, gpytorch.models.ExactGP, Preprocessor
):
    """
    Gaussian Process Exact Emulator
    This class implements an exact Gaussian Process emulator using the GPyTorch library

    It supports:
    - multi-task Gaussian processes
    - custom mean and kernel specification

    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        likelihood_cls: type[MultitaskGaussianLikelihood] = MultitaskGaussianLikelihood,
        mean_module_fn: MeanModuleFn = constant_mean,
        covar_module_fn: CovarModuleFn = rbf,
        preprocessor_cls: type[Preprocessor] | None = None,
        epochs: int = 50,
        batch_size: int = 16,
        activation: type[nn.Module] = nn.ReLU,
        lr: float = 2e-1,
        device: DeviceLike | None = None,
    ):
        # Init device
        TorchDeviceMixin.__init__(self, device=device)

        x, y = self._convert_to_tensors(x, y)
        x, y = self._move_tensors_to_device(x, y)

        # Initialize the mean and covariance modules
        # TODO: consider refactoring to only pass torch tensors x and y
        mean_module = mean_module_fn(tuple(x.shape)[1], torch.Size([y.shape[1]]))
        covar_module = covar_module_fn(tuple(x.shape)[1], torch.Size([y.shape[1]]))

        # If the combined kernel is not a ScaleKernel, wrap it in one
        covar_module = (
            covar_module
            if isinstance(covar_module, ScaleKernel)
            else ScaleKernel(
                covar_module,
                batch_shape=torch.Size([y.shape[1]]),
            )
        )

        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1

        # Init preprocessor
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

        # Init likelihood
        likelihood = likelihood_cls(num_tasks=tuple(y.shape)[1])
        likelihood = likelihood.to(self.device)

        # Init must be called with preprocessed data
        x_preprocessed = self.preprocess(x)
        gpytorch.models.ExactGP.__init__(
            self,
            train_inputs=x_preprocessed,
            train_targets=y,
            likelihood=likelihood,
        )
        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.to(self.device)

    @staticmethod
    def is_multioutput():
        return True

    def preprocess(self, x: TensorLike) -> TensorLike:
        """Preprocess the input data using the preprocessor."""
        if self.preprocessor is not None:
            x = self.preprocessor.preprocess(x)
        return x

    def forward(self, x: TensorLike):
        mean = self.mean_module(x)
        assert isinstance(mean, torch.Tensor)
        covar = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(
            MultivariateNormal(mean, covar)
        )

    def log_epoch(self, epoch: int, loss: TensorLike):
        logger = logging.getLogger(__name__)
        assert self.likelihood.noise is not None
        msg = (
            f"Epoch: {epoch + 1:{int(np.log10(self.epochs) + 1)}.0f}/{self.epochs}; "
            f"MLL: {-loss:4.3f}; noise: {self.likelihood.noise.item():4.3f}"
        )
        logger.info(msg)

    def _fit(self, x: TensorLike, y: TensorLike):
        self.train()
        self.likelihood.train()
        x, y = self._move_tensors_to_device(x, y)

        # TODO: move conversion out of _fit() and instead rely on for impl check
        x, y = self._convert_to_tensors(x, y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        x = self.preprocess(x)

        # Set the training data in case changed since init
        self.set_train_data(x, y, strict=False)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = mll(output, y)
            assert isinstance(loss, torch.Tensor)
            loss = -loss
            loss.backward()
            self.log_epoch(epoch, loss)
            optimizer.step()

    def _predict(self, x: TensorLike) -> GaussianProcessLike:
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
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
            "likelihood_cls": [MultitaskGaussianLikelihood],
        }


class CorrGPModule(GaussianProcessExact):
    """
    Multioutput GP module for correlated outputs, see Bonilla et al. 2008.
    """

    # TODO: refactor the init as similar to exact GP base class
    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        likelihood_cls: type[MultitaskGaussianLikelihood] = MultitaskGaussianLikelihood,
        mean_module_fn: MeanModuleFn = constant_mean,
        covar_module_fn: CovarModuleFn = rbf,
        epochs: int = 50,
        batch_size: int = 16,
        activation: type[nn.Module] = nn.ReLU,
        lr: float = 2e-1,
        device: DeviceLike | None = None,
    ):
        # Init device
        TorchDeviceMixin.__init__(self, device=device)

        # Convert to 2D tensors if needed and move to device
        x, y = self._move_tensors_to_device(*self._convert_to_tensors(x, y))

        # Initialize the mean and covariance modules
        mean_module = mean_module_fn(tuple(x.shape)[1], None)
        covar_module = covar_module_fn(tuple(x.shape)[1], None)

        # If the combined kernel is not a ScaleKernel, wrap it in one
        covar_module = (
            covar_module
            if isinstance(covar_module, ScaleKernel)
            else ScaleKernel(
                covar_module,
            )
        )

        # Mean and covariance modules for multitask
        num_tasks = tuple(y.shape)[1]
        mean_module = MultitaskMean(mean_module, num_tasks=num_tasks)
        covar_module = MultitaskKernel(covar_module, num_tasks=num_tasks, rank=1)

        # Init likelihood
        likelihood = likelihood_cls(num_tasks=num_tasks)
        likelihood = likelihood.to(self.device)

        # Init must be called with preprocessed data
        gpytorch.models.ExactGP.__init__(
            self,
            train_inputs=x,
            train_targets=y,
            likelihood=likelihood,
        )
        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.activation = activation
        self.to(self.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        assert isinstance(mean_x, TensorLike)
        covar_x = self.covar_module(x)
        return GaussianProcessLike(mean_x, covar_x)
