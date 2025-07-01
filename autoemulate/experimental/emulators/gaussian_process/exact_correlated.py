import gpytorch
from gpytorch.kernels import MultitaskKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import MultitaskMean
from torch import nn

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.gaussian_process import (
    CovarModuleFn,
    MeanModuleFn,
)
from autoemulate.experimental.types import (
    DeviceLike,
    GaussianProcessLike,
    TensorLike,
)

from .exact import GaussianProcessExact
from .kernel import rbf
from .mean import (
    constant_mean,
)


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
