import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import MultitaskKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import MultitaskMean
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from autoemulate.experimental.callbacks.early_stopping import (
    EarlyStopping,
    EarlyStoppingException,
)
from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import GaussianProcessEmulator
from autoemulate.experimental.emulators.gaussian_process import (
    CovarModuleFn,
    MeanModuleFn,
)
from autoemulate.experimental.transforms.utils import make_positive_definite
from autoemulate.experimental.types import (
    DeviceLike,
    GaussianLike,
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
from .mean import constant_mean, linear_mean, poly_mean, zero_mean


class GaussianProcessExact(GaussianProcessEmulator, gpytorch.models.ExactGP):
    """
    Gaussian Process Exact Emulator

    This class implements an exact Gaussian Process emulator using the GPyTorch library

    It supports:
    - multi-task Gaussian processes
    - custom mean and kernel specification
    """

    # TODO: refactor to work more like PyTorchBackend once any subclasses implemented
    optimizer_cls: type[optim.Optimizer] = optim.AdamW
    optimizer: optim.Optimizer
    lr: float = 1e-1
    scheduler_cls: type[LRScheduler] | None = None

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        likelihood_cls: type[MultitaskGaussianLikelihood] = MultitaskGaussianLikelihood,
        mean_module_fn: MeanModuleFn = constant_mean,
        covar_module_fn: CovarModuleFn = rbf_plus_constant,
        posterior_predictive: bool = False,
        epochs: int = 50,
        activation: type[nn.Module] = nn.ReLU,
        lr: float = 2e-1,
        early_stopping: EarlyStopping | None = None,
        device: DeviceLike | None = None,
        **kwargs,
    ):
        """
        Initialize the GaussianProcessExact emulator.

        Parameters
        ----------
        x: TensorLike
            Input features, expected to be a 2D tensor of shape (n_samples, n_features).
        y: TensorLike
            Target values, expected to be a 2D tensor of shape (n_samples, n_tasks).
        likelihood_cls: type[MultitaskGaussianLikelihood],
            default=MultitaskGaussianLikelihood
            Likelihood class to use for the model. Defaults to
            `MultitaskGaussianLikelihood`.
        mean_module_fn: MeanModuleFn, default=constant_mean
            Function to create the mean module.
        covar_module_fn: CovarModuleFn, default=rbf
            Function to create the covariance module.
        posterior_predictive: bool, default=False
            If True, the model will return the posterior predictive distribution that
            by default includes observation noise (both global and task-specific). If
            False, it will return the posterior distribution over the modelled function.
        epochs: int, default=50
            Number of training epochs.
        activation: type[nn.Module], default=nn.ReLU
            Activation function to use in the model.
        lr: float, default=2e-1
            Learning rate for the optimizer.
        early_stopping: EarlyStopping | None
            An optional EarlyStopping callback. Defaults to None.
        device: DeviceLike | None, default=None
            Device to run the model on. If None, uses the default device (usually CPU or
            GPU).
        """
        # Init device
        TorchDeviceMixin.__init__(self, device=device)

        x, y = self._convert_to_tensors(x, y)
        x, y = self._move_tensors_to_device(x, y)

        # Local variables for number of features and tasks
        n_features = x.shape[1]
        num_tasks = y.shape[1]
        num_tasks_torch = torch.Size([num_tasks])

        # Initialize the mean and covariance modules
        mean_module = mean_module_fn(n_features, num_tasks_torch)
        covar_module = covar_module_fn(n_features, num_tasks_torch)

        # If the combined kernel is not a ScaleKernel, wrap it in one
        covar_module = (
            covar_module
            if isinstance(covar_module, ScaleKernel)
            else ScaleKernel(covar_module, batch_shape=num_tasks_torch)
        )

        # Init likelihood
        likelihood = likelihood_cls(num_tasks=num_tasks)
        likelihood = likelihood.to(self.device)

        # Init must be called with preprocessed data
        gpytorch.models.ExactGP.__init__(
            self, train_inputs=x, train_targets=y, likelihood=likelihood
        )
        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.epochs = epochs
        self.lr = lr
        self.activation = activation
        self.optimizer = self.optimizer_cls(self.parameters(), lr=self.lr)  # type: ignore[call-arg] since all optimizers include lr
        self.scheduler_setup(kwargs)
        self.early_stopping = early_stopping
        self.posterior_predictive = posterior_predictive
        self.num_tasks = num_tasks
        self.to(self.device)

    @staticmethod
    def is_multioutput():
        return True

    def forward(self, x: TensorLike):
        mean = self.mean_module(x)
        assert isinstance(mean, torch.Tensor)
        covar = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(
            MultivariateNormal(mean, covar)
        )

    def _fit(self, x: TensorLike, y: TensorLike):
        self.train()
        self.likelihood.train()
        x, y = self._move_tensors_to_device(x, y)

        # TODO: move conversion out of _fit() and instead rely on for impl check
        x, y = self._convert_to_tensors(x, y)

        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        # Set the training data in case changed since init
        self.set_train_data(x, y, strict=False)

        # Initialize early stopping
        if self.early_stopping is not None:
            self.early_stopping.on_train_begin()

        # Avoid `"epoch" is possibly unbound` type error at the end
        epoch = 0
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self(x)
            loss = mll(output, y)
            assert isinstance(loss, torch.Tensor)
            loss = -loss
            loss.backward()
            self.optimizer.step()

            if self.early_stopping is not None:
                try:
                    # TODO: use validation loss instead, see #589
                    self.early_stopping.on_epoch_end(self, epoch, loss.item())
                except EarlyStoppingException:
                    # EarlyStopping prints a message if this happens
                    break

            # Update learning rate if scheduler is defined
            if self.scheduler is not None:
                self.scheduler.step()

        if self.early_stopping is not None:
            self.early_stopping.on_train_end(self, epoch)

        # Update learning rate if scheduler is defined
        if self.scheduler is not None:
            self.scheduler.step()

    def _predict(self, x: TensorLike, with_grad: bool):
        with torch.set_grad_enabled(with_grad), gpytorch.settings.fast_pred_var():
            self.eval()
            self.likelihood.eval()
            x = x.to(self.device)

            num_points = x.shape[0]
            num_tasks = self.num_tasks
            max_batch_size = 128

            # Lists to store batches of means and covs
            means_list = []
            covs_list = []

            # Loop over batches of points for predictionss
            for i in range(0, num_points, max_batch_size):
                x_batch = x[i : i + max_batch_size]

                # Get predictive output with full covariance between points and tasks
                output = self(x)
                if self.posterior_predictive:
                    output = self.likelihood(output)
                assert isinstance(output, GaussianProcessLike)

                mean = output.mean
                cov = output.covariance_matrix
                assert isinstance(mean, TensorLike)
                assert isinstance(cov, TensorLike)

                num_batch = x_batch.shape[0]

                # Reshape mean
                mean = mean.reshape(num_batch, num_tasks)  # (num_batch, num_tasks)

                # Task covariance blocks for each point
                cov_blocks = cov.reshape(num_batch, num_tasks, num_batch, num_tasks)

                # Take diagonal along batch to only keep task covariance
                task_covs = cov_blocks[
                    torch.arange(num_batch), :, torch.arange(num_batch), :
                ]  # (num_batch, num_tasks, num_tasks)

                means_list.append(mean)
                covs_list.append(task_covs)

            # Concatenate batches
            means = torch.cat(means_list, dim=0)  # (num_points, num_tasks)
            covs = torch.cat(covs_list, dim=0)  # (num_points, num_tasks, num_tasks)

            # TODO: consider if clamp_eigval is  correct or should be applied within
            # transforms
            return GaussianLike(means, make_positive_definite(covs, clamp_eigvals=True))

    @staticmethod
    def get_tune_config():
        scheduler_config = GaussianProcessExact.scheduler_config()
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
            "epochs": [50, 100, 200],
            "lr": [5e-1, 1e-1, 5e-2, 1e-2],
            "likelihood_cls": [MultitaskGaussianLikelihood],
            "scheduler_cls": scheduler_config["scheduler_cls"],
            "scheduler_kwargs": scheduler_config["scheduler_kwargs"],
        }


class GaussianProcessExactCorrelated(GaussianProcessExact):
    """
    Multioutput exact GP implementation with correlated task covariance.

    This class extends the `GaussianProcessExact` to support correlated task covariance
    by using a `MultitaskKernel` with a rank-1 covariance factor and a `MultitaskMean`
    for the mean function.

    It is designed to handle multi-task Gaussian processes where the tasks are
    correlated, allowing for more flexible modeling of multi-output data.
    """

    # TODO: refactor the init as similar to exact GP base class
    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        likelihood_cls: type[MultitaskGaussianLikelihood] = MultitaskGaussianLikelihood,
        mean_module_fn: MeanModuleFn = constant_mean,
        covar_module_fn: CovarModuleFn = rbf_plus_constant,
        posterior_predictive: bool = False,
        epochs: int = 50,
        activation: type[nn.Module] = nn.ReLU,
        lr: float = 2e-1,
        early_stopping: EarlyStopping | None = None,
        seed: int | None = None,
        device: DeviceLike | None = None,
        **kwargs,
    ):
        """
        Initialize the GaussianProcessExactCorrelated emulator.

        Parameters
        ----------
        x: TensorLike
            Input features, expected to be a 2D tensor of shape (n_samples, n_features).
        y: TensorLike
            Target values, expected to be a 2D tensor of shape (n_samples, n_tasks).
        likelihood_cls: type[MultitaskGaussianLikelihood],
            default=MultitaskGaussianLikelihood
            Likelihood class to use for the model. Defaults to
            `MultitaskGaussianLikelihood`.
        mean_module_fn: MeanModuleFn, default=constant_mean
            Function to create the mean module.
        covar_module_fn: CovarModuleFn, default=rbf
            Function to create the covariance module.
        posterior_predictive: bool, default=False
            If True, the model will return the posterior predictive distribution that
            by default includes observation noise (both global and task-specific). If
            False, it will return the posterior distribution over the modelled function.
        epochs: int, default=50
            Number of training epochs.
        activation: type[nn.Module], default=nn.ReLU
            Activation function to use in the model.
        lr: float, default=2e-1
            Learning rate for the optimizer.
        early_stopping: EarlyStopping | None
            An optional EarlyStopping callback. Defaults to None.
        seed: int | None, default=None
            Random seed for reproducibility. If None, no seed is set.
        device: DeviceLike | None, default=None
            Device to run the model on. If None, uses the default device (usually CPU or
            GPU).
        """

        # Init device
        TorchDeviceMixin.__init__(self, device=device)

        if seed is not None:
            set_random_seed(seed)

        # Convert to 2D tensors if needed and move to device
        x, y = self._move_tensors_to_device(*self._convert_to_tensors(x, y))

        # Initialize the mean and covariance modules
        n_features = tuple(x.shape)[1]
        mean_module = mean_module_fn(n_features, None)
        covar_module = covar_module_fn(n_features, None)

        # Mean and covariance modules for multitask
        num_tasks = tuple(y.shape)[1]
        mean_module = MultitaskMean(mean_module, num_tasks=num_tasks)
        covar_module = MultitaskKernel(covar_module, num_tasks=num_tasks, rank=1)

        # TODO: identify if initialization of task covariance factor is needed for
        # deterministic behavior
        # with torch.no_grad():
        #     mean_module.task_covar_factor.fill_(1.0)  # type: ignore  # noqa: PGH003
        #     covar_module.task_covar_factor.fill_(1.0)  # type: ignore  # noqa: PGH003

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
        self.activation = activation
        self.optimizer = self.optimizer_cls(self.parameters(), lr=self.lr)  # type: ignore[call-arg] since all optimizers include lr
        self.scheduler_setup(kwargs)
        self.early_stopping = early_stopping
        self.posterior_predictive = posterior_predictive
        self.num_tasks = num_tasks
        self.to(self.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        assert isinstance(mean_x, TensorLike)
        covar_x = self.covar_module(x)
        return GaussianProcessLike(mean_x, covar_x)
