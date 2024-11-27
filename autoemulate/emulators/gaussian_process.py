import gpytorch
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from skorch.callbacks import LRScheduler
from skorch.probabilistic import ExactGPRegressor

from autoemulate.emulators.gaussian_process_utils import EarlyStoppingCustom
from autoemulate.emulators.gaussian_process_utils import PolyMean
from autoemulate.emulators.neural_networks.gp_module import GPModule
from autoemulate.utils import set_random_seed


class GaussianProcess(RegressorMixin, BaseEstimator):
    """Exact Gaussian Process emulator build with GPyTorch.

    Batched Multi-Output GP, treating outputs independently.

    Parameters
    ----------
    mean_module : GP mean, defaults to gpytorch.means.ConstantMean() when None
    covar_module : GP covariance, defaults to gpytorch.kernels.RBFKernel() when None
    lr : learning rate, default=1e-1
    optimizer : optimizer, default=torch.optim.AdamW
    max_epochs : maximum number of epochs, default=30
    normalize_y : whether to normalize the target values, default=True
    device : device to use, defaults to "cuda" if available, otherwise "cpu"
    random_state : random seed, default=None
    """

    def __init__(
        self,
        # architecture
        mean_module=None,
        covar_module=None,
        # training
        lr=2e-1,
        optimizer=torch.optim.AdamW,
        max_epochs=50,
        normalize_y=True,
        # misc
        device="cpu",
        random_state=None,
    ):
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.lr = lr
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.normalize_y = normalize_y
        self.device = device
        self.random_state = random_state

    def _get_module(self, module, default_module, n_features, n_outputs):
        """
        Get mean and kernel modules.

        We can't default the modules in the constructor because 'fit' modifies them which
        fails scikit-learn estimator tests. Therefore, we deepcopy if module is given or return the default class
        if not.
        """
        if module is None:
            return default_module
        if callable(module):
            # torch.Size is needed to specify the batch shape
            return module(n_features, torch.Size([n_outputs]))
        else:
            ValueError("module must be callable or None")

    def fit(self, X, y):
        """Fit the emulator to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples, )
            The output data.
        Returns
        -------
        self : object
            Returns self.
        """
        if self.random_state is not None:
            set_random_seed(self.random_state)

        X, y = check_X_y(
            X,
            y,
            y_numeric=True,
            multi_output=True,
            dtype=np.float32,
            copy=True,
            ensure_2d=True,
        )
        self.y_dim_ = y.ndim
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1
        y = y.astype(np.float32)

        # GP's work better when the target values are normalized
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)
            y = (y - self._y_train_mean) / self._y_train_std

        # default modules
        default_mean_module = gpytorch.means.ConstantMean(
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

        default_covar_module = gpytorch.kernels.ScaleKernel(
            combined, batch_shape=torch.Size([self.n_outputs_])
        )

        mean_module = self._get_module(
            self.mean_module, default_mean_module, self.n_features_in_, self.n_outputs_
        )
        covar_module = self._get_module(
            self.covar_module,
            default_covar_module,
            self.n_features_in_,
            self.n_outputs_,
        )

        # wrapping in ScaleKernel is generally good, as it adds an outputscale parameter
        if not isinstance(covar_module, gpytorch.kernels.ScaleKernel):
            covar_module = gpytorch.kernels.ScaleKernel(
                covar_module, batch_shape=torch.Size([self.n_outputs_])
            )

        # model
        self.model_ = ExactGPRegressor(
            GPModule,
            module__mean=mean_module,
            module__covar=covar_module,
            likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.n_outputs_
            ),
            max_epochs=self.max_epochs,
            lr=self.lr,
            optimizer=self.optimizer,
            callbacks=[
                (
                    "lr_scheduler",
                    LRScheduler(policy="ReduceLROnPlateau", patience=5, factor=0.5),
                ),
                (
                    "early_stopping",
                    EarlyStoppingCustom(
                        monitor="train_loss",
                        patience=10,
                        threshold=1e-3,
                        load_best=True,
                    ),
                ),
            ],
            verbose=0,
            device=self.device,
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        """Predict the output of the emulator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        return_std : bool, default=False
            Whether to return the standard deviation.

        Returns
        -------
        y : array-like of shape (n_samples, )
            The predicted output.
        """

        # checks
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)

        # predict
        mean, std = self.model_.predict(X, return_std=True)

        # sklearn: regression models should return float64
        mean = mean.astype(np.float64)
        std = std.astype(np.float64)

        # output shape should be same as input shape
        # when input dim is 1D, make sure output is 1D
        if mean.ndim == 2 and self.y_dim_ == 1:
            mean = mean.squeeze()
            std = std.squeeze()

        # undo normalization
        if self.normalize_y:
            mean = mean * self._y_train_std + self._y_train_mean
            std = std * self._y_train_std

        if return_std:
            return mean, std
        return mean

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters for the emulator."""

        if search_type == "random":
            param_space = {
                "covar_module": [
                    rbf,
                    matern_5_2_kernel,
                    matern_3_2_kernel,
                    rq_kernel,
                    rbf_plus_constant,
                    rbf_plus_linear,
                    rbf_times_linear,
                    matern_5_2_plus_rq,
                ],
                "mean_module": [
                    constant_mean,
                    zero_mean,
                    linear_mean,
                    poly_mean,
                ],
                "optimizer": [torch.optim.AdamW, torch.optim.Adam],
                "lr": [5e-1, 1e-1, 5e-2, 1e-2],
                "max_epochs": [50, 100, 200],
            }
        return param_space

    @property
    def model_name(self):
        return self.__class__.__name__

    def _more_tags(self):
        # TODO: is it really non-deterministic?
        return {"multioutput": True, "non_deterministic": True}


# kernel functions for parameter search have to be outside the class so that pickle can find them
def rbf(n_features, n_outputs):
    return gpytorch.kernels.RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ).initialize(lengthscale=torch.ones(n_features) * 1.5)


def matern_5_2_kernel(n_features, n_outputs):
    return gpytorch.kernels.MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def matern_3_2_kernel(n_features, n_outputs):
    return gpytorch.kernels.MaternKernel(
        nu=1.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rq_kernel(n_features, n_outputs):
    return gpytorch.kernels.RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rbf_plus_constant(n_features, n_outputs):
    return (
        gpytorch.kernels.RBFKernel(
            ard_num_dims=n_features,
            batch_shape=n_outputs,
        ).initialize(lengthscale=torch.ones(n_features) * 1.5)
        + gpytorch.kernels.ConstantKernel()
    )


# combinations
def rbf_plus_linear(n_features, n_outputs):
    return gpytorch.kernels.RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) + gpytorch.kernels.LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def matern_5_2_plus_rq(n_features, n_outputs):
    return gpytorch.kernels.MaternKernel(
        nu=2.5,
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) + gpytorch.kernels.RQKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


def rbf_times_linear(n_features, n_outputs):
    return gpytorch.kernels.RBFKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    ) * gpytorch.kernels.LinearKernel(
        ard_num_dims=n_features,
        batch_shape=n_outputs,
    )


# means
def constant_mean(n_features, n_outputs):
    return gpytorch.means.ConstantMean(batch_shape=n_outputs)


def zero_mean(n_features, n_outputs):
    return gpytorch.means.ZeroMean(batch_shape=n_outputs)


def linear_mean(n_features, n_outputs):
    return gpytorch.means.LinearMean(input_size=n_features, batch_shape=n_outputs)


def poly_mean(n_features, n_outputs):
    return PolyMean(degree=2, input_size=n_features, batch_shape=n_outputs)
