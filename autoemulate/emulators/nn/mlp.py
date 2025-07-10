import torch
from torch import nn, optim

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, GaussianLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.transforms.standardize import StandardizeTransform
from autoemulate.transforms.utils import make_positive_definite

from ..base import DropoutTorchBackend, GaussianEmulator


class MLP(DropoutTorchBackend):
    """
    Multi-Layer Perceptron (MLP) emulator.

    MLP provides a simple deterministic emulator with optional model stochasticity
    provided by different weight initialization and dropout.
    """

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = True,
        standardize_y: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
        loss_fn_cls: type[nn.Module] = nn.MSELoss,
        epochs: int = 100,
        batch_size: int = 16,
        layer_dims: list[int] | None = None,
        weight_init: str = "default",
        scale: float = 1.0,
        bias_init: str = "default",
        dropout_prob: float | None = None,
        lr: float = 1e-2,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        **scheduler_kwargs,
    ):
        """
        Multi-Layer Perceptron (MLP) emulator.

        MLP provides a simple deterministic emulator with optional model stochasticity
        provided by different weight initialization and dropout.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike
            Target values.
        activation_cls: type[nn.Module]
            Activation function to use in the hidden layers. Defaults to `nn.ReLU`.
        layer_dims: list[int] | None
            Dimensions of the hidden layers. If None, defaults to [32, 16].
            Defaults to None.
        weight_init: str
            Weight initialization method. Options are "default", "normal", "uniform",
            "zeros", "ones", "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal". Defaults to "default".
        scale: float
            Scale parameter for weight initialization methods. Used as:
            - gain for Xavier methods
            - std for normal distribution
            - bound for uniform distribution (range: [-scale, scale])
            - ignored for Kaiming methods (uses optimal scaling)
            Defaults to 1.0.
        bias_init: str
            Bias initialization method. Options: "zeros", "default":
                - "zeros" initializes biases to zero
                - "default" uses PyTorch's default uniform initialization
        dropout_prob: float | None
            Dropout probability for regularization. If None, no dropout is applied.
            Defaults to None.
        lr: float
            Learning rate for the optimizer. Defaults to 1e-2.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to None.
        **scheduler_kwargs: dict
            Additional keyword arguments related to the scheduler.

        Raises
        ------
        ValueError
            If the input dimensions of `x` and `y` are not matrices.
        """
        TorchDeviceMixin.__init__(self, device=device)
        nn.Module.__init__(self)

        if random_seed is not None:
            set_random_seed(seed=random_seed)

        # Ensure x and y are tensors with correct dimensions
        x, y = self._convert_to_tensors(x, y)

        # Construct the MLP layers
        layer_dims = [x.shape[1], *layer_dims] if layer_dims else [x.shape[1], 32, 16]
        layers = []
        for idx, dim in enumerate(layer_dims[1:]):
            layers.append(nn.Linear(layer_dims[idx], dim, device=self.device))
            layers.append(activation_cls())
            if dropout_prob is not None:
                layers.append(nn.Dropout(p=dropout_prob))

        # Add final layer without activation
        num_tasks = y.shape[1]
        layers.append(nn.Linear(layer_dims[-1], num_tasks, device=self.device))
        self.nn = nn.Sequential(*layers)

        # Finalize initialization
        self._initialize_weights(weight_init, scale, bias_init)
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
        self.epochs = epochs
        self.loss_fn = loss_fn_cls()
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = self.optimizer_cls(self.nn.parameters(), lr=self.lr)  # type: ignore[call-arg] since all optimizers include lr
        self.scheduler_setup(scheduler_kwargs)
        self.to(self.device)

    def forward(self, x):
        """Forward pass for the MLP."""
        return self.nn(x)

    @staticmethod
    def is_multioutput() -> bool:
        """MLP supports multi-output."""
        return True

    @staticmethod
    def get_tune_params():
        """Return a dictionary of hyperparameters to tune."""
        scheduler_params = MLP.scheduler_params()
        return {
            "epochs": [100, 200],
            "layer_dims": [[8, 4], [16, 8], [32, 16], [64, 32, 16]],
            "lr": [5e-1, 2e-1, 1e-1, 1e-2, 1e-3],
            "batch_size": [16, 32],
            "weight_init": ["default", "normal"],
            "scale": [0.1, 1.0],
            "bias_init": ["default", "zeros"],
            "dropout_prob": [0.3, None],
            "scheduler_cls": scheduler_params["scheduler_cls"],
            "scheduler_kwargs": scheduler_params["scheduler_kwargs"],
        }


class GaussianMLP(DropoutTorchBackend, GaussianEmulator):
    """Multi-Layer Perceptron (MLP) emulator with Gaussian outputs."""

    def __init__(
        self,
        x,
        y,
        activation_cls=nn.ReLU,
        loss_fn_cls=nn.MSELoss,
        optimizer_cls=optim.Adam,
        scheduler_cls=None,
        scheduler_kwargs=None,
        epochs=100,
        layer_dims=None,
        weight_init="default",
        scale=1,
        bias_init="default",
        dropout_prob=None,
        lr=1e-2,
        random_seed=None,
        device=None,
        **kwargs,
    ):
        TorchDeviceMixin.__init__(self, device=device)
        nn.Module.__init__(self)

        if random_seed is not None:
            set_random_seed(seed=random_seed)

        # Ensure x and y are tensors with correct dimensions
        x, y = self._convert_to_tensors(x, y)

        # Construct the MLP layers
        # Total params required for last layer: mean + tril covariance
        num_params = y.shape[1] + (y.shape[1] * (y.shape[1] + 1)) // 2
        layer_dims = (
            [x.shape[1], *layer_dims]
            if layer_dims
            else [x.shape[1], 4 * num_params, 2 * num_params]
        )
        layers = []
        for idx, dim in enumerate(layer_dims[1:]):
            layers.append(nn.Linear(layer_dims[idx], dim, device=self.device))
            layers.append(activation_cls())
            if dropout_prob is not None:
                layers.append(nn.Dropout(p=dropout_prob))

        # Add final layer without activation
        layers.append(nn.Linear(layer_dims[-1], num_params, device=self.device))
        self.nn = nn.Sequential(*layers)

        # Finalize initialization
        self._initialize_weights(weight_init, scale, bias_init)
        self.epochs = epochs
        self.loss_fn = loss_fn_cls()
        self.num_tasks = y.shape[1]
        self.optimizer = optimizer_cls(self.nn.parameters(), lr=lr)
        self.scheduler_cls = scheduler_cls
        self.scheduler_setup(scheduler_kwargs)
        self.to(device)

    def _predict(self, x, with_grad=False):
        """Predict using the MLP model."""
        with torch.set_grad_enabled(with_grad):
            self.nn.eval()
            return self(x)

    def forward(self, x):
        """Forward pass for the Gaussian MLP."""
        y = self.nn(x)
        mean = y[..., : self.num_tasks]

        # Use Cholesky decomposition to guarantee PSD covariance matrix
        num_chol_params = (self.num_tasks * (self.num_tasks + 1)) // 2
        chol_params = y[..., self.num_tasks : self.num_tasks + num_chol_params]

        # Assign params to matrix
        scale_tril = torch.zeros(
            *y.shape[:-1], self.num_tasks, self.num_tasks, device=y.device
        )
        tril_indices = torch.tril_indices(
            self.num_tasks, self.num_tasks, device=y.device
        )
        scale_tril[..., tril_indices[0], tril_indices[1]] = chol_params

        # Ensure positive variance
        diag_idxs = torch.arange(self.num_tasks)
        diag = (
            torch.nn.functional.softplus(scale_tril[..., diag_idxs, diag_idxs]) + 1e-6
        )
        scale_tril[..., diag_idxs, diag_idxs] = diag

        covariance_matrix = scale_tril @ scale_tril.transpose(-1, -2)

        # TODO: for large covariance martrices, numerical instability remains
        return GaussianLike(mean, make_positive_definite(covariance_matrix))

    def loss_func(self, y_pred, y_true):
        """Negative log likelihood loss function."""
        return -y_pred.log_prob(y_true).mean()

    @staticmethod
    def is_multioutput() -> bool:
        """GaussianMLP supports multi-output."""
        return True

    @staticmethod
    def get_tune_config():
        """Return a dictionary of hyperparameters to tune."""
        return {
            "epochs": [50, 100, 200],
            "layer_dims": [[64, 64], [128, 128]],
            "lr": [1e-1, 1e-2, 1e-3],
            "batch_size": [16, 32],
            "weight_init": ["default", "normal"],
            "scale": [0.1, 1.0],
            "bias_init": ["default", "zeros"],
            "dropout_prob": [0.3, 0.5, None],
        }
