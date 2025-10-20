import torch
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, GaussianLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.base import GaussianEmulator
from autoemulate.emulators.nn.mlp import MLP
from autoemulate.transforms.standardize import StandardizeTransform
from autoemulate.transforms.utils import make_positive_definite
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


class GaussianMLP(GaussianEmulator, MLP):
    """Multi-Layer Perceptron (MLP) emulator with Gaussian outputs."""

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = True,
        standardize_y: bool = True,
        activation_cls: type[nn.Module] = nn.ReLU,
        epochs: int = 100,
        batch_size: int = 16,
        layer_dims: list[int] | None = None,
        weight_init: str = "default",
        scale: float = 1.0,
        full_covariance: bool = False,
        bias_init: str = "default",
        dropout_prob: float | None = None,
        lr: float = 5e-3,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_params: dict | None = None,
    ):
        """
        Multi-Layer Perceptron (MLP) emulator with Gaussian outputs.

        GaussianMLP extends the standard MLP to output Gaussian distributions with
        either diagonal or full covariance matrices, allowing for uncertainty
        quantification in predictions.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike
            Target values.
        standardize_x: bool
            Whether to standardize the input features. Defaults to True.
        standardize_y: bool
            Whether to standardize the target values. Defaults to True.
        batch_size: int
            Batch size for training. Defaults to 16.
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
        full_covariance: bool
            If True, the emulator predicts full covariance matrices for the outputs. If
            False, only variance is predicted. Defaults to False.
        bias_init: str
            Bias initialization method. Options: "zeros", "default":
                - "zeros" initializes biases to zero
                - "default" uses PyTorch's default uniform initialization
        dropout_prob: float | None
            Dropout probability for regularization. If None, no dropout is applied.
            Defaults to None.
        lr: float
            Learning rate for the optimizer. Defaults to 5e-3.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        device: DeviceLike | None
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to None.
        scheduler_cls: type[LRScheduler] | None
            Learning rate scheduler class. If None, no scheduler is used. Defaults to
            None.
        scheduler_params: dict | None
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
        # Total params required for last layer
        num_params = (
            y.shape[1] + (y.shape[1] * (y.shape[1] + 1)) // 2  # mean + tril covariance
            if full_covariance
            else 2 * y.shape[1]  # mean + variance (diag covariance)
        )
        layer_dims = (
            [x.shape[1], *layer_dims]
            if layer_dims
            else [x.shape[1], 32 * num_params, 16 * num_params]
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
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
        self.epochs = epochs
        self.lr = lr
        self.num_tasks = y.shape[1]
        self.batch_size = batch_size
        self.full_covariance = full_covariance
        self.optimizer = self.optimizer_cls(self.nn.parameters(), lr=lr)  # type: ignore  # noqa: PGH003
        self.scheduler_cls = scheduler_cls
        self.scheduler_params = scheduler_params or {}
        self.scheduler_setup(self.scheduler_params)
        self.to(device)

    def forward(self, x):
        """Forward pass for the Gaussian MLP."""
        y = self.nn(x)
        mean = y[..., : self.num_tasks]

        if self.full_covariance:
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
                torch.nn.functional.softplus(scale_tril[..., diag_idxs, diag_idxs])
                + 1e-6
            )
            scale_tril[..., diag_idxs, diag_idxs] = diag

            covariance_matrix = scale_tril @ scale_tril.transpose(-1, -2)

            # TODO: for large covariance matrices, numerical instability remains
            return GaussianLike(mean, make_positive_definite(covariance_matrix))

        # Diagonal covariance case
        return GaussianLike(
            mean,
            torch.diag_embed(
                torch.nn.functional.softplus(y[..., self.num_tasks :]) + 1e-6
            ),
        )

    def _predict(self, x: TensorLike, with_grad: bool) -> GaussianLike:
        """Predict method that returns GaussianLike distribution.

        The method provides the implementation from PyTorchBackend base class but is
        required to be implemented here to satisfy the type signature.
        """
        self.eval()
        with torch.set_grad_enabled(with_grad):
            return self(x)

    def loss_func(self, y_pred, y_true):
        """Negative log likelihood loss function."""
        return -y_pred.log_prob(y_true).mean()
