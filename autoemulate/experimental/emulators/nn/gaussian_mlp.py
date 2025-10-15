import torch
from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, GaussianLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.base import GaussianEmulator
from autoemulate.emulators.nn.mlp import MLP
from autoemulate.transforms.standardize import StandardizeTransform
from autoemulate.transforms.utils import make_positive_definite
from torch import nn


class GaussianMLP(GaussianEmulator, MLP):
    """Multi-Layer Perceptron (MLP) emulator with Gaussian outputs."""

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
        full_covariance: bool = False,
        bias_init: str = "default",
        dropout_prob: float | None = None,
        lr: float = 1e-1,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        **scheduler_kwargs,
    ):
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
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
        self.epochs = epochs
        self.loss_fn = loss_fn_cls()
        self.lr = lr
        self.num_tasks = y.shape[1]
        self.batch_size = batch_size
        self.full_covariance = full_covariance
        self.optimizer = self.optimizer_cls(self.nn.parameters(), lr=lr)  # type: ignore  # noqa: PGH003
        self.scheduler_setup(scheduler_kwargs)
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
