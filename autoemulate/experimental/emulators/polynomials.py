import torch
from sklearn.preprocessing import PolynomialFeatures
from torch import nn

from autoemulate.experimental.core.device import TorchDeviceMixin
from autoemulate.experimental.core.types import DeviceLike, TensorLike
from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.emulators.base import PyTorchBackend
from autoemulate.experimental.transforms.standardize import StandardizeTransform


class PolynomialRegression(PyTorchBackend):
    """
    PolynomialRegression emulator.

    Implements a linear model including all main effects, interactions,
    and quadratic terms.
    """

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = False,
        standardize_y: bool = False,
        degree: int = 2,
        lr: float = 0.1,
        epochs: int = 500,
        batch_size: int = 16,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
        **kwargs,
    ):
        """Initialize a PolynomialRegression emulator.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike
            Target values.
        standardize_x: bool, default=False
            Whether to standardize input features.
        standardize_y: bool, default=False
            Whether to standardize target values.
        degree: int, default=2
            Degree of the polynomial features to be generated.
        lr: float, default=0.1
            Learning rate for the optimizer.
        epochs: int, default=500
            Number of training epochs.
        batch_size: int, default=16
            Batch size for training.
        random_seed: int | None, default=None
            Random seed for reproducibility.
        device: DeviceLike | None, default=None
            Device to run the model on. If None, uses the default device.
        **kwargs: dict
            Additional keyword arguments.
        """
        super().__init__()
        TorchDeviceMixin.__init__(self, device=device)
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
        self.degree = degree
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Fit the polynomial feature transformer on the input data
        self.poly = PolynomialFeatures(degree=self.degree)
        x_np, _ = self._convert_to_numpy(x)
        x_poly = self.poly.fit_transform(x_np)
        self.n_outputs_ = y.shape[1] if y.ndim > 1 else 1
        self.linear = nn.Linear(x_poly.shape[1], self.n_outputs_, bias=False).to(
            self.device
        )
        self.optimizer = self.optimizer_cls(self.linear.parameters(), lr=self.lr)  # type: ignore[call-arg] since all optimizers include lr
        self.scheduler_setup(kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through for polynomial regression."""
        # Transform input using the fitted PolynomialFeatures
        x_np, _ = self._convert_to_numpy(x)
        x_poly = self.poly.transform(x_np)
        x_poly_tensor = torch.tensor(x_poly, dtype=torch.float32, device=self.device)
        return self.linear(x_poly_tensor)

    @staticmethod
    def is_multioutput() -> bool:
        """Polynomial regression supports multi-output."""
        return True

    @staticmethod
    def get_tune_params():
        """Return a dictionary of hyperparameters to tune."""
        scheduler_params = PolynomialRegression.scheduler_params()
        return {
            "lr": [1e-3, 1e-2, 1e-1, 2e-1],
            "epochs": [50, 100, 200, 500, 1000],
            "batch_size": [8, 16, 32],
            "scheduler_cls": scheduler_params["scheduler_cls"],
            "scheduler_kwargs": scheduler_params["scheduler_kwargs"],
        }
