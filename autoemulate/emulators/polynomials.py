import torch
from torch import nn

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators.base import PyTorchBackend
from autoemulate.feature_generation.polynomial_features import PolynomialFeatures
from autoemulate.transforms.standardize import StandardizeTransform


class PolynomialRegression(PyTorchBackend):
    """
    PolynomialRegression emulator.

    Implements a linear model including all main effects, interactions,
    and quadratic terms.
    """

    def __init__(
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
        standardize_x: bool
            Whether to standardize input features. Defaults to False.
        standardize_y: bool
            Whether to standardize target values. Defaults to False.
        degree: int
            Degree of the polynomial features to be generated. Defaults to 2.
        lr: float
            Learning rate for the optimizer. Defaults to 0.1.
        epochs: int
            Number of training epochs. Defaults to 500.
        batch_size: int
            Batch size for training. Defaults to 16.
        random_seed: int | None
            Random seed for reproducibility. Defaults to None.
        device: DeviceLike | None
            Device to run the model on. If None, uses the default device. Defaults to
            None.
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
        self.n_features = x.shape[1]
        self.n_outputs = y.shape[1] if y.ndim > 1 else 1

        # includes bias term by default
        self.poly = PolynomialFeatures(
            self.n_features, degree=self.degree, device=self.device
        )
        self.linear = nn.Linear(
            self.poly.n_output_features, self.n_outputs, bias=False
        ).to(self.device)
        self.optimizer = self.optimizer_cls(self.linear.parameters(), lr=self.lr)  # type: ignore[call-arg] since all optimizers include lr
        self.scheduler_setup(kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through for polynomial regression."""
        # Transform input using the fitted PolynomialFeatures
        x_poly = self.poly.transform(x)
        return self.linear(x_poly)

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
