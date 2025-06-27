import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from torch import nn, optim

from autoemulate.experimental.data.utils import set_random_seed
from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import PyTorchBackend, SklearnBackend
from autoemulate.experimental.types import DeviceLike, TensorLike


class PolynomialRegression(PyTorchBackend):
    """PolynomialRegression emulator.

    Implements a linear model including all main effects, interactions,
    and quadratic terms.
    """

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,
        y: TensorLike,
        degree: int = 2,
        lr: float = 1e-2,
        epochs: int = 50,
        batch_size: int = 16,
        optimizer_cls: type[optim.Optimizer] = optim.Adam,
        random_seed: int | None = None,
        device: DeviceLike | None = None,
    ):
        super().__init__()
        TorchDeviceMixin.__init__(self, device=device)
        if random_seed is not None:
            set_random_seed(seed=random_seed)
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
        self.optimizer = optimizer_cls(self.linear.parameters(), lr=self.lr)  # type: ignore[call-arg] since all optimizers include lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform input using the fitted PolynomialFeatures
        x_np, _ = self._convert_to_numpy(x)
        x_poly = self.poly.transform(x_np)
        x_poly_tensor = torch.tensor(x_poly, dtype=torch.float32, device=self.device)
        return self.linear(x_poly_tensor)

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config():
        return {
            "lr": [1e-3, 1e-2, 1e-1],
            "epochs": [50, 100, 200],
            "batch_size": [8, 16, 32],
        }


class PolynomialRegressionOld(SklearnBackend):
    """Second order polynomial emulator.

    Creates a second order polynomial emulator. This is a linear model
    including all main effects, interactions and quadratic terms.
    """

    def __init__(
        self,
        x: TensorLike,
        y: TensorLike,
        degree: int = 2,
        device: DeviceLike = "cpu",
    ):
        """Initializes a SecondOrderPolynomial object."""
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
        self.degree = degree
        self.model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=self.degree)),
                ("model", LinearRegression()),
            ]
        )

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config():
        return {}
