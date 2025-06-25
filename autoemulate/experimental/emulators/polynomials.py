from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import DeviceLike, TensorLike


class SecondOrderPolynomial(SklearnBackend):
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
        """Initializes a RandomForest object."""
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
