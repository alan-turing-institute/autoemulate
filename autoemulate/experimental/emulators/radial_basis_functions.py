import random

import numpy as np
from torchrbf import RBFInterpolator

from autoemulate.experimental.core.device import TorchDeviceMixin
from autoemulate.experimental.core.types import DeviceLike, OutputLike, TensorLike
from autoemulate.experimental.emulators.base import PyTorchBackend
from autoemulate.experimental.transforms.standardize import StandardizeTransform


class RadialBasisFunctions(PyTorchBackend):
    """
    Radial basis function Emulator.

    Wraps the Radial Basis Function Interpolation in PyTorch.
    """

    supports_grad = False

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,  # noqa: ARG002
        y: TensorLike,  # noqa: ARG002
        standardize_x: bool = False,
        standardize_y: bool = False,
        smoothing: float = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: float = 1.0,
        degree: int = 1,
        device: DeviceLike | None = None,
    ):
        """Initialize a RadialBasisFunctions emulator.

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
        smoothing: float, default=0.0
            Smoothing parameter for the RBF interpolator.
        kernel: str, default="thin_plate_spline"
            Kernel type for the RBF interpolator. Options are:
            "linear", "multiquadric", "thin_plate_spline", "cubic", "quintic",
            "gaussian".
        epsilon: float, default=1.0
            Epsilon parameter for the RBF interpolator.
        degree: int, default=1
            Degree of the polynomial to be added to the RBF interpolator.
        device: DeviceLike | None, default=None
            Device to run the model on. If None, uses the default device.
        """
        super().__init__()
        TorchDeviceMixin.__init__(self, device=device)

        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.device = device

    def _fit(self, x: TensorLike, y: TensorLike):
        self.model = RBFInterpolator(
            x,
            y,
            smoothing=self.smoothing,
            kernel=self.kernel,
            epsilon=self.epsilon,
            degree=self.degree,
            device=self.device,  # type: ignore PGH003
        )

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass for the radial basis function emulator."""
        return self.model(x)

    def _predict(self, x: TensorLike, with_grad: bool) -> OutputLike:
        if with_grad:
            msg = "Gradient calculation is not supported."
            raise ValueError(msg)
        self.eval()
        x = self.preprocess(x)
        return self(x)

    @staticmethod
    def is_multioutput() -> bool:
        """Radial basis functions support multi-output."""
        return True

    @staticmethod
    def get_tune_config():
        """Return a dictionary of hyperparameters to tune."""
        all_params = [
            {
                "kernel": ["linear", "multiquadric"],
                "degree": [np.random.randint(0, 3)],  # Degrees valid for these kernels
                "smoothing": [np.random.uniform(0.0, 1.0)],
            },
            {
                "kernel": ["thin_plate_spline", "cubic"],
                "degree": [np.random.randint(1, 3)],
                "smoothing": [np.random.uniform(0.0, 1.0)],
            },
            {
                "kernel": ["quintic"],
                "degree": [np.random.randint(2, 3)],
                "smoothing": [np.random.uniform(0.0, 1.0)],
            },
            {
                "kernel": ["gaussian"],
                "degree": [np.random.randint(-1, 3)],
                "smoothing": [np.random.uniform(0.0, 1.0)],
            },
        ]
        # Randomly select one of the parameter sets
        return random.choice(all_params)
