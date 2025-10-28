import random

import numpy as np
from torchrbf import RBFInterpolator

from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.types import DeviceLike, TensorLike
from autoemulate.emulators.base import PyTorchBackend
from autoemulate.transforms.standardize import StandardizeTransform


class RadialBasisFunctions(PyTorchBackend):
    """
    Radial basis function Emulator.

    Wraps the Radial Basis Function Interpolation in PyTorch.
    """

    supports_grad = True

    def __init__(
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
        standardize_x: bool
            Whether to standardize input features. Defaults to False.
        standardize_y: bool
            Whether to standardize target values. Defaults to False.
        smoothing: float
            Smoothing parameter for the RBF interpolator. Defaults to 0.0.
        kernel: str
            Kernel type for the RBF interpolator.
            Kernel type for the RBF interpolator. Options are:
            "linear", "multiquadric", "thin_plate_spline", "cubic", "quintic",
            "gaussian".
            Defaults to "thin_plate_spline".
        epsilon: float
            Epsilon parameter for the RBF interpolator. Defaults to 1.0.
        degree: int
            Degree of the polynomial to be added to the RBF interpolator. Defaults to 1.
        device: DeviceLike | None
            Device to run the model on. If None, uses the default device. Defaults to
            None.
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

    @staticmethod
    def is_multioutput() -> bool:
        """Radial basis functions support multi-output."""
        return True

    @staticmethod
    def get_tune_params():
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
