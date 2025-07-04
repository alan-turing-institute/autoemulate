import random

import numpy as np
from torchrbf import RBFInterpolator

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import PyTorchBackend
from autoemulate.experimental.types import DeviceLike, TensorLike


class RadialBasisFunctions(PyTorchBackend):
    """Radial basis function Emulator.

    Wraps the Radial Basis Function Interpolation in PyTorch.
    """

    def __init__(  # noqa: PLR0913
        self,
        x: TensorLike,  # noqa: ARG002
        y: TensorLike,  # noqa: ARG002
        smoothing: float = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: float = 1.0,
        degree: int = 1,
        device: DeviceLike | None = None,
    ):
        """Initializes a RadialBasisFunctions object."""
        super().__init__()
        TorchDeviceMixin.__init__(self, device=device)
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
        return self.model(x)

    @staticmethod
    def is_multioutput() -> bool:
        return True

    @staticmethod
    def get_tune_config():
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
