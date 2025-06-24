import random

import numpy as np
from scipy.interpolate import RBFInterpolator

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import DeviceLike, TensorLike


class RadialBasisFunctions(SklearnBackend):
    """Radial basis function Emulator.

    Wraps the RBF interpolator from scipy.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        smoothing: float = 0.0,
        kernel: str = "thin_plate_spline",
        epsilon: float = 1.0,
        degree: int = 1,
        device: DeviceLike = "cpu",
    ):
        """Initializes a RadialBasisFunctions object."""
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.model = RBFInterpolator(
            x,
            y,
            smoothing=self.smoothing,
            kernel=self.kernel,
            epsilon=self.epsilon,
            degree=self.degree,
        )

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
