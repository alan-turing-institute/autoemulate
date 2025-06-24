import random

import numpy as np
from scipy.interpolate import RBFInterpolator

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import DeviceLike, OutputLike, TensorLike


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

    def _fit(self, x: TensorLike, y: TensorLike):
        if self.normalize_y:
            y, y_mean, y_std = self._normalize(y)
            self.y_mean = y_mean
            self.y_std = y_std
        x_np, y_np = self._convert_to_numpy(x, y)
        assert isinstance(x_np, np.ndarray)
        assert isinstance(y_np, np.ndarray)
        self.n_features_in_ = x_np.shape[1]
        self._model_specific_check(x_np, y_np)
        self.model = RBFInterpolator(
            x,
            y,
            smoothing=self.smoothing,
            kernel=self.kernel,
            epsilon=self.epsilon,
            degree=self.degree,
        )

    def _predict(self, x: TensorLike) -> OutputLike:
        x_np, _ = self._convert_to_numpy(x, None)
        y_pred = self.model(x_np)  # type: ignore PGH003
        _, y_pred = self._move_tensors_to_device(*self._convert_to_tensors(x, y_pred))
        if self.normalize_y:
            y_pred = self._denormalize(y_pred, self.y_mean, self.y_std)
        return y_pred

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
