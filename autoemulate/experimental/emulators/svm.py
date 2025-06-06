import numpy as np
from sklearn.svm import SVR
from sklearn.utils.validation import check_X_y

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import DeviceLike, NumpyLike, TensorLike


class SupportVectorMachine(SklearnBackend):
    """Support Vector Machines Emulator.

    Wraps Support Vector Regressor from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike | None,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        C: float = 1.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        cache_size: int = 200,
        verbose: bool = False,
        max_iter: int = 100,
        normalise_y: bool = True,
        random_seed: int | None = None,
        device: DeviceLike = "cpu",
    ):
        """Initializes a SupportVectorMachines object."""
        if random_seed is not None:
            self.set_random_seed(random_seed)
        _, _, _ = x, y, device  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.normalise_y = normalise_y
        self.model = SVR(
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
        )

    @staticmethod
    def is_multioutput() -> bool:
        return False

    def _model_specific_check(self, x: NumpyLike, y: NumpyLike):
        check_X_y(x, y, ensure_min_samples=2)

    @staticmethod
    def get_tune_config():
        return {
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "degree": [np.random.randint(2, 6)],
            "gamma": ["scale", "auto"],
            "coef0": [np.random.uniform(0.0, 1.0)],
            "tol": [np.random.uniform(1e-5, 1e-3)],
            "C": [np.random.uniform(1.0, 3.0)],
            "epsilon": [np.random.uniform(0.1, 0.3)],
            "shrinking": [True, False],
            "max_iter": [-1],
        }
