from typing import Literal

import numpy as np
from sklearn.svm import SVR
from sklearn.utils.validation import check_X_y

from autoemulate.experimental.core.device import TorchDeviceMixin
from autoemulate.experimental.core.types import DeviceLike, NumpyLike, TensorLike
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.transforms.standardize import StandardizeTransform


class SupportVectorMachine(SklearnBackend):
    """
    Support Vector Machines Emulator.

    Wraps Support Vector Regressor from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike | None,
        standardize_x: bool = False,
        standardize_y: bool = False,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        degree: int = 3,
        gamma: Literal["scale", "auto"] | float = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        C: float = 1.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200.0,
        verbose: bool = False,
        max_iter: int = 100,
        normalise_y: bool = True,
        device: DeviceLike = "cpu",
    ):
        """Initialize a SupportVectorMachine emulator.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike | None
            Target values. If None, the model will be fitted later.
        standardize_x: bool
            Whether to standardize input features. Defaults to False.
        standardize_y: bool
            Whether to standardize target values. Defaults to False.
        kernel: {"linear", "poly", "rbf", "sigmoid", "precomputed"}
            Kernel type to be used in the algorithm. Options are:
            "linear", "poly", "rbf", "sigmoid", "precomputed". Defaults to "rbf".
        degree: int
            Degree of the polynomial kernel function ('poly'). Ignored by other kernels.
            Defaults to 3.
        gamma: {"scale", "auto"} | float
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If 'scale', it is set
            to 1 / (n_features * X.var()) as default; if 'auto', it is set to
            1 / n_features. If float, must be non-negative.
            Defaults to 'scale'.
        coef0: float
            Independent term in kernel function. It is only significant in 'poly' and
            'sigmoid'. Defaults to 0.0.
        tol: float
            Tolerance for stopping criteria. Defaults to 1e-3.
        C: float
            Regularization parameter. The strength of the regularization is inversely
            proportional to C. Must be strictly positive. Defaults to 1.0.
        epsilon: float
            Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which
            no penalty is associated in the training loss function with points predicted
            within a distance epsilon from the actual value. Defaults to 0.1.
        shrinking: bool
            Whether to use the shrinking heuristic. Defaults to True.
        cache_size: float
            Specify the size of the kernel cache (in MB). Defaults to 200.
        verbose: bool
            Enable verbose output. Defaults to False.
        max_iter: int
            Hard limit on iterations within solver, or -1 for no limit. Defaults to 100.
        normalise_y: bool
            Whether to normalize the target values y before fitting the model. Defaults
            to True.
        device: DeviceLike
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Note: SVR
            always runs on CPU, this parameter is for compatibility with the base class.
        """
        _, _, _ = x, y, device  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
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
            gamma=self.gamma,  # type: ignore reportArgumentType
            coef0=self.coef0,
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            cache_size=self.cache_size,  # type: ignore reportArgumentType
            verbose=self.verbose,
            max_iter=self.max_iter,
        )

    @staticmethod
    def is_multioutput() -> bool:
        """Support Vector Machines do not support multi-output."""
        return False

    def _model_specific_check(self, x: NumpyLike, y: NumpyLike):
        check_X_y(x, y, ensure_min_samples=2)

    @staticmethod
    def get_tune_params():
        """Return a dictionary of hyperparameters to tune."""
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
