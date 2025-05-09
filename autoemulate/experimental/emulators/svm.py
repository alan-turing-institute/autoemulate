import numpy as np
from sklearn.svm import SVR
from sklearn.utils.validation import check_X_y
from torch import Tensor

from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import OutputLike, TensorLike
from autoemulate.utils import _denormalise_y, _normalise_y


class SupportVectorMachines(SklearnBackend):
    """Support Vector Machines Emulator.

    Wraps Support Vector Regressor from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike | None,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        C=1.0,
        epsilon=0.1,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=100,
        normalise_y=True,
    ):
        """Initializes a SupportVectorMachines object."""
        _, _ = x, y  # ignore unused arguments
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

    def _model_specific_check(self, x, y):
        print("IN SVM")
        print(x.shape, y.shape)
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
