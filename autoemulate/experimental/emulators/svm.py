import numpy as np
from sklearn.svm import SVR
from sklearn.utils.validation import check_X_y
from torch import Tensor

from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import InputLike, OutputLike
from autoemulate.utils import _denormalise_y, _normalise_y


class SupportVectorMachines(SklearnBackend):
    """Support Vector Machines Emulator.

    Wraps Support Vector Regressor from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: InputLike,
        y: InputLike | None,
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
        max_iter=-1,
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

    def fit(self, x: InputLike, y: InputLike | None):
        """Fits the emulator to the data."""
        self.n_iter_ = self.max_iter if self.max_iter > 0 else 1
        x, y = self.sklearn_fit_checks(x, y)

        y = y.ravel()  # Ensure y is 1-dimensional

        x, y = check_X_y(
            x,
            y,
            y_numeric=True,
            ensure_min_samples=2,
        )
        if self.normalise_y:
            y, self.y_mean_, self.y_std_ = _normalise_y(y)
        elif y is not None and isinstance(y, np.ndarray):
            self.y_mean_ = np.zeros(y.shape[1]) if y.ndim > 1 else 0
            self.y_std_ = np.ones(y.shape[1]) if y.ndim > 1 else 1
        else:
            msg = "Input 'y' must be a non-None NumPy array."
            raise ValueError(msg)

        self._fit(x, y)

    def predict(self, x: InputLike) -> OutputLike:
        """Predicts the output of the emulator for a given input."""
        y_pred = self._predict(x)

        if self.normalise_y:
            y_pred = _denormalise_y(y_pred, self.y_mean_, self.y_std_)

        # Ensure the output is a 2D tensor array with shape (n_samples, 1)
        return Tensor(y_pred.reshape(-1, 1))  # type: ignore PGH003

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

    @property
    def model_name(self):
        return self.__class__.__name__
