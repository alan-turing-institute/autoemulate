import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from scipy.interpolate import RBFInterpolator
from scipy.stats import uniform, randint
from skopt.space import Real, Categorical, Integer


class RBF(BaseEstimator, RegressorMixin):
    """Radial basis function Emulator.

    Wraps the RBF interpolator from scipy.
    """

    def __init__(
        self,
        smoothing=0.0,
        kernel="thin_plate_spline",
        epsilon=1.0,
        degree=1,
    ):
        """Initializes an RBF object."""
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree

    def fit(self, X, y):
        """Fits the emulator to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            dtype=np.float64,
            ensure_min_samples=2,
        )
        self.n_features_in_ = X.shape[1]
        self.model_ = RBFInterpolator(
            X,
            y,
            smoothing=self.smoothing,
            kernel=self.kernel,
            epsilon=self.epsilon,
            degree=self.degree,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predicts the output of the emulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted values (real numbers).
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.model_(X)

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters of the emulator."""
        # param_grid_random = {
        #     #"smoothing": uniform(0.0, 1.0),
        #     "kernel": ["linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric", "gaussian"],
        #     #"epsilon": uniform(0.0, 1.0),
        #     "degree": randint(0, 5),
        # }
        param_grid_random = [
            {
                "kernel": ["linear", "multiquadric"],
                "degree": randint(0, 3),  # Degrees valid for these kernels
                "smoothing": uniform(0.0, 1.0),
            },
            {
                "kernel": ["thin_plate_spline", "cubic"],
                "degree": randint(1, 3),  # Degrees valid for the 'quintic' kernel
                "smoothing": uniform(0.0, 1.0),
            },
            {
                "kernel": ["quintic"],
                "degree": randint(2, 3),
                "smoothing": uniform(0.0, 1.0),
            },
            {
                "kernel": ["gaussian"],
                "degree": randint(-1, 3),
                "smoothing": uniform(0.0, 1.0),
            },
        ]

        param_grid_bayes = [
            {
                "kernel": Categorical(["linear", "multiquadric"]),
                "degree": Integer(0, 4),  # Degrees valid for these kernels
                "smoothing": Real(0.0, 1.0),
            },
            {
                "kernel": Categorical(["thin_plate_spline", "cubic"]),
                "degree": Integer(1, 4),  # Degrees valid for the 'quintic' kernel
                "smoothing": Real(0.0, 1.0),
            },
            {
                "kernel": Categorical(["quintic"]),
                "degree": Integer(2, 4),
                "smoothing": Real(0.0, 1.0),
            },
            {
                "kernel": Categorical(["gaussian"]),
                "degree": Integer(-1, 4),
                "smoothing": Real(0.0, 1.0),
            },
        ]

        if search_type == "random":
            param_grid = param_grid_random
        elif search_type == "bayes":
            param_grid = param_grid_bayes

        return param_grid

    def _more_tags(self):
        return {"multioutput": True}
