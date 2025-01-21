from scipy.stats import loguniform
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y

from autoemulate.utils import _suppress_convergence_warnings


class GaussianProcessSklearn(BaseEstimator, RegressorMixin):
    """Gaussian Process Emulator.

    Wraps GaussianProcessRegressor from scikit-learn.
    """

    def __init__(
        self,
        kernel=RBF(),
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=20,
        normalize_y=True,
        copy_X_train=True,
        random_state=None,
    ):
        """Initializes a GaussianProcess object."""
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

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

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.model_ = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
        )
        with _suppress_convergence_warnings():
            self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        return_std : bool
            If True, returns a touple with two ndarrays,
            one with the mean and one with the standard deviations of the prediction.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Model predictions.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict(X, return_std=return_std)

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters of the emulator."""
        if search_type == "random":
            param_space = {
                "kernel": [
                    RBF(),
                    Matern(),
                    RationalQuadratic(),
                    # DotProduct(),
                ],
                "optimizer": ["fmin_l_bfgs_b"],
                "alpha": loguniform(1e-10, 1e-2),
                "normalize_y": [True],
            }
        return param_space

    @property
    def model_name(self):
        return self.__class__.__name__

    def _more_tags(self):
        return {"multioutput": True}
