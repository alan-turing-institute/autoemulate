import gpytorch
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from skorch.probabilistic import ExactGPRegressor

from autoemulate.emulators.neural_networks.gp_module import GPModule


class GPTorch(RegressorMixin, BaseEstimator):
    def __init__(
        self,
    ):
        pass

    def fit(self, X, y):
        """Fit the emulator to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples, )
            The output data.
        Returns
        -------
        self : object
            Returns self.
        """

        X, y = check_X_y(X, y, y_numeric=True)  # multi-output = True
        self.n_features_in_ = X.shape[1]
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        self.model_ = ExactGPRegressor(GPModule, likelihood=likelihood, max_epochs=100)
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        """Predict the output of the emulator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        return_std : bool, default=False
            Whether to return the standard deviation.

        Returns
        -------
        y : array-like of shape (n_samples, )
            The predicted output.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict(X, return_std=return_std)

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters for the emulator."""
        pass

    @property
    def model_name(self):
        return self.__class__.__name__

    def _more_tags(self):
        return {"multioutput": True}
