import numpy as np
import mogp_emulator
from autoemulate.emulators import Emulator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class GaussianProcess(BaseEstimator, RegressorMixin):
    """Gaussian process Emulator.

    Implements GaussianProcess regression from the mogp_emulator package.
    """

    def __init__(self, nugget="fit"):
        """Initializes a GaussianProcess object."""
        self.nugget = nugget

    def fit(self, X, y):
        """Fits the emulator to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.model_ = mogp_emulator.GaussianProcess(X, y, nugget=self.nugget)
        self.model_ = mogp_emulator.fit_GP_MAP(self.model_, n_tries=15)
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
        if return_std:
            mean, var, _ = self.model_.predict(X)
            return np.asarray(mean), np.asarray(np.sqrt(var))
        else:
            return np.asarray(self.model_.predict(X).mean)

    def get_grid_params(self):
        """Returns the grid parameters of the emulator."""
        param_grid = {"model__nugget": ["fit", "adaptive", "pivot"]}
        return param_grid

    def _more_tags(self):
        return {"multioutput": False}

    # def score(self, X, y, metric):
    #     """Returns the score of the emulator.

    #     Parameters
    #     ----------
    #     X : numpy.ndarray
    #         Input data (simulation input).
    #     y : numpy.ndarray
    #         Target data (simulation output).

    #     Returns
    #     -------
    #     rmse : float
    #         Root mean squared error of the emulator.
    #     """
    #     prediction_means = self.predict(X)
    #     return metric(y, prediction_means)
