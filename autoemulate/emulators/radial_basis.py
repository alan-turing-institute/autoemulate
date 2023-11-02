import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from smt.surrogate_models import RBF


class RadialBasis(BaseEstimator, RegressorMixin):
    """Radial basis function Emulator.

    Wraps the RBF surrogate model from SMT.
    """

    def __init__(
        self,
        d0=1.0,
        poly_degree=-1,
        reg=1e-10,
    ):
        """Initializes a RadialBasis object."""
        self.d0 = d0
        self.poly_degree = poly_degree
        self.reg = reg

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
        self.model_ = RBF(d0=self.d0, poly_degree=self.poly_degree, reg=self.reg)
        self.model_.set_training_values(X, y)
        self.model_.train()
        return self

    def predict(self, X, return_std=False):
        """Predicts the output of the emulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        return_std : bool, default=False
            Whether to return the standard deviation of the prediction.

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            The predicted values.
        """
        X = check_array(X)
        check_is_fitted(self)
        if return_std:
            return self.model_.predict_values(X), self.model_.predict_variances(X)
        else:
            return self.model_.predict_values(X)

    def get_grid_params(self):
        """Returns the grid parameters of the emulator."""
        param_grid = {
            "d0": [1.0, 2.0, 3.0],
            "poly_degree": [-1, 0, 1],
            "reg": [1e-10, 1e-5, 1e-2],
        }

    def _more_tags(self):
        return {"multioutput": True}
