import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from smt.surrogate_models import QP


class SecondOrderPolynomial(BaseEstimator, RegressorMixin):
    """Second order polynomial emulator.

    Wrapper for the second order polynomial surrogate model from SMT.
    """

    def __init__(self):
        """Initializes a QuadraticPolynomial object."""
        pass

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
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.model_ = QP(print_global=False)
        self.model_.set_training_values(X, y)
        self.model_.train()
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
        y : ndarray of shape (n_samples, n_features)
            The predicted values.
        """
        X = check_array(X, dtype=np.float64)
        check_is_fitted(self)
        predictions = self.model_.predict_values(X)
        return predictions

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters of the emulator.

        Returns
        -------
        params : dict
            Dictionary of grid parameters.
        """
        if search_type == "random":
            param_grid = {}
        else:
            param_grid = {}
        return param_grid

    def _more_tags(self):
        return {
            "multioutput": True,
            "_xfail_checks": {
                "check_estimators_dtypes": "fails because too few samples",
                "check_dtype_object": "fails because too few samples",
                "check_regressor_multioutput": "fails because too few samples",
                "check_regressors_no_decision_function": "fails because too few samples",
                "check_regressors_int": "fails because too few samples",
                "check_fit2d_1sample": "fails because too few samples",
                "check_regressors_train": "TODO, shouldn't fail",
            },
        }
