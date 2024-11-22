import numpy as np
from scipy.stats import loguniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y

from autoemulate.utils import _suppress_convergence_warnings


class NeuralNetSk(BaseEstimator, RegressorMixin):
    """Multi-layer perceptron Emulator.

    Wraps MLPRegressor from scikit-learn.
    """

    def __init__(
        self,
        hidden_layer_sizes=(
            100,
            100,
        ),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=200,
        tol=1e-4,
        random_state=None,
    ):
        """Initializes an MLPRegressor object."""
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        """Fits the emulator to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        with _suppress_convergence_warnings():
            self.model_.fit(X, y)
        # expose n_iter_ attribute to be consistent with sklearn estimators
        self.n_iter_ = self.model_.n_iter_
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Model predictions.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict(X)

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters of the emulator."""
        if search_type == "random":
            param_space = {
                "hidden_layer_sizes": [
                    (50,),
                    (100,),
                    (100, 50),
                    (100, 100),
                    (100, 100, 100),
                ],
                "activation": ["relu"],
                "solver": ["adam", "lbfgs"],
                "alpha": loguniform(1e-5, 1e-1),
                "learning_rate_init": loguniform(1e-4, 1e-2),
            }
        return param_space

    @property
    def model_name(self):
        return self.__class__.__name__

    def _more_tags(self):
        return {"multioutput": True}
