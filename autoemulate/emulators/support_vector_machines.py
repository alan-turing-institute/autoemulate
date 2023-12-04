from sklearn.svm import SVR
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class SupportVectorMachines(BaseEstimator, RegressorMixin):
    """Support Vector Machines Emulator.

    Wraps Support Vector Regressor from scikit-learn.
    """

    def __init__(
        self,
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
    ):
        """Initializes a SupportVectorMachines object."""
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
            X, y, multi_output=self._more_tags()["multioutput"], y_numeric=True
        )

        self.n_features_in_ = X.shape[1]
        self.n_iter_ = self.max_iter if self.max_iter > 0 else 1

        self.model_ = SVR(
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
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predicts the output for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The predicted output values.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.model_.predict(X)

    def get_grid_params(self):
        """Returns the grid paramaters for the emulator."""
        param_grid = {
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "degree": [2, 3, 4, 5],
            "gamma": ["scale", "auto"],
            "coef0": [0.0, 0.5, 1.0],
            "tol": [1e-3, 1e-4, 1e-5],
            "C": [1.0, 2.0, 3.0],
            "epsilon": [0.1, 0.2, 0.3],
            "shrinking": [True, False],
            "cache_size": [200, 300, 400],
            "verbose": [False],
            "max_iter": [-1],
        }
        return param_grid

    def _more_tags(self):
        return {"multioutput": False}
