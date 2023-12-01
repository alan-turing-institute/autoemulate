from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class GradientBoosting(BaseEstimator, RegressorMixin):
    """Gradient Boosting Emulator.

    Wraps Gradient Boosting regression from scikit-learn.
    """

    def __init__(
        self,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        max_features=None,
        ccp_alpha=0.0,
        n_iter_no_change=None,
        random_state=None,
    ):
        """Initializes a GradientBoosting object."""
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.n_iter_no_change = n_iter_no_change
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
        X, y = check_X_y(
            X, y, multi_output=self._more_tags()["multioutput"], y_numeric=True
        )

        self.n_features_in_ = X.shape[1]
        self.model_ = GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            ccp_alpha=self.ccp_alpha,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
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
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict(X)

    def get_grid_params(self):
        """Returns the grid parameters of the emulator."""
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 6],
            "subsample": [0.6, 0.8, 1.0],
            "max_features": ["sqrt", "log2", None],
            "ccp_alpha": [0.0, 0.01, 0.1],
        }
        return param_grid

    def _more_tags(self):
        return {"multioutput": False}
