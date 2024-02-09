from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real


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

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters of the emulator."""
        param_space_random = {
            "learning_rate": loguniform(0.01, 0.2),
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 8),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 6),
            "subsample": uniform(0.6, 0.4),  # 0.4 is the range width (1.0 - 0.6)
            "max_features": ["sqrt", "log2", None],
            "ccp_alpha": loguniform(0.01, 0.1),
        }

        param_space_bayes = {
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(3, 8),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 6),
            "subsample": Real(0.6, 1.0),
            "max_features": Categorical(["sqrt", "log2", None]),
            "ccp_alpha": Real(0.01, 0.1, prior="log-uniform"),
        }

        if search_type == "random":
            param_space = param_space_random
        elif search_type == "bayes":
            param_space = param_space_bayes

        return param_space

    def _more_tags(self):
        return {"multioutput": False}
