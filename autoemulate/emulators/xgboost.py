import numpy as np
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real
from xgboost import XGBRegressor

from ..types import Any
from ..types import ArrayLike
from ..types import Literal
from ..types import MatrixLike
from ..types import Optional
from ..types import Self
from ..types import Union


class XGBoost(BaseEstimator, RegressorMixin):
    """XGBoost Emulator.

    Wraps XGBoost regression from scikit-learn.

    Parameters
    ----------
    booster : {'gbtree', 'gblinear', 'dart'}, default='gbtree'
        Which booster to use.
    verbosity : int, default=0
        The degree of verbosity.
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=6
        The maximum depth of the tree.
    max_leaves : int, default=0
        The maximum number of leaves for base learners.
    learning_rate : float, default=0.3
        The learning rate.
    gamma : float, default=0
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : float, default=1
        Minimum sum of instance weight (hessian) needed in a child.
    max_delta_step : float, default=0
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float, default=1
        Subsample ratio of the training instance.
    colsample_bytree : float, default=1
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float, default=1
        Subsample ratio of columns for each level.
    colsample_bynode : float, default=1
        Subsample ratio of columns for each split.
    reg_alpha : float, default=0
        L1 regularization term on weights.
    reg_lambda : float, default=1
        L2 regularization term on weights.
    objective : str, default='reg:squarederror'
        Specify the learning task and the corresponding learning objective or a custom objective function.
    tree_method : {'auto', 'exact', 'approx', 'hist', 'gpu_hist'}, default='auto'
        The tree construction algorithm used in XGBoost.
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if bootstrap=True) and the sampling of the features
        to consider when looking for the best split at each node (if
        max_features < n_features).
    n_jobs : int, default=None
        Number of parallel threads used to run XGBoost.
    """

    def __init__(
        self,
        # general parameters
        booster: Literal["gbtree", "gblinear", "dart"] = "gbtree",
        verbosity: int = 0,
        # tree booster parameters
        n_estimators: int = 100,
        max_depth: int = 6,
        max_leaves: int = 0,  # no limit
        learning_rate: float = 0.3,
        gamma: int = 0,
        min_child_weight: int = 1,
        max_delta_step: int = 0,
        subsample: int = 1,
        colsample_bytree: int = 1,
        colsample_bylevel: int = 1,
        colsample_bynode: int = 1,
        reg_alpha: int = 0,
        reg_lambda: int = 1,
        objective: str = "reg:squarederror",
        tree_method: Literal["auto", "exact", "approx", "hist", "gpu_hist"] = "auto",
        random_state=None,  # TODO: add correct type
        n_jobs: Optional[int] = None,
    ):
        """Initializes a XGBoost object."""
        self.booster = booster
        self.verbosity = verbosity
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: MatrixLike, y: ArrayLike) -> Self:
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

        self.model_ = XGBRegressor(
            booster=self.booster,
            verbosity=self.verbosity,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: MatrixLike) -> ArrayLike:
        """Predicts the output of the emulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Model predictions.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        # XGBoost returns float32, but we want float64 to pass sklearn estimator checks
        y_pred = self.model_.predict(X).astype(np.float64)
        return y_pred

    def get_grid_params(
        self, search_type: Literal["random", "bayes"] = "random"
    ) -> dict[str, Any]:
        """Returns the grid parameters of the emulator."""
        param_space_random = {
            "booster": ["gbtree", "dart"],
            "n_estimators": randint(100, 1000),
            "max_depth": randint(3, 10),
            "learning_rate": loguniform(0.001, 0.5),
            "gamma": loguniform(0.01, 1),
            "min_child_weight": randint(1, 10),
            "max_delta_step": randint(0, 10),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "colsample_bylevel": uniform(0.5, 0.5),
            "colsample_bynode": uniform(0.5, 0.5),
            "reg_alpha": loguniform(0.01, 1),
            "reg_lambda": loguniform(0.01, 1),
        }

        param_space_bayes = {
            "booster": Categorical(["gbtree", "dart"]),
            "n_estimators": Integer(100, 1000),
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
            "gamma": Real(0.01, 1, prior="log-uniform"),
            "min_child_weight": Integer(1, 10),
            "max_delta_step": Integer(0, 10),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "colsample_bylevel": Real(0.5, 1.0),
            "colsample_bynode": Real(0.5, 1.0),
            "reg_alpha": Real(0.01, 1, prior="log-uniform"),
            "reg_lambda": Real(0.01, 1, prior="log-uniform"),
        }

        if search_type == "random":
            param_space = param_space_random
        elif search_type == "bayes":
            param_space = param_space_bayes

        # TODO: Should this raise an error if the search type is not recognised?

        return param_space

    def _more_tags(self):
        """Returns more tags for the estimator.

        Returns
        -------
        dict
            The tags for the estimator.
        """
        return {"multioutput": True}
