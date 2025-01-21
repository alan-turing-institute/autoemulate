import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y


class LightGBM(BaseEstimator, RegressorMixin):
    """LightGBM Emulator.

    Wraps LightGBM regression from LightGBM.
    """

    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1.0,
        # subsample_freq=0.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=1,
        importance_type="split",
        verbose=-1,
    ):
        """Initializes a LightGBM object."""
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        # self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fits the emulator to the data."""
        X, y = check_X_y(
            X, y, multi_output=self._more_tags()["multioutput"], y_numeric=True
        )

        self.n_features_in_ = X.shape[1]

        self.model_ = LGBMRegressor(
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample_for_bin=self.subsample_for_bin,
            objective=self.objective,
            class_weight=self.class_weight,
            min_split_gain=self.min_split_gain,
            min_child_weight=self.min_child_weight,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            importance_type=self.importance_type,
            verbose=self.verbose,
        )

        self.model_.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predicts the output of the emulator for a given input."""
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        y_pred = self.model_.predict(X)
        return y_pred

    def get_grid_params(self, search_type="random"):
        """Returns the grid parameters of the emulator."""
        if search_type == "random":
            param_space = {
                "boosting_type": ["gbdt"],
                "num_leaves": randint(10, 100),
                "max_depth": randint(-1, 12),
                "learning_rate": loguniform(0.001, 0.1),
                "n_estimators": randint(50, 1000),
                # "colsample_bytree": uniform(0.5, 1.0),
                "reg_alpha": loguniform(0.001, 1),
                "reg_lambda": loguniform(0.001, 1),
            }
        return param_space

    @property
    def model_name(self):
        return self.__class__.__name__

    def _more_tags(self):
        return {"multioutput": False}
