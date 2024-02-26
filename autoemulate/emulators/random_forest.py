from scipy.stats import randint
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real

from ..types import Any
from ..types import ArrayLike
from ..types import Literal
from ..types import MatrixLike
from ..types import Optional
from ..types import Self
from ..types import Union


class RandomForest(BaseEstimator, RegressorMixin):
    """Random forest Emulator.

    Implements Random Forests regression from scikit-learn.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {"squared_error", "mse", "mae"}, default="squared_error"
        The function to measure the quality of a split.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and
          int(max_features * n_features) features are considered at each split.
        - If "auto", then max_features=sqrt(n_features).
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X to train
        each base estimator. If None (default), then draw X.shape[0] samples.
        If int, then draw max_samples samples.
        If float, then draw max_samples * X.shape[0] samples. Thus, max_samples
        should be in the interval (0, 1).
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if bootstrap=True) and the sampling of the features
        to consider when looking for the best split at each node (if
        max_features < n_features).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Literal["squared_error", "mse", "mae"] = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[
            Union[Literal["auto", "sqrt", "log2"], int, float]
        ] = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        max_samples: Union[int, float] = None,
        random_state=None,  # TODO: set correct type
    ):
        """Initializes a RandomForest object."""
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: MatrixLike, y: ArrayLike) -> Self:
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
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: MatrixLike) -> ArrayLike:
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

    def get_grid_params(
        self, search_type: Literal["random", "bayes"] = "random"
    ) -> dict[str, Any]:
        """Returns the grid parameters of the emulator."""

        param_space_random = {
            "n_estimators": randint(50, 500),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": [None, "sqrt", "log2"],
            "bootstrap": [True, False],
            "oob_score": [True, False],
            # # "max_depth": [None] + list(range(3, 20)),  # None plus a range of depths
            "max_samples": [None, 0.5, 0.75],
        }

        param_space_bayes = {
            "n_estimators": Integer(50, 500),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Categorical([None, "sqrt", "log2"]),
            "bootstrap": Categorical([True, False]),
            "oob_score": Categorical([True, False]),
            # "max_depth": Categorical([None] + list(range(3, 20))),  # None plus a range of depths
            "max_samples": Categorical([None, 0.5, 0.75]),
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

    # def score(self, X: ArrayLike, y: ArrayLike, metric:Literal["rsme", "r2"]) -> float:
    #     """Returns the score of the emulator.

    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_features)
    #         Simulation input.
    #     y : array-like, shape (n_samples, n_outputs)
    #         Simulation output.
    #     metric : str
    #         Name of the metric to use, currently either rsme or r2.
    #     Returns
    #     -------
    #     metric : float
    #         Metric of the emulator.

    #     """
    #     predictions = self.predict(X)
    #     return metric(y, predictions)

    # def _more_tags(self):
    #     """Returns more tags for the estimator.
    #     Returns
    #     -------
    #     dict
    #         The tags for the estimator.
    #     """
    #     return {'non_deterministic': True,
    #             'multioutput': True}
