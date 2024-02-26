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

from ..types import Any
from ..types import ArrayLike
from ..types import Literal
from ..types import MatrixLike
from ..types import Optional
from ..types import Self
from ..types import Union


class GradientBoosting(BaseEstimator, RegressorMixin):
    """Gradient Boosting Emulator.

    Wraps Gradient Boosting regression from scikit-learn.

    Parameters
    ----------
    loss : {'squared_error', 'ls', 'lad', 'huber', 'quantile'}, default='squared_error'
        The loss function to be optimized. 'squared_error' refers to the
        ordinary least squares fit. 'ls' refers to least squares fit. 'lad'
        refers to least absolute deviation fit. 'huber' is a combination of
        least squares and least absolute deviation. 'quantile' allows quantile
        regression (use alpha to specify the quantile).
    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each tree. There is a
        trade-off between learning_rate and n_estimators.
    n_estimators : int, default=100
        The number of boosting stages to be run. Gradient boosting is fairly
        robust to over-fitting so a large number usually results in better
        performance.
    max_depth : int, default=3
        The maximum depth of the individual estimators. The maximum depth
        limits the number of nodes in the tree. Tune this parameter for best
        performance; the best value depends on the interaction of the input
        variables.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. subsample interacts with the parameter n_estimators. Choosing
        subsample < 1.0 leads to a reduction of variance and an increase in
        bias.
    max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and
          int(max_features * n_features) features are considered at each split.
        - If 'auto', then max_features=sqrt(n_features).
        - If 'sqrt', then max_features=sqrt(n_features).
        - If 'log2', then max_features=log2(n_features).
        - If None, then max_features=n_features.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ccp_alpha will be chosen. By default, no pruning is performed.
    n_iter_no_change : int, default=None
        Number of iterations with no improvement to wait before stopping
        fitting. Convergence is checked against the training loss or the
        validation loss depending on the early_stopping parameter.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each base learner at each boosting
        iteration. In addition, it controls the random permutation of the
        features at each split. It also controls the random spliting of the
        training data to obtain a validation set if n_iter_no_change is not
        None. Pass an int for reproducible output across multiple function
        calls.
    """

    def __init__(
        self,
        loss: Literal[
            "squared_error", "ls", "lad", "huber", "quantile"
        ] = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features: Optional[
            Union[Literal["auto", "sqrt", "log2"], int, float]
        ] = None,
        ccp_alpha: float = 0.0,
        n_iter_no_change: Optional[int] = None,
        random_state: Optional[int] = None,
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

    def predict(self, X: MatrixLike) -> ArrayLike:
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

    def get_grid_params(
        self, search_type: Literal["random", "bayes"] = "random"
    ) -> dict[str, Any]:
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

        # TODO: Should this raise an error if the search type is not recognised?

        return param_space

    def _more_tags(self) -> dict[str, bool]:
        """Returns more tags for the estimator.

        Returns
        -------
        dict
            Dictionary of tags.
        """
        return {"multioutput": False}
