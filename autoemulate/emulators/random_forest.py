from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RandomForest(BaseEstimator, RegressorMixin):
    """Random forest Emulator.

    Implements Random Forests regression from scikit-learn.
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        oob_score=False,
        max_samples=None,
        random_state=None,
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

        self.native_multioutput = True

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

    def predict(self, X):
        """Predicts the output of the simulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        return_std : bool
            If True, returns a touple with two ndarrays,
            one with the mean and one with the standard deviations of the prediction.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Model predictions.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return self.model_.predict(X)

    def get_grid_params(self):
        """Returns the grid parameters of the emulator."""
        param_grid = {
            "model__n_estimators": randint(50, 500),  # broader range
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": [1.0, "sqrt", "log2"],  # instead of fixed values
            "model__bootstrap": [True, False],
            "model__oob_score": [True, False],
            # "model__max_depth": [None] + list(range(3, 20)),  # None plus a range of depths
            "model__max_samples": [None, 0.5, 0.75],  # assuming max_samples is relevant
        }
        return param_grid

    def _more_tags(self):
        return {"multioutput": True}

    # def score(self, X, y, metric):
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
    #     return {'non_deterministic': True,
    #             'multioutput': True}
