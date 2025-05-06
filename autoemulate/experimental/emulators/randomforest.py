import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y

from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import TensorLike


class RandomForest(SklearnBackend):
    """Random forest Emulator.

    Implements Random Forests regression from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike | None,
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
        _, _ = x, y  # ignore unused arguments
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
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
        )

    @staticmethod
    def is_multioutput() -> bool:
        return False

    def fit(self, x: TensorLike, y: TensorLike | None):
        """Fits the emulator to the data."""
        x, y = self._convert_to_numpy(x, y)
        self.n_features_in_ = x.shape[1]

        # y = y.ravel()  # Ensure y is 1-dimensional

        x, y = check_X_y(x, y, multi_output=True, y_numeric=True)
        self._fit(x, y)

    @staticmethod
    def get_tune_config():
        return {
            "n_estimators": [np.random.randint(50, 500)],
            "min_samples_split": [np.random.randint(2, 20)],
            "min_samples_leaf": [np.random.randint(1, 10)],
            "max_features": ["sqrt", "log2", None, 1.0],
            "bootstrap": [True, False],
            "oob_score": [True, False],
            "max_depth": [None, *list(range(5, 30, 5))],  # None plus a range of depths
            "max_samples": [None, 0.5, 0.7, 0.9],
        }
