import numpy as np
from sklearn.ensemble import RandomForestRegressor

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import DeviceLike, TensorLike


class RandomForest(SklearnBackend):
    """Random forest Emulator.

    Implements Random Forests regression from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        max_samples: int | None = None,
        random_seed: int | None = None,
        device: DeviceLike = "cpu",
    ):
        """Initializes a RandomForest object."""
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_seed,
        )

    @staticmethod
    def is_multioutput() -> bool:
        return True

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
