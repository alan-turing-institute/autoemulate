from typing import Literal

import numpy as np
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingRegressor

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.types import DeviceLike, TensorLike


class GradientBoosting(SklearnBackend):
    """Gradient Boosting Emulator.

    Wraps Gradient Boosting regression from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        loss: Literal[
            "squared_error", "absolute_error", "huber", "quantile"
        ] = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features: float | None = None,
        ccp_alpha: float = 0.0,
        n_iter_no_change: int | None = None,
        random_seed: int | None = None,
        device: DeviceLike = "cpu",
    ):
        """Initializes a GradientBoosting object."""
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
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
        self.random_seed = random_seed
        self.model = GradientBoostingRegressor(
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
            random_state=self.random_seed,
        )

    @staticmethod
    def is_multioutput() -> bool:
        return False

    @staticmethod
    def get_tune_config():
        return {
            "learning_rate": [loguniform(0.01, 0.2).rvs()],
            "n_estimators": [np.random.randint(100, 500)],
            "max_depth": [np.random.randint(3, 8)],
            "min_samples_split": [np.random.randint(2, 20)],
            "min_samples_leaf": [np.random.randint(1, 6)],
            "subsample": [np.random.uniform(0.6, 1.0)],
            "max_features": ["sqrt", "log2", None],
            "ccp_alpha": [loguniform(0.001, 0.1).rvs()],
        }
