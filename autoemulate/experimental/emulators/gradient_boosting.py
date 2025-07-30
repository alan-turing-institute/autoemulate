from typing import Literal

import numpy as np
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingRegressor

from autoemulate.experimental.core.device import TorchDeviceMixin
from autoemulate.experimental.core.types import DeviceLike, TensorLike
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.transforms.standardize import StandardizeTransform


class GradientBoosting(SklearnBackend):
    """
    Gradient Boosting Emulator.

    Wraps Gradient Boosting regression from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = False,
        standardize_y: bool = False,
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
        """Initialize a GradientBoosting object.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike
            Target values.
        standardize_x: bool
            Whether to standardize input features. Defaults to False.
        standardize_y: bool
            Whether to standardize target values. Defaults to False.
        loss: str
            Loss function to be optimized. Defaults to "squared_error".
        learning_rate: float
            Learning rate shrinks the contribution of each new tree. Defaults to 0.1.
        n_estimators: int
            The number of boosting stages to be run. Defaults to 100.
        max_depth: int
            Maximum depth of the individual regression estimators. Defaults to 3.
        min_samples_split: int
            Minimum number of samples required to split an internal node. Defaults to 2.
        min_samples_leaf: int
            Minimum number of samples required to be at a leaf node. Defaults to 1.
        subsample: float
            The fraction of samples to be used for fitting the individual base learners.
            Defaults to 1.0.
        max_features: float | None
            The number of features to consider when looking for the best split. Defaults
            to None.
        ccp_alpha: float
            Complexity parameter used for Minimal Cost-Complexity Pruning. Defaults to
            0.0.
        n_iter_no_change: int | None
            If not None, the number of iterations with no improvement to wait before
            stopping. Defaults to None.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set. Defaults to None.
        device: DeviceLike
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to "cpu".
        """
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
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
        """GradientBoosting does not support multi-output."""
        return False

    @staticmethod
    def get_tune_params():
        """Return a dictionary of hyperparameters to tune."""
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
