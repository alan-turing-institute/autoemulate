from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import SklearnBackend
from autoemulate.experimental.transforms.standardize import StandardizeTransform
from autoemulate.experimental.types import DeviceLike, TensorLike


class RandomForest(SklearnBackend):
    """
    Random forest Emulator.

    Implements Random Forests regression from scikit-learn.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike,
        y: TensorLike,
        standardize_x: bool = False,
        standardize_y: bool = False,
        n_estimators: int = 100,
        criterion: Literal[
            "squared_error", "absolute_error", "friedman_mse", "poisson"
        ] = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Literal["sqrt", "log2"] | int | float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        max_samples: int | None = None,
        random_seed: int | None = None,
        device: DeviceLike = "cpu",
        **kwargs,
    ):
        """Initialize a RandomForest emulator.

        Parameters
        ----------
        x: TensorLike
            Input features.
        y: TensorLike
            Target values.
        standardize_x: bool, default=False
            Whether to standardize input features.
        standardize_y: bool, default=False
            Whether to standardize target values.
        n_estimators: int, default=100
            Number of trees in the forest.
        criterion: {"squared_error", "absolute_error", "friedman_mse", "poisson"},
                default="squared_error"
            The function to measure the quality of a split. "squared_error" for
            the mean squared error, "absolute_error" for the mean absolute error,
            "friedman_mse" for mean squared error with Friedman's improvement score,
            and "poisson" for Poisson deviance.
        max_depth: int | None, default=None
            The maximum depth of the tree. If None, nodes are expanded until all leaves
            are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split: int, default=2
            The minimum number of samples required to split an internal node.
        min_samples_leaf: int, default=1
            The minimum number of samples required to be at a leaf node.
        max_features: "sqrt" | "log2" | int | float, default=1.0
            The number of features to consider when looking for the best split.
            If int, then consider max_features features at each split.
            If float, then max_features is a fraction and
            max(1, int(max_features * n_features_in_)) features are considered.
            If "sqrt", then max_features=sqrt(n_features).
            If "log2", then max_features=log2(n_features).
            If 1.0, then max_features=n_features.
        bootstrap: bool, default=True
            Whether bootstrap samples are used when building trees.
        oob_score: bool, default=False
            Whether to use out-of-bag samples to estimate the generalization accuracy.
        max_samples: int | None, default=None
            If bootstrap is True, the number of samples to draw from X to train each
            base estimator. If None, then draw n_samples.
        random_seed: int | None, default=None
            Random seed for reproducibility. If None, no seed is set.
        device: DeviceLike, default="cpu"
            Device to run the model on. If None, uses the default device.
        **kwargs: dict
            Additional keyword arguments.

        """
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device, cpu_only=True)
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
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
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,  # type: ignore reportArgumentType
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            max_samples=self.max_samples,
            random_state=self.random_seed,
        )

    @staticmethod
    def is_multioutput() -> bool:
        """Random forests support multi-output."""
        return True

    @staticmethod
    def get_tune_params():
        """Return a dictionary of hyperparameters to tune."""
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
