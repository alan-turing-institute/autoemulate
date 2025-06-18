import numpy as np
from lightgbm import LGBMRegressor
from scipy.sparse import spmatrix

from autoemulate.experimental.device import TorchDeviceMixin
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.types import DeviceLike, OutputLike, TensorLike


class LightGBM(Emulator):
    """LightGBM Emulator.

    Wraps LightGBM regression from LightGBM.
    See https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    for more details.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike | None = None,
        y: TensorLike | None = None,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: str | None = None,
        class_weight: dict | str | None = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 0.001,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_seed: int | None = None,
        n_jobs: int | None = 1,
        importance_type: str = "split",
        verbose: int = -1,
        device: DeviceLike = "cpu",
    ):
        """Initializes a LightGBM object."""
        if random_seed is not None:
            self.set_random_seed(random_seed)
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device)
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
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self.verbose = verbose
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
            random_state=self.random_seed,
            n_jobs=self.n_jobs,
            importance_type=self.importance_type,
            verbose=self.verbose,
        )

    @staticmethod
    def is_multioutput() -> bool:
        return False

    def _fit(self, x: TensorLike, y: TensorLike):
        """
        Fits the emulator to the data.
        The model expects the input data to be:
            x (features): 2D array
            y (target): 1D array
        """
        x_np, y_np = self._convert_to_numpy(x, y)
        self.n_features_in_ = x_np.shape[1]
        self.model_.fit(x_np, y_np)

    def _predict(self, x: TensorLike) -> OutputLike:
        """Predicts the output of the emulator for a given input."""
        y_pred = self.model_.predict(x)
        assert not isinstance(y_pred, spmatrix | list)
        _, y = self._convert_to_tensors(x, y_pred)
        return y

    @staticmethod
    def get_tune_config():
        # Note: 10 ** np.random.uniform(-3, 0)
        # is equivalent to scipy.stats.loguniform(0.001, 0.1)
        return {
            "num_leaves": [np.random.randint(10, 100)],
            "max_depth": [np.random.randint(-1, 12)],
            "learning_rate": [10 ** np.random.uniform(-3, -1)],
            "n_estimators": [np.random.randint(50, 1000)],
            "reg_alpha": [10 ** np.random.uniform(-3, 0)],
            "reg_lambda": [10 ** np.random.uniform(-3, 0)],
        }
