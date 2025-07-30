import numpy as np
from lightgbm import LGBMRegressor
from scipy.sparse import spmatrix

from autoemulate.experimental.core.device import TorchDeviceMixin
from autoemulate.experimental.core.types import DeviceLike, TensorLike
from autoemulate.experimental.emulators.base import DeterministicEmulator
from autoemulate.experimental.transforms.standardize import StandardizeTransform


class LightGBM(DeterministicEmulator):
    """LightGBM Emulator.

    Wraps LightGBM regression from LightGBM.
    See https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    for more details.
    """

    def __init__(  # noqa: PLR0913 allow too many arguments since all currently required
        self,
        x: TensorLike | None = None,
        y: TensorLike | None = None,
        standardize_x: bool = False,
        standardize_y: bool = False,
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
        """Initialize a LightGBM object.

        Parameters
        ----------
        x: TensorLike | None
            Input features. If None, the model will be fitted later. Defaults to None.
        y: TensorLike | None
            Target values. If None, the model will be fitted later. Defaults to None.
        standardize_x: bool
            Whether to standardize input features. Defaults to None.
        standardize_y: bool
            Whether to standardize target values. Defaults to None.
        boosting_type: str
            Type of boosting to use. Options are "gbdt", "dart", "goss", "rf". Defaults
            to "gbdt".
        num_leaves: int
            Maximum number of leaves in one tree. Defaults to 31.
        max_depth: int
            Maximum depth of the tree. -1 means no limit. Defaults to -1.
        learning_rate: float
            Learning rate shrinks the contribution of each new tree. Defaults to 0.1.
        n_estimators: int
            The number of boosting stages to be run. Defaults to 100.
        subsample_for_bin: int
            Number of samples for constructing bins. Defaults to 200000.
        objective: str | None
            Objective function to be optimized. If None, defaults to "regression".
            Defaults to None.
        class_weight: dict | str | None
            Class weights for multi-class classification. If None, all classes are
            assumed to have equal weight. Defaults to None.
        min_split_gain: float
            Minimum loss reduction required to make a further partition on a leaf node.
            Defaults to 0.0.
        min_child_weight: float
            Minimum sum of instance weight (hessian) needed in a child.
            Defaults to 0.001.
        min_child_samples: int
            Minimum number of data points in a child. Defaults to 20.
        subsample: float
            Fraction of samples to be used for fitting the individual base learners.
            Defaults to 1.0.
        colsample_bytree: float
            Fraction of features to be used for fitting the individual base learners.
            Defaults to 1.0.
        reg_alpha: float
            L1 regularization term on weights. Defaults to 0.0.
        reg_lambda: float
            L2 regularization term on weights. Defaults to 0.0.
        random_seed: int | None
            Random seed for reproducibility. If None, no seed is set.
        n_jobs: int | None
            Number of parallel threads used to run LightGBM. If None, uses all available
            cores. Defaults to 1.
        importance_type: str
            Type of feature importance to be calculated. Options are "split", "gain",
            "cover", "total_gain", "total_cover". Defaults to "split".
        verbose: int
            Verbosity of the output. -1 means no output, 0 means warnings only,
            1 means info, 2 means debug.
        device: DeviceLike
            Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to "cpu".
        """
        _, _ = x, y  # ignore unused arguments
        TorchDeviceMixin.__init__(self, device=device)
        self.x_transform = StandardizeTransform() if standardize_x else None
        self.y_transform = StandardizeTransform() if standardize_y else None
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
        self.supports_grad = False
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
        """LightGBM does not support multi-output."""
        return False

    def _fit(self, x: TensorLike, y: TensorLike):
        x_np, y_np = self._convert_to_numpy(x, y)
        self.n_features_in_ = x_np.shape[1]
        self.model_.fit(x_np, y_np)

    def _predict(self, x: TensorLike, with_grad: bool) -> TensorLike:
        if with_grad:
            msg = "Gradient calculation is not supported."
            raise ValueError(msg)
        y_pred = self.model_.predict(x)
        assert not isinstance(y_pred, spmatrix | list)
        _, y = self._convert_to_tensors(x, y_pred)
        return y

    @staticmethod
    def get_tune_params():
        """Return a dictionary of hyperparameters to tune."""
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
