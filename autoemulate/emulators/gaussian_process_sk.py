from scipy.stats import loguniform
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real

from ..types import ArrayLike
from ..types import Literal
from ..types import MatrixLike
from ..types import Optional
from ..types import Self
from autoemulate.utils import suppress_convergence_warnings


class GaussianProcessSk(BaseEstimator, RegressorMixin):
    """Gaussian Process Emulator.

    Wraps GaussianProcessRegressor from scikit-learn.
    """

    def __init__(
        self,
        kernel: RBF = RBF(),  # TODO: Is this the correct type then?
        alpha: float = 1e-10,
        optimizer: str = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 20,
        normalize_y: bool = True,
        copy_X_train: bool = True,
        random_state: Optional[int] = None,
    ):
        """Initializes a GaussianProcess object.

        Parameters
        ----------
        kernel : kernel object
            The kernel specifying the covariance function of the GP.
        alpha : float
            Value added to the diagonal of the kernel matrix during fitting.
            This can prevent a potential numerical issue during fitting.
        optimizer : str
            The optimizer to use for optimizing the kernel's parameters.
        n_restarts_optimizer : int
            The number of restarts of the optimizer for finding the kernel's parameters.
        normalize_y : bool
            Whether to normalize the target values.
        copy_X_train : bool
            Whether to make a copy of the training data.
        random_state : int
            The random state to use for the emulator.
        """
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    # TODO: Should X be MatrixLike here?
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
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
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.model_ = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
        )
        with suppress_convergence_warnings():
            self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: MatrixLike, return_std: bool = False) -> ArrayLike:
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
        return self.model_.predict(X, return_std=return_std)

    def get_grid_params(
        self, search_type: Literal["random", "bayes"] = "random"
    ) -> dict[str, list]:
        """Returns the grid parameters of the emulator."""
        param_space_random = {
            "kernel": [
                RBF(),
                Matern(),
                RationalQuadratic(),
                # DotProduct(),
            ],
            "optimizer": ["fmin_l_bfgs_b"],
            "alpha": loguniform(1e-10, 1e-2),
            "normalize_y": [True],
        }
        param_space_bayes = {
            # "kernel": Categorical([RBF(), Matern()]), # unhashable type
            "optimizer": Categorical(["fmin_l_bfgs_b"]),
            "alpha": Real(1e-10, 1e-2, prior="log-uniform"),
            "normalize_y": Categorical([True]),
        }

        if search_type == "random":
            param_space = param_space_random
        elif search_type == "bayes":
            param_space = param_space_bayes

        # TODO: Should this raise an error if the search type is not recognised?

        return param_space

    def _more_tags(self) -> dict:
        """Returns more tags for the estimator.

        Returns
        -------
        dict
            The multioutput tag.
        """
        return {"multioutput": True}
