# experimental version of a PyTorch neural network emulator wrapped in Skorch
# to make it compatible with scikit-learn. Works with cross_validate and GridSearchCV,
# but doesn't pass tests, because we're subclassing
import warnings

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import NotFittedError
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback

from ..types import ArrayLike
from ..types import List
from ..types import Literal
from ..types import MatrixLike
from ..types import Optional
from ..types import Self
from ..types import Union
from autoemulate.emulators.neural_networks import get_module
from autoemulate.utils import set_random_seed


class InputShapeSetter(Callback):
    """Callback to set input and output layer sizes dynamically.

    This is needed to support dynamic input size.

    Parameters
    ----------
    net : NeuralNetRegressor
        The neural network regressor.
    X : Optional[Union[torch.Tensor, np.ndarray]], optional
        The input samples, by default None.
    y : Optional[Union[torch.Tensor, np.ndarray]], optional
        The target values, by default None.
    """

    def on_train_begin(
        self,
        net: NeuralNetRegressor,
        X: Optional[Union[torch.Tensor, np.ndarray]] = None,
        y: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs,
    ):
        if hasattr(net, "n_features_in_") and net.n_features_in_ != X.shape[-1]:
            raise ValueError(
                f"Mismatch number of features, "
                f"expected {net.n_features_in_}, received {X.shape[-1]}."
            )
        if not hasattr(net, "n_features_in_"):
            output_size = 1 if y.ndim == 1 else y.shape[1]
            net.set_params(
                module__input_size=X.shape[1], module__output_size=output_size
            )


class NeuralNetTorch(NeuralNetRegressor):
    """
    Wrap PyTorch modules in Skorch to make them compatible with scikit-learn.

    module__input_size and module__output_size must be provided to define the
    input and output dimension of the data.

    Parameters
    ----------
    module : str, optional
        The module to use, by default "mlp"
    criterion : torch.nn.Module, optional
        The loss function, by default torch.nn.MSELoss
    optimizer : torch.optim.Optimizer, optional
        The optimizer to use, by default torch.optim.AdamW
    lr : float, optional
        The learning rate, by default 1e-3
    batch_size : int, optional
        The batch size, by default 128
    max_epochs : int, optional
        The maximum number of epochs, by default 1
    module__input_size : int, optional
        The input size, by default 2
    module__output_size : int, optional
        The output size, by default 1
    optimizer__weight_decay : float, optional
        The weight decay, by default 0.0
    iterator_train__shuffle : bool, optional
        Whether to shuffle the training data, by default True
    callbacks : List[Callback], optional
        The callbacks to use, by default [InputShapeSetter()]
    train_split : bool, optional
        Whether to split the data, by default False
    verbose : int, optional
        The verbosity level, by default 0
    random_state : int, optional
        The random state, by default None
    **kwargs
        Additional keyword arguments to pass to the neural network regressor.
    """

    def __init__(
        self,
        module: str = "mlp",
        criterion: torch.nn.Module = torch.nn.MSELoss,  # TODO: verify type here
        optimizer: torch.optim.Optimizer = torch.optim.AdamW,  # TODO: verify type here
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 1,
        module__input_size: int = 2,
        module__output_size: int = 1,
        optimizer__weight_decay: float = 0.0,
        iterator_train__shuffle: bool = True,
        callbacks: List[Callback] = [InputShapeSetter()],
        train_split: bool = False,  # to run cross_validate without splitting the data
        verbose: int = 0,
        **kwargs,
    ):
        if "random_state" in kwargs:
            setattr(self, "random_state", kwargs.pop("random_state"))
            set_random_seed(self.random_state)
        super().__init__(
            module=get_module(module),
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            module__input_size=module__input_size,
            module__output_size=module__output_size,
            optimizer__weight_decay=optimizer__weight_decay,
            iterator_train__shuffle=iterator_train__shuffle,
            callbacks=callbacks,
            train_split=train_split,
            verbose=verbose,
            **kwargs,
        )
        self.initialize()

    def set_params(self, **params) -> Self:
        """Set the parameters of the neural network regressor.

        Parameters
        ----------
        **params
            The parameters to set. If `random_state` is provided, it is set as
            an attribute of the class.

        Returns
        -------
        self
            The neural network regressor with the new parameters.
        """
        if "random_state" in params:
            random_state = params.pop("random_state")
            if hasattr(self, "random_state"):
                self.random_state = random_state
            else:
                setattr(self, "random_state", random_state)
            set_random_seed(self.random_state)
            self.initialize()
        return super().set_params(**params)

    def initialize_module(self, reason: Optional[str] = None) -> Self:
        """Initializes the module.

        Parameters
        ----------
        reason : str, optional
            The reason for initializing the module, by default None

        Returns
        -------
        self
            The neural network regressor with the initialized module.
        """
        kwargs = self.get_params_for("module")
        if hasattr(self, "random_state"):
            kwargs["random_state"] = self.random_state
        module = self.initialized_instance(self.module, kwargs)
        self.module_ = module
        return self

    def get_grid_params(
        self,
        search_type: Literal[
            "random", "bayes"
        ] = "random",  # TODO: Verify search_type types
    ) -> dict[str, Union[bool, dict[str, str]]]:
        return self.module_.get_grid_params(search_type)

    def __sklearn_is_fitted__(self) -> bool:
        """Private method to check if the model is fitted. This is used by scikit-learn."""
        return hasattr(self, "n_features_in_")

    def _more_tags(self) -> dict[str, Union[bool, dict[str, str]]]:
        """Returns more tags for the estimator.

        Returns
        -------
        dict
            The tags of the neural network regressor.
        """
        return {
            "multioutput": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_no_attributes_set_in_init": "skorch initialize attributes in __init__.",
                "check_regressors_no_decision_function": "skorch NeuralNetRegressor class implements the predict_proba.",
                "check_parameters_default_constructible": "skorch NeuralNet class callbacks parameter expects a list of callables.",
                "check_methods_subset_invariance": "the assert_allclose check is done in float64 while Torch models operate in float32. The max absolute difference is 1.1920929e-07.",
                "check_dont_overwrite_parameters": "the change of public attribute module__input_size is needed to support dynamic input size.",
            },
        }

    def check_data(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """Check the data for the neural network regressor.

        Parameters
        ----------
        X : np.ndarray
            The input samples.
        y : np.ndarray, optional
            The target values, by default None

        Returns
        -------
        np.ndarray
            The input samples.
        """
        if isinstance(y, np.ndarray):
            if np.iscomplex(X).any():
                raise ValueError("Complex data not supported")
        else:
            X = np.array(X)
        X = X.astype(np.float32)
        if issparse(X):
            raise ValueError("Sparse data not supported")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("NaNs and inf values are not supported.")
        if X.size == 0:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum "
                f"of {self.module__input_size} is required."
            )
        if X.ndim == 1:
            raise ValueError("Reshape your data")
        if y is not None:
            if isinstance(y, np.ndarray):
                if np.iscomplex(y).any():
                    raise ValueError("Complex data not supported")
            else:
                y = np.array(y)
            y = y.astype(np.float32)
            if np.isnan(y).any() or np.isinf(y).any():
                raise ValueError("NaNs and inf values are not supported.")
            if y.ndim == 2 and y.shape[1] == 1:
                warnings.warn(
                    DataConversionWarning(
                        "A column-vector y was passed when a 1d array was expected"
                    )
                )
                y = np.squeeze(y, axis=-1)
        return X, y

    def fit_loop(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        epochs: Optional[int] = None,
        **fit_params,
    ) -> Self:
        """Loop to fit the neural network regressor.

        Parameters
        ----------
        X : np.ndarray
            The input samples.
        y : np.ndarray, optional
            The target values, by default None
        epochs : int, optional
            The number of epochs, by default None
        **fit_params
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        self
            The neural network regressor fitted to the data.
        """
        X, y = self.check_data(X, y)
        return super().fit_loop(X, y, epochs, **fit_params)

    def partial_fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        classes: ArrayLike = None,
        **fit_params,
    ) -> Self:
        """Partially fit the neural network regressor.

        Parameters
        ----------
        X : np.ndarray
            The input samples.
        y : np.ndarray, optional
            The target values, by default None
        classes : np.ndarray, optional
            The classes, by default None
        **fit_params
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        self
            The neural network regressor partially fitted to the data.
        """
        X, y = self.check_data(X, y)
        return super().partial_fit(X, y, classes, **fit_params)

    def fit(self, X: MatrixLike, y: Optional[ArrayLike], **fit_params) -> Self:
        """
        Fits the emulator to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).
        **fit_params
            Additional keyword arguments to pass to the fit method.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = self.check_data(X, y)
        return super().fit(X, y, **fit_params)

    @torch.inference_mode()
    def predict_proba(self, X: MatrixLike) -> np.ndarray:
        """
        Predicts the output of the emulator for a given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples, n_features)
            The predicted values.
        """
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError
        dtype = X.dtype if hasattr(X, "dtype") else None
        X, _ = self.check_data(X)
        y_pred = super().predict_proba(X)
        if dtype is not None:
            y_pred = y_pred.astype(dtype)
        return y_pred

    def infer(self, x: torch.Tensor, **fit_params) -> np.ndarray:
        """Predicts the output of the emulator for a given input.

        Parameters
        ----------
        x : torch.Tensor
            The input samples.
        **fit_params
            Additional keyword arguments to pass to the infer method.

        Returns
        -------
        y_pred : np.ndarray
            The predicted values.
        """
        if not hasattr(self, "n_features_in_"):
            setattr(self, "n_features_in_", x.size(1))
        y_pred = super().infer(x, **fit_params)
        if self.module__output_size == 1 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred
