# experimental version of a PyTorch neural network emulator wrapped in Skorch
# to make it compatible with scikit-learn. Works with cross_validate and GridSearchCV,
# but doesn't pass tests, because we're subclassing
import warnings
from typing import List

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import NotFittedError
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback

from autoemulate.emulators.neural_networks import get_module
from autoemulate.utils import set_random_seed


class NeuralNetTorch(NeuralNetRegressor):
    """
    Wrap PyTorch modules in Skorch to make them compatible with scikit-learn.

    module__input_size and module__output_size must be provided to define the
    input and output dimension of the data.
    """

    def __init__(
        self,
        module: str = "MultiLayerPerceptron",
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.AdamW,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 1,
        module__input_size: int = None,
        module__output_size: int = None,
        optimizer__weight_decay: float = 0.0,
        iterator_train__shuffle: bool = True,
        callbacks: List[Callback] = None,
        train_split: bool = False,  # to run cross_validate without splitting the data
        verbose: int = 0,
        **kwargs,
    ):
        self.module_name = module
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
        if module__input_size is not None and module__output_size is not None:
            self.initialize()

    def input_output_sizes_are_set(self) -> bool:
        return (
            self.module__input_size is not None and self.module__output_size is not None
        )

    def initialize_optimizer(self, triggered_directly=None):
        if self.optimizer == torch.optim.LBFGS and hasattr(
            self, "optimizer__weight_decay"
        ):
            # LBFGS does not support weight_decay
            del self.optimizer__weight_decay
        return super().initialize_optimizer(triggered_directly)

    def initialize_module(self, reason=None):
        kwargs = self.get_params_for("module")
        if hasattr(self, "random_state"):
            kwargs["random_state"] = self.random_state
        if self.input_output_sizes_are_set():
            self.module_ = self.initialized_instance(self.module, kwargs)
        return self

    def get_grid_params(self, search_type="random"):
        return self.module.get_grid_params(search_type)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_features_in_")

    @property
    def model_name(self):
        return f"Torch{self.module_name}"

    def _more_tags(self):
        return {
            "multioutput": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_no_attributes_set_in_init": "skorch initialize attributes in __init__.",
                "check_regressors_no_decision_function": "skorch NeuralNetRegressor class implements the predict_proba.",
                "check_parameters_default_constructible": "skorch NeuralNet class callbacks parameter expects a list of callables.",
                "check_dont_overwrite_parameters": "the change of public attribute module__input_size is needed to support dynamic input size.",
                "check_estimators_overwrite_params": "in order to support dynamic input and output size, we have to overwrite module__input_size and module__output_size during fit.",
                "check_estimators_empty_data_messages": "the error message cannot print module__input_size the module has not been initialized",
                "check_set_params": "_params_to_validate must be a list or set, while check_set_params set it to a float which causes AttributeError",
            },
        }

    def check_data(self, X: np.ndarray, y: np.ndarray = None):
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

    def check_initialized(self, X: np.ndarray, y: np.ndarray):
        if not self.warm_start or not self.initialized_:
            self.module__input_size = X.shape[1]
            self.module__output_size = y.shape[1] if y.ndim > 1 else 1
            self.initialize()

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        X, y = self.check_data(X, y)
        return super().fit_loop(X, y, epochs, **fit_params)

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        X, y = self.check_data(X, y)
        self.check_initialized(X, y)
        return super().partial_fit(X, y, classes, **fit_params)

    def fit(self, X, y, **fit_params):
        X, y = self.check_data(X, y)
        self.check_initialized(X, y)
        return super().fit(X, y, **fit_params)

    @torch.inference_mode()
    def predict_proba(self, X):
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError
        dtype = X.dtype if hasattr(X, "dtype") else None
        X, _ = self.check_data(X)
        y_pred = super().predict_proba(X)
        if dtype is not None:
            y_pred = y_pred.astype(dtype)
        return y_pred

    def infer(self, x: torch.Tensor, **fit_params):
        if hasattr(self, "n_features_in_"):
            if self.n_features_in_ != x.shape[-1]:
                raise ValueError(
                    f"Mismatch number of features, "
                    f"expected {self.n_features_in_}, received {x.shape[-1]}."
                )
        else:
            setattr(self, "n_features_in_", x.size(1))
        y_pred = super().infer(x, **fit_params)
        if self.module__output_size == 1 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred
