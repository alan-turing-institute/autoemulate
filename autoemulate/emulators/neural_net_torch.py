# experimental version of a PyTorch neural network emulator wrapped in Skorch
# to make it compatible with scikit-learn. Works with cross_validate and GridSearchCV,
# but doesn't pass tests, because we're subclassing
import random
import warnings
from typing import List, Tuple

import numpy as np
import torch
from scipy.sparse import issparse
from scipy.stats import loguniform
from sklearn.exceptions import DataConversionWarning
from skopt.space import Integer, Real
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback
from torch import nn

from autoemulate.emulators.neural_networks import get_module
from autoemulate.utils import set_random_seed


class InputShapeSetter(Callback):
    """Callback to set input and output layer sizes dynamically."""

    def on_train_begin(
        self,
        net: NeuralNetRegressor,
        X: torch.Tensor | np.ndarray = None,
        y: torch.Tensor | np.ndarray = None,
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


# Step 2: Create the Skorch wrapper for the NeuralNetRegressor
class NeuralNetTorch(NeuralNetRegressor):
    def __init__(
        self,
        module: str = "mlp",
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr: float = 0.01,
        batch_size: int = 128,
        max_epochs: int = 1,
        module__input_size: int = 2,
        module__output_size: int = 1,
        optimizer__weight_decay: float = 0.0001,
        iterator_train__shuffle: bool = True,
        callbacks: List[Callback] = [InputShapeSetter()],
        train_split: bool = False,  # to run cross_validate without splitting the data
        verbose: int = 0,
        **kwargs,
    ):
        if "random_state" in kwargs:
            setattr(self, "random_state", kwargs.pop("random_state"))
            set_random_seed(self.random_state)
        # get all arguments for module initialization
        module_args = {
            "input_size": module__input_size,
            "output_size": module__output_size,
        }
        for k, v in kwargs.items():
            if k.startswith("module__"):
                module_args[k.replace("module__", "")] = v
                kwargs.pop(k)

        super().__init__(
            module=get_module(module, module_args),
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

    def set_params(self, **params):
        if "random_state" in params:
            random_state = params.pop("random_state")
            if hasattr(self, "random_state"):
                self.random_state = random_state
            else:
                setattr(self, "random_state", random_state)
            set_random_seed(self.random_state)
            self._initialize_module()
            self._initialize_optimizer()
        return super().set_params(**params)

    def initialize_module(self, reason=None):
        kwargs = self.get_params_for("module")
        if hasattr(self, "random_state"):
            kwargs["random_state"] = self.random_state
        module = self.initialized_instance(self.module, kwargs)
        self.module_ = module
        return self

    def get_grid_params(self, search_type="random"):
        param_space_random = {
            "lr": loguniform(1e-4, 1e-2),
            "max_epochs": [10, 20, 30],
            "module__hidden_sizes": [
                (50,),
                (100,),
                (100, 50),
                (100, 100),
                (200, 100),
            ],
        }

        param_space_bayes = {
            "lr": Real(1e-4, 1e-2, prior="log-uniform"),
            "max_epochs": Integer(10, 30),
        }

        if search_type == "random":
            param_space = param_space_random
        elif search_type == "bayes":
            param_space = param_space_bayes

        return param_space

    def __sklearn_is_fitted__(self):
        return hasattr(self, "n_features_in_")

    def _more_tags(self):
        return {
            "multioutput": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_no_attributes_set_in_init": "skorch initialize attributes in __init__.",
                "check_regressors_no_decision_function": "skorch NeuralNetRegressor class implements the predict_proba.",
                "check_parameters_default_constructible": "skorch NeuralNet class callbacks parameter expects a list of callables.",
                "check_dont_overwrite_parameters": "the change of public attribute module__input_size is needed to support dynamic input size.",
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

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        X, y = self.check_data(X, y)
        return super().fit_loop(X, y, epochs, **fit_params)

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        X, y = self.check_data(X, y)
        return super().partial_fit(X, y, classes, **fit_params)

    def fit(self, X, y, **fit_params):
        X, y = self.check_data(X, y)
        return super().fit(X, y, **fit_params)

    @torch.inference_mode()
    def predict_proba(self, X):
        dtype = X.dtype if hasattr(X, "dtype") else None
        X, _ = self.check_data(X)
        y_pred = super().predict_proba(X)
        if dtype is not None:
            y_pred = y_pred.astype(dtype)
        return y_pred

    def infer(self, x: torch.Tensor, **fit_params):
        if not hasattr(self, "n_features_in_"):
            setattr(self, "n_features_in_", x.size(1))
        y_pred = super().infer(x, **fit_params)
        if self.module__output_size == 1 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)
        return y_pred
