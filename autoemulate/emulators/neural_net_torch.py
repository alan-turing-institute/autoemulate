# experimental version of a PyTorch neural network emulator wrapped in Skorch
# to make it compatible with scikit-learn. Works with cross_validate and GridSearchCV,
# but doesn't pass tests, because we're subclassing

import random
from typing import List, Tuple

import numpy as np
import skorch
import torch
from scipy.sparse import issparse
from scipy.stats import loguniform
from skopt.space import Integer, Real
from skorch import NeuralNet, NeuralNetRegressor
from skorch.callbacks import Callback
from torch import nn


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for Python, Numpy and PyTorch.
    Args:
        seed: int, the random seed to use.
        deterministic: bool, use "deterministic" algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


class InputShapeSetter(Callback):
    """Callback to set input and output layer sizes dynamically."""

    def on_train_begin(
        self,
        net,
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


# Step 1: Define the PyTorch Module for the MLP
class MLPModule(nn.Module):
    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
        hidden_size: int = 100,
    ):
        super(MLPModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

    def forward(self, X: torch.Tensor):
        return self.model(X)


# Step 2: Create the Skorch wrapper for the NeuralNetRegressor
class NeuralNetTorch(NeuralNetRegressor):
    def __init__(
        self,
        module: nn.Module = MLPModule,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr: float = 0.01,
        batch_size: int = 128,
        max_epochs: int = 10,
        module__input_size: int = 10,
        module__output_size: int = 1,
        module__hidden_size: int = 100,
        optimizer__weight_decay: float = 0.0001,
        iterator_train__shuffle: bool = True,
        callbacks: List[Callback] = [InputShapeSetter()],
        train_split: bool = False,  # to run cross_validate without splitting the data
        verbose: int = 0,
        **kwargs,
    ):
        if "random_state" in kwargs:
            set_random_seed(kwargs.pop("random_state"))
        super(NeuralNetTorch, self).__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            module__input_size=module__input_size,
            module__output_size=module__output_size,
            module__hidden_size=module__hidden_size,
            optimizer__weight_decay=optimizer__weight_decay,
            iterator_train__shuffle=iterator_train__shuffle,
            callbacks=callbacks,
            train_split=train_split,
            verbose=verbose,
            **kwargs,
        )

    def set_params(self, **params):
        if "random_state" in params:
            set_random_seed(params.pop("random_state"))
            self._initialize_module()
            self._initialize_criterion()
            self._initialize_optimizer()
            return self
        else:
            return super(NeuralNetTorch, self).set_params(**params)

    def get_grid_params(self, search_type="random"):
        param_grid_random = {
            "lr": loguniform(1e-4, 1e-2),
            "max_epochs": [10, 20, 30],
            "module__hidden_layer_sizes": [
                (50,),
                (100,),
                (100, 50),
                (100, 100),
                (200, 100),
            ],
        }

        param_grid_bayes = {
            "lr": Real(1e-4, 1e-2, prior="log-uniform"),
            "max_epochs": Integer(10, 30),
        }

        if search_type == "random":
            param_grid = param_grid_random
        elif search_type == "bayes":
            param_grid = param_grid_bayes

        return param_grid

    def _more_tags(self):
        return {"multioutput": True, "stateless": True, "poor_score": True}

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
        return X, y

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        X, y = self.check_data(X, y)
        if not hasattr(self, "n_features_in_"):
            setattr(self, "n_features_in_", X.shape[1])
        return super(NeuralNetRegressor, self).fit_loop(X, y, epochs, **fit_params)

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        X, y = self.check_data(X, y)
        return super(NeuralNetRegressor, self).partial_fit(X, y, classes, **fit_params)

    def fit(self, X, y, **fit_params):
        X, y = self.check_data(X, y)
        super(NeuralNetRegressor, self).fit(X, y, **fit_params)
        return self

    @torch.inference_mode()
    def predict_proba(self, X):
        dtype = X.dtype if hasattr(X, "dtype") else None
        X, _ = self.check_data(X)
        y_pred = super(NeuralNetRegressor, self).predict_proba(X)
        if self.module__output_size == 1 and y_pred.shape[-1] == 1:
            y_pred = np.squeeze(y_pred, axis=-1)
        if dtype is not None:
            y_pred = y_pred.astype(dtype)
        return y_pred
