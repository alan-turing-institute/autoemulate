# experimental version of a PyTorch neural network emulator wrapped in Skorch
# to make it compatible with scikit-learn. Works with cross_validate and GridSearchCV,
# but doesn't pass tests, because we're subclassing

import numpy as np
import skorch
import torch
from scipy.stats import loguniform
from skopt.space import Integer, Real
from skorch import NeuralNet, NeuralNetRegressor
from torch import nn


class InputShapeSetter(skorch.callbacks.Callback):
    """Callback to set input and output layer sizes dynamically."""

    def on_train_begin(
        self,
        net,
        X: torch.Tensor | np.ndarray = None,
        y: torch.Tensor | np.ndarray = None,
        **kwargs
    ):
        if X.ndim == 1:
            raise ValueError("Reshape your data")
        output_size = 1 if y.ndim == 1 else y.shape[1]
        net.set_params(module__input_size=X.shape[1], module__output_size=output_size)


# Step 1: Define the PyTorch Module for the MLP
class MLPModule(nn.Module):
    def __init__(self, input_size=10, hidden_layer_sizes=(50,), output_size=1):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.output_layer = None
        self.input_size = input_size
        if input_size is not None and output_size is not None:
            self.build_module(input_size, output_size, hidden_layer_sizes)

    def build_module(self, input_size, output_size, hidden_layer_sizes):
        hs = [input_size] + list(hidden_layer_sizes)
        for i in range(len(hs) - 1):
            self.hidden_layers.append(nn.Linear(hs[i], hs[i + 1]))
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, X: torch.Tensor):
        X = X.to(torch.float32)
        for layer in self.hidden_layers:
            X = torch.relu(layer(X))
        if self.output_layer is not None:
            X = self.output_layer(X)
        return X


# Step 2: Create the Skorch wrapper for the NeuralNetRegressor
class NeuralNetTorch(NeuralNetRegressor):
    def __init__(
        self,
        module=MLPModule,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.01,
        batch_size=128,
        max_epochs=10,
        module__input_size=10,
        module__output_size=1,
        module__hidden_layer_sizes=(100,),
        optimizer__weight_decay=0.0001,
        iterator_train__shuffle=True,
        callbacks=[InputShapeSetter()],
        train_split=False,  # to run cross_validate without splitting the data
        verbose=0,
        **kwargs
    ):
        if "random_state" in kwargs:
            torch.random.manual_seed(kwargs.pop("random_state"))
        super(NeuralNetTorch, self).__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            module__input_size=module__input_size,
            module__output_size=module__output_size,
            module__hidden_layer_sizes=module__hidden_layer_sizes,
            optimizer__weight_decay=optimizer__weight_decay,
            iterator_train__shuffle=iterator_train__shuffle,
            callbacks=callbacks,
            train_split=train_split,
            verbose=verbose,
            **kwargs
        )

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
        return {"multioutput": True, "stateless": True}

    def fit(self, X, y, **fit_params):
        y = y.astype(np.float32)
        estimator = super(NeuralNetRegressor, self).fit(X, y, **fit_params)
        if not hasattr(estimator, "n_features_in_"):
            setattr(estimator, "n_features_in_", X.shape[1])
        return estimator

    def predict(self, X):
        if X.ndim == 1:
            raise ValueError("Reshape your data")
        return super(NeuralNetRegressor, self).predict(X)

    def predict_proba(self, X):
        if X.ndim == 1:
            raise ValueError("Reshape your data")
        return super(NeuralNetRegressor, self).predict_proba(X)
