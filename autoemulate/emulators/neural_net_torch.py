# experimental version of a PyTorch neural network emulator wrapped in Skorch
# to make it compatible with scikit-learn. Works with cross_validate and GridSearchCV,
# but doesn't pass tests, because we're subclassing

import torch
import numpy as np
import skorch
from torch import nn
from skorch import NeuralNetRegressor
from scipy.stats import loguniform


class InputShapeSetter(skorch.callbacks.Callback):
    """Callback to set input and output layer sizes dynamically."""

    def on_train_begin(self, net, X, y):
        output_size = 1 if y.ndim == 1 else y.shape[1]
        net.set_params(module__input_size=X.shape[1], module__output_size=output_size)


# Step 1: Define the PyTorch Module for the MLP
class MLPModule(nn.Module):
    def __init__(self, input_size=10, hidden_layer_sizes=(50,), output_size=1):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.output_layer = None

        if input_size is not None and output_size is not None:
            self.build_module(input_size, output_size, hidden_layer_sizes)

    def build_module(self, input_size, output_size, hidden_layer_sizes):
        hs = [input_size] + list(hidden_layer_sizes)
        for i in range(len(hs) - 1):
            self.hidden_layers.append(nn.Linear(hs[i], hs[i + 1]))
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, X):
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
        super().__init__(
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

    def get_grid_params(self):
        return {
            "model__lr": loguniform(1e-4, 1e-2),
            "model__max_epochs": [10, 20, 30],
            "model__module__hidden_layer_sizes": [
                (50,),
                (100,),
                (100, 50),
                (100, 100),
                (200, 100),
            ],
        }

    def _more_tags(self):
        return {"multioutput": True}
