from typing import Tuple

import numpy as np
import torch
from scipy.stats import loguniform
from skopt.space import Integer
from skopt.space import Real
from torch import nn

from autoemulate.emulators.neural_networks.base import TorchModule


class MLPModule(TorchModule):
    """Multi-layer perceptron module for NeuralNetRegressor"""

    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
        random_state: int = None,
        hidden_layers: int = 1,
        hidden_size: int = 100,
        hidden_activation: Tuple[callable] = nn.ReLU,
    ):
        super(MLPModule, self).__init__(
            module_name="mlp",
            input_size=input_size,
            output_size=output_size,
            random_state=random_state,
        )
        modules = []
        assert hidden_layers >= 1
        for _ in range(hidden_layers):
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            modules.append(hidden_activation())
            input_size = hidden_size
        modules.append(nn.Linear(in_features=input_size, out_features=output_size))
        self.model = nn.Sequential(*modules)

    def get_grid_params(self, search_type: str = "random"):
        param_space = {
            "max_epochs": np.arange(10, 110, 10).tolist(),
            "batch_size": np.arange(2, 128, 2).tolist(),
            "module__hidden_layers": np.arange(1, 4).tolist(),
            "module__hidden_size": np.arange(50, 250, 50).tolist(),
            "module__hidden_activation": [
                nn.ReLU,
                nn.Tanh,
                nn.Sigmoid,
                nn.GELU,
            ],
            "optimizer__weight_decay": (1 / 10 ** np.arange(1, 9)).tolist(),
        }
        match search_type:
            case "random":
                param_space |= {"lr": loguniform(1e-06, 1e-2)}
            case "bayes":
                param_space |= {"lr": Real(1e-06, 1e-2, prior="log-uniform")}
            case _:
                raise ValueError(f"Invalid search type: {search_type}")

        return param_space

    def forward(self, X: torch.Tensor):
        return self.model(X)
