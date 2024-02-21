from typing import Tuple
from typing import Union

import numpy as np
import torch
from scipy.stats import loguniform
from skopt.space import Integer
from skopt.space import Real
from torch import nn

from autoemulate.emulators.neural_networks.base import TorchModule


class CNN1dModule(TorchModule):
    """1-dimensional CNN module for NeuralNetRegressor"""

    def __init__(
        self,
        input_size: Union[int, Tuple] = None,
        output_size: int = None,
        random_state: int = None,
        hidden_layers: int = 1,
        hidden_size: int = 100,
        kernel_size: int = 3,
        dropout: float = 0.0,
        hidden_activation: Tuple[callable] = nn.ReLU,
    ):
        super(CNN1dModule, self).__init__(
            module_name="cnn1d",
            input_size=input_size,
            output_size=output_size,
            random_state=random_state,
        )
        in_channel, length = input_size
        assert hidden_layers >= 1
        modules = []
        for _ in range(hidden_layers):
            modules.append(
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            modules.append(hidden_activation())
            modules.append(nn.Dropout1d(p=dropout))
            in_channel = hidden_size
        in_channel = in_channel * length
        modules.append(nn.Flatten())
        modules.append(nn.Linear(in_features=in_channel, out_features=output_size))
        self.model = nn.Sequential(*modules)

    def check_input_size(self):
        assert (
            len(self.input_size) == 2
        ), "CNN1dModule input_size should has format (features, length)"

    def get_grid_params(self, search_type: str = "random"):
        param_space = {
            "max_epochs": np.arange(10, 110, 10).tolist(),
            "batch_size": np.arange(2, 128, 2).tolist(),
            "module__hidden_layers": np.arange(1, 4).tolist(),
            "module__hidden_size": np.arange(50, 250, 50).tolist(),
            "module__kernel_size": np.arange(2, 6).tolist(),
            "module__hidden_activation": [
                nn.ReLU,
                nn.Tanh,
                nn.Sigmoid,
                nn.GELU,
            ],
            "module__dropout": np.arange(0, 0.55, 0.05).tolist(),
            "optimizer": [torch.optim.AdamW, torch.optim.SGD],
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
