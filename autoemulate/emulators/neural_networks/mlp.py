from typing import Tuple

import torch
from scipy.stats import loguniform
from skopt.space import Integer
from skopt.space import Real
from torch import nn

from autoemulate.emulators.neural_networks.neural_networks import register
from autoemulate.emulators.neural_networks.neural_networks import TorchModule


@register("mlp")
class MLPModule(TorchModule):
    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
        random_state: int = None,
        hidden_sizes: Tuple[int] = (100,),
    ):
        super(MLPModule, self).__init__(
            module_name="mlp",
            input_size=input_size,
            output_size=output_size,
            random_state=random_state,
        )
        modules = []
        for hidden_size in hidden_sizes:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            modules.append(nn.ReLU())
            input_size = hidden_size
        modules.append(nn.Linear(in_features=input_size, out_features=output_size))
        self.model = nn.Sequential(*modules)

    def get_grid_params(self, search_type: str = "random"):
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

        match search_type:
            case "random":
                param_space = param_space_random
            case "bayes":
                param_space = param_space_bayes
            case _:
                raise ValueError(f"Invalid search type: {search_type}")

        return param_space

    def forward(self, X: torch.Tensor):
        return self.model(X)
