from typing import Tuple

import torch
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

    def forward(self, X: torch.Tensor):
        return self.model(X)
