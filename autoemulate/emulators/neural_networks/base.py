from typing import Tuple
from typing import Union

import torch
from torch import nn

from autoemulate.utils import set_random_seed


class TorchModule(nn.Module):
    """
    Basic structure of a `module` for `NeuralNetRegressor`
    """

    def __init__(
        self,
        module_name: str,
        input_size: Union[int, Tuple] = None,
        output_size: int = None,
        random_state: int = None,
    ):
        super(TorchModule, self).__init__()

        if random_state is not None:
            set_random_seed(random_state)
        self.module_name = module_name
        self.input_size = input_size if type(input_size) == tuple else (input_size,)
        self.output_size = output_size
        self.check_input_size()

    def check_input_size(self):
        raise NotImplementedError("check_input_size method not implemented.")

    def get_grid_params(self, search_type: str = "random"):
        """Return the hyperparameter search space for the module"""
        raise NotImplementedError("get_grid_params method not implemented.")

    def forward(self, X: torch.Tensor):
        raise NotImplementedError("forward method not implemented.")
