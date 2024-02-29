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
        input_size: int = None,
        output_size: int = None,
        random_state: int = None,
    ):
        super(TorchModule, self).__init__()
        if random_state is not None:
            set_random_seed(random_state)
        self.module_name = module_name
        self.input_size = input_size
        self.output_size = output_size
        self.initialized = False

    def initialize(self):
        raise NotImplementedError("initialize method not implemented.")

    @staticmethod
    def get_grid_params(search_type: str = "random"):
        """Return the hyperparameter search space for the module"""
        raise NotImplementedError("get_grid_params method not implemented.")

    def forward(self, X: torch.Tensor):
        raise NotImplementedError("forward method not implemented.")
