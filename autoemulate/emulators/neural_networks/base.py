import torch
from torch import nn

from ...types import Optional
from autoemulate.utils import set_random_seed


class TorchModule(nn.Module):
    """
    Basic structure of a `module` for `NeuralNetRegressor`
    """

    def __init__(
        self,
        module_name: str,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        super(TorchModule, self).__init__()
        if random_state is not None:
            set_random_seed(random_state)
        self.module_name = module_name
        self.input_size = input_size
        self.output_size = output_size

    def get_grid_params(self, search_type: str = "random"):
        """Return the hyperparameter search space for the module"""
        raise NotImplementedError("get_grid_params method not implemented.")

    def forward(self, X: torch.Tensor):
        """Forward pass through the module"""
        raise NotImplementedError("forward method not implemented.")
