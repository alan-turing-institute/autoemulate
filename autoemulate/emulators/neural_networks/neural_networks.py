import torch
from torch import nn

from autoemulate.utils import set_random_seed

_MODULES = dict()


def register(name):
    def add_to_dict(fn):
        global _MODULES
        _MODULES[name] = fn
        return fn

    return add_to_dict


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

    def get_grid_params(self, search_type: str = "random"):
        """Return the hyperparameter search space for the module"""
        raise NotImplementedError("get_grid_params method not implemented.")

    def forward(self, X: torch.Tensor):
        raise NotImplementedError("forward method not implemented.")


def get_module(module: str | TorchModule, module_args) -> TorchModule:
    """
    Return the module instance for NeuralNetRegressor. If `module` is a string,
    then initialize a TorchModule with the same registered name. If `module` is
    already a TorchModule, then return it as is.
    """
    if isinstance(module, TorchModule):
        return module
    if module not in _MODULES:
        raise NotImplementedError(f"Module {module} not implemented.")
    return _MODULES[module](**module_args)
