from autoemulate.emulators.neural_networks import TorchModule
from autoemulate.emulators.neural_networks.mlp import MLPModule


def get_module(module: str | TorchModule, module_args) -> TorchModule:
    """
    Return the module instance for NeuralNetRegressor. If `module` is a string,
    then initialize a TorchModule with the same registered name. If `module` is
    already a TorchModule, then return it as is.
    """
    if not isinstance(module, TorchModule):
        match module:
            case "mlp":
                module = MLPModule(**module_args)
            case _:
                raise NotImplementedError(f"Module {module} not implemented.")
    return module
