from autoemulate.emulators.neural_networks import TorchModule
from autoemulate.emulators.neural_networks.mlp import MLPModule


def get_module(module: str | TorchModule) -> TorchModule:
    """
    Return the module class for NeuralNetRegressor. If `module` is
    already a TorchModule, then return it as is.
    """
    if not isinstance(module, str):
        return module
    match module:
        case "mlp":
            module = MLPModule
        case _:
            raise NotImplementedError(f"Module {module} not implemented.")
    return module
