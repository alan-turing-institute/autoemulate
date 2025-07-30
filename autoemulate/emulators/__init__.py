from .base import Emulator
from .ensemble import EnsembleMLP, EnsembleMLPDropout
from .gaussian_process.exact import GaussianProcess, GaussianProcessCorrelated
from .lightgbm import LightGBM
from .nn.mlp import MLP
from .radial_basis_functions import RadialBasisFunctions
from .random_forest import RandomForest
from .svm import SupportVectorMachine
from .transformed.base import TransformedEmulator

# from .neural_processes.conditional_neural_process import CNPModule

ALL_EMULATORS: list[type[Emulator]] = [
    GaussianProcess,
    GaussianProcessCorrelated,
    LightGBM,
    # CNPModule,
    SupportVectorMachine,
    RadialBasisFunctions,
    RandomForest,
    MLP,
    EnsembleMLP,
    EnsembleMLPDropout,
]

EMULATOR_REGISTRY = {em_cls.model_name().lower(): em_cls for em_cls in ALL_EMULATORS}
EMULATOR_REGISTRY_SHORT_NAME = {em_cls.short_name(): em_cls for em_cls in ALL_EMULATORS}


def get_emulator_class(name: str) -> type[Emulator]:
    """
    Get the emulator class by name.

    Parameters
    ----------
    name: str
        The name of the emulator class.

    Returns
    -------
    type[Emulator] | None
        The emulator class if found, None otherwise.
    """
    emulator_cls = EMULATOR_REGISTRY.get(
        name.lower()
    ) or EMULATOR_REGISTRY_SHORT_NAME.get(name.lower())

    if emulator_cls is None:
        raise ValueError(
            f"Unknown emulator name: {name}.Available: {list(EMULATOR_REGISTRY.keys())}"
        )

    return emulator_cls


__all__ = [
    "MLP",
    "EnsembleMLP",
    "EnsembleMLPDropout",
    "GaussianProcess",
    "GaussianProcessCorrelated",
    "LightGBM",
    "RadialBasisFunctions",
    "RandomForest",
    "SupportVectorMachine",
    "TransformedEmulator",
]
