from .base import Emulator, GaussianProcessEmulator
from .ensemble import EnsembleMLP, EnsembleMLPDropout
from .gaussian_process.exact import (
    GaussianProcessCorrelatedMatern32,
    GaussianProcessCorrelatedRBF,
    GaussianProcessMatern32,
    GaussianProcessRBF,
)
from .lightgbm import LightGBM
from .nn.mlp import MLP
from .polynomials import PolynomialRegression
from .radial_basis_functions import RadialBasisFunctions
from .random_forest import RandomForest
from .svm import SupportVectorMachine
from .transformed.base import TransformedEmulator

DEFAULT_EMULATORS: list[type[Emulator]] = [
    GaussianProcessMatern32,
    GaussianProcessRBF,
    RadialBasisFunctions,
    PolynomialRegression,
    MLP,
    EnsembleMLP,
]

# listing non pytorch emulators as we do not expect this list to grow
NON_PYTORCH_EMULATORS: list[type[Emulator]] = [
    LightGBM,
    SupportVectorMachine,
    RandomForest,
]

ALL_EMULATORS: list[type[Emulator]] = [
    *DEFAULT_EMULATORS,
    *NON_PYTORCH_EMULATORS,
    GaussianProcessCorrelatedMatern32,
    GaussianProcessCorrelatedRBF,
    EnsembleMLPDropout,
]

PYTORCH_EMULATORS: list[type[Emulator]] = [
    emulator for emulator in ALL_EMULATORS if emulator not in NON_PYTORCH_EMULATORS
]
GAUSSIAN_PROCESS_EMULATORS: list[type[Emulator]] = [
    emulator
    for emulator in ALL_EMULATORS
    if issubclass(emulator, GaussianProcessEmulator)
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
    "GaussianProcessCorrelatedMatern32",
    "GaussianProcessCorrelatedRBF",
    "GaussianProcessMatern32",
    "GaussianProcessRBF",
    "LightGBM",
    "PolynomialRegression",
    "RadialBasisFunctions",
    "RandomForest",
    "SupportVectorMachine",
    "TransformedEmulator",
]
