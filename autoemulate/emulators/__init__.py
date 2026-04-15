from .base import Emulator
from .conformal import ConformalMLP
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
from .registry import Registry, _default_registry, get_emulator_class, register
from .svm import SupportVectorMachine
from .transformed.base import TransformedEmulator

# Module-level constants for backward compatibility and simplified public access
DEFAULT_EMULATORS = _default_registry._default_emulators
NON_PYTORCH_EMULATORS = _default_registry._non_pytorch_emulators
ALL_EMULATORS = _default_registry._all_emulators
PYTORCH_EMULATORS = _default_registry._pytorch_emulators
GAUSSIAN_PROCESS_EMULATORS = _default_registry._gaussian_process_emulators
EMULATOR_REGISTRY = _default_registry._emulator_registry
EMULATOR_REGISTRY_SHORT_NAME = _default_registry._emulator_registry_short_name

__all__ = [
    "MLP",
    "ConformalMLP",
    "Emulator",
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
    "Registry",
    "SupportVectorMachine",
    "TransformedEmulator",
    "get_emulator_class",
    "register",
]
