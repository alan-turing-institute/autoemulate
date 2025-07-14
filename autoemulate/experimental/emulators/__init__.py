from .ensemble import EnsembleMLP, EnsembleMLPDropout
from .gaussian_process.exact import GaussianProcessExact, GaussianProcessExactCorrelated
from .lightgbm import LightGBM

# from .neural_processes.conditional_neural_process import CNPModule
from .nn.mlp import MLP
from .radial_basis_functions import RadialBasisFunctions
from .random_forest import RandomForest
from .svm import SupportVectorMachine
from .transformed.base import TransformedEmulator

ALL_EMULATORS = [
    GaussianProcessExact,
    GaussianProcessExactCorrelated,
    LightGBM,
    # CNPModule,
    SupportVectorMachine,
    RadialBasisFunctions,
    RandomForest,
    MLP,
    EnsembleMLP,
    EnsembleMLPDropout,
]

__all__ = [
    "MLP",
    "EnsembleMLP",
    "EnsembleMLPDropout",
    "GaussianProcessExact",
    "GaussianProcessExactCorrelated",
    "LightGBM",
    "RadialBasisFunctions",
    "RandomForest",
    "SupportVectorMachine",
    "TransformedEmulator",
]
