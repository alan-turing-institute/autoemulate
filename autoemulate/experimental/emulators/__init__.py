from .ensemble import EnsembleMLP, EnsembleMLPDropout
from .gaussian_process.exact import GaussianProcessExact
from .lightgbm import LightGBM

# from .neural_processes.conditional_neural_process import CNPModule
from .nn.mlp import MLP
from .random_forest import RandomForest
from .svm import SupportVectorMachine

ALL_EMULATORS = [
    GaussianProcessExact,
    LightGBM,
    # CNPModule,
    SupportVectorMachine,
    RandomForest,
    MLP,
    EnsembleMLP,
    EnsembleMLPDropout,
]
