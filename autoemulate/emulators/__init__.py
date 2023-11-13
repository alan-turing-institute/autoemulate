from .base import Emulator
from .gaussian_process import GaussianProcess
from .gaussian_process_sk import GaussianProcessSk
from .neural_net_sk import NeuralNetSk
from .random_forest import RandomForest
from .radial_basis import RadialBasis
from .neural_net_torch import NeuralNetTorch

MODEL_REGISTRY = {
    # "GaussianProcess": GaussianProcess,
    "GaussianProcessSk": GaussianProcessSk,
    "NeuralNetSk": NeuralNetSk,
    "RandomForest": RandomForest,
    "RadialBasis": RadialBasis,
    "NeuralNetTorch": NeuralNetTorch,
}
