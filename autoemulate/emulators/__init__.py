from .base import Emulator
from .gaussian_process import GaussianProcess
from .gaussian_process_sk import GaussianProcessSk
from .neural_network import NeuralNetwork
from .random_forest import RandomForest

MODEL_REGISTRY = {
    "GaussianProcess": GaussianProcess,
    "GaussianProcessSk": GaussianProcessSk,
    "NeuralNetwork": NeuralNetwork,
    "RandomForest": RandomForest,
}
