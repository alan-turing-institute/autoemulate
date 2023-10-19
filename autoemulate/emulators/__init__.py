from .base import Emulator
from .gaussian_process import GaussianProcess
from .gaussian_process2 import GaussianProcess2
from .neural_network import NeuralNetwork
from .random_forest import RandomForest

MODEL_REGISTRY = {
    "GaussianProcess": GaussianProcess,
    "GaussianProcess2": GaussianProcess2,
    "NeuralNetwork": NeuralNetwork,
    "RandomForest": RandomForest,
}
