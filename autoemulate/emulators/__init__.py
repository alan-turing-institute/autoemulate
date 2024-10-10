from ..model_registry import ModelRegistry
from .conditional_neural_process import ConditionalNeuralProcess
from .gaussian_process import GaussianProcess
from .gaussian_process_mogp import GaussianProcessMOGP
from .gaussian_process_sklearn import GaussianProcessSklearn
from .gaussian_process_torch import GaussianProcessTorch
from .gradient_boosting import GradientBoosting
from .light_gbm import LightGBM
from .neural_net_sk import NeuralNetSk
from .polynomials import SecondOrderPolynomial
from .radial_basis_functions import RadialBasisFunctions
from .random_forest import RandomForest
from .support_vector_machines import SupportVectorMachines

model_registry = ModelRegistry()

# core models
model_registry.register_model(
    SecondOrderPolynomial().model_name, SecondOrderPolynomial, is_core=True
)
model_registry.register_model(
    RadialBasisFunctions().model_name, RadialBasisFunctions, is_core=True
)
model_registry.register_model(RandomForest().model_name, RandomForest, is_core=True)
model_registry.register_model(
    GradientBoosting().model_name, GradientBoosting, is_core=True
)
model_registry.register_model(LightGBM().model_name, LightGBM, is_core=True)
model_registry.register_model(
    SupportVectorMachines().model_name, SupportVectorMachines, is_core=True
)
model_registry.register_model(
    GaussianProcessTorch().model_name, GaussianProcessTorch, is_core=True
)
model_registry.register_model(
    ConditionalNeuralProcess().model_name, ConditionalNeuralProcess, is_core=True
)
model_registry.register_model(
    GaussianProcess().model_name, GaussianProcess, is_core=True
)

# non-core models
model_registry.register_model(
    GaussianProcessSklearn().model_name, GaussianProcessSklearn, is_core=False
)
model_registry.register_model(NeuralNetSk().model_name, NeuralNetSk, is_core=False)
