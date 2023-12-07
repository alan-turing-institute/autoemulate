from .gaussian_process import GaussianProcess
from .gaussian_process_sk import GaussianProcessSk
from .neural_net_sk import NeuralNetSk
from .random_forest import RandomForest
from .radial_basis import RadialBasis
from .neural_net_torch import NeuralNetTorch
from .second_order_polynomials import SecondOrderPolynomial
from .gradient_boosting import GradientBoosting
from .support_vector_machines import SupportVectorMachines
from .xgboost import XGBoost

MODEL_REGISTRY = {
    "SecondOrderPolynomial": SecondOrderPolynomial,
    "RandomForest": RandomForest,
    "GradientBoosting": GradientBoosting,
    "GaussianProcessSk": GaussianProcessSk,
    "SupportVectorMachines": SupportVectorMachines,
    "XGBoost": XGBoost,
    "NeuralNetSk": NeuralNetSk,
    "RadialBasis": RadialBasis,
    # "NeuralNetTorch": NeuralNetTorch,
    # "GaussianProcess": GaussianProcess,
}
