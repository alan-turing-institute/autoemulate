from .gaussian_process import GaussianProcess
from .gaussian_process_sk import GaussianProcessSk
from .neural_net_sk import NeuralNetSk
from .random_forest import RandomForest
from .neural_net_torch import NeuralNetTorch
from .gradient_boosting import GradientBoosting
from .support_vector_machines import SupportVectorMachines
from .xgboost import XGBoost
from .rbf import RBF
from .polynomials import SecondOrderPolynomial

# REGISTRY keys are the class names (i.e. type(model).__name__)
MODEL_REGISTRY = {
    "SecondOrderPolynomial": SecondOrderPolynomial,
    "RBF": RBF,
    "RandomForest": RandomForest,
    "GradientBoosting": GradientBoosting,
    "GaussianProcessSk": GaussianProcessSk,
    "SupportVectorMachines": SupportVectorMachines,
    "XGBoost": XGBoost,
    "NeuralNetSk": NeuralNetSk,
    "NeuralNetTorch": NeuralNetTorch,
    # "GaussianProcess": GaussianProcess,
}
