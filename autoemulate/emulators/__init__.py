from .gaussian_process import GaussianProcess
from .gaussian_process_sk import GaussianProcessSk
from .gradient_boosting import GradientBoosting
from .neural_net_sk import NeuralNetSk
from .neural_net_torch import NeuralNetTorch
from .polynomials import SecondOrderPolynomial
from .random_forest import RandomForest
from .rbf import RBF
from .support_vector_machines import SupportVectorMachines
from .xgboost import XGBoost

# REGISTRY keys are the class names (i.e. type(model).__name__)
MODEL_REGISTRY = {
    "SecondOrderPolynomial": SecondOrderPolynomial(),
    "RBF": RBF(),
    "RandomForest": RandomForest(),
    "GradientBoosting": GradientBoosting(),
    "GaussianProcessSk": GaussianProcessSk(),
    "SupportVectorMachines": SupportVectorMachines(),
    "XGBoost": XGBoost(),
    "NeuralNetSk": NeuralNetSk(),
    "NeuralNetTorch": NeuralNetTorch(module="mlp"),
    # "GaussianProcess": GaussianProcess,
}
