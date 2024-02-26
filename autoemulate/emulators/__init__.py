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
    "poly": SecondOrderPolynomial(),
    "rbf": RBF(),
    "rf": RandomForest(),
    "gb": GradientBoosting(),
    "gp": GaussianProcessSk(),
    "svm": SupportVectorMachines(),
    "xgboost": XGBoost(),
    "mlpsk": NeuralNetSk(),
    "mlp": NeuralNetTorch(module="mlp"),
    # "GaussianProcess": GaussianProcess,
}
