from .gaussian_process import GaussianProcess
from .gaussian_process_sk import GaussianProcessSk
from .neural_net_sk import NeuralNetSk
from .random_forest import RandomForest
from .neural_net_torch import NeuralNetTorch
from .second_order_polynomials import SecondOrderPolynomial
from .gradient_boosting import GradientBoosting
from .support_vector_machines import SupportVectorMachines
from .xgboost import XGBoost
from .rbf import RBF
from .polynomials import SecondOrderPolynomials

MODEL_REGISTRY = {
    "SecondOrderPolynomials": SecondOrderPolynomials,
    "SecondOrderPolynomial": SecondOrderPolynomial,
    # "RBF": RBF,
    # "RandomForest": RandomForest,
    # "GradientBoosting": GradientBoosting,
    # "GaussianProcessSk": GaussianProcessSk,
    # "SupportVectorMachines": SupportVectorMachines,
    # "XGBoost": XGBoost,
    # "NeuralNetSk": NeuralNetSk,
    # # "NeuralNetTorch": NeuralNetTorch,
    # "GaussianProcess": GaussianProcess,
}
