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
    SecondOrderPolynomial().model_name: SecondOrderPolynomial(),
    RBF().model_name: RBF(),
    RandomForest().model_name: RandomForest(),
    GradientBoosting().model_name: GradientBoosting(),
    GaussianProcessSk().model_name: GaussianProcessSk(),
    SupportVectorMachines().model_name: SupportVectorMachines(),
    XGBoost().model_name: XGBoost(),
    NeuralNetTorch(module="mlp").model_name: NeuralNetTorch(module="mlp"),
    NeuralNetTorch(module="rbf").model_name: NeuralNetTorch(module="rbf"),
    NeuralNetSk().model_name: NeuralNetSk(),
    # "GaussianProcess": GaussianProcess,
}
