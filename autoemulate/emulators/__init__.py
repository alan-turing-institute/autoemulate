from .gaussian_process import GaussianProcess
from .gaussian_process_mogp import GaussianProcessMOGP
from .gradient_boosting import GradientBoosting
from .neural_net_sk import NeuralNetSk
from .neural_net_torch import NeuralNetTorch
from .polynomials import SecondOrderPolynomial
from .random_forest import RandomForest
from .rbf import RadialBasisFunctions
from .support_vector_machines import SupportVectorMachines

# from .light_gbm import LightGBM

MODEL_REGISTRY = {
    SecondOrderPolynomial().model_name: SecondOrderPolynomial(),
    RadialBasisFunctions().model_name: RadialBasisFunctions(),
    RandomForest().model_name: RandomForest(),
    GradientBoosting().model_name: GradientBoosting(),
    GaussianProcess().model_name: GaussianProcess(),
    SupportVectorMachines().model_name: SupportVectorMachines(),
    NeuralNetSk().model_name: NeuralNetSk(),
}
