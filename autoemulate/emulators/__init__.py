from .conditional_neural_process import ConditionalNeuralProcess
from .gaussian_process import GaussianProcess
from .gaussian_process_mogp import GaussianProcessMOGP
from .gaussian_process_torch import GaussianProcessTorch
from .gradient_boosting import GradientBoosting
from .light_gbm import LightGBM
from .neural_net_sk import NeuralNetSk
from .polynomials import SecondOrderPolynomial
from .radial_basis_functions import RadialBasisFunctions
from .random_forest import RandomForest
from .support_vector_machines import SupportVectorMachines

MODEL_REGISTRY = {
    SecondOrderPolynomial().model_name: SecondOrderPolynomial(),
    RadialBasisFunctions().model_name: RadialBasisFunctions(),
    RandomForest().model_name: RandomForest(),
    GradientBoosting().model_name: GradientBoosting(),
    LightGBM().model_name: LightGBM(),
    SupportVectorMachines().model_name: SupportVectorMachines(),
    GaussianProcess().model_name: GaussianProcess(),
    NeuralNetSk().model_name: NeuralNetSk(),
    ConditionalNeuralProcess().model_name: ConditionalNeuralProcess(),
    GaussianProcessTorch().model_name: GaussianProcessTorch(),
}
