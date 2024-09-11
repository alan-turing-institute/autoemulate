from .conditional_neural_process import ConditionalNeuralProcess
from .gaussian_process_torch import GaussianProcessTorch
from .gradient_boosting import GradientBoosting
from .light_gbm import LightGBM
from .polynomials import SecondOrderPolynomial
from .radial_basis_functions import RadialBasisFunctions
from .random_forest import RandomForest
from .support_vector_machines import SupportVectorMachines

# from .gaussian_process_mogp import GaussianProcessMOGP
# from .gaussian_process import GaussianProcess
# from .neural_net_sk import NeuralNetSk

MODEL_REGISTRY = {
    SecondOrderPolynomial().model_name: SecondOrderPolynomial(),
    RadialBasisFunctions().model_name: RadialBasisFunctions(),
    RandomForest().model_name: RandomForest(),
    GradientBoosting().model_name: GradientBoosting(),
    LightGBM().model_name: LightGBM(),
    SupportVectorMachines().model_name: SupportVectorMachines(),
    GaussianProcessTorch().model_name: GaussianProcessTorch(),
    ConditionalNeuralProcess().model_name: ConditionalNeuralProcess(),
    # currently not in use
    # NeuralNetSk().model_name: NeuralNetSk(),
    # GaussianProcess().model_name: GaussianProcess(),
    # GaussianProcessMOGP().model_name: GaussianProcessMOGP(),
}
