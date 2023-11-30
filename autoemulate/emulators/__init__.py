from .gaussian_process import GaussianProcess
from .gaussian_process_sk import GaussianProcessSk
from .neural_net_sk import NeuralNetSk
from .random_forest import RandomForest
from .radial_basis import RadialBasis
from .neural_net_torch import NeuralNetTorch
from .second_order_polynomials import SecondOrderPolynomial
from .gradient_boosting import GradientBoosting

MODEL_REGISTRY = {
    # "GaussianProcess": GaussianProcess,
    "GaussianProcessSk": GaussianProcessSk,
    "NeuralNetSk": NeuralNetSk,
    "RandomForest": RandomForest,
    "GradientBoosting": GradientBoosting,
    # "RadialBasis": RadialBasis,
    "SecondOrderPolynomial": SecondOrderPolynomial,
    # "NeuralNetTorch": NeuralNetTorch,
}
