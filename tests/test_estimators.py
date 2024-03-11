# scikit-learn estimator tests
# new emulator models should pass these tests to be fully compatible with scikit-learn
# see https://scikit-learn.org/stable/developers/develop.html
# and https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py
from functools import partial

from sklearn.utils.estimator_checks import _yield_all_checks
from sklearn.utils.estimator_checks import parametrize_with_checks

from autoemulate.emulators import GaussianProcess
from autoemulate.emulators import GaussianProcessMOGP
from autoemulate.emulators import GradientBoosting
from autoemulate.emulators import LightGBM
from autoemulate.emulators import NeuralNetSk
from autoemulate.emulators import NeuralNetTorch
from autoemulate.emulators import RadialBasisFunctions
from autoemulate.emulators import RandomForest
from autoemulate.emulators import SecondOrderPolynomial
from autoemulate.emulators import SupportVectorMachines


@parametrize_with_checks(
    [
        SupportVectorMachines(),
        RandomForest(random_state=42),
        GaussianProcess(random_state=1337),
        NeuralNetSk(random_state=13),
        GradientBoosting(random_state=42),
        SecondOrderPolynomial(),
        RadialBasisFunctions(),
        NeuralNetTorch(module="mlp", random_state=42),
        NeuralNetTorch(module="rbf", random_state=42),
        LightGBM(),
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)
