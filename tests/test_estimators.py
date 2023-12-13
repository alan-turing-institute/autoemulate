# scikit-learn estimator tests
# new emulator models should pass these tests to be fully compatible with scikit-learn
# see https://scikit-learn.org/stable/developers/develop.html
# and https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py

from sklearn.utils.estimator_checks import parametrize_with_checks, _yield_all_checks
from autoemulate.emulators import (
    RandomForest,
    GaussianProcessSk,
    NeuralNetSk,
    GaussianProcess,
    NeuralNetTorch,
    SecondOrderPolynomial,
    GradientBoosting,
    SupportVectorMachines,
    XGBoost,
    RBF,
)
from functools import partial


@parametrize_with_checks(
    [
        SupportVectorMachines(),
        RandomForest(random_state=42),
        GaussianProcessSk(random_state=1337),
        NeuralNetSk(random_state=13),
        GradientBoosting(random_state=42),
        SecondOrderPolynomial(),
        XGBoost(),
        RBF(),
        # NeuralNetTorch(random_state=42), # fails because it subclasses
        # GaussianProcess()
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)
