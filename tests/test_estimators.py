# scikit-learn estimator tests
# new emulator models should pass these tests to be fully compatible with scikit-learn
# see https://scikit-learn.org/stable/developers/develop.html
# and https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py

from functools import partial

from sklearn.utils.estimator_checks import _yield_all_checks, parametrize_with_checks

from autoemulate.emulators import (
    RBF,
    GaussianProcess,
    GaussianProcessSk,
    GradientBoosting,
    NeuralNetSk,
    NeuralNetTorch,
    RandomForest,
    SecondOrderPolynomial,
    SupportVectorMachines,
    XGBoost,
)


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
        NeuralNetTorch(random_state=42),
        # GaussianProcess()
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)
