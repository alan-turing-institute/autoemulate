"""Test whether emulators are sklearn-compatible."""

from sklearn.utils.estimator_checks import parametrize_with_checks, _yield_all_checks
from autoemulate.emulators import (
    RandomForest,
    GaussianProcessSk,
    NeuralNetSk,
    GaussianProcess,
    RadialBasis,
    NeuralNetTorch,
    SecondOrderPolynomial,
    GradientBoosting,
    SupportVectorMachines,
)
from functools import partial


@parametrize_with_checks(
    [  # GaussianProcess()
        SupportVectorMachines(),
        # RandomForest(random_state=42),
        # GaussianProcessSk(random_state=1337),
        # NeuralNetSk(random_state=13),
        # RadialBasis(),
        # GradientBoosting(random_state=42),
        # #NeuralNetTorch(random_state=42), # fails because it subclasses
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)


# checks for SecondOrderPolynomial
# needs minimum sample size, which increases with increasing number of features
# thus excluding several tests
excluded_tests = [
    "check_estimators_dtypes",
    "check_dtype_object",
    "check_regressor_multioutput",
    "check_regressors_no_decision_function",
    "check_regressors_int",
    "check_fit2d_1sample",
    "check_regressors_train",
]


@parametrize_with_checks([SecondOrderPolynomial()])
def test_sklearn_compatible_estimator(estimator, check):
    # Access the original function if 'check' is a functools.partial
    check_func = check.func if isinstance(check, partial) else check

    if check_func.__name__ not in excluded_tests:
        check(estimator)
