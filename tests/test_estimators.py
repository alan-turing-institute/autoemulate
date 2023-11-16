"""Test whether emulators are sklearn-compatible."""

from sklearn.utils.estimator_checks import parametrize_with_checks, _yield_all_checks
from autoemulate.emulators import (
    RandomForest,
    GaussianProcessSk,
    NeuralNetSk,
    GaussianProcess,
    RadialBasis,
    NeuralNetTorch,
)
from functools import partial


@parametrize_with_checks(
    [  # GaussianProcess(),
        RandomForest(random_state=42),
        GaussianProcessSk(random_state=1337),
        NeuralNetSk(random_state=13),
        RadialBasis(),
        # NeuralNetTorch(random_state=42), # fails because it subclasses
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)
