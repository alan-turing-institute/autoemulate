# scikit-learn estimator tests
# new emulator models should pass these tests to be fully compatible with scikit-learn
# see https://scikit-learn.org/stable/developers/develop.html
# and https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py
from functools import partial

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import _yield_all_checks
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.estimator_checks import check_estimators_dtypes
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import set_random_state

from autoemulate.emulators import ConditionalNeuralProcess
from autoemulate.emulators import GaussianProcessMOGP
from autoemulate.emulators import GaussianProcessSklearn
from autoemulate.emulators import GaussianProcessTorch
from autoemulate.emulators import GradientBoosting
from autoemulate.emulators import LightGBM
from autoemulate.emulators import NeuralNetSk
from autoemulate.emulators import RadialBasisFunctions
from autoemulate.emulators import RandomForest
from autoemulate.emulators import SecondOrderPolynomial
from autoemulate.emulators import SupportVectorMachines


@parametrize_with_checks(
    [
        SupportVectorMachines(),
        RandomForest(random_state=42),
        GaussianProcessSklearn(random_state=1337),
        NeuralNetSk(random_state=13),
        GradientBoosting(random_state=42),
        SecondOrderPolynomial(),
        RadialBasisFunctions(),
        LightGBM(),
        ConditionalNeuralProcess(random_state=42),
        GaussianProcessTorch(random_state=42),
    ]
)
def test_check_estimator(estimator, check):
    check(estimator)
