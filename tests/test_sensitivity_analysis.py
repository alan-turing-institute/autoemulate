import numpy as np
import pytest
from sklearn.datasets import make_regression

from autoemulate.emulators import RandomForest
from autoemulate.experimental_design import LatinHypercube
from autoemulate.sensitivity_analysis import sobol_analysis
from autoemulate.sensitivity_analysis import sobol_results_to_df
from autoemulate.simulations.projectile import simulate_projectile
from autoemulate.simulations.projectile import simulate_projectile_multioutput


@pytest.fixture
def Xy_1d():
    lhd = LatinHypercube([(-5.0, 1.0), (0.0, 1000.0)])
    X = lhd.sample(100)
    y = np.array([simulate_projectile(x) for x in X])
    return X, y


@pytest.fixture
def Xy_2d():
    lhd = LatinHypercube([(-5.0, 1.0), (0.0, 1000.0)])
    X = lhd.sample(100)
    y = np.array([simulate_projectile_multioutput(x) for x in X])
    return X, y


@pytest.fixture
def model_1d(Xy_1d):
    X, y = Xy_1d
    rf = RandomForest()
    rf.fit(X, y)
    return rf


@pytest.fixture
def model_2d(Xy_2d):
    X, y = Xy_2d
    rf = RandomForest()
    rf.fit(X, y)
    return rf


def test_sensitivity_analysis(model_2d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }

    Si = sobol_analysis(model_2d, problem)
    df = sobol_results_to_df(Si)
    print(df)
