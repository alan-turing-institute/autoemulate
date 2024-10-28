import numpy as np
import pandas as pd
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


def test_sobol_analysis(model_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }

    Si = sobol_analysis(model_1d, problem)
    assert isinstance(Si, dict)
    assert "y1" in Si
    assert all(
        key in Si["y1"] for key in ["S1", "S1_conf", "S2", "S2_conf", "ST", "ST_conf"]
    )


def test_sobol_analysis_2d(model_2d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    Si = sobol_analysis(model_2d, problem)
    assert isinstance(Si, dict)
    assert ["y1", "y2"] == list(Si.keys())


@pytest.fixture
def sobol_results_1d(model_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    return sobol_analysis(model_1d, problem)


def test_sobol_results_to_df(sobol_results_1d):
    df = sobol_results_to_df(sobol_results_1d)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == [
        "output",
        "parameter",
        "index",
        "value",
        "confidence",
    ]
    assert ["X1", "X2", "X1-X2"] in df["parameter"].unique()
    assert all(isinstance(x, float) for x in df["value"])
    assert all(isinstance(x, float) for x in df["confidence"])
