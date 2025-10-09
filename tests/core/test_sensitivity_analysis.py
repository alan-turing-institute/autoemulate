import numpy as np
import pandas as pd
import pytest
import torch
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis
from autoemulate.emulators.random_forest import RandomForest
from autoemulate.simulations.projectile import Projectile, ProjectileMultioutput


@pytest.fixture
def xy_1d():
    sim = Projectile()
    x = sim.sample_inputs(100)
    y, _ = sim.forward_batch(x)
    return x, y


@pytest.fixture
def xy_2d():
    sim = ProjectileMultioutput()
    x = sim.sample_inputs(100)
    y, _ = sim.forward_batch(x)
    return x, y


@pytest.fixture
def model_1d(xy_1d):
    x, y = xy_1d
    rf = RandomForest(x, y)
    rf.fit(x, y)
    return rf


@pytest.fixture
def model_2d(xy_2d):
    x, y = xy_2d
    rf = RandomForest(x, y)
    rf.fit(x, y)
    return rf


@pytest.fixture
def sa_1d(model_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    return SensitivityAnalysis(model_1d, problem=problem)


# test problem checking ----------------------------------------------------------------
def test_check_problem(sa_1d):
    problem = sa_1d._check_problem(sa_1d.problem)
    assert problem["num_vars"] == 2
    assert problem["names"] == ["c", "v0"]
    assert problem["bounds"] == [(-5.0, 1.0), (0.0, 1000.0)]


def test_check_problem_invalid(model_1d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
    }
    with pytest.raises(ValueError, match="problem must contain 'bounds'."):
        SensitivityAnalysis(model_1d, problem=problem)


def test_check_problem_bounds(model_1d):
    problem = {"num_vars": 2, "names": ["c", "v0"], "bounds": [(-5.0, 1.0)]}
    with pytest.raises(ValueError, match="Length of 'bounds' must match 'num_vars'."):
        SensitivityAnalysis(model_1d, problem=problem)


# test output name retrieval --------------------------------------------------
def test_get_output_names_default(sa_1d):
    output_names = sa_1d._get_output_names(1)
    assert output_names == ["y1"]


def test_get_output_names_custom(model_2d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
        "output_names": ["lol1", "lol2"],
    }
    sa = SensitivityAnalysis(model_2d, problem=problem)
    output_names = sa._get_output_names(2)
    assert output_names == ["lol1", "lol2"]


def test_get_output_names_invalid(model_2d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
        "output_names": "lol",
    }
    sa = SensitivityAnalysis(model_2d, problem=problem)
    with pytest.raises(ValueError, match="'output_names' must be a list of strings."):
        sa._get_output_names(1)


# test Sobol analysis ------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "expected_names",
    [["c", "v0", "c", "v0", ["c", "v0"]]],
)
def test_sobol_results_to_df(sa_1d, expected_names):
    df = sa_1d.run("sobol")

    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == [
        "output",
        "parameter",
        "index",
        "value",
        "confidence",
    ]
    assert expected_names == df["parameter"].to_list()
    assert all(isinstance(x, float) for x in df["value"])
    assert all(isinstance(x, float) for x in df["confidence"])


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sobol_analysis_2d(model_2d):
    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    sa = SensitivityAnalysis(model_2d, problem=problem)
    df = sa.run("sobol")

    assert isinstance(df, pd.DataFrame)
    assert df["output"].nunique() == 2
    assert "y1" in list(df["output"].unique())
    assert "y2" in list(df["output"].unique())

    assert len(sa.top_n_sobol_params(df, 1)) == 1
    assert len(sa.top_n_sobol_params(df, 2)) == 2
    assert sa.top_n_sobol_params(df, 1)[0] == "c"
    assert sa.top_n_sobol_params(df, 2)[-1] == "v0"


# test _generate_problem -----------------------------------------------------


def test_generate_problem(sa_1d):
    x = np.array([[0, 0], [1, 1], [2, 2]])
    problem = sa_1d._generate_problem(torch.tensor(x))
    assert problem["num_vars"] == 2
    assert problem["names"] == ["x1", "x2"]
    assert problem["bounds"] == [[0, 2], [0, 2]]
