import numpy as np
import pandas as pd
import pytest
import torch
from autoemulate.core.sensitivity_analysis import SensitivityAnalysis
from autoemulate.emulators import GaussianProcessRBF as GaussianProcess
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


# test prediction with variance ------------------------------------------------


@pytest.mark.filterwarnings("ignore::gpytorch.utils.warnings.GPInputWarning")
def test_predict_with_variance(xy_1d):
    """Test that _predict can return variance for UQ-enabled emulators."""
    x, y = xy_1d
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    sa = SensitivityAnalysis(gp, problem=problem)

    # Test with return_variance=False (default)
    y_pred = sa._predict(x.numpy())
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (100, 1)

    # Test with return_variance=True
    y_pred, y_var = sa._predict(x.numpy(), return_variance=True)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_var, np.ndarray)
    assert y_pred.shape == (100, 1)
    assert y_var.shape == (100, 1)
    assert np.all(y_var >= 0)  # Variance should be non-negative


def test_predict_with_variance_no_uq(xy_1d):
    """Test that _predict returns None variance for non-UQ emulators."""
    x, y = xy_1d
    rf = RandomForest(x, y)
    rf.fit(x, y)

    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    sa = SensitivityAnalysis(rf, problem=problem)

    # Test with return_variance=True
    y_pred, y_var = sa._predict(x.numpy(), return_variance=True)
    assert isinstance(y_pred, np.ndarray)
    assert y_var is None


# test sensitivity analysis with prediction variance ----------------------------


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    ("method", "n_samples", "expected_columns"),
    [
        (
            "sobol",
            256,
            ["output", "parameter", "index", "value", "confidence"],
        ),
        (
            "morris",
            10,
            ["output", "parameter", "mu", "mu_star", "sigma", "mu_star_conf"],
        ),
    ],
)
def test_with_prediction_variance(xy_1d, method, n_samples, expected_columns):
    """Test sensitivity analysis with prediction variance incorporation."""
    x, y = xy_1d
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    sa = SensitivityAnalysis(gp, problem=problem)

    # Run with prediction variance
    df = sa.run(
        method=method,
        n_samples=n_samples,
        include_prediction_variance=True,
        n_bootstrap=10,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_columns
    # Check confidence intervals are valid
    if method == "sobol":
        assert all(isinstance(x, float) for x in df["value"])
        assert all(df["confidence"] >= 0)
    else:  # morris
        assert all(df["mu_star_conf"] >= 0)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(("method", "n_samples"), [("sobol", 256), ("morris", 10)])
def test_with_variance_vs_without(xy_1d, method, n_samples):
    """Compare sensitivity analysis results with and without prediction variance."""
    x, y = xy_1d
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    sa = SensitivityAnalysis(gp, problem=problem)

    # Standard analysis
    df_standard = sa.run(method=method, n_samples=n_samples)

    # With variance
    df_with_var = sa.run(
        method=method,
        n_samples=n_samples,
        include_prediction_variance=True,
        n_bootstrap=10,
    )

    # Both should have the same structure
    assert df_standard.columns.tolist() == df_with_var.columns.tolist()
    assert len(df_standard) == len(df_with_var)

    # With low bootstrap samples, CIs can vary - just check they're valid
    if method == "sobol":
        assert all(df_with_var["confidence"] >= 0)
        assert all(df_standard["confidence"] >= 0)
    else:  # morris
        assert all(df_with_var["mu_star_conf"] >= 0)
        assert all(df_standard["mu_star_conf"] >= 0)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(("method", "n_samples"), [("sobol", 256), ("morris", 10)])
def test_with_variance_no_uq_emulator(xy_1d, method, n_samples):
    """Test that prediction variance is ignored for non-UQ emulators."""
    x, y = xy_1d
    rf = RandomForest(x, y)
    rf.fit(x, y)

    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    sa = SensitivityAnalysis(rf, problem=problem)

    # Should run but ignore variance (falls back to standard)
    df = sa.run(
        method=method,
        n_samples=n_samples,
        include_prediction_variance=True,
        n_bootstrap=10,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(("method", "n_samples"), [("sobol", 256), ("morris", 10)])
def test_with_variance_2d(xy_2d, method, n_samples):
    """Test sensitivity analysis with variance for multi-output."""
    x, y = xy_2d
    gp = GaussianProcess(x, y)
    gp.fit(x, y)

    problem = {
        "num_vars": 2,
        "names": ["c", "v0"],
        "bounds": [(-5.0, 1.0), (0.0, 1000.0)],
    }
    sa = SensitivityAnalysis(gp, problem=problem)

    df = sa.run(
        method=method,
        n_samples=n_samples,
        include_prediction_variance=True,
        n_bootstrap=10,
    )

    # Should have entries for both outputs
    assert len(list(df["output"].unique())) == 2
    if method == "sobol":
        assert all(df["confidence"] >= 0)
    else:  # morris
        assert all(df["mu_star_conf"] >= 0)
