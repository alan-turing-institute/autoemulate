import numpy as np
import pandas as pd
import pytest
import torch

from autoemulate.compare import AutoEmulate
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.utils import get_model_name


@pytest.fixture()
def ae():
    return AutoEmulate()


@pytest.fixture()
def Xy():
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    return X, y


@pytest.fixture()
def Xy_multioutput():
    X = np.random.rand(20, 5)
    y = np.random.rand(20, 2)
    return X, y


@pytest.fixture()
def ae_run(ae, Xy):
    X, y = Xy
    ae.setup(X, y)
    ae.compare()
    return ae


@pytest.fixture()
def ae_run_multioutput(ae, Xy_multioutput):
    X, y = Xy_multioutput
    ae.setup(X, y)
    ae.compare()
    return ae


def test_setup_data(ae, Xy):
    X, y = Xy
    ae.setup(X, y)
    # assert that auto_emulate.X has nearly same values as X (but dtype can be different)
    assert np.allclose(ae.X, X)
    assert np.allclose(ae.y, y)


def test_error_if_not_setup():
    ae = AutoEmulate()
    # should raise runtime error if compare is called before setup
    with pytest.raises(RuntimeError):
        ae.compare()


# -----------------------test _check_input-----------------------------#
# test whether different inputs are correctly converted to numpy arrays
data_types = [np.array, list, pd.DataFrame, torch.tensor]
data = ([1, 2], [3, 4], [5, 6]), ([7, 8], [9, 10], [11, 12])


@pytest.mark.parametrize(
    "input_X, input_y", [(dt(data[0]), dt(data[1])) for dt in data_types]
)
def test__check_input(ae, input_X, input_y):
    X_converted, y_converted = ae._check_input(input_X, input_y)
    # test that X and y are numpy arrays
    assert isinstance(X_converted, np.ndarray)
    assert isinstance(y_converted, np.ndarray)
    # test that X and y have correct shapes
    assert X_converted.shape == (3, 2)
    assert y_converted.shape == (3, 2)
    # test that X and y have correct dtypes
    assert X_converted.dtype == np.float32
    assert y_converted.dtype == np.float32


# test whether X and y with different shapes raise an error
def test__check_input_errors_with_different_shapes(ae):
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6]])
    with pytest.raises(ValueError):
        ae._check_input(X, y)


# test whether X and y with different dtypes raise an error
def test__check_input_errors_with_different_dtypes(ae):
    X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5], [6]], dtype=np.int32)
    X_converted, y_converted = ae._check_input(X, y)
    assert X_converted.dtype == np.float32
    assert y_converted.dtype == np.float32


# test whether NA values in X and y raise an error
def test__check_input_errors_with_NA_values_in_y(ae):
    X = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [np.nan, 8]], dtype=np.float32)
    with pytest.raises(ValueError):
        ae._check_input(X, y)


#
def test__check_input_errors_with_NA_values_in_X(ae):
    X = np.array([[1, 2], [np.nan, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    with pytest.raises(ValueError):
        ae._check_input(X, y)


# -----------------------test _get_metric-------------------#
def test__get_metrics(ae):
    metrics = ae._get_metrics(METRIC_REGISTRY)
    assert isinstance(metrics, list)
    metric_names = [metric.__name__ for metric in metrics]
    # check all metric names are in METRIC_REGISTRY
    assert all([metric_name in METRIC_REGISTRY for metric_name in metric_names])


# -----------------------test get_model-------------------#
def test_get_model_by_name(ae_run):
    model = ae_run.get_model(name="RandomForest")
    assert get_model_name(model) == "RandomForest"


def test_get_model_by_short_name(ae_run):
    model = ae_run.get_model(name="rf")
    print("intest", model)
    assert get_model_name(model) == "RandomForest"


def test_get_model_by_invalid_name(ae_run):
    with pytest.raises(ValueError):
        ae_run.get_model(name="invalid_name")


def test_get_model_by_rank(ae_run):
    model = ae_run.get_model(rank=1)
    assert model is not None


def test_get_model_with_invalid_rank(ae_run):
    with pytest.raises(RuntimeError):
        ae_run.get_model(rank=0)


def test_get_model_before_compare(ae):
    # Test getting a model before running compare
    with pytest.raises(RuntimeError):
        ae.get_model()


def test_get_model_with_invalid_metric(ae_run):
    # Test getting a model with an invalid metric
    with pytest.raises(ValueError):
        ae_run.get_model(metric="invalid_metric")


# -----------------------test evaluate-------------------#


def test_evaluate_singleoutput(ae_run):
    model = ae_run.get_model(rank=1)
    scores_df = ae_run.evaluate(model=model, multioutput="uniform_average")
    expected_cols = {"model", "short", "preprocessing"}.union(
        {metric.__name__ for metric in ae_run.metrics}
    )
    assert set(scores_df.columns) == expected_cols
    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape == (
        1,
        len(ae_run.metrics) + 3,
    )  # 3 columns: model, short, target
    assert all(metric.__name__ in scores_df.columns for metric in ae_run.metrics)


def test_evaluate_multioutput(ae_run_multioutput):
    model = ae_run_multioutput.get_model(rank=1)
    scores_df = ae_run_multioutput.evaluate(model=model, multioutput="uniform_average")
    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape == (1, len(ae_run_multioutput.metrics) + 3)


def test_evaluate_singleoutput_raw(ae_run):
    model = ae_run.get_model(rank=1)
    scores_df = ae_run.evaluate(model=model, multioutput="raw_values")
    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape == (1, len(ae_run.metrics) + 4)


def test_evaluate_multioutput_raw(ae_run_multioutput):
    model = ae_run_multioutput.get_model(rank=1)
    scores_df = ae_run_multioutput.evaluate(model=model, multioutput="raw_values")
    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape == (2, len(ae_run_multioutput.metrics) + 4)


"""
The method has a default model=None parameter
If no model is provided, it tries to use self.best_model
Only raises ValueError if self.best_model doesn't exist and no model was provided

def test_score_without_model(ae_run):
    with pytest.raises(ValueError):
        ae_run.evaluate()
"""


# -----------------------test refit-------------------#


def test_refit(ae_run):
    model = ae_run.get_model(rank=1)
    refitted_model = ae_run.refit(model)
    assert refitted_model is not None


# --------------- test correct hyperparameter updating ------------------
def test_param_search_updates_models(ae, Xy):
    X, y = Xy

    ae.setup(X, y, models=["RandomForest"], param_search=True, param_search_iters=1)

    # Get initial parameters from the pipeline's model
    params_before = ae.ae_pipeline.models_piped[0].regressor.steps[-1][1].get_params()

    ae.compare()

    # Get parameters from the best model after comparison
    best_model = ae.preprocessing_results["None"]["best_model"]

    # Handle both pipeline structures
    if hasattr(best_model, "regressor"):  # TransformedTargetRegressor case
        params_after = best_model.regressor.steps[-1][1].get_params()
    else:  # Direct pipeline case
        params_after = best_model.steps[-1][1].get_params()

    # Verify parameters changed (since param_search=True)
    assert params_before != params_after


def test_model_params_equal_wo_param_search(ae, Xy):
    X, y = Xy

    # Setup without parameter search and with output scaling disabled
    ae.setup(
        X,
        y,
        models=["RandomForest"],
        param_search=False,
        scale_output=False,  # Disable output scaling
        scaler_output=None,  # Explicitly no scaler
    )

    # Get initial parameters directly from the model (not pipeline)
    initial_model = ae.ae_pipeline.models_piped[0]
    if hasattr(initial_model, "regressor"):  # If wrapped in a pipeline
        initial_params = initial_model.regressor.get_params()
    else:
        initial_params = initial_model.get_params()

    ae.compare()

    # Get final parameters
    final_model = ae.preprocessing_results["None"]["models"][0]
    if hasattr(final_model, "regressor"):  # If wrapped in a pipeline
        final_params = final_model.regressor.get_params()
    else:
        final_params = final_model.get_params()

    # Filter out sklearn internal parameters and random states
    skip_params = {"random_state", "n_jobs", "verbose"}  # Add others as needed
    filtered_before = {
        k: v
        for k, v in initial_params.items()
        if not (k.endswith("_") or k in skip_params)
    }
    filtered_after = {
        k: v
        for k, v in final_params.items()
        if not (k.endswith("_") or k in skip_params)
    }

    # Compare only the parameters that should remain constant
    for param in filtered_before:
        assert param in filtered_after, f"Parameter {param} missing after compare"
        assert filtered_before[param] == filtered_after[param], (
            f"Parameter {param} changed unexpectedly\n"
            f"Before: {filtered_before[param]}\n"
            f"After: {filtered_after[param]}"
        )


# -----------------------test summarize_cv-------------------#
def test_cv_summary_all_models(ae_run):
    """Test that summarize_cv returns aggregated results for all models."""
    summary = ae_run.summarize_cv()

    assert isinstance(summary, pd.DataFrame)

    # Get the unique model names from the summary (accounting for preprocessing)
    summary_model_names = summary["model"].unique()

    # Get the expected model names from model_registry
    expected_model_names = set(ae_run.model_names.keys())

    # Verify all expected models are present in the results
    assert (
        set(summary_model_names) == expected_model_names
    ), f"Expected models {expected_model_names}, got {set(summary_model_names)}"


def test_setup_preprocessing_methods(ae, Xy):
    """Test setup with custom preprocessing methods."""
    X, y = Xy
    preprocessing_methods = [
        {"name": "None", "params": {}},
        {"name": "PCA", "params": {"reduced_dim": 1}},
    ]

    ae.setup(
        X, y, preprocessing_methods=preprocessing_methods, verbose=0, print_setup=False
    )

    assert ae.preprocessing_methods == preprocessing_methods


def test_compare_with_preprocessing(ae, Xy):
    """Test model comparison with preprocessing methods."""
    X, y = Xy

    preprocessing_methods = [
        {"name": "None", "params": {}},
        {
            "name": "PCA",
            "params": {"reduced_dim": 1},
        },  # Changed from reduced_dim to n_components
    ]

    # Setup with reduced configuration for testing
    ae.setup(
        X=X,
        y=y,
        models=["RandomForest"],
        preprocessing_methods=preprocessing_methods,
        param_search=False,  # Disable for faster testing
        scale_output=False,  # Disable output scaling
        verbose=0,
        print_setup=False,
    )

    best_combo = ae.compare()

    # Basic type and structure checks
    assert isinstance(best_combo, dict)
    assert set(best_combo.keys()) == {"preprocessing", "model", "transformer"}

    # Verify preprocessing method was applied
    assert best_combo["preprocessing"] in ["None", "PCA"]
    assert hasattr(ae, "preprocessing_results")

    # Check both preprocessing methods were processed
    assert "None" in ae.preprocessing_results
    assert "PCA" in ae.preprocessing_results

    # Verify models exist for each preprocessing method
    assert len(ae.preprocessing_results["None"]["models"]) == 1
    assert len(ae.preprocessing_results["PCA"]["models"]) == 1

    # Additional check for PCA transformer
    if best_combo["preprocessing"] == "PCA":
        assert ae.preprocessing_results["PCA"]["transformer"] is not None
        assert hasattr(ae.preprocessing_results["PCA"]["transformer"], "transform")
