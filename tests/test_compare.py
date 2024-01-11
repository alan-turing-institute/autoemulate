import pytest
import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.experimental_design import ExperimentalDesign, LatinHypercube
from autoemulate.emulators import GaussianProcess, RandomForest
from autoemulate.compare import AutoEmulate
from autoemulate.metrics import METRIC_REGISTRY
from autoemulate.emulators import MODEL_REGISTRY
from autoemulate.cv import CV_REGISTRY


@pytest.fixture()
def ae():
    return AutoEmulate()


@pytest.fixture()
def Xy():
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    return X, y


@pytest.fixture()
def ae_run(ae, Xy):
    X, y = Xy
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


# -----------------------test _update_scores_df-----------------------------#
def test__update_scores_df(ae_run):
    # Check that scores_df is not empty after running compare
    assert not ae_run.scores_df.empty
    # Check that scores_df has the expected columns
    assert ae_run.scores_df.columns.tolist() == ["model", "metric", "fold", "score"]
    # # Check that scores_df has the expected number of rows
    assert (
        len(ae_run.scores_df)
        == len(ae_run.models) * len(METRIC_REGISTRY) * ae_run.cv.n_splits
    )
    # Check that all scores are floats
    assert ae_run.scores_df["score"].dtype == np.float64


# -----------------------test _print_results-----------------------------#
def test_print_results_all_models(ae_run, capsys):
    ae_run.print_results()
    captured = capsys.readouterr()
    assert "Average scores across all models:" in captured.out
    assert "model" in captured.out
    assert "metric" in captured.out


def test_print_results_single_model(ae_run, capsys):
    ae_run.print_results("GaussianProcessSk")
    captured = capsys.readouterr()
    assert "Scores for GaussianProcessSk across all folds:" in captured.out
    assert "fold" in captured.out
    assert "metric" in captured.out


def test_print_results_invalid_model(ae_run):
    with pytest.raises(ValueError):
        ae_run.print_results("InvalidModel")
