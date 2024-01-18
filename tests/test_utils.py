import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.emulators import GradientBoosting
from autoemulate.utils import add_prefix_to_param_grid
from autoemulate.utils import add_prefix_to_single_grid
from autoemulate.utils import adjust_param_grid
from autoemulate.utils import denormalise_y
from autoemulate.utils import get_model_name
from autoemulate.utils import get_model_param_grid
from autoemulate.utils import normalise_y


# test retrieving model name ---------------------------------------------------
@pytest.fixture()
def model_name():
    return "GradientBoostingRegressor"


def test_basic(model_name):
    gb = GradientBoostingRegressor()
    assert get_model_name(gb) == model_name


def test_multioutput(model_name):
    gb = MultiOutputRegressor(GradientBoostingRegressor())
    assert get_model_name(gb) == model_name


def test_pipeline(model_name):
    gb = Pipeline([("model", GradientBoostingRegressor())])
    assert get_model_name(gb) == model_name


def test_pipeline_multioutput():
    gb = Pipeline([("model", MultiOutputRegressor(GradientBoostingRegressor()))])
    assert get_model_name(gb) == "GradientBoostingRegressor"


def test_pipeline_with_scaler(model_name):
    gb = Pipeline(
        [("scaler", StandardScaler()), ("model", GradientBoostingRegressor())]
    )
    assert get_model_name(gb) == model_name


# test retrieving and adjusting parameter grids ---------------------------------


@pytest.fixture
def model():
    return GradientBoosting()


# gets all the parameters of the model with the sklearn get_params() method
@pytest.fixture
def param_grid(model):
    return model.get_params()


@pytest.fixture
def model_in_pipeline(model):
    return Pipeline([("model", model)])


@pytest.fixture
def model_in_pipe_w_scaler(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


@pytest.fixture
def model_multiout(model):
    return MultiOutputRegressor(model)


@pytest.fixture
def model_multiout_pipe(model):
    return Pipeline([("model", MultiOutputRegressor(model))])


@pytest.fixture
def model():
    return GradientBoosting()


# gets all the parameters of the model with the sklearn get_params() method
@pytest.fixture
def param_grid(model):
    return model.get_params()


@pytest.fixture
def model_in_pipe(model):
    return Pipeline([("model", model)])


@pytest.fixture
def model_in_pipe_w_scaler(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


@pytest.fixture
def model_multiout(model):
    return MultiOutputRegressor(model)


@pytest.fixture
def model_multiout_pipe(model):
    return Pipeline([("model", MultiOutputRegressor(model))])


# test get_model_param_grid ----------------------------------------------------
def test_param_basic_random(model, param_grid):
    model_grid = get_model_param_grid(model, search_type="random")
    # check that all keys in model_grid are in param_grid
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_basic_bayes(model, param_grid):
    model_grid = get_model_param_grid(model, search_type="bayes")
    # check that all keys in model_grid are in param_grid
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_pipe(model_in_pipe, param_grid):
    model_grid = get_model_param_grid(model_in_pipe)
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_pipe_scaler(model_in_pipe_w_scaler, param_grid):
    model_grid = get_model_param_grid(model_in_pipe_w_scaler)
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_multiout(model_multiout, param_grid):
    model_grid = get_model_param_grid(model_multiout)
    assert all(key in param_grid.keys() for key in model_grid.keys())


def test_param_multiout_pipe(model_multiout_pipe, param_grid):
    model_grid = get_model_param_grid(model_multiout_pipe)
    assert all(key in param_grid.keys() for key in model_grid.keys())


# test adjust_param_grid -------------------------------------------------------
def test_adj_param_basic(model, param_grid):
    adjusted_param_grid = adjust_param_grid(model, param_grid)
    assert all(key in param_grid.keys() for key in adjusted_param_grid.keys())


def test_adj_param_pipe(model_in_pipe, param_grid):
    adjusted_param_grid = adjust_param_grid(model_in_pipe, param_grid)
    assert all(key.startswith("model__") for key in adjusted_param_grid.keys())


def test_adj_param_pipe_scaler(model_in_pipe_w_scaler, param_grid):
    adjusted_param_grid = adjust_param_grid(model_in_pipe_w_scaler, param_grid)
    assert all(key.startswith("model__") for key in adjusted_param_grid.keys())


def test_adj_param_multiout(model_multiout, param_grid):
    adjusted_param_grid = adjust_param_grid(model_multiout, param_grid)
    assert all(key.startswith("estimator__") for key in adjusted_param_grid.keys())


def test_adj_param_multiout_pipe(model_multiout_pipe, param_grid):
    adjusted_param_grid = adjust_param_grid(model_multiout_pipe, param_grid)
    assert all(
        key.startswith("model__estimator__") for key in adjusted_param_grid.keys()
    )


# test normalise_y and denormalise_y -------------------------------------------


def test_normalise_1d():
    y = np.array([1, 2, 3, 4, 5])
    y_norm, y_mean, y_std = normalise_y(y)

    assert np.isclose(np.mean(y_norm), 0, atol=1e-5)
    assert np.isclose(np.std(y_norm), 1, atol=1e-5)


def test_normalise_2d():
    y = np.array([[1, 2], [3, 4], [5, 6]])
    y_norm, y_mean, y_std = normalise_y(y)

    for i in range(y_norm.shape[1]):
        assert np.isclose(np.mean(y_norm[:, i]), 0, atol=1e-5)
        assert np.isclose(np.std(y_norm[:, i]), 1, atol=1e-5)


def test_denormalise_1d():
    y = np.array([1, 2, 3, 4, 5])
    y_norm, y_mean, y_std = normalise_y(y)
    y_denorm = denormalise_y(y_norm, y_mean, y_std)

    np.testing.assert_array_almost_equal(y, y_denorm)


def test_denormalise_2d():
    y = np.array([[1, 2], [3, 4], [5, 6]])
    y_norm, y_mean, y_std = normalise_y(y)
    y_denorm = denormalise_y(y_norm, y_mean, y_std)

    np.testing.assert_array_almost_equal(y, y_denorm)


# test add_prefix_to_param_grid ------------------------------------------------


@pytest.fixture
def grid():
    return {"param1": [1, 2, 3], "param2": [4, 5, 6], "param3": [7, 8, 9]}


@pytest.fixture
def grid_list():
    return [
        {"param1": [1, 2, 3], "param2": [4, 5, 6]},
        {"param3": [7, 8, 9], "param4": [10, 11, 12]},
    ]


@pytest.fixture
def grid_list_of_tuples():
    return [
        ({"param1": [1, 2, 3]}, 1),
        ({"param2": [4, 5, 6]}, 1),
    ]


@pytest.fixture
def prefix():
    return "prefix_"


def test_add_prefix_to_param_grid_dict(grid, prefix):
    """
    Test whether add_prefix_to_param_grid correctly adds a prefix to each key
    in a parameter grid dictionary.
    """
    expected_result = {
        "prefix_param1": [1, 2, 3],
        "prefix_param2": [4, 5, 6],
        "prefix_param3": [7, 8, 9],
    }
    assert (
        add_prefix_to_param_grid(grid, prefix) == expected_result
    ), "Prefix not correctly added to param grid dict"


def test_add_prefix_to_param_grid_list(grid_list, prefix):
    """
    Test whether add_prefix_to_param_grid correctly adds a prefix to each key
    in a list of parameter grid dictionaries.
    """
    expected_result = [
        {"prefix_param1": [1, 2, 3], "prefix_param2": [4, 5, 6]},
        {"prefix_param3": [7, 8, 9], "prefix_param4": [10, 11, 12]},
    ]
    assert (
        add_prefix_to_param_grid(grid_list, prefix) == expected_result
    ), "Prefix not correctly added to param grid list"  # test add_prefix_to_single_grid ------------------------------------------------


def test_add_prefix_to_param_grid_list_of_tuples(grid_list_of_tuples, prefix):
    """
    Test whether add_prefix_to_param_grid correctly adds a prefix to each key
    in a list of parameter grid dictionaries.
    """
    expected_result = [
        ({"prefix_param1": [1, 2, 3]}, 1),
        ({"prefix_param2": [4, 5, 6]}, 1),
    ]
    assert (
        add_prefix_to_param_grid(grid_list_of_tuples, prefix) == expected_result
    ), "Prefix not correctly added to param grid list of tuples"


def test_add_prefix_to_single_grid(grid, prefix):
    expected_result = {
        "prefix_param1": [1, 2, 3],
        "prefix_param2": [4, 5, 6],
        "prefix_param3": [7, 8, 9],
    }
    assert (
        add_prefix_to_single_grid(grid, prefix) == expected_result
    ), "Prefix not correctly added to single grid dictionary"
