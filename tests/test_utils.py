import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.emulators import GradientBoosting
from autoemulate.emulators import RandomForest
from autoemulate.utils import _add_prefix_to_param_space
from autoemulate.utils import _add_prefix_to_single_grid
from autoemulate.utils import _adjust_param_space
from autoemulate.utils import _denormalise_y
from autoemulate.utils import _get_full_model_name
from autoemulate.utils import _get_model_names_dict
from autoemulate.utils import _normalise_y
from autoemulate.utils import get_mean_scores
from autoemulate.utils import get_model_name
from autoemulate.utils import get_model_param_space
from autoemulate.utils import get_short_model_name


# test retrieving model name ---------------------------------------------------
@pytest.fixture()
def model_name():
    return "GradientBoosting"


@pytest.fixture
def models():
    return {
        "GradientBoosting": GradientBoosting(),
        "RandomForest": RandomForest(),
    }


def test_basic_models(model_name, models):
    gb = GradientBoosting()
    assert get_model_name(gb) == model_name


# test retrieving and adjusting parameter grids ---------------------------------


@pytest.fixture
def model():
    return GradientBoosting()


# gets all the parameters of the model with the sklearn get_params() method
@pytest.fixture
def param_space(model):
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
def param_space(model):
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


# test get_model_param_space ----------------------------------------------------
def test_param_basic_random(model, param_space):
    model_grid = get_model_param_space(model, search_type="random")
    # check that all keys in model_grid are in param_space
    assert all(key in param_space.keys() for key in model_grid.keys())


def test_param_basic_bayes(model, param_space):
    model_grid = get_model_param_space(model, search_type="bayes")
    # check that all keys in model_grid are in param_space
    assert all(key in param_space.keys() for key in model_grid.keys())


def test_param_pipe(model_in_pipe, param_space):
    model_grid = get_model_param_space(model_in_pipe)
    assert all(key in param_space.keys() for key in model_grid.keys())


def test_param_pipe_scaler(model_in_pipe_w_scaler, param_space):
    model_grid = get_model_param_space(model_in_pipe_w_scaler)
    assert all(key in param_space.keys() for key in model_grid.keys())


def test_param_multiout(model_multiout, param_space):
    model_grid = get_model_param_space(model_multiout)
    assert all(key in param_space.keys() for key in model_grid.keys())


def test_param_multiout_pipe(model_multiout_pipe, param_space):
    model_grid = get_model_param_space(model_multiout_pipe)
    assert all(key in param_space.keys() for key in model_grid.keys())


# test adjust_param_space -------------------------------------------------------
def test_adj_param_basic(model, param_space):
    adjusted_param_space = _adjust_param_space(model, param_space)
    assert all(key in param_space.keys() for key in adjusted_param_space.keys())


def test_adj_param_pipe(model_in_pipe, param_space):
    adjusted_param_space = _adjust_param_space(model_in_pipe, param_space)
    assert all(key.startswith("model__") for key in adjusted_param_space.keys())


def test_adj_param_pipe_scaler(model_in_pipe_w_scaler, param_space):
    adjusted_param_space = _adjust_param_space(model_in_pipe_w_scaler, param_space)
    assert all(key.startswith("model__") for key in adjusted_param_space.keys())


def test_adj_param_multiout(model_multiout, param_space):
    adjusted_param_space = _adjust_param_space(model_multiout, param_space)
    assert all(key.startswith("estimator__") for key in adjusted_param_space.keys())


def test_adj_param_multiout_pipe(model_multiout_pipe, param_space):
    adjusted_param_space = _adjust_param_space(model_multiout_pipe, param_space)
    assert all(
        key.startswith("model__estimator__") for key in adjusted_param_space.keys()
    )


# test normalise_y and denormalise_y -------------------------------------------


def test_normalise_1d():
    y = np.array([1, 2, 3, 4, 5])
    y_norm, y_mean, y_std = _normalise_y(y)

    assert np.isclose(np.mean(y_norm), 0, atol=1e-5)
    assert np.isclose(np.std(y_norm), 1, atol=1e-5)


def test_normalise_2d():
    y = np.array([[1, 2], [3, 4], [5, 6]])
    y_norm, y_mean, y_std = _normalise_y(y)

    for i in range(y_norm.shape[1]):
        assert np.isclose(np.mean(y_norm[:, i]), 0, atol=1e-5)
        assert np.isclose(np.std(y_norm[:, i]), 1, atol=1e-5)


def test_denormalise_1d():
    y = np.array([1, 2, 3, 4, 5])
    y_norm, y_mean, y_std = _normalise_y(y)
    y_denorm = _denormalise_y(y_norm, y_mean, y_std)

    np.testing.assert_array_almost_equal(y, y_denorm)


def test_denormalise_2d():
    y = np.array([[1, 2], [3, 4], [5, 6]])
    y_norm, y_mean, y_std = _normalise_y(y)
    y_denorm = _denormalise_y(y_norm, y_mean, y_std)

    np.testing.assert_array_almost_equal(y, y_denorm)


# test add_prefix_to_param_space ------------------------------------------------


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


def test_add_prefix_to_param_space_dict(grid, prefix):
    """
    Test whether add_prefix_to_param_space correctly adds a prefix to each key
    in a parameter grid dictionary.
    """
    expected_result = {
        "prefix_param1": [1, 2, 3],
        "prefix_param2": [4, 5, 6],
        "prefix_param3": [7, 8, 9],
    }
    assert (
        _add_prefix_to_param_space(grid, prefix) == expected_result
    ), "Prefix not correctly added to param grid dict"


def test_add_prefix_to_param_space_list(grid_list, prefix):
    """
    Test whether add_prefix_to_param_space correctly adds a prefix to each key
    in a list of parameter grid dictionaries.
    """
    expected_result = [
        {"prefix_param1": [1, 2, 3], "prefix_param2": [4, 5, 6]},
        {"prefix_param3": [7, 8, 9], "prefix_param4": [10, 11, 12]},
    ]
    assert (
        _add_prefix_to_param_space(grid_list, prefix) == expected_result
    ), "Prefix not correctly added to param grid list"


# test add_prefix_to_single_grid ------------------------------------------------


def test_add_prefix_to_param_space_list_of_tuples(grid_list_of_tuples, prefix):
    """
    Test whether add_prefix_to_param_space correctly adds a prefix to each key
    in a list of parameter grid dictionaries.
    """
    expected_result = [
        ({"prefix_param1": [1, 2, 3]}, 1),
        ({"prefix_param2": [4, 5, 6]}, 1),
    ]
    assert (
        _add_prefix_to_param_space(grid_list_of_tuples, prefix) == expected_result
    ), "Prefix not correctly added to param grid list of tuples"


def test_add_prefix_to_single_grid(grid, prefix):
    expected_result = {
        "prefix_param1": [1, 2, 3],
        "prefix_param2": [4, 5, 6],
        "prefix_param3": [7, 8, 9],
    }
    assert (
        _add_prefix_to_single_grid(grid, prefix) == expected_result
    ), "Prefix not correctly added to single grid dictionary"


# mean scores -------------------------------------------------------------------
def test_get_mean_scores_r2():
    scores_df = pd.DataFrame(
        {
            "model": ["ModelA", "ModelB", "ModelA", "ModelB"],
            "short": ["ma", "mb", "ma", "mb"],
            "metric": ["r2", "r2", "r2", "r2"],
            "fold": [1, 2, 1, 2],
            "score": [0.8, 0.9, 0.7, 0.6],
        }
    )
    expected_result = pd.DataFrame(
        {"model": ["ModelA", "ModelB"], "short": ["ma", "mb"], "r2": [0.75, 0.75]}
    )
    assert get_mean_scores(scores_df, "r2").equals(expected_result)


def test_get_mean_scores_rmse():
    scores_df = pd.DataFrame(
        {
            "model": ["ModelA", "ModelB", "ModelA", "ModelB"],
            "short": ["ma", "mb", "ma", "mb"],
            "metric": ["rmse", "rmse", "rmse", "rmse"],
            "fold": [1, 2, 1, 2],
            "score": [1.0, 0.5, 0.8, 0.6],
        }
    )
    expected_result = pd.DataFrame(
        {"model": ["ModelB", "ModelA"], "short": ["mb", "ma"], "rmse": [0.55, 0.9]}
    )
    assert get_mean_scores(scores_df, "rmse").equals(expected_result)


def test_get_mean_scores_unsupported_metric():
    scores_df = pd.DataFrame(
        {
            "model": ["Model A", "Model B"],
            "metric": ["mae", "mae"],
            "fold": [1, 2],
            "score": [0.5, 0.6],
        }
    )
    with pytest.raises(RuntimeError):
        get_mean_scores(scores_df, "mae")


def test_get_mean_scores_metric_not_found():
    scores_df = pd.DataFrame(
        {
            "model": ["Model A", "Model B"],
            "metric": ["r2", "r2"],
            "fold": [1, 2],
            "score": [0.8, 0.9],
        }
    )
    with pytest.raises(ValueError):
        get_mean_scores(scores_df, "rmse")


# test model name getters ------------------------------------------------------
def test_get_model_name():
    model = RandomForest()
    assert get_model_name(model) == "RandomForest"

    model = GradientBoosting()
    assert get_model_name(model) == "GradientBoosting"


def test_get_model_name_pipeline():
    model = Pipeline([("model", RandomForest())])
    assert get_model_name(model) == "RandomForest"


def test_get_model_name_multiout():
    model = MultiOutputRegressor(RandomForest())
    assert get_model_name(model) == "RandomForest"


def test_get_model_name_pipeline_multiout():
    model = Pipeline([("model", MultiOutputRegressor(RandomForest()))])
    assert get_model_name(model) == "RandomForest"


def test_get_short_model_name():
    model = RandomForest()
    assert get_short_model_name(model) == "rf"

    model = GradientBoosting()
    assert get_short_model_name(model) == "gb"


def test__get_full_model_name():
    model_names_dict = {"GradientBoosting": "gb", "RandomForest": "rf"}
    assert _get_full_model_name("gb", model_names_dict) == "GradientBoosting"
    assert _get_full_model_name("RandomForest", model_names_dict) == "RandomForest"
    # test that it raises an error if the model name is not in the dictionary
    with pytest.raises(ValueError):
        _get_full_model_name("GaussianProcess", model_names_dict)


# test _get_model_names_dict ---------------------------------------------------


def test__get_model_names_dict():
    models = {
        "GradientBoosting": GradientBoosting(),
        "RandomForest": RandomForest(),
    }
    model_names_dict = {
        "GradientBoosting": "gb",
        "RandomForest": "rf",
    }
    assert _get_model_names_dict(models) == model_names_dict


def test__get_model_names_dict_w_subset():
    models = {
        "GradientBoosting": GradientBoosting(),
        "RandomForest": RandomForest(),
    }
    model_names_dict = {
        "GradientBoosting": "gb",
        "RandomForest": "rf",
    }
    # test that it works with short names and full names
    assert _get_model_names_dict(models, ["GradientBoosting", "rf"]) == model_names_dict
    # test that it raises an error if the model name is not in the dictionary
    with pytest.raises(ValueError):
        _get_model_names_dict(
            models, ["GradientBoosting", "RandomForest", "GaussianProcess"]
        )
