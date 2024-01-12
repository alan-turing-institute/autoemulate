import pytest
from autoemulate.hyperparam_searching import (
    check_param_grid,
    optimize_params,
    process_param_grid,
)
from autoemulate.emulators import RandomForest
from autoemulate.utils import get_model_name

from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
import logging


@pytest.fixture
def Xy():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    return X, y


# fixture for random forest model in pipeline
@pytest.fixture
def model():
    return Pipeline([("model", RandomForest())])


# param grid for random forest
@pytest.fixture
def param_grid():
    return {
        "n_estimators": [10, 20],
        "max_depth": [None, 3],
    }


def test_optimize_params(Xy, model):
    X, y = Xy
    #    print(f"this is the model name: {get_model_name(model)}")
    best_estimator = optimize_params(
        X,
        y,
        cv=3,
        model=model,
        search_type="random",
        niter=3,
        logger=logging.getLogger(__name__),
    )
    assert best_estimator is not None
    assert type(best_estimator) == Pipeline
    assert best_estimator.named_steps["model"].is_fitted_ == True


def test_process_param_grid_none(model, param_grid):
    search_type = "random"
    param_grid = process_param_grid(model, search_type, param_grid=None)
    # check that param_grid has been populated
    assert type(param_grid) == dict


def test_process_param_grid(model, param_grid):
    search_type = "random"
    param_grid = process_param_grid(model, search_type, param_grid)
    # check that param_grid has been populated
    assert type(param_grid) == dict


def test_process_param_grid_invalid(model, param_grid):
    search_type = "random"
    param_grid = {"invalid_param": [1, 2]}
    with pytest.raises(ValueError):
        param_grid = process_param_grid(model, search_type, param_grid)


def test_check_param_grid(param_grid, model):
    # param_grid should be a dictionary
    with pytest.raises(TypeError):
        check_param_grid(model, [])
    # keys in param_grid should be strings
    with pytest.raises(TypeError):
        check_param_grid({1: []}, model)
    # values in param_grid should be lists
    with pytest.raises(TypeError):
        check_param_grid({"n_estimators": 10}, model)
    #     # model__ prefixed keys should be actual parameters in the model
    with pytest.raises(ValueError):
        check_param_grid({"model__invalid_param": [1, 2]}, model)
    # check param_grid returned if valid
    assert check_param_grid(param_grid, model) == param_grid
