import logging

import pytest
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline

from autoemulate.emulators import GaussianProcessSklearn
from autoemulate.emulators import RandomForest
from autoemulate.hyperparam_searching import _check_param_space
from autoemulate.hyperparam_searching import _optimize_params
from autoemulate.hyperparam_searching import _process_param_space


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
def param_space():
    return {
        "n_estimators": [10, 20],
        "max_depth": [None, 3],
    }


def test_optimize_params(Xy, model):
    X, y = Xy
    best_estimator = _optimize_params(
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


def test_process_param_space_none(model, param_space):
    search_type = "random"
    param_space = _process_param_space(model, search_type, param_space=None)
    # check that param_space has been populated
    assert type(param_space) == dict


def test_process_param_space(model, param_space):
    search_type = "random"
    param_space = _process_param_space(model, search_type, param_space)
    # check that param_space has been populated
    assert type(param_space) == dict


def test_process_param_space_invalid(model, param_space):
    search_type = "random"
    param_space = {"invalid_param": [1, 2]}
    with pytest.raises(ValueError):
        param_space = _process_param_space(model, search_type, param_space)


def test_check_param_space(param_space, model):
    # param_space should be a dictionary
    with pytest.raises(TypeError):
        _check_param_space(model, [])
    # keys in param_space should be strings
    with pytest.raises(TypeError):
        _check_param_space({1: []}, model)
    # values in param_space should be lists
    with pytest.raises(TypeError):
        _check_param_space({"n_estimators": 10}, model)
    #     # model__ prefixed keys should be actual parameters in the model
    with pytest.raises(ValueError):
        _check_param_space({"model__invalid_param": [1, 2]}, model)
    # check param_space returned if valid
    assert _check_param_space(param_space, model) == param_space
