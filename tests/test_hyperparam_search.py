import pytest
from autoemulate.hyperparam_search import HyperparamSearcher
from autoemulate.emulators import RandomForest
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging


@pytest.fixture
def X_y():
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


# fixture for hyperparameter search object
@pytest.fixture
def hyperparam_search(X_y):
    X, y = X_y
    return HyperparamSearcher(X, y, cv=3, n_jobs=1, logger=logging.getLogger(__name__))


def test_check_param_grid(hyperparam_search, model, param_grid):
    # param_grid should be a dictionary
    with pytest.raises(TypeError):
        hyperparam_search.check_param_grid([], model)
    # keys in param_grid should be strings
    with pytest.raises(TypeError):
        hyperparam_search.check_param_grid({1: []}, model)
    # values in param_grid should be lists
    with pytest.raises(TypeError):
        hyperparam_search.check_param_grid({"n_estimators": 10}, model)
    # model__ prefixed keys should be actual parameters in the model
    with pytest.raises(ValueError):
        hyperparam_search.check_param_grid({"model__invalid_param": [1, 2]}, model)
    # check param_grid returned if valid
    assert hyperparam_search.check_param_grid(param_grid, model) == param_grid


def test_search(model, hyperparam_search):
    # check that the best_params attribute is populated after search
    best_params = hyperparam_search.search(model)
    assert best_params != {}
    # # check that the best_params attribute is a dictionary
    assert type(best_params) == dict
    # # check that best param keys are strings
    assert all([type(key) == str for key in best_params.keys()])
