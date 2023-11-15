import pytest
from autoemulate.hyperparam_search import HyperparamSearch
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
def rf_pipeline():
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
    return HyperparamSearch(X, y, cv=3, n_jobs=1, logger=logging.getLogger(__name__))


# test prepare_param_grid
def test_prepare_param_grid(rf_pipeline, param_grid, hyperparam_search):
    param_grid = hyperparam_search.prepare_param_grid(rf_pipeline)
    # check that the parameter grid is prefixed with "model__"
    assert all([key.startswith("model__") for key in param_grid.keys()])


# test
def test_search(rf_pipeline, hyperparam_search):
    # check that the best_params attribute is empty before search
    assert hyperparam_search.best_params == {}
    # check that the best_params attribute is populated after search
    hyperparam_search.search(rf_pipeline)
    assert hyperparam_search.best_params != {}
    # # check that the best_params attribute is a dictionary
    assert type(hyperparam_search.best_params) == dict
    # # check that best param keys are strings
    assert all([type(key) == str for key in hyperparam_search.best_params.keys()])
