import logging
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.compare import AutoEmulate
from autoemulate.cross_validate import _run_cv
from autoemulate.cross_validate import _sum_cv
from autoemulate.cross_validate import _sum_cvs
from autoemulate.data_splitting import _split_data
from autoemulate.emulators import RandomForest
from autoemulate.metrics import METRIC_REGISTRY


@pytest.fixture(scope="module")
def Xy():
    X, y = make_regression(n_samples=20, n_features=2, random_state=0)
    return X, y


@pytest.fixture(scope="module")
def model():
    model = RandomForest()
    return model


@pytest.fixture(scope="module")
def metrics():
    return [metric for metric in METRIC_REGISTRY.values()]


@pytest.fixture(scope="module")
def scorers(metrics):
    return {metric.__name__: make_scorer(metric) for metric in metrics}


@pytest.fixture(scope="module")
def cv():
    cv = KFold(n_splits=5, shuffle=True)
    return cv


@pytest.fixture()
def scores_df():
    scores_df = pd.DataFrame(columns=["model", "metric", "fold", "score"]).astype(
        {"model": "object", "metric": "object", "fold": "int64", "score": "float64"}
    )
    return scores_df


@pytest.fixture()
def model_name():
    return "rf"


@pytest.fixture()
def cv_result(Xy, model, cv, scorers):
    X, y = Xy
    cv_result = cross_validate(
        model, X, y, scoring=scorers, cv=cv, return_estimator=True, return_indices=True
    )
    return cv_result


@pytest.fixture()
def cv_results(Xy):
    X, y = Xy
    em = AutoEmulate()
    em.setup(X, y, models=["rbf", "rf"])
    em.compare()
    return em.preprocessing_results['None']["cv_results"]


def test_run_cv(Xy, cv, metrics, model):
    X, y = Xy
    _run_cv(X, y, cv, model, metrics)


def test_cv_results(Xy, cv, metrics, model):
    X, y = Xy
    _, cv_results = _run_cv(X, y, cv, model, metrics)

    assert isinstance(cv_results, dict)

    assert "test_r2" in cv_results.keys()
    assert "test_rmse" in cv_results.keys()

    assert isinstance(cv_results["test_r2"], np.ndarray)
    assert isinstance(cv_results["test_rmse"], np.ndarray)

    assert len(cv_results["test_r2"]) == 5
    assert len(cv_results["test_rmse"]) == 5


def test_fitted_model(Xy, cv, model, metrics):
    X, y = Xy
    fitted_model, _ = _run_cv(X, y, cv, model, metrics)
    assert isinstance(fitted_model, BaseEstimator)
    # check that score does not raise an error (i.e. model is fitted)
    fitted_model.score(X, y)


def test_sum_cv(cv_result, metrics):
    cv_sum = _sum_cv(cv_result)
    assert isinstance(cv_sum, pd.DataFrame)
    assert cv_sum.shape[1] == 3
    # check that all metrics are present
    assert all(metric.__name__ in cv_sum.columns for metric in metrics)
    # check that metrics are numeric
    assert pd.api.types.is_numeric_dtype(cv_sum.iloc[:, 1]), "Column 1 is not numeric"
    assert pd.api.types.is_numeric_dtype(cv_sum.iloc[:, 2]), "Column 2 is not numeric"


def test_sum_cvs(cv_results):
    cv_all = _sum_cvs(cv_results)
    assert isinstance(cv_all, pd.DataFrame)
    assert all(col in cv_all.columns for col in ["model", "short", "r2", "rmse"])
    assert pd.api.types.is_numeric_dtype(cv_all["r2"]), "r2 column is not numeric"
    assert pd.api.types.is_numeric_dtype(cv_all["rmse"]), "rmse column is not numeric"
    assert cv_all.shape[0] == 2, "Should be one row per model"


def test_sum_cvs_invalid_sort_by(cv_results):
    with pytest.raises(ValueError):
        _sum_cvs(cv_results, sort_by="invalid_metric")


def test_sum_cvs_sorted_correctly(cv_results):
    cv_all = _sum_cvs(cv_results, sort_by="r2")
    # check that the r2 column is sorted in descending order
    assert np.all(np.diff(cv_all["r2"]) <= 0)
    # check that the rmse column is sorted in ascending order
    assert np.all(np.diff(cv_all["rmse"]) >= 0)
