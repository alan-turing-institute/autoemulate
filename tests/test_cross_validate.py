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
from autoemulate.cross_validate import _get_cv_results
from autoemulate.cross_validate import _get_cv_sum
from autoemulate.cross_validate import _get_mean_scores
from autoemulate.cross_validate import _get_model_scores
from autoemulate.cross_validate import _run_cv
from autoemulate.cross_validate import _update_scores_df
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


def test_get_cv_sum(cv_result, metrics):
    print(cv_result)
    print(_get_cv_sum(cv_result))
    cv_sum = _get_cv_sum(cv_result)
    assert isinstance(cv_sum, pd.DataFrame)
    assert cv_sum.shape[1] == 3
    # check that all metrics are present
    assert all(metric.__name__ in cv_sum.columns for metric in metrics)
    # check that metrics are numeric
    assert pd.api.types.is_numeric_dtype(cv_sum.iloc[:, 1]), "Column 1 is not numeric"
    assert pd.api.types.is_numeric_dtype(cv_sum.iloc[:, 2]), "Column 2 is not numeric"


def test_update_scores_df(Xy, cv, model, metrics, scores_df, model_name):
    X, y = Xy
    _, cv_results = _run_cv(X, y, cv, model, metrics)
    scores_df_updated = _update_scores_df(scores_df, model_name, cv_results)
    assert isinstance(scores_df_updated, pd.DataFrame)

    # 5 columns: model, short, metric, fold, score
    assert scores_df_updated.shape[1] == 4
    assert all(scores_df_updated["model"] == "rf")
    # check that all metrics are present
    assert set(scores_df_updated["metric"]) == set(
        [metric.__name__ for metric in metrics]
    )
    # check that score is numeric
    assert pd.api.types.is_numeric_dtype(scores_df_updated["score"])
    # check that fold contains all values from 0 to 4
    assert set(scores_df_updated["fold"]) == set(range(5))


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
    assert _get_mean_scores(scores_df, "r2").equals(expected_result)


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
    assert _get_mean_scores(scores_df, "rmse").equals(expected_result)


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
        _get_mean_scores(scores_df, "mae")


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
        _get_mean_scores(scores_df, "rmse")


# test _get_model_scores ------------------------------------------------------
def test_get_model_scores():
    scores_df = pd.DataFrame(
        {
            "model": ["Model A", "Model B"],
            "metric": ["r2", "r2"],
            "fold": [1, 2],
            "score": [0.8, 0.9],
        }
    )
