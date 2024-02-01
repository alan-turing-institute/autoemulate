import logging
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoemulate.compare import AutoEmulate
from autoemulate.cross_validate import run_cv
from autoemulate.cross_validate import update_scores_df
from autoemulate.emulators import RandomForest
from autoemulate.metrics import METRIC_REGISTRY

# import make_regression_data

X, y = make_regression(n_samples=20, n_features=2, random_state=0)
cv = KFold(n_splits=5, shuffle=True)
model = Pipeline([("scaler", StandardScaler()), ("model", RandomForest())])
metrics = [metric for metric in METRIC_REGISTRY.values()]
n_jobs = 1
logger = logging.getLogger(__name__)
scores_df = pd.DataFrame(columns=["model", "metric", "fold", "score"]).astype(
    {"model": "object", "metric": "object", "fold": "int64", "score": "float64"}
)


@pytest.fixture()
def cv_results():
    return run_cv(X, y, cv, model, metrics, n_jobs, logger)


def test_cv(cv_results):
    assert isinstance(cv_results, dict)
    # check that it contains scores
    assert "test_r2" in cv_results.keys()
    assert "test_rmse" in cv_results.keys()

    assert isinstance(cv_results["test_r2"], np.ndarray)
    assert isinstance(cv_results["test_rmse"], np.ndarray)

    assert len(cv_results["test_r2"]) == 5
    assert len(cv_results["test_rmse"]) == 5


def test_update_scores_df(cv_results):
    scores_df_new = update_scores_df(scores_df, model, cv_results)
    assert isinstance(scores_df_new, pd.DataFrame)

    assert scores_df_new.shape[0] == 10
    assert scores_df_new.shape[1] == 4
    assert scores_df_new["model"][0] == "RandomForest"
